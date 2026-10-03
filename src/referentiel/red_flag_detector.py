"""
red_flag_detector.py

Module de détection déterministe de red flags cliniques, basé sur le
référentiel FRENCH (SFMU, V1.1, juin 2018).

Implémente l'Étape 2 (détection déterministe) et le mécanisme de croisement
avec l'Étape 3 (labellisation LLM) décrits dans le pipeline de génération
du dataset SFT/DPO du POC CHSA.

IMPORTANT : ce module s'appuie sur un sous-ensemble NON EXHAUSTIF du
référentiel FRENCH (cf. french_referentiel.json). Il doit être complété
et validé par un professionnel de santé avant tout usage en conditions
réelles. Il est conçu pour un usage POC / génération de données, pas pour
une décision clinique réelle.

--- Ce qui a changé au lot 1 (sept. 2026) ---
1. DEUX PLANCHERS SÉPARÉS :
   - plancher des CONSTANTES VITALES (et règle absolue du nourrisson fébrile) :
     INVIOLABLE. Le label final ne peut jamais être moins urgent.
   - plancher du MOTIF (tri médian FRENCH ajusté par les modulateurs) :
     RÉVISABLE. Il ne force plus le label ; si l'Étape 3 descend en dessous,
     le cas est marqué "sous_plancher_motif" pour relecture humaine.
2. MODULATEURS À LA BAISSE : comme dans FRENCH (tri a posteriori "au-dessus ou
   en dessous du tri a priori"), un modulateur déclenché remplace le tri de
   base, même s'il est moins urgent. Chaque application à la baisse est
   listée dans `modulateurs_a_la_baisse` (à valider par un urgentiste).
3. CONTRÔLE DE PRÉSENCE : si le texte de la vignette est fourni
   (`texte_source`), une constante vitale n'est retenue que si son nombre
   figure dans ce texte. Sinon elle est ignorée (une valeur inventée par le
   LLM ne peut plus forcer un plancher inviolable) et listée dans
   `constantes_non_verifiees`.
4. Glycémie : signalée seulement au-dessus de 20 mmol/l (sans effet sur le
   niveau : la cétose n'est pas modélisée).
5. Pédiatrie (< 15 ans) : avant, AUCUNE constante n'était contrôlée chez l'enfant.
   Maintenant : SpO2 et GCS avec les seuils adulte ; PAS / FC / FR avec les seuils
   par tranche d'âge du référentiel (tri 2, à valider par un pédiatre) ; bradycardie
   (FC <= 80 avant 1 an, <= 60 après) d'après le motif FRENCH correspondant.
Le shock index n'est volontairement PAS implémenté (cf. french_referentiel.json,
meta.validation.non_implemente_volontairement).
"""

import json
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


ORDRE_URGENCE = {"1": 1, "2": 2, "3A": 3, "3B": 4, "4": 5, "5": 6}

MAPPING_CATEGORIES_BRIEF = {
    "1": "urgence_maximale",
    "2": "urgence_maximale",
    "3A": "moderee",
    "3B": "moderee",
    "4": "differee",
    "5": "differee",
}

ORDRE_CATEGORIE_BRIEF = {"urgence_maximale": 1, "moderee": 2, "differee": 3}

# Seuils pédiatriques par tranche d'âge, recopiés de
# french_referentiel.json > constantes_vitales_pediatrie.tranches_age (un test
# vérifie que les deux restent identiques). Chaque tuple :
#   (âge maximal EXCLU en années, PAS hypotension si <, FC tachycardie si >, FR polypnée si >)
# Aux bornes (1 mois, 2 ans, 10 ans), l'enfant bascule dans la tranche la plus
# âgée, dont les seuils sont les plus sensibles : choix prudent.
TRANCHES_PEDIATRIQUES = [
    (1 / 12, 50, 180, 60),        # < 1 mois
    (2, 65, 160, 40),             # 1 mois - 2 ans
    (10, 70, 130, 30),            # 2 - 10 ans
    (float("inf"), 80, 120, 20),  # > 10 ans (jusqu'à 15 ans, seuil de l'âge pédiatrique)
]
# Le référentiel ne donne pas de niveau de tri pour ces seuils : tri 2 retenu
# (comme la tachycardie 130-180 de l'adulte). À valider par un pédiatre.
TRI_CONSTANTE_PEDIATRIQUE = "2"


def _tranche_pediatrique(age_annees: float) -> tuple:
    """Seuils (PAS, FC, FR) de la tranche d'âge de l'enfant."""
    for age_max, pas_min, fc_max, fr_max in TRANCHES_PEDIATRIQUES:
        if age_annees < age_max:
            return pas_min, fc_max, fr_max
    raise ValueError(f"âge hors tranches : {age_annees}")


def charger_referentiel(chemin: str = "french_referentiel.json") -> dict:
    """Charge le référentiel structuré depuis le fichier JSON."""
    with open(chemin, "r", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class CasClinique:
    """Représentation structurée d'un cas clinique (sortie de l'Étape 1
    du pipeline : extraction structurée à partir de la vignette source)."""
    motif_id: Optional[str] = None
    age_annees: Optional[float] = None
    pas_mmhg: Optional[float] = None          # pression artérielle systolique
    fc_min: Optional[float] = None             # fréquence cardiaque
    spo2_pct: Optional[float] = None
    fr_min: Optional[float] = None              # fréquence respiratoire
    glycemie_mmol_l: Optional[float] = None
    gcs: Optional[int] = None                   # score de Glasgow
    fievre: bool = False
    criteres_presents: list = field(default_factory=list)  # ex. ["ECG anormal typique de SCA"]
    # Valeurs telles qu'ÉCRITES dans la vignette avant conversion d'unité
    # (ex. {"pas_mmhg": 16.5} pour "16,5/10 cm Hg"). Sert au contrôle de présence.
    valeurs_brutes: dict = field(default_factory=dict)


@dataclass
class ResultatDetection:
    # Champs historiques (compatibles avec le code et les tests d'avant le lot 1) :
    # niveau le plus urgent trouvé, toutes règles confondues.
    tri_minimal_force: Optional[str]
    categorie_brief_minimale: Optional[str]
    flags_declenches: list
    # Nouveaux champs du lot 1
    tri_constantes: Optional[str] = None          # plancher INVIOLABLE
    categorie_constantes: Optional[str] = None
    tri_motif: Optional[str] = None               # plancher RÉVISABLE
    categorie_motif: Optional[str] = None
    modulateurs_a_la_baisse: list = field(default_factory=list)
    constantes_non_verifiees: list = field(default_factory=list)


def _plus_urgent(tri_a: Optional[str], tri_b: Optional[str]) -> Optional[str]:
    """Retourne le niveau de tri le plus urgent entre deux niveaux (le plus
    petit numéro d'ordre = le plus urgent). Ignore les valeurs None."""
    candidats = [t for t in (tri_a, tri_b) if t is not None]
    if not candidats:
        return None
    return min(candidats, key=lambda t: ORDRE_URGENCE[t])


# --------------------------------------------------------------------------
# Contrôle de présence d'une constante dans le texte de la vignette
# --------------------------------------------------------------------------

def _nombres_du_texte(texte: str) -> list:
    """Tous les nombres du texte, virgule décimale comprise ("16,5" -> 16.5)."""
    t = texte.replace(",", ".")
    nombres = []
    for m in re.finditer(r"\d+(?:\.\d+)?", t):
        try:
            nombres.append(float(m.group()))
        except ValueError:
            pass
    return nombres


def _valeur_presente(champ: str, valeur: float, texte: str, valeurs_brutes: dict) -> bool:
    """Vrai si le nombre extrait (ou sa forme brute avant conversion) figure
    dans le texte. Tolérant : la PAS peut être écrite en cmHg (165 <-> 16,5),
    la glycémie en g/l (10 mmol/l <-> 1,8 g/l)."""
    candidats = {float(valeur)}
    if champ in valeurs_brutes and valeurs_brutes[champ] is not None:
        candidats.add(float(valeurs_brutes[champ]))
    if champ == "pas_mmhg":
        candidats.add(float(valeur) / 10.0)
    if champ == "glycemie_mmol_l":
        candidats.add(float(valeur) / 5.55)
    nombres = _nombres_du_texte(texte)
    return any(abs(n - c) <= 0.06 for n in nombres for c in candidats)


# --------------------------------------------------------------------------
# Tri du motif (tri médian a priori, ajusté par les modulateurs)
# --------------------------------------------------------------------------

def _tri_du_motif(motif: dict, criteres_presents: list):
    """Retourne (tri, modulateurs_declenches, a_la_baisse).

    FRENCH : le tri médian du motif est un a priori ; les modulateurs le
    ajustent, au-dessus OU en dessous. Si plusieurs modulateurs sont
    déclenchés, le plus urgent l'emporte. Sans modulateur déclenché, on garde
    le tri de base."""
    declenches = [m for m in motif["modulateurs"] if m["critere"] in criteres_presents]
    if not declenches:
        return motif["tri_base"], [], False
    tri = None
    for m in declenches:
        tri = _plus_urgent(tri, m["tri"])
    a_la_baisse = ORDRE_URGENCE[tri] > ORDRE_URGENCE[motif["tri_base"]]
    return tri, declenches, a_la_baisse


def detecter_red_flags(cas: CasClinique, referentiel: dict, texte_source: Optional[str] = None) -> ResultatDetection:
    """
    Étape 2 du pipeline : détection déterministe, indépendante du LLM.

    Applique successivement :
    1. Les seuils de constantes vitales : PAS / FC / FR adulte, ou par tranche
       d'âge avant 15 ans ; SpO2 / GCS / glycémie identiques à tout âge.
    2. Les règles pédiatriques spécifiques (ex. fièvre <= 3 mois).
    3. Le tri du motif de recours identifié en Étape 1 (base + modulateurs).

    Les résultats des points 1-2 forment le plancher INVIOLABLE (constantes),
    ceux du point 3 le plancher RÉVISABLE (motif). Voir l'en-tête du module.

    Si `texte_source` est fourni, chaque constante vitale doit figurer dans ce
    texte pour être retenue (sinon : ignorée et listée dans
    `constantes_non_verifiees`).
    """
    flags = []
    tri_constantes = None
    non_verifiees = []

    est_pediatrique = cas.age_annees is not None and cas.age_annees < 15

    def _retenue(champ: str):
        """Valeur de la constante, ou None si absente ou non retrouvée dans le texte."""
        valeur = getattr(cas, champ)
        if valeur is None:
            return None
        if texte_source is not None and not _valeur_presente(champ, valeur, texte_source, cas.valeurs_brutes):
            non_verifiees.append(f"{champ}={valeur}")
            return None
        return valeur

    pas = _retenue("pas_mmhg")
    fc = _retenue("fc_min")
    spo2 = _retenue("spo2_pct")
    fr = _retenue("fr_min")
    gcs = _retenue("gcs")
    glycemie = _retenue("glycemie_mmol_l")

    # --- 1a. Constantes vitales adulte (PAS, FC, FR) ---
    # Règles codées explicitement (plus sûr qu'un parsing automatique des
    # conditions textuelles du JSON, qui restent indicatives / documentaires).
    if not est_pediatrique:
        if pas is not None:
            if pas < 70:
                tri_constantes = _plus_urgent(tri_constantes, "1")
                flags.append("PAS < 70 mmHg")
            elif pas <= 90 or (pas <= 100 and (fc or 0) > 100):
                tri_constantes = _plus_urgent(tri_constantes, "2")
                flags.append("PAS <= 90 mmHg (ou <=100 avec FC>100)")

        if fc is not None:
            if fc > 180 or fc < 40:
                tri_constantes = _plus_urgent(tri_constantes, "1")
                flags.append("FC > 180 ou < 40/min")
            elif 130 <= fc <= 180:
                tri_constantes = _plus_urgent(tri_constantes, "2")
                flags.append("FC 130-180/min")

        if fr is not None:
            # NB : "30-40 = tri 2" est appliqué ici à tous les motifs, comme dans
            # le JSON ; à valider (cf. meta.validation du référentiel).
            if fr > 40:
                tri_constantes = _plus_urgent(tri_constantes, "1")
                flags.append("FR > 40/min")
            elif 30 <= fr <= 40:
                tri_constantes = _plus_urgent(tri_constantes, "2")
                flags.append("FR 30-40/min")

    # --- 1b. Constantes vitales pédiatriques (PAS, FC, FR selon la tranche d'âge) ---
    if est_pediatrique:
        pas_min, fc_max, fr_max = _tranche_pediatrique(cas.age_annees)
        tri_p = TRI_CONSTANTE_PEDIATRIQUE
        if pas is not None and pas < pas_min:
            tri_constantes = _plus_urgent(tri_constantes, tri_p)
            flags.append(f"Pédiatrie : PAS < {pas_min} mmHg pour l'âge")
        if fc is not None and fc > fc_max:
            tri_constantes = _plus_urgent(tri_constantes, tri_p)
            flags.append(f"Pédiatrie : FC > {fc_max}/min pour l'âge")
        # Bradycardie : critère du motif FRENCH « bradycardie pédiatrie ≤ 2 ans »
        # (avant 1 an FC ≤ 80, après 1 an FC ≤ 60 -> tri 2), étendu à tout âge pédiatrique.
        seuil_brady = 80 if cas.age_annees < 1 else 60
        if fc is not None and fc <= seuil_brady:
            tri_constantes = _plus_urgent(tri_constantes, "2")
            flags.append(f"Pédiatrie : FC <= {seuil_brady}/min (bradycardie)")
        if fr is not None and fr > fr_max:
            tri_constantes = _plus_urgent(tri_constantes, tri_p)
            flags.append(f"Pédiatrie : FR > {fr_max}/min pour l'âge")

    # --- 1c. Constantes sans seuil lié à l'âge (SpO2, GCS, glycémie) : tous âges ---
    if spo2 is not None:
        if spo2 < 86:
            tri_constantes = _plus_urgent(tri_constantes, "1")
            flags.append("SpO2 < 86%")
        elif spo2 <= 90:
            tri_constantes = _plus_urgent(tri_constantes, "2")
            flags.append("SpO2 86-90%")

    if gcs is not None:
        if gcs <= 8:
            tri_constantes = _plus_urgent(tri_constantes, "1")
            flags.append("GCS <= 8")
        elif 9 <= gcs <= 13:
            tri_constantes = _plus_urgent(tri_constantes, "2")
            flags.append("GCS 9-13")

    if glycemie is not None and glycemie > 20:
        # Simplifié : la cétose n'est pas modélisée (donnée rarement
        # disponible à l'accueil). Signalement seul, sans effet sur le niveau.
        flags.append("Glycémie > 20 mmol/l (à confirmer avec cétose)")

    # --- 2. Règles pédiatriques spécifiques ---
    # (la règle « diarrhée/vomissements <= 6 mois » du référentiel n'est pas une
    # constante : elle est couverte par le modulateur « ≤ 6 mois » du motif
    # diarrhee_vomissements_du_nourrisson_24_mois.)
    if est_pediatrique:
        for regle in referentiel["constantes_vitales_pediatrie"]["regles_specifiques"]:
            if regle["id"] == "fievre_nourrisson_3_mois":
                if cas.fievre and cas.age_annees <= 0.25:  # 3 mois = 0.25 an
                    tri_constantes = _plus_urgent(tri_constantes, regle["tri_force"])
                    flags.append(f"Red flag pédiatrique : {regle['id']}")

    # --- 3. Tri du motif de recours (plancher RÉVISABLE) ---
    tri_motif = None
    a_la_baisse = []
    if cas.motif_id:
        motif = next(
            (m for m in referentiel["motifs"] if m["id"] == cas.motif_id), None
        )
        if motif:
            tri_motif, declenches, baisse = _tri_du_motif(motif, cas.criteres_presents)
            for modulateur in declenches:
                flags.append(f"Motif '{motif['libelle']}' : {modulateur['critere']}")
            if not declenches:
                # Traçabilité : même sans modulateur déclenché, le tri de
                # base du motif contribue au résultat — il doit apparaître
                # dans les flags, pas rester invisible pour l'audit.
                flags.append(f"Motif '{motif['libelle']}' : tri de base ({motif['tri_base']}), aucun modulateur déclenché")
            if baisse:
                criteres = " ; ".join(m["critere"] for m in declenches)
                a_la_baisse.append(
                    f"{motif['id']} : le modulateur « {criteres} » abaisse le tri de {motif['tri_base']} à {tri_motif} "
                    f"(à valider par un urgentiste)")

    tri_force = _plus_urgent(tri_constantes, tri_motif)
    categorie = MAPPING_CATEGORIES_BRIEF.get(tri_force) if tri_force else None

    return ResultatDetection(
        tri_minimal_force=tri_force,
        categorie_brief_minimale=categorie,
        flags_declenches=flags,
        tri_constantes=tri_constantes,
        categorie_constantes=MAPPING_CATEGORIES_BRIEF.get(tri_constantes) if tri_constantes else None,
        tri_motif=tri_motif,
        categorie_motif=MAPPING_CATEGORIES_BRIEF.get(tri_motif) if tri_motif else None,
        modulateurs_a_la_baisse=a_la_baisse,
        constantes_non_verifiees=non_verifiees,
    )


def appliquer_override(
    categorie_llm: str, resultat_deterministe: ResultatDetection
) -> dict:
    """
    Croisement Étape 2 / Étape 3 (lot 1).

    - Plancher des CONSTANTES : inviolable. Si la catégorie du plancher est
      plus urgente que celle du LLM, le label final est relevé (override_applique=True).
    - Plancher du MOTIF : révisable. Il ne force plus le label. Si le label
      final reste moins urgent que ce plancher, `sous_plancher_motif` vaut True
      (le cas doit être relu par un humain).
    - Le LLM peut toujours monter au-dessus des deux planchers.

    Retourne le label final ainsi que les métadonnées de traçabilité
    nécessaires à l'audit (exigence du brief CHSA).
    """
    ordre = ORDRE_CATEGORIE_BRIEF

    cat_constantes = getattr(resultat_deterministe, "categorie_constantes", None)
    cat_motif = getattr(resultat_deterministe, "categorie_motif", None)
    if cat_constantes is None and cat_motif is None:
        # Résultat au format d'avant le lot 1 : on retombe sur l'ancien
        # comportement (le plancher global est inviolable).
        cat_constantes = resultat_deterministe.categorie_brief_minimale

    label_final = categorie_llm
    override_applique = False
    if cat_constantes is not None and ordre[cat_constantes] < ordre[label_final]:
        label_final = cat_constantes
        override_applique = True

    sous_plancher_motif = cat_motif is not None and ordre[label_final] > ordre[cat_motif]

    return {
        "label_final": label_final,
        "label_llm_initial": categorie_llm,
        "categorie_deterministe_minimale": resultat_deterministe.categorie_brief_minimale,
        "categorie_plancher_constantes": cat_constantes,
        "categorie_plancher_motif": cat_motif,
        "override_applique": override_applique,
        "sous_plancher_motif": sous_plancher_motif,
        "modulateurs_a_la_baisse": list(getattr(resultat_deterministe, "modulateurs_a_la_baisse", [])),
        "constantes_non_verifiees": list(getattr(resultat_deterministe, "constantes_non_verifiees", [])),
        "red_flags_detectes": resultat_deterministe.flags_declenches,
        "tri_french_minimal_force": resultat_deterministe.tri_minimal_force,
    }


# --------------------------------------------------------------------------
# Exemple d'utilisation
# --------------------------------------------------------------------------
if __name__ == "__main__":
    referentiel = charger_referentiel(
        str(Path(__file__).parent / "french_referentiel.json")
    )

    # Cas exemple : patient avec douleur thoracique, ECG anormal typique SCA,
    # mais le LLM (hypothétique) aurait sous-évalué le cas en "modérée".
    cas = CasClinique(
        motif_id="douleur_thoracique_sca",
        age_annees=68,
        criteres_presents=["ECG anormal typique de SCA"],
    )

    resultat = detecter_red_flags(cas, referentiel)
    print("Résultat détection déterministe :", resultat)

    decision = appliquer_override(categorie_llm="moderee", resultat_deterministe=resultat)
    print("\nDécision finale avec traçabilité :")
    for cle, valeur in decision.items():
        print(f"  {cle}: {valeur}")
    print("\n(Le plancher du motif ne force plus le label : ici le label reste 'moderee' "
          "et sous_plancher_motif vaut True -> le cas est à relire.)")
