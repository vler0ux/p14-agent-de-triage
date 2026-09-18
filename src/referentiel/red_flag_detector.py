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
"""

import json
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


@dataclass
class ResultatDetection:
    tri_minimal_force: Optional[str]
    categorie_brief_minimale: Optional[str]
    flags_declenches: list


def _plus_urgent(tri_a: Optional[str], tri_b: Optional[str]) -> Optional[str]:
    """Retourne le niveau de tri le plus urgent entre deux niveaux (le plus
    petit numéro d'ordre = le plus urgent). Ignore les valeurs None."""
    candidats = [t for t in (tri_a, tri_b) if t is not None]
    if not candidats:
        return None
    return min(candidats, key=lambda t: ORDRE_URGENCE[t])


def detecter_red_flags(cas: CasClinique, referentiel: dict) -> ResultatDetection:
    """
    Étape 2 du pipeline : détection déterministe, indépendante du LLM.

    Applique successivement :
    1. Les seuils de constantes vitales (adulte ou pédiatrie selon l'âge).
    2. Les règles pédiatriques spécifiques (ex. fièvre <= 3 mois).
    3. Les modulateurs du motif de recours identifié en Étape 1.

    Retourne le niveau de tri minimal forcé (le plus urgent trouvé parmi
    toutes les règles déclenchées) et la liste des flags déclenchés,
    pour traçabilité.
    """
    flags = []
    tri_force = None

    est_pediatrique = cas.age_annees is not None and cas.age_annees < 15

    # --- 1. Constantes vitales adulte ---
    if not est_pediatrique:
        cv = referentiel["constantes_vitales_adulte"]

        def _check_seuils(valeur, seuils, nom_param):
            nonlocal tri_force
            if valeur is None:
                return
            # Les seuils sont ordonnés du plus grave au moins grave ;
            # on prend la première règle numériquement compatible.
            # Ici on reste volontairement explicite plutôt que de parser
            # les chaînes de condition (qui sont indicatives), pour éviter
            # une fausse sécurité de parsing automatique non validé.
            pass  # cf. note ci-dessous

        # Exemple de règles codées explicitement (plus sûr qu'un parsing
        # automatique des conditions textuelles du JSON, qui restent
        # indicatives / documentaires) :
        if cas.pas_mmhg is not None:
            if cas.pas_mmhg < 70:
                tri_force = _plus_urgent(tri_force, "1")
                flags.append("PAS < 70 mmHg")
            elif cas.pas_mmhg <= 90 or (cas.pas_mmhg <= 100 and (cas.fc_min or 0) > 100):
                tri_force = _plus_urgent(tri_force, "2")
                flags.append("PAS <= 90 mmHg (ou <=100 avec FC>100)")

        if cas.fc_min is not None:
            if cas.fc_min > 180 or cas.fc_min < 40:
                tri_force = _plus_urgent(tri_force, "1")
                flags.append("FC > 180 ou < 40/min")
            elif 130 <= cas.fc_min <= 180:
                tri_force = _plus_urgent(tri_force, "2")
                flags.append("FC 130-180/min")

        if cas.spo2_pct is not None:
            if cas.spo2_pct < 86:
                tri_force = _plus_urgent(tri_force, "1")
                flags.append("SpO2 < 86%")
            elif cas.spo2_pct <= 90:
                tri_force = _plus_urgent(tri_force, "2")
                flags.append("SpO2 86-90%")

        if cas.fr_min is not None:
            if cas.fr_min > 40:
                tri_force = _plus_urgent(tri_force, "1")
                flags.append("FR > 40/min")
            elif 30 <= cas.fr_min <= 40:
                tri_force = _plus_urgent(tri_force, "2")
                flags.append("FR 30-40/min")

        if cas.gcs is not None:
            if cas.gcs <= 8:
                tri_force = _plus_urgent(tri_force, "1")
                flags.append("GCS <= 8")
            elif 9 <= cas.gcs <= 13:
                tri_force = _plus_urgent(tri_force, "2")
                flags.append("GCS 9-13")

        if cas.glycemie_mmol_l is not None and cas.glycemie_mmol_l <= 20:
            # Simplifié : la cétose n'est pas modélisée ici (donnée
            # rarement disponible au questionnaire initial)
            flags.append("Glycémie <= 20 mmol/l (à confirmer avec cétose)")

    # --- 2. Règles pédiatriques spécifiques ---
    if est_pediatrique:
        for regle in referentiel["constantes_vitales_pediatrie"]["regles_specifiques"]:
            if regle["id"] == "fievre_nourrisson_3_mois":
                if cas.fievre and cas.age_annees <= 0.25:  # 3 mois = 0.25 an
                    tri_force = _plus_urgent(tri_force, regle["tri_force"])
                    flags.append(f"Red flag pédiatrique : {regle['id']}")

    # --- 3. Modulateurs du motif de recours ---
    if cas.motif_id:
        motif = next(
            (m for m in referentiel["motifs"] if m["id"] == cas.motif_id), None
        )
        if motif:
            tri_motif = motif["tri_base"]
            modulateur_declenche = False
            for modulateur in motif["modulateurs"]:
                if modulateur["critere"] in cas.criteres_presents:
                    tri_motif = _plus_urgent(tri_motif, modulateur["tri"])
                    flags.append(f"Motif '{motif['libelle']}' : {modulateur['critere']}")
                    modulateur_declenche = True
            if not modulateur_declenche:
                # Traçabilité : même sans modulateur déclenché, le tri de
                # base du motif contribue au résultat — il doit apparaître
                # dans les flags, pas rester invisible pour l'audit.
                flags.append(f"Motif '{motif['libelle']}' : tri de base ({motif['tri_base']}), aucun modulateur déclenché")
            tri_force = _plus_urgent(tri_force, tri_motif)

    categorie = MAPPING_CATEGORIES_BRIEF.get(tri_force) if tri_force else None

    return ResultatDetection(
        tri_minimal_force=tri_force,
        categorie_brief_minimale=categorie,
        flags_declenches=flags,
    )


def appliquer_override(
    categorie_llm: str, resultat_deterministe: ResultatDetection
) -> dict:
    """
    Mécanisme de croisement Étape 2 / Étape 3 : le label du LLM ne peut
    jamais être revu À LA BAISSE par rapport à ce que force la détection
    déterministe. Il peut en revanche être conservé s'il est plus urgent
    que ce que la détection déterministe a trouvé (le LLM peut capter des
    éléments cliniques non couverts par la checklist).

    Retourne le label final ainsi que les métadonnées de traçabilité
    nécessaires à l'audit (exigence du brief CHSA).
    """
    ordre_categorie = {"urgence_maximale": 1, "moderee": 2, "differee": 3}

    categorie_deterministe = resultat_deterministe.categorie_brief_minimale

    if categorie_deterministe is None:
        label_final = categorie_llm
        override_applique = False
    else:
        if ordre_categorie[categorie_deterministe] < ordre_categorie[categorie_llm]:
            label_final = categorie_deterministe
            override_applique = True
        else:
            label_final = categorie_llm
            override_applique = False

    return {
        "label_final": label_final,
        "label_llm_initial": categorie_llm,
        "categorie_deterministe_minimale": categorie_deterministe,
        "override_applique": override_applique,
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
