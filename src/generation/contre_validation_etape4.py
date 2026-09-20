"""
contre_validation_etape4.py

Étape 4 du pipeline : audit du label retenu après le croisement
Étape 2 / Étape 3, avec un second appel LLM en posture d'AUDITEUR.

L'auditeur reçoit la VIGNETTE SOURCE en plus du cas structuré : c'est la
seule étape qui puisse repérer une erreur de l'Étape 1 (mauvais motif, signe
grave omis), puisque les Étapes 2 et 3 partagent la même extraction.

--- Ce qui a changé au lot 1 (sept. 2026) ---
- L'Étape 4 NE MODIFIE PLUS le label : elle SIGNALE. Un signalement alimente la
  liste de relecture humaine ("a_relire") ; le label reste celui d'après les
  Étapes 2 et 3. Raison : sur 19 cas de test, l'auditeur relevait 4 cas de
  "modérée" à "urgence maximale" avec des arguments hypothétiques (scénarios
  possibles plutôt que signes présents dans la vignette).
- Pour signaler, l'auditeur doit CITER MOT POUR MOT un passage de la vignette.
  Le CODE vérifie que cette citation figure bien dans la vignette. Une
  citation absente marque le signalement "non vérifié" : il sort de la file
  de relecture prioritaire.
- "extraction_incoherente" est redéfini comme à l'Étape 3 (motif ou valeur
  chiffrée qui contredit la vignette ; l'absence d'antécédents ou d'examens
  n'en est pas une).
- Le sens est inchangé : l'auditeur ne cherche QUE le risque de SOUS-triage.
  Le sur-triage est mesuré autrement (répartition des catégories, évaluation).

Fournisseur par défaut : Groq, volontairement DIFFÉRENT de l'Étape 3 (Gemini)
et de l'Étape 1 (Anthropic), pour réduire la corrélation des erreurs.

Usage (mode test, contre le jeu de cas de référence) :
    python contre_validation_etape4.py --golden ../../tests/golden_etape1_extraction.json

Usage (mode réel, sur la sortie de labellisation_etape3.py) :
    python contre_validation_etape4.py \
        --input ../../data/normalized/mediqal_mcqm_labellisation_test.jsonl \
        --output ../../data/normalized/mediqal_mcqm_etape4_test.jsonl
"""

import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))
from llm_client import completer  # noqa: E402

CATEGORIES_VALIDES = {"urgence_maximale", "moderee", "differee"}
ORDRE_CATEGORIES = {"urgence_maximale": 1, "moderee": 2, "differee": 3}

FOURNISSEUR_DEFAUT = "groq"
MODELE_DEFAUT = "openai/gpt-oss-120b"   # à vérifier dans ta console avant un gros lot

# Champs du cas structuré montrés à l'auditeur
CHAMPS_VISIBLES = ["motif_id", "age_annees", "pas_mmhg", "fc_min", "spo2_pct", "fr_min",
                   "glycemie_mmol_l", "gcs", "fievre_statut", "criteres_presents"]

PROMPT_SYSTEME = """Tu es un auditeur clinique indépendant, pour un projet d'agent de triage médical. Ton rôle N'EST PAS de refaire la classification depuis zéro, mais de SIGNALER si un niveau de priorité déjà attribué te semble sous-estimer la gravité réelle du cas. Tu ne modifies rien : ton signalement sera relu par un humain.

On te donne : la VIGNETTE SOURCE (texte clinique original, qui fait foi), le cas structuré extrait automatiquement de cette vignette (résumé partiel, qui peut être incomplet ou comporter des erreurs, notamment sur le motif de recours), et le niveau de priorité attribué (parmi "urgence_maximale", "moderee", "differee", du plus au moins urgent).

Ta tâche : identifie UNIQUEMENT s'il existe un risque concret de SOUS-triage de la présentation aiguë INITIALE du patient. Tu ne dois JAMAIS signaler un risque de SUR-triage.

RÈGLES IMPÉRATIVES :
1. Ne signale un risque QUE si un élément clinique PRÉSENT dans la vignette le justifie. Un scénario possible, une maladie qu'on pourrait imaginer, une complication qui pourrait survenir ne sont PAS des éléments de la vignette : ne les signale pas.
2. Pour signaler un risque, tu dois CITER, mot pour mot, le passage de la vignette qui le justifie (entre 4 et 30 mots, copié exactement, sans le reformuler). Si tu ne peux pas citer de passage, ne signale rien.
3. N'invente AUCUN élément clinique absent de la vignette.
4. Juge la gravité de la présentation INITIALE, pas l'état après traitement ni les résultats d'examens réalisés ensuite (imagerie, ECG, bilans) : un patient décrit comme stabilisé ou hospitalisé a pu arriver dans un état grave. Ne cite pas un résultat d'examen comme signe d'arrivée.
5. Si le niveau attribué te semble déjà adapté ou trop prudent, ne signale rien.
6. Dans le cas structuré, "fievre_statut" vaut "oui", "non" ou "non_mentionne". "non_mentionne" signifie que la vignette n'en parle pas : ce n'est PAS l'absence de fièvre.

Réponds UNIQUEMENT avec un objet JSON, sans texte avant ni après :
{"risque_sous_triage": true ou false, "niveau_suggere": "urgence_maximale" ou "moderee" ou "differee" ou null si risque_sous_triage est false, "citation": "passage copié mot pour mot de la vignette, ou null si aucun risque", "justification": "une phrase expliquant pourquoi ce passage justifie un niveau plus élevé, ou null si aucun risque", "extraction_incoherente": true ou false, "detail_extraction": "une phrase décrivant la contradiction, ou null si extraction_incoherente vaut false"}
où "extraction_incoherente" vaut true UNIQUEMENT si le motif du cas structuré est manifestement différent de la plainte décrite dans la vignette, ou si une valeur chiffrée du cas structuré contredit la vignette. Ne le mets PAS à true pour l'absence d'antécédents, d'examens complémentaires ou de symptômes secondaires : le cas structuré ne les contient pas volontairement. Toute explication doit figurer DANS les champs prévus : n'écris AUCUN texte en dehors de l'objet JSON.
"""


def construire_contenu(cas_structure: dict, vignette_source: str, categorie_attribuee: str) -> str:
    """Contenu envoyé à l'auditeur. Partagé avec orchestrer_pipeline.py pour
    que les deux ne divergent jamais."""
    visible = {k: cas_structure.get(k) for k in CHAMPS_VISIBLES if k in cas_structure}
    contenu = {"vignette_source": vignette_source, "cas_structure": visible,
               "niveau_deja_attribue": categorie_attribuee}
    return json.dumps(contenu, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------
# Vérification de la citation (faite par le CODE, pas par le LLM)
# --------------------------------------------------------------------------

def _normaliser(texte: str) -> str:
    """Minuscules, sans accents, sans ponctuation, espaces réduits."""
    t = unicodedata.normalize("NFKD", texte or "")
    t = "".join(c for c in t if not unicodedata.combining(c)).lower()
    t = t.replace("œ", "oe").replace("’", "'").replace("‘", "'")
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return t.strip()


def verifier_citation(citation, vignette: str, min_mots: int = 4, min_mots_fragment: int = 3) -> bool:
    """Vrai si la citation figure mot pour mot dans la vignette (à la
    ponctuation, aux accents et à la casse près). Une citation en plusieurs
    fragments séparés par « ... » est acceptée si chaque fragment (3 mots ou
    plus) figure dans la vignette. Moins de 4 mots au total : refusée."""
    if not isinstance(citation, str) or not citation.strip() or not vignette:
        return False
    fragments = [f for f in re.split(r"\.\.\.|…|\[\.\.\.\]", citation) if _normaliser(f)]
    if not fragments:
        return False
    vign = " " + _normaliser(vignette) + " "
    total_mots = 0
    for f in fragments:
        f_norm = _normaliser(f)
        mots = f_norm.split()
        if len(mots) < min_mots_fragment:
            return False
        if (" " + f_norm + " ") not in vign:
            return False
        total_mots += len(mots)
    return total_mots >= min_mots

def plus_long_passage_litteral(citation, vignette: str):
    """Retourne (n, total) : n = plus longue suite de mots CONSÉCUTIFS de la citation retrouvée telle quelle
    dans la vignette (à la ponctuation, aux accents et à la casse près) ; total = nombre de mots de la citation."""
    mots = _normaliser(citation).split() if isinstance(citation, str) else []
    if not mots or not vignette:
        return 0, len(mots)
    vign = " " + _normaliser(vignette) + " "
    meilleur = 0
    for i in range(len(mots)):
        j = i + 1
        while j <= len(mots) and (" " + " ".join(mots[i:j]) + " ") in vign:
            j += 1
        meilleur = max(meilleur, j - 1 - i)
    return meilleur, len(mots)


def verifier_citation_critere(citation, vignette: str, min_mots: int = 3, part_min: float = 0.5) -> bool:
    """Vérification TOLÉRANTE, pour les critères cochés par l'Appel B de l'Étape 1 : la citation est acceptée si une
    suite d'au moins `min_mots` mots consécutifs figure telle quelle dans la vignette ET couvre au moins `part_min`
    de la citation. Une citation longue dont seul un petit bout est authentique reste refusée."""
    n, total = plus_long_passage_litteral(citation, vignette)
    return total > 0 and n >= min_mots and n / total >= part_min


def contre_valider(cas_structure: dict, vignette_source: str, categorie_attribuee: str,
                   fournisseur: str, modele: str) -> dict:
    contenu_json = construire_contenu(cas_structure, vignette_source, categorie_attribuee)
    resultat = completer(PROMPT_SYSTEME, contenu_json, fournisseur, modele, max_tokens=2048)

    if "_erreur_parsing" in resultat:
        return resultat

    if resultat.get("niveau_suggere") is not None and resultat.get("niveau_suggere") not in CATEGORIES_VALIDES:
        resultat["_erreur_categorie_invalide"] = resultat["niveau_suggere"]

    if resultat.get("risque_sous_triage") is True:
        resultat["_citation_verifiee"] = verifier_citation(resultat.get("citation"), vignette_source)

    return resultat


def resoudre_contre_validation(categorie_initiale: str, verdict: dict, vignette_source: str = None) -> dict:
    """L'Étape 4 ne modifie JAMAIS le label : `categorie_finale` est toujours
    la catégorie reçue. Elle produit un signalement (`signal_sous_triage`)
    avec sa priorité de relecture."""
    base = {"categorie_finale": categorie_initiale, "contre_validation_a_modifie": False,
            "verdict_etape4": verdict}

    if not verdict.get("risque_sous_triage"):
        return {**base, "signal_sous_triage": False}

    niveau = verdict.get("niveau_suggere")
    if niveau not in CATEGORIES_VALIDES or ORDRE_CATEGORIES[niveau] >= ORDRE_CATEGORIES[categorie_initiale]:
        # Risque affirmé mais niveau absent, invalide ou pas plus urgent : incohérent, on ne signale pas.
        return {**base, "signal_sous_triage": False, "_incoherence_auditeur": True}

    citation_ok = verdict.get("_citation_verifiee")
    if citation_ok is None and vignette_source is not None:
        citation_ok = verifier_citation(verdict.get("citation"), vignette_source)

    return {**base, "signal_sous_triage": True, "niveau_suggere": niveau,
            "citation_verifiee": bool(citation_ok),
            "priorite_relecture": "haute" if citation_ok else "basse"}


def executer_mode_test(golden_path: str, fournisseur: str, modele: str, pause: float):
    with open(golden_path, encoding="utf-8") as f:
        golden = json.load(f)

    for cas in golden["cas"]:
        cas_structure = cas["extraction_attendue"]
        categorie_initiale = cas.get("categorie_attendue")

        print(f"\n=== {cas['id']} ===")
        print(f"  Niveau initial (à auditer) : {categorie_initiale}")

        verdict = contre_valider(cas_structure, cas["vignette_source"], categorie_initiale, fournisseur, modele)
        if "_erreur_parsing" in verdict:
            print(f"  ❌ ERREUR DE PARSING : {verdict['_erreur_parsing']}")
            time.sleep(pause)
            continue

        decision = resoudre_contre_validation(categorie_initiale, verdict, cas["vignette_source"])
        if decision["signal_sous_triage"]:
            etat = "citation vérifiée" if decision["citation_verifiee"] else "citation NON retrouvée dans la vignette"
            print(f"  ⚠️  SIGNAL DE SOUS-TRIAGE ({etat}) : {categorie_initiale} -> {decision['niveau_suggere']} suggéré")
            print(f"  Citation : {verdict.get('citation')}")
            print(f"  Justification : {verdict.get('justification')}")
            print("  (le label n'est PAS modifié : le cas irait en relecture)")
        else:
            print(f"  ✅ Aucun signal de sous-triage, niveau conservé : {categorie_initiale}")

        time.sleep(pause)


def charger_ids_deja_traites(output_file: Path) -> set:
    """Lit le fichier de sortie déjà existant pour connaître les ids déjà
    traités lors d'une exécution précédente (mécanisme de reprise)."""
    ids = set()
    if output_file.exists():
        with open(output_file, encoding="utf-8") as f:
            for ligne in f:
                ligne = ligne.strip()
                if ligne:
                    ids.add(json.loads(ligne)["id"])
    return ids


def executer_mode_reel(input_path: str, output_path: str, fournisseur: str, modele: str, pause: float):
    lignes = []
    with open(input_path, encoding="utf-8") as f:
        for ligne in f:
            ligne = ligne.strip()
            if ligne:
                lignes.append(json.loads(ligne))

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    ids_deja_traites = charger_ids_deja_traites(output_file)
    if ids_deja_traites:
        print(f"Reprise détectée : {len(ids_deja_traites)} vignettes déjà traitées, elles seront sautées.",
              file=sys.stderr)
    lignes = [l for l in lignes if l["id"] not in ids_deja_traites]

    n_signaux, n_ignorees = 0, 0
    with open(output_file, "a", encoding="utf-8") as f_out:
        for i, entree in enumerate(lignes, 1):
            label = entree.get("label_etape3", {})
            categorie_initiale = label.get("categorie")
            if categorie_initiale not in CATEGORIES_VALIDES:
                n_ignorees += 1
                continue

            cas_structure = entree["cas_structure"]
            vignette_source = entree.get("vignette_source")
            if not vignette_source:
                n_ignorees += 1
                continue
            verdict = contre_valider(cas_structure, vignette_source, categorie_initiale, fournisseur, modele)
            if "_erreur_parsing" in verdict:
                n_ignorees += 1
                continue

            decision = resoudre_contre_validation(categorie_initiale, verdict, vignette_source)
            if decision["signal_sous_triage"]:
                n_signaux += 1

            sortie = {"id": entree["id"], "cas_id": entree.get("cas_id"), "cas_structure": cas_structure, **decision}
            f_out.write(json.dumps(sortie, ensure_ascii=False) + "\n")
            f_out.flush()

            if i % 10 == 0 or i == len(lignes):
                print(f"  {i}/{len(lignes)} traitées — {n_signaux} signalées, "
                      f"{n_ignorees} ignorées", file=sys.stderr)

            time.sleep(pause)

    print(f"\nTerminé -> {output_file}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Étape 4 : audit du label retenu (signalement seul)")
    parser.add_argument("--golden", default=None)
    parser.add_argument("--input", default=None, help="Mode réel : sortie JSONL de labellisation_etape3.py")
    parser.add_argument("--output", default=None)
    parser.add_argument("--fournisseur", choices=["groq", "gemini", "anthropic"], default=FOURNISSEUR_DEFAUT,
                         help="Défaut : groq (différent des Étapes 1 et 3)")
    parser.add_argument("--modele", default=None,
                         help=f"Défaut : {MODELE_DEFAUT} (à vérifier dans ta console)")
    parser.add_argument("--pause", type=float, default=3.0)
    args = parser.parse_args()

    modele = args.modele or MODELE_DEFAUT

    if args.input:
        if not args.output:
            print("--output est requis avec --input", file=sys.stderr)
            sys.exit(1)
        executer_mode_reel(args.input, args.output, args.fournisseur, modele, args.pause)
    else:
        golden_path = args.golden or str(Path(__file__).parent.parent.parent / "tests" / "golden_etape1_extraction.json")
        executer_mode_test(golden_path, args.fournisseur, modele, args.pause)


if __name__ == "__main__":
    main()
