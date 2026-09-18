"""
contre_validation_etape4.py

Étape 4 du pipeline : challenge le label retenu après le croisement
Étape 2 / Étape 3, avec un second appel LLM en posture d'AUDITEUR.

Par défaut, ce script utilise GROQ (pas Gemini) — volontairement DIFFÉRENT
du fournisseur par défaut de l'Étape 3, pour réduire la corrélation des
erreurs entre les deux jugements (cf. point de vigilance identifié tôt dans
le projet). Le prompt de cette étape est court, donc l'impact sur le quota
Groq du filtre en cours reste minime.

Usage (mode test, contre le jeu de cas de référence) :
    python contre_validation_etape4.py --golden ../../tests/golden_etape1_extraction.json

Usage (mode réel, sur la sortie de labellisation_etape3.py) :
    python contre_validation_etape4.py \
        --input ../../data/normalized/mediqal_mcqm_labellisation_test.jsonl \
        --output ../../data/normalized/mediqal_mcqm_etape4_test.jsonl
"""

import argparse
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))
from llm_client import completer  # noqa: E402

CATEGORIES_VALIDES = {"urgence_maximale", "moderee", "differee"}
ORDRE_CATEGORIES = {"urgence_maximale": 1, "moderee": 2, "differee": 3}

PROMPT_SYSTEME = """Tu es un auditeur clinique indépendant, pour un projet d'agent de triage médical. Ton rôle N'EST PAS de refaire la classification depuis zéro, mais de VÉRIFIER si un niveau de priorité déjà attribué ne sous-estime pas la gravité réelle du cas.

On te donne : le cas clinique structuré, et le niveau de priorité qui lui a été attribué (parmi "urgence_maximale", "moderee", "differee", du plus au moins urgent).

Ta tâche : identifie UNIQUEMENT s'il existe un risque concret de SOUS-triage. Tu ne dois JAMAIS signaler un risque de SUR-triage — seul le sous-triage est un danger à corriger à cette étape.

RÈGLES IMPÉRATIVES :
1. Ne remets en cause le niveau attribué QUE si tu identifies un élément clinique concret dans le cas structuré qui le justifie — pas une prudence générique.
2. N'invente AUCUN élément clinique absent du cas structuré fourni.
3. Si le niveau attribué te semble déjà adapté ou trop prudent, ne signale rien.

Réponds UNIQUEMENT avec un objet JSON, sans texte avant ni après :
{"risque_sous_triage": true ou false, "niveau_suggere": "urgence_maximale" ou "moderee" ou "differee" ou null si risque_sous_triage est false, "justification": "une phrase citant l'élément clinique concerné, ou null si aucun risque"}
"""


def contre_valider(cas_structure: dict, categorie_attribuee: str, fournisseur: str, modele: str) -> dict:
    contenu = {"cas_structure": cas_structure, "niveau_deja_attribue": categorie_attribuee}
    contenu_json = json.dumps(contenu, ensure_ascii=False, indent=2)
    resultat = completer(PROMPT_SYSTEME, contenu_json, fournisseur, modele, max_tokens=2048)

    if resultat.get("niveau_suggere") is not None and resultat.get("niveau_suggere") not in CATEGORIES_VALIDES:
        resultat["_erreur_categorie_invalide"] = resultat["niveau_suggere"]

    return resultat


def resoudre_contre_validation(categorie_initiale: str, verdict: dict) -> dict:
    if not verdict.get("risque_sous_triage"):
        return {"categorie_finale": categorie_initiale, "contre_validation_a_modifie": False, "verdict_etape4": verdict}

    niveau_suggere = verdict.get("niveau_suggere")
    if niveau_suggere not in CATEGORIES_VALIDES:
        return {"categorie_finale": categorie_initiale, "contre_validation_a_modifie": False,
                "verdict_etape4": verdict, "_necessite_relecture_humaine": True}

    if ORDRE_CATEGORIES[niveau_suggere] < ORDRE_CATEGORIES[categorie_initiale]:
        return {"categorie_finale": niveau_suggere, "contre_validation_a_modifie": True, "verdict_etape4": verdict}

    return {"categorie_finale": categorie_initiale, "contre_validation_a_modifie": False,
            "verdict_etape4": verdict, "_incoherence_auditeur": True}


def executer_mode_test(golden_path: str, fournisseur: str, modele: str, pause: float):
    with open(golden_path, encoding="utf-8") as f:
        golden = json.load(f)

    for cas in golden["cas"]:
        cas_structure = cas["extraction_attendue"]
        categorie_initiale = cas.get("categorie_attendue")

        print(f"\n=== {cas['id']} ===")
        print(f"  Niveau initial (à challenger) : {categorie_initiale}")

        verdict = contre_valider(cas_structure, categorie_initiale, fournisseur, modele)
        if "_erreur_parsing" in verdict:
            print(f"  ❌ ERREUR DE PARSING : {verdict['_erreur_parsing']}")
            time.sleep(pause)
            continue

        decision = resoudre_contre_validation(categorie_initiale, verdict)
        if decision["contre_validation_a_modifie"]:
            print(f"  ⚠️  RISQUE DE SOUS-TRIAGE SIGNALÉ ET APPLIQUÉ : "
                  f"{categorie_initiale} -> {decision['categorie_finale']}")
            print(f"  Justification : {verdict.get('justification')}")
        elif verdict.get("risque_sous_triage"):
            print(f"  ℹ️  Risque signalé mais non appliqué (incohérent ou non exploitable) : {verdict}")
        else:
            print(f"  ✅ Aucun risque de sous-triage signalé, niveau conservé : {categorie_initiale}")

        time.sleep(pause)


def charger_ids_deja_traites(output_file: Path) -> set:
    """Lit le fichier de sortie déjà existant pour connaître les ids déjà
    traités lors d'une exécution précédente (mécanisme de reprise, cf.
    filtre_pertinence.py — évite les doublons en cas d'interruption)."""
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

    n_modifies, n_ignorees = 0, 0
    with open(output_file, "a", encoding="utf-8") as f_out:
        for i, entree in enumerate(lignes, 1):
            label = entree.get("label_etape3", {})
            categorie_initiale = label.get("categorie")
            if categorie_initiale not in CATEGORIES_VALIDES:
                n_ignorees += 1
                continue

            cas_structure = entree["cas_structure"]
            verdict = contre_valider(cas_structure, categorie_initiale, fournisseur, modele)
            if "_erreur_parsing" in verdict:
                n_ignorees += 1
                continue

            decision = resoudre_contre_validation(categorie_initiale, verdict)
            if decision["contre_validation_a_modifie"]:
                n_modifies += 1

            sortie = {"id": entree["id"], "cas_structure": cas_structure, **decision}
            f_out.write(json.dumps(sortie, ensure_ascii=False) + "\n")
            f_out.flush()

            if i % 10 == 0 or i == len(lignes):
                print(f"  {i}/{len(lignes)} traitées — {n_modifies} corrigées par l'Étape 4, "
                      f"{n_ignorees} ignorées", file=sys.stderr)

            time.sleep(pause)

    print(f"\nTerminé -> {output_file}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Étape 4 : contre-validation du label retenu")
    parser.add_argument("--golden", default=None)
    parser.add_argument("--input", default=None, help="Mode réel : sortie JSONL de labellisation_etape3.py")
    parser.add_argument("--output", default=None)
    parser.add_argument("--fournisseur", choices=["groq", "gemini"], default="groq",
                         help="Défaut : groq (différent de l'Étape 3, prompt court -> impact quota minime)")
    parser.add_argument("--modele", default=None,
                         help="Défaut selon le fournisseur : openai/gpt-oss-20b ou gemini-3.6-flash")
    parser.add_argument("--pause", type=float, default=3.0)
    args = parser.parse_args()

    modele = args.modele or ("qwen/qwen3.8-27b" if args.fournisseur == "groq" else "gemini-3.6-flash")
    
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