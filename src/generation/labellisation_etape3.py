"""
labellisation_etape3.py

Étape 3 du pipeline : à partir du cas clinique structuré (sortie de
l'Étape 1), le LLM propose un niveau de priorité parmi les 3 catégories
du brief (urgence_maximale / moderee / differee), avec justification.

Cette proposition n'est PAS le label final : elle est ensuite croisée avec
le résultat de l'Étape 2 (détection déterministe, red_flag_detector.py) via
appliquer_override(), puis challengée à l'Étape 4.

Par défaut, ce script utilise Gemini (cf. llm_client.py) pour ne pas entrer
en compétition avec le quota Groq du filtre de pertinence.

Usage (mode test, contre le jeu de cas de référence) :
    python labellisation_etape3.py --golden ../../tests/golden_etape1_extraction.json

Usage (mode réel, sur la sortie de extraction_etape1.py) :
    python labellisation_etape3.py \
        --input ../../data/normalized/mediqal_mcqm_extraction_test.jsonl \
        --output ../../data/normalized/mediqal_mcqm_labellisation_test.jsonl
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

PROMPT_SYSTEME = """Tu es un assistant d'aide à la classification de triage clinique, pour un projet d'agent de triage médical.

À partir du cas clinique structuré suivant (déjà extrait d'une vignette source), détermine le niveau de priorité parmi exactement ces 3 catégories :
- "urgence_maximale" : pronostic vital ou fonctionnel engagé à court terme, prise en charge sans délai ou très rapide nécessaire
- "moderee" : atteinte potentielle nécessitant une évaluation médicale dans la journée, sans urgence vitale immédiate
- "differee" : motif de recours légitime aux urgences mais sans risque de dégradation à court terme

RÈGLES IMPÉRATIVES :
1. En cas de doute entre deux niveaux, choisis TOUJOURS le niveau le plus élevé (principe de précaution).
2. Justifie ta réponse en citant explicitement les éléments du cas structuré qui motivent ce choix.
3. N'invente AUCUN élément clinique absent du cas structuré fourni — base-toi uniquement sur les champs donnés.
4. Un motif de recours plausible mais bénin (ex. traumatisme mineur, symptôme chronique stable) doit être classé "differee", pas "moderee" par excès de prudence non justifié.

Réponds UNIQUEMENT avec un objet JSON, sans texte avant ni après :
{"categorie": "urgence_maximale" ou "moderee" ou "differee", "justification": "une ou deux phrases citant les éléments du cas qui motivent ce choix"}
"""


def labelliser(cas_structure: dict, fournisseur: str, modele: str) -> dict:
    cas_json = json.dumps(cas_structure, ensure_ascii=False, indent=2)
    resultat = completer(PROMPT_SYSTEME, f"Cas structuré :\n{cas_json}", fournisseur, modele, max_tokens=2048)

    if "_erreur_parsing" not in resultat and resultat.get("categorie") not in CATEGORIES_VALIDES:
        resultat["_erreur_categorie_invalide"] = resultat.get("categorie")

    return resultat


def executer_mode_test(golden_path: str, fournisseur: str, modele: str, pause: float):
    with open(golden_path, encoding="utf-8") as f:
        golden = json.load(f)

    n_ok, n_total = 0, 0
    for cas in golden["cas"]:
        n_total += 1
        resultat = labelliser(cas["extraction_attendue"], fournisseur, modele)

        print(f"\n=== {cas['id']} ===")
        if "_erreur_parsing" in resultat:
            print(f"❌ ERREUR DE PARSING : {resultat['_erreur_parsing']}")
            time.sleep(pause)
            continue

        categorie_attendue = cas.get("categorie_attendue", "(non définie)")
        categorie_obtenue = resultat.get("categorie")
        ok = categorie_obtenue == categorie_attendue

        marqueur = "✅" if ok else "❌"
        print(f"  {marqueur} attendu={categorie_attendue!r}, obtenu={categorie_obtenue!r}")
        print(f"  Justification LLM : {resultat.get('justification')}")
        if "_erreur_categorie_invalide" in resultat:
            print(f"  ⚠️  GARDE-FOU DÉCLENCHÉ : catégorie hors des 3 valeurs autorisées "
                  f"({resultat['_erreur_categorie_invalide']!r})")

        if ok:
            n_ok += 1
        time.sleep(pause)

    print(f"\n{'=' * 50}\n{n_ok}/{n_total} catégories correctes")


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

    n_ok, n_erreurs, n_ignorees = 0, 0, 0
    with open(output_file, "a", encoding="utf-8") as f_out:
        for i, entree in enumerate(lignes, 1):
            cas_structure = entree.get("extraction", {})
            if "_erreur_parsing" in cas_structure:
                n_ignorees += 1
                continue

            resultat = labelliser(cas_structure, fournisseur, modele)
            if "_erreur_parsing" in resultat:
                n_erreurs += 1
            else:
                n_ok += 1

            sortie = {"id": entree["id"], "cas_structure": cas_structure, "label_etape3": resultat}
            f_out.write(json.dumps(sortie, ensure_ascii=False) + "\n")
            f_out.flush()

            if i % 10 == 0 or i == len(lignes):
                print(f"  {i}/{len(lignes)} traitées — {n_ok} ok, {n_erreurs} erreurs, "
                      f"{n_ignorees} ignorées (Étape 1 en échec)", file=sys.stderr)

            time.sleep(pause)

    print(f"\nTerminé -> {output_file}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Étape 3 : labellisation du niveau de priorité")
    parser.add_argument("--golden", default=None, help="Mode test : chemin du jeu de cas de référence")
    parser.add_argument("--input", default=None, help="Mode réel : sortie JSONL de extraction_etape1.py")
    parser.add_argument("--output", default=None, help="Mode réel : fichier de sortie JSONL")
    parser.add_argument("--fournisseur", choices=["groq", "gemini"], default="gemini")
    parser.add_argument("--modele", default=None,
                         help="Défaut selon le fournisseur : gemini-3.6-flash ou openai/gpt-oss-20b")
    parser.add_argument("--pause", type=float, default=3.0)
    args = parser.parse_args()

    modele = args.modele or ("gemini-3.6-flash" if args.fournisseur == "gemini" else "openai/gpt-oss-20b")

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