"""
load_frenchmedmcqa.py

Script d'ingestion du corpus FrenchMedMCQA (qanastek/frenchmedmcqa,
licence Apache 2.0) depuis Hugging Face, et normalisation vers le format
commun "vignette normalisée" (cf. schema.py).

Structure réelle du dataset source (vérifiée sur
huggingface.co/datasets/qanastek/frenchmedmcqa) :
    Pas de subset : un seul jeu de données.
    Splits : train (2171) / validation (312) / test (622)
    Champs : id, question, answer_a..answer_e, correct_answers (liste de
             lettres minuscules, ex. ["c","d","e"]), subject_name (toujours
             "pharmacie" — corpus limité à cette spécialité), choice_type
             ("single" | "multiple")
    Pas de champ "contexte clinique" — confirme que ce corpus est composé
    de questions d'examen isolées, sans vignette patient.

⚠️ Ce dataset utilise un script de chargement personnalisé côté HuggingFace
(nécessite trust_remote_code=True). À documenter dans le rapport comme
exécution de code tiers.

Usage :
    python load_frenchmedmcqa.py --split train \
        --output ../../data/normalized/frenchmedmcqa_train.jsonl
"""

import argparse
import json
import urllib.request
from pathlib import Path

from schema import VignetteNormalisee, valider_vignette

BASE_URL = "https://raw.githubusercontent.com/qanastek/FrenchMedMCQA/main/corpus"

NOM_FICHIER_PAR_SPLIT = {
    "train": "train.json",
    "validation": "dev.json",
    "test": "test.json",
}

def normaliser_ligne_frenchmedmcqa(ligne: dict, split: str) -> dict:
    reponses_brutes = ligne.get("answers", {})
    choix = {
        "a": reponses_brutes.get("a"),
        "b": reponses_brutes.get("b"),
        "c": reponses_brutes.get("c"),
        "d": reponses_brutes.get("d"),
        "e": reponses_brutes.get("e"),
    }

    correct = ligne.get("correct_answers") or []
    reponse_correcte = ",".join(r.upper() for r in correct) if correct else None
    
      # correct_answers peut arriver sous forme de liste (["c","d"]) ou de
    # chaîne selon la version du loader — on normalise vers "C,D" pour
    # rester cohérent avec le format déjà utilisé pour MediQAl.
    

    vignette = VignetteNormalisee(
        id=f"frenchmedmcqa_{ligne.get('id')}",
        source="FrenchMedMCQA",
        subset=None,
        split=split,
        langue="fr",
        contexte_clinique=None,
        question=ligne.get("question"),
        choix=choix,
        reponse_correcte=reponse_correcte,
        specialite=ligne.get("subject_name"),
        type_question=ligne.get("type"),
        task="QCM",
    )
    return vignette.to_dict()

  


def charger_depuis_github(split: str):
    if split not in NOM_FICHIER_PAR_SPLIT:
        raise ValueError(f"Split inconnu : {split}")
    url = f"{BASE_URL}/{NOM_FICHIER_PAR_SPLIT[split]}"
    with urllib.request.urlopen(url) as reponse:
        return json.loads(reponse.read().decode("utf-8"))


def charger_depuis_fichier_local(chemin: str):
    chemin = Path(chemin)
    if chemin.suffix == ".jsonl":
        with open(chemin, encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]
    elif chemin.suffix == ".json":
        with open(chemin, encoding="utf-8") as f:
            return json.load(f)
    elif chemin.suffix == ".parquet":
        import pandas as pd
        return pd.read_parquet(chemin).to_dict(orient="records")
    else:
        raise ValueError(f"Format de fichier local non supporté : {chemin.suffix}")


def main():
    parser = argparse.ArgumentParser(description="Ingestion et normalisation du corpus FrenchMedMCQA")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="train")
    parser.add_argument(
        "--local-file", default=None,
        help="Fichier local (jsonl/json/parquet) à utiliser au lieu du téléchargement HuggingFace"
    )
    parser.add_argument("--output", required=True, help="Chemin du fichier JSONL de sortie")
    parser.add_argument("--limit", type=int, default=None, help="Limiter le nombre de lignes traitées (debug)")
    args = parser.parse_args()

    if args.local_file:
        print(f"Chargement depuis le fichier local : {args.local_file}")
        lignes_brutes = charger_depuis_fichier_local(args.local_file)
    else:
        print(f"Téléchargement depuis GitHub : qanastek/FrenchMedMCQA [{NOM_FICHIER_PAR_SPLIT[args.split]}]")
        lignes_brutes = charger_depuis_github(args.split)

    if args.limit:
        lignes_brutes = lignes_brutes[: args.limit]

    lignes_normalisees = []
    lignes_invalides = 0
    for ligne in lignes_brutes:
        vignette = normaliser_ligne_frenchmedmcqa(ligne, args.split)
        manquants = valider_vignette(vignette)
        if manquants:
            lignes_invalides += 1
            continue
        lignes_normalisees.append(vignette)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for vignette in lignes_normalisees:
            f.write(json.dumps(vignette, ensure_ascii=False) + "\n")

    print(f"\n{len(lignes_normalisees)} vignettes normalisées écrites dans {output_path}")
    if lignes_invalides:
        print(f"  {lignes_invalides} lignes ignorées (champs obligatoires manquants)")

    print("  Rappel : ce corpus ne contient aucun contexte clinique (0% attendu) — "
          "normal, à ne pas confondre avec une erreur d'extraction.")


if __name__ == "__main__":
    main()
