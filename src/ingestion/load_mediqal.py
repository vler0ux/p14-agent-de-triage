"""
load_mediqal.py

Script d'ingestion du corpus MediQAl (ANR-MALADES, licence CC-BY-4.0) depuis
Hugging Face, et normalisation vers le format commun "vignette normalisée"
(cf. schema.py) utilisé par le reste du pipeline.

Structure réelle du dataset source (vérifiée sur huggingface.co/datasets/ANR-MALADES/MediQAl) :
    Subsets : mcqm (10.6k, QCM multi-réponses), mcqu (17k, QCM réponse unique),
              oeq (4.97k, questions ouvertes — schéma non encore vérifié, à confirmer
              avant usage).
    Splits  : train / validation / test
    Colonnes (mcqm / mcqu) : id, clinical_case, question, answer_a..answer_e,
              correct_answers, task, medical_subject, question_type

Usage :
    python load_mediqal.py --subset mcqm --split train \
        --output ../../data/normalized/mediqal_mcqm_train.jsonl

Note réseau : ce script télécharge le dataset depuis huggingface.co via la
librairie `datasets`. Si vous travaillez dans un environnement sans accès
direct à huggingface.co, utilisez --local-file pour pointer vers un export
déjà téléchargé (jsonl / json / parquet) plutôt que --subset/--split.
"""

import argparse
import json
from pathlib import Path

from schema import VignetteNormalisee, valider_vignette


def normaliser_ligne_mediqal(ligne: dict, subset: str, split: str) -> dict:
    """Transforme une ligne brute MediQAl (subset mcqm ou mcqu) en vignette normalisée."""
    choix = None
    if subset in ("mcqm", "mcqu"):
        choix = {
            "a": ligne.get("answer_a"),
            "b": ligne.get("answer_b"),
            "c": ligne.get("answer_c"),
            "d": ligne.get("answer_d"),
            "e": ligne.get("answer_e"),
        }

    vignette = VignetteNormalisee(
        id=f"mediqal_{subset}_{ligne.get('id')}",
        source="MediQAl",
        subset=subset,
        split=split,
        langue="fr",
        contexte_clinique=ligne.get("clinical_case"),
        question=ligne.get("question"),
        choix=choix,
        reponse_correcte=ligne.get("correct_answers"),
        specialite=ligne.get("medical_subject"),
        type_question=ligne.get("question_type"),
        task=ligne.get("task"),
    )
    return vignette.to_dict()


def charger_depuis_huggingface(subset: str, split: str):
    from datasets import load_dataset
    ds = load_dataset("ANR-MALADES/MediQAl", subset, split=split)
    return list(ds)


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
    parser = argparse.ArgumentParser(description="Ingestion et normalisation du corpus MediQAl")
    parser.add_argument("--subset", choices=["mcqm", "mcqu", "oeq"], default="mcqm")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="train")
    parser.add_argument(
        "--local-file", default=None,
        help="Fichier local (jsonl/json/parquet) à utiliser au lieu du téléchargement HuggingFace"
    )
    parser.add_argument("--output", required=True, help="Chemin du fichier JSONL de sortie")
    parser.add_argument("--limit", type=int, default=None, help="Limiter le nombre de lignes traitées (debug)")
    args = parser.parse_args()

    if args.subset == "oeq":
        print(
            "ATTENTION : le schéma exact du subset 'oeq' (questions ouvertes) n'a pas "
            "encore été vérifié dans ce script — vérifiez les noms de colonnes réels "
            "avant de l'utiliser en production."
        )

    if args.local_file:
        print(f"Chargement depuis le fichier local : {args.local_file}")
        lignes_brutes = charger_depuis_fichier_local(args.local_file)
    else:
        print(f"Téléchargement depuis Hugging Face : ANR-MALADES/MediQAl [{args.subset}/{args.split}]")
        lignes_brutes = charger_depuis_huggingface(args.subset, args.split)

    if args.limit:
        lignes_brutes = lignes_brutes[: args.limit]

    lignes_normalisees = []
    lignes_invalides = 0
    for ligne in lignes_brutes:
        vignette = normaliser_ligne_mediqal(ligne, args.subset, args.split)
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

    avec_contexte = sum(1 for v in lignes_normalisees if v["contexte_clinique"])
    if lignes_normalisees:
        pct = avec_contexte / len(lignes_normalisees) * 100
        print(f"  dont {avec_contexte} avec contexte clinique ({pct:.1f}%) — ce sont les plus "
              f"utiles pour la génération de vignettes de triage")


if __name__ == "__main__":
    main()
