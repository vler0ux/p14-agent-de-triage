"""
echantillonner_vignettes.py

Tire un échantillon aléatoire reproductible depuis un fichier de vignettes
normalisées (JSONL), pour inspection manuelle ou constitution d'un jeu de
cas de référence.

Reproductibilité : la seed est fixée par défaut (42) et toujours affichée
dans la sortie, pour que n'importe qui puisse retirer exactement le même
échantillon à partir du même fichier source — exigence de traçabilité du
projet (audit des choix méthodologiques).

Usage :
    python echantillonner_vignettes.py --input ../../data/normalized/mediqal_mcqm_train.jsonl --n 15

    # Ne garder que les vignettes avec un contexte clinique renseigné :
    python echantillonner_vignettes.py --input ../../data/normalized/mediqal_mcqm_train.jsonl \
        --n 15 --avec-contexte-clinique

    # Écrire l'échantillon dans un fichier plutôt que l'afficher :
    python echantillonner_vignettes.py --input ../../data/normalized/mediqal_mcqm_train.jsonl \
        --n 15 --avec-contexte-clinique \
        --output ../../data/normalized/echantillons/mediqal_echantillon_15.jsonl
"""

import argparse
import json
import random
import sys
from pathlib import Path


def charger_vignettes(chemin: Path, avec_contexte_clinique: bool) -> list:
    vignettes = []
    with open(chemin, encoding="utf-8") as f:
        for ligne in f:
            ligne = ligne.strip()
            if not ligne:
                continue
            v = json.loads(ligne)
            if avec_contexte_clinique and not v.get("contexte_clinique"):
                continue
            vignettes.append(v)
    return vignettes


def main():
    parser = argparse.ArgumentParser(
        description="Tire un échantillon aléatoire reproductible depuis un fichier de vignettes normalisées."
    )
    parser.add_argument("--input", required=True, help="Fichier JSONL de vignettes normalisées")
    parser.add_argument("--n", type=int, default=15, help="Taille de l'échantillon (défaut : 15)")
    parser.add_argument("--seed", type=int, default=42, help="Graine aléatoire, pour reproductibilité (défaut : 42)")
    parser.add_argument(
        "--avec-contexte-clinique", action="store_true",
        help="Ne garder que les vignettes dont le champ contexte_clinique est renseigné"
    )
    parser.add_argument("--output", default=None, help="Fichier de sortie JSONL (sinon, affiche sur stdout)")
    args = parser.parse_args()

    input_path = Path(args.input)
    vignettes = charger_vignettes(input_path, args.avec_contexte_clinique)

    print(f"# Source : {input_path}", file=sys.stderr)
    print(f"# {len(vignettes)} vignettes disponibles"
          f"{' (avec contexte clinique)' if args.avec_contexte_clinique else ''}", file=sys.stderr)
    print(f"# Seed utilisée : {args.seed} (relancer avec --seed {args.seed} reproduit exactement cet échantillon)",
          file=sys.stderr)

    if len(vignettes) < args.n:
        print(f"# ATTENTION : seulement {len(vignettes)} vignettes disponibles, "
              f"moins que les {args.n} demandées — tout le pool sera retourné.", file=sys.stderr)

    random.seed(args.seed)
    echantillon = random.sample(vignettes, min(args.n, len(vignettes)))

    lignes_json = [json.dumps(v, ensure_ascii=False) for v in echantillon]

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lignes_json) + "\n")
        print(f"# {len(echantillon)} vignettes écrites dans {output_path}", file=sys.stderr)
    else:
        for ligne in lignes_json:
            print(ligne)


if __name__ == "__main__":
    main()