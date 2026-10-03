"""
nettoyer_labels_perimes.py

Retire de dialogues_473.jsonl les dialogues déjà générés (variante réussie)
dont le label (categorie_finale) ne correspond plus au label ACTUEL du même
cas dans le fichier de pipeline (Étapes 1-4).

Pourquoi ce script existe : la logique de reprise de generation_dialogue_etape5.py
(deja_traites) ne regarde que le couple (id, variante) déjà présent dans le
fichier de sortie -- elle ne vérifie jamais si le label associé est toujours
celui du pipeline. Si le pipeline a été relabellisé entre deux runs de
génération (ex. correction du référentiel), d'anciens dialogues restent dans
le fichier avec un label périmé, invisibles à la reprise.

Ce que fait ce script :
  1. Charge le label actuel de chaque cas depuis le fichier de pipeline
     (categorie_finale).
  2. Parcourt dialogues_473.jsonl : toute entrée réussie (_erreur absent) dont
     le label ne correspond plus au pipeline actuel est retirée du fichier ET
     conservée à part, dans un fichier "_perimes.jsonl" -- rien n'est perdu.
  3. Les entrées en échec (_erreur) sont laissées telles quelles : la reprise
     de generation_dialogue_etape5.py les retentera de toute façon.
  4. Une copie de sauvegarde du fichier d'origine est écrite avant toute
     modification (suffixe .avant_nettoyage).

Après ce nettoyage, relancez generation_dialogue_etape5.py avec le MÊME
--output : les entrées retirées ne seront plus "déjà faites" et seront
régénérées avec le label à jour.

Usage :
    python nettoyer_labels_perimes.py \
        --dialogues ../../data/normalized/dialogues_473.jsonl \
        --pipeline ../../data/normalized/mediqal_mcqm_pipeline_complet.jsonl
"""

import argparse
import json
import shutil
import sys
from pathlib import Path


def charger_labels_actuels(chemin: str) -> dict:
    """id -> categorie_finale, pour tous les cas du fichier de pipeline qui ont un label."""
    labels = {}
    with open(chemin, encoding="utf-8") as f:
        for ligne in f:
            ligne = ligne.strip()
            if not ligne:
                continue
            d = json.loads(ligne)
            categorie = d.get("categorie_finale")
            if categorie:
                labels[d["id"]] = categorie
    return labels


def main():
    parser = argparse.ArgumentParser(
        description="Retire de dialogues_473.jsonl les dialogues dont le label ne correspond plus au pipeline actuel"
    )
    parser.add_argument("--dialogues", required=True, help="Fichier de dialogues à nettoyer (modifié sur place)")
    parser.add_argument("--pipeline", required=True, help="Fichier de pipeline à jour (source des labels actuels)")
    args = parser.parse_args()

    dialogues_path = Path(args.dialogues)
    labels_actuels = charger_labels_actuels(args.pipeline)
    print(f"{len(labels_actuels)} cas avec un label dans {args.pipeline}", file=sys.stderr)

    conservees = []
    perimees = []
    n_erreurs = 0
    n_ok = 0
    with open(dialogues_path, encoding="utf-8") as f:
        for ligne in f:
            ligne_brute = ligne.strip()
            if not ligne_brute:
                continue
            entree = json.loads(ligne_brute)

            if "_erreur" in entree:
                conservees.append(ligne_brute)
                n_erreurs += 1
                continue

            n_ok += 1
            label_actuel = labels_actuels.get(entree.get("id"))
            if label_actuel is not None and label_actuel != entree.get("categorie_finale"):
                perimees.append(ligne_brute)
            else:
                conservees.append(ligne_brute)

    # Sauvegarde de sécurité avant d'écraser le fichier d'origine.
    sauvegarde = dialogues_path.with_suffix(dialogues_path.suffix + ".avant_nettoyage")
    shutil.copy2(dialogues_path, sauvegarde)
    print(f"Sauvegarde de l'original : {sauvegarde}", file=sys.stderr)

    with open(dialogues_path, "w", encoding="utf-8") as f:
        for ligne_brute in conservees:
            f.write(ligne_brute + "\n")

    chemin_perimees = dialogues_path.with_name(dialogues_path.stem + "_perimes.jsonl")
    with open(chemin_perimees, "w", encoding="utf-8") as f:
        for ligne_brute in perimees:
            f.write(ligne_brute + "\n")

    print(f"\n{n_ok} dialogues réussis examinés, {n_erreurs} échecs laissés tels quels (retentés au prochain run)",
          file=sys.stderr)
    print(f"{len(perimees)} dialogues périmés retirés -> {chemin_perimees}", file=sys.stderr)
    print(f"{len(conservees)} lignes conservées dans {dialogues_path}", file=sys.stderr)
    print("\nRelancez maintenant generation_dialogue_etape5.py avec le même --output : "
          "les entrées retirées seront régénérées avec le label à jour.", file=sys.stderr)


if __name__ == "__main__":
    main()
