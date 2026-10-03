"""
nettoyer_conclusions_ambigues.py

Ré-applique le contrôle de cohérence CORRIGÉ (mots-clés mutuellement exclusifs
entre niveaux, cf. correctif du 27/09) à chaque dialogue déjà généré dans
dialogues_473.jsonl. Retire ceux dont la conclusion mélange le vocabulaire de
plusieurs niveaux -- l'ancien contrôle les avait laissés passer à tort.

Les entrées retirées ne sont plus "déjà faites" : relancer
generation_dialogue_etape5.py les régénère automatiquement (mécanisme de
reprise existant), avec le prompt calibré et le contrôle corrigé.

Usage :
    python nettoyer_conclusions_ambigues.py --dialogues ../../data/normalized/dialogues_473.jsonl
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

MOTS_CLES_CONCLUSION = {
    "urgence_maximale": ["tout de suite", "immédiat", "sans attendre", "maintenant", "sans délai"],
    "moderee": ["rapidement", "dans la journée", "assez vite", "bientôt", "sous peu"],
    "differee": ["patienter", "pas urgent", "attendre", "pas d'urgence"],
}
EXCLUSIONS_MOTS_CLES = {"attendre": ["sans attendre"]}


def _mot_present(mot: str, texte: str) -> bool:
    for expr in EXCLUSIONS_MOTS_CLES.get(mot, []):
        texte = texte.replace(expr, "")
    return mot in texte


def conclusion_coherente(conclusion: str, categorie: str) -> bool:
    bas = conclusion.lower()
    mots_autres = [m for c, mots in MOTS_CLES_CONCLUSION.items() if c != categorie for m in mots]
    return (any(_mot_present(k, bas) for k in MOTS_CLES_CONCLUSION.get(categorie, []))
            and not any(_mot_present(k, bas) for k in mots_autres))


def extraire_conclusion(entree: dict) -> str:
    """La conclusion est le dernier message 'agent' du dialogue (avant ajout de la balise par preparer_dataset_sft.py)."""
    dialogue = entree.get("dialogue")
    if not isinstance(dialogue, list) or not dialogue:
        return ""
    dernier = dialogue[-1]
    if isinstance(dernier, dict) and dernier.get("role") == "agent":
        return str(dernier.get("content", ""))
    # Selon la structure exacte sauvegardée, la conclusion peut aussi être un champ séparé.
    return str(entree.get("conclusion", ""))


def main():
    parser = argparse.ArgumentParser(description="Retire les dialogues dont la conclusion mélange le vocabulaire de plusieurs niveaux")
    parser.add_argument("--dialogues", required=True)
    args = parser.parse_args()

    dialogues_path = Path(args.dialogues)
    conserves, perimes = [], []
    n_ok, n_erreur = 0, 0

    with open(dialogues_path, encoding="utf-8") as f:
        for ligne in f:
            ligne_brute = ligne.strip()
            if not ligne_brute:
                continue
            entree = json.loads(ligne_brute)

            if "_erreur" in entree:
                conserves.append(ligne_brute)  # déjà retenté au prochain run, rien à faire
                n_erreur += 1
                continue

            n_ok += 1
            categorie = entree.get("categorie_finale")
            conclusion = extraire_conclusion(entree)
            if categorie in MOTS_CLES_CONCLUSION and conclusion and not conclusion_coherente(conclusion, categorie):
                perimes.append(ligne_brute)
            else:
                conserves.append(ligne_brute)

    sauvegarde = dialogues_path.with_suffix(dialogues_path.suffix + ".avant_nettoyage_conclusions")
    shutil.copy2(dialogues_path, sauvegarde)
    print(f"Sauvegarde de l'original : {sauvegarde}", file=sys.stderr)

    with open(dialogues_path, "w", encoding="utf-8") as f:
        for ligne_brute in conserves:
            f.write(ligne_brute + "\n")

    chemin_perimes = dialogues_path.with_name(dialogues_path.stem + "_conclusions_ambigues.jsonl")
    with open(chemin_perimes, "w", encoding="utf-8") as f:
        for ligne_brute in perimes:
            f.write(ligne_brute + "\n")

    print(f"\n{n_ok} dialogues réussis examinés, {n_erreur} échecs laissés tels quels", file=sys.stderr)
    print(f"{len(perimes)} conclusions ambiguës retirées -> {chemin_perimes}", file=sys.stderr)
    print(f"{len(conserves)} lignes conservées dans {dialogues_path}", file=sys.stderr)
    print("\nRelancez generation_dialogue_etape5.py (avec le prompt calibré) : "
          "les entrées retirées seront régénérées.", file=sys.stderr)


if __name__ == "__main__":
    main()
