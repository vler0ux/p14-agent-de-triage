"""
preparer_dataset_sft.py

Transforme les dialogues générés (sortie de generation_dialogue_etape5.py)
en dataset conversationnel au format attendu par TRL (SFTTrainer) et
compatible avec le chat template de Qwen3.

Format de sortie, une ligne JSONL par exemple :
    {
      "messages": [
        {"role": "system", "content": "..."},
        {"role": "assistant", "content": "..."},   # première question de l'agent
        {"role": "user", "content": "..."},        # réponse du patient
        ...
        {"role": "assistant", "content": "..."}    # conclusion finale
      ],
      "id": "...",                    # conservé pour traçabilité, ignoré par l'entraînement
      "categorie_finale": "..."       # idem
    }

Note de conception : dans ce format, "assistant" = l'agent de triage (le
rôle que le modèle final devra jouer en production), "user" = le patient.
C'est l'inverse de la convention utilisée pendant la GÉNÉRATION (Étape 5),
où "agent"/"patient" étaient de simples étiquettes de mise en scène — ici
on les traduit dans le vocabulaire role-based qu'attend l'entraînement.

Usage :
    python preparer_dataset_sft.py \
        --input ../../data/normalized/mediqal_mcqm_dialogues.jsonl \
        --output-dir ../../data/sft \
        --val-ratio 0.1
"""

import argparse
import json
import random
import sys
from pathlib import Path

SYSTEM_PROMPT = (
    "Tu es un agent de triage médical au Centre Hospitalier Saint-Aurélien (CHSA). "
    "Ton rôle est de recueillir les symptômes du patient par des questions ciblées, "
    "claires et rassurantes, puis de conclure en indiquant le niveau de prise en "
    "charge nécessaire, dans un langage humain et compréhensible — jamais de jargon "
    "médical ou administratif."
)


def convertir_en_messages(entree: dict) -> list:
    """Convertit une entrée de dialogue (format Étape 5) en liste de messages
    role-based, prête pour l'entraînement."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for tour in entree["dialogue"]:
        role_source = tour.get("role")
        contenu = tour.get("content", "").strip()
        if not contenu:
            continue
        if role_source == "agent":
            messages.append({"role": "assistant", "content": contenu})
        elif role_source == "patient":
            messages.append({"role": "user", "content": contenu})
        # tout autre rôle inattendu est ignoré plutôt que de planter —
        # préférable de perdre un exemple que de casser tout le dataset

    conclusion = entree.get("conclusion", "").strip()
    if conclusion:
        messages.append({"role": "assistant", "content": conclusion})

    return messages


def valider_exemple(messages: list) -> list:
    """Vérifications minimales avant d'inclure un exemple dans le dataset
    final — mieux vaut exclure un exemple douteux que polluer l'entraînement."""
    problemes = []
    if len(messages) < 3:  # system + au moins 1 échange
        problemes.append("trop court (moins de 2 tours utiles)")
    roles = [m["role"] for m in messages]
    if "assistant" not in roles:
        problemes.append("aucun tour 'assistant'")
    if "user" not in roles:
        problemes.append("aucun tour 'user' (patient)")
    # alternance : deux messages consécutifs ne devraient pas avoir le même rôle
    # (hors le tout premier message 'system')
    for i in range(1, len(messages) - 1):
        if messages[i]["role"] == messages[i + 1]["role"]:
            problemes.append(f"rupture d'alternance des rôles en position {i}")
            break
    return problemes


def main():
    parser = argparse.ArgumentParser(description="Préparation du dataset SFT à partir des dialogues générés")
    parser.add_argument("--input", required=True, nargs="+",
                         help="Un ou plusieurs fichiers JSONL de dialogues (sortie de generation_dialogue_etape5.py)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Proportion réservée à la validation")
    parser.add_argument("--seed", type=int, default=42, help="Graine aléatoire, pour reproductibilité du split")
    args = parser.parse_args()

    exemples_valides = []
    n_ignores = 0
    n_total = 0

    for chemin_input in args.input:
        with open(chemin_input, encoding="utf-8") as f:
            for ligne in f:
                ligne = ligne.strip()
                if not ligne:
                    continue
                n_total += 1
                entree = json.loads(ligne)

                if "dialogue" not in entree or "conclusion" not in entree:
                    n_ignores += 1  # cas en échec (ex. _erreur), pas un vrai dialogue
                    continue

                messages = convertir_en_messages(entree)
                problemes = valider_exemple(messages)
                if problemes:
                    n_ignores += 1
                    print(f"  Exemple {entree.get('id')} ignoré : {', '.join(problemes)}", file=sys.stderr)
                    continue

                exemples_valides.append({
                    "messages": messages,
                    "id": entree.get("id"),
                    "categorie_finale": entree.get("categorie_finale"),
                })

    print(f"\n{n_total} entrées lues, {len(exemples_valides)} exemples valides, {n_ignores} ignorés", file=sys.stderr)

    if len(exemples_valides) < 20:
        print(f"  ⚠️  ATTENTION : seulement {len(exemples_valides)} exemples valides — "
              f"beaucoup trop peu pour un entraînement réel (cible du brief : ~5000). "
              f"Ce dataset ne devrait servir qu'à un test de mécanique du pipeline, "
              f"pas à un entraînement de qualité.", file=sys.stderr)

    random.seed(args.seed)
    random.shuffle(exemples_valides)

    n_val = max(1, round(len(exemples_valides) * args.val_ratio)) if len(exemples_valides) >= 10 else 0
    val_set = exemples_valides[:n_val]
    train_set = exemples_valides[n_val:]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for nom, jeu in [("train", train_set), ("validation", val_set)]:
        chemin = output_dir / f"{nom}.jsonl"
        with open(chemin, "w", encoding="utf-8") as f:
            for ex in jeu:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"  {nom} : {len(jeu)} exemples -> {chemin}", file=sys.stderr)

    # Répartition des catégories, utile pour repérer un déséquilibre avant d'entraîner
    from collections import Counter
    repartition = Counter(ex["categorie_finale"] for ex in exemples_valides)
    print(f"\nRépartition des catégories (train+val) : {dict(repartition)}", file=sys.stderr)


if __name__ == "__main__":
    main()