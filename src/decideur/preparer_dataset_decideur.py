#!/usr/bin/env python3
"""Prépare le dataset du décideur (encodeur ModernCamemBERT-bio) à partir des dialogues SFT.

Pour chaque dialogue de data/sft/{train,validation}.jsonl :
  - le label est lu dans la balise [[PRIORITE: ...]] du dernier tour de l'agent ;
  - ce dernier tour (phrase de conclusion + balise) est retiré ;
  - la conversation restante devient une transcription texte (« Agent : … / Patient : … ») ;
  - l'exemple produit est {"text", "label", "label_id", "id", "cas_id", "variante"}.

Deux variantes d'entrée sont produites :
  - complet : tours de l'agent et du patient ;
  - patient : messages du patient uniquement (test des indices laissés par l'agent).

Aucun fichier existant n'est modifié. Le découpage train / validation est conservé tel
quel (pas de suréchantillonnage, pas de pondération, pas d'aléatoire).

Usage (depuis la racine du projet) :
    python src/decideur/preparer_dataset_decideur.py
    python src/decideur/preparer_dataset_decideur.py --sans-tokenizer
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

# Ordre FIXE des classes : label_id 0, 1, 2. Réutilisé par l'entraînement et l'inférence.
LABELS = ("urgence_maximale", "moderee", "differee")
LABEL2ID = {lab: i for i, lab in enumerate(LABELS)}
RE_BALISE = re.compile(r"\[\[PRIORITE:\s*([a-z_]+)\s*\]\]")
VARIANTES = ("complet", "patient")
FICHIERS = ("train.jsonl", "validation.jsonl")
MODELE_ENCODEUR = "almanach/ModernCamemBERT-bio-v2-base"


def formater_transcription(messages, variante="complet"):
    """Transcription texte des tours (hors système). Réutilisée telle quelle à l'inférence."""
    lignes = []
    for m in messages:
        if m["role"] == "assistant":
            if variante == "complet":
                lignes.append(f"Agent : {m['content']}")
        elif m["role"] == "user":
            lignes.append(f"Patient : {m['content']}")
    return "\n".join(lignes)


def convertir(exemple, variante):
    """Transforme un dialogue SFT en exemple du décideur. Lève ValueError si incohérent."""
    messages = exemple["messages"]
    dernier = messages[-1]
    if dernier["role"] != "assistant":
        raise ValueError("le dernier message n'est pas celui de l'agent")
    trouve = RE_BALISE.search(dernier["content"])
    if not trouve:
        raise ValueError("balise [[PRIORITE]] absente du dernier message")
    label = trouve.group(1)
    if label not in LABEL2ID:
        raise ValueError(f"label inconnu : {label}")
    attendu = exemple.get("categorie_finale")
    if attendu and attendu != label:
        raise ValueError(f"balise ({label}) différente de categorie_finale ({attendu})")

    corps = [m for m in messages[:-1] if m["role"] != "system"]
    if any(RE_BALISE.search(m["content"]) for m in corps):
        raise ValueError("balise présente avant la conclusion (fuite du label)")
    if not corps or corps[-1]["role"] != "user":
        raise ValueError("la conversation ne se termine pas par un message du patient")

    return {
        "text": formater_transcription(corps, variante),
        "label": label,
        "label_id": LABEL2ID[label],
        "id": exemple.get("id"),
        "cas_id": exemple.get("cas_id"),
        "variante": exemple.get("variante"),
    }


def mesurer_longueurs(chemin_sortie, modele):
    """Longueur en tokens des transcriptions, pour choisir max_length sans tronquer."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(modele)
    print(f"\nLongueur des transcriptions en tokens ({modele}) :")
    for variante in VARIANTES:
        longueurs = []
        for nom in FICHIERS:
            with open(chemin_sortie / variante / nom, encoding="utf-8") as f:
                for l in f:
                    if l.strip():
                        longueurs.append(len(tok(json.loads(l)["text"])["input_ids"]))
        longueurs.sort()
        p95 = longueurs[int(0.95 * (len(longueurs) - 1))]
        print(f"  {variante:<8} min {longueurs[0]:>4} | médiane {longueurs[len(longueurs) // 2]:>4}"
              f" | p95 {p95:>4} | max {longueurs[-1]:>4}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--entree", default="data/sft", help="dossier des dialogues SFT")
    p.add_argument("--sortie", default="data/decideur", help="dossier de sortie")
    p.add_argument("--modele", default=MODELE_ENCODEUR, help="tokenizer pour la mesure des longueurs")
    p.add_argument("--sans-tokenizer", action="store_true", help="ne pas mesurer les longueurs")
    p.add_argument("--ecraser", action="store_true", help="autoriser l'écrasement de --sortie")
    args = p.parse_args()

    entree, sortie = Path(args.entree), Path(args.sortie)
    if sortie.exists() and any(sortie.iterdir()) and not args.ecraser:
        sys.exit(f"{sortie} existe déjà et n'est pas vide (utiliser --ecraser).")

    erreurs = 0
    exemples = {}
    for variante in VARIANTES:
        (sortie / variante).mkdir(parents=True, exist_ok=True)
        for nom in FICHIERS:
            compte, cas = Counter(), {lab: set() for lab in LABELS}
            with open(entree / nom, encoding="utf-8") as f_in, \
                 open(sortie / variante / nom, "w", encoding="utf-8") as f_out:
                for n, ligne in enumerate(f_in, 1):
                    if not ligne.strip():
                        continue
                    ex = json.loads(ligne)
                    try:
                        out = convertir(ex, variante)
                    except ValueError as e:
                        erreurs += 1
                        print(f"  ERREUR {nom}:{n} ({ex.get('id')}) : {e}")
                        continue
                    f_out.write(json.dumps(out, ensure_ascii=False) + "\n")
                    compte[out["label"]] += 1
                    cas[out["label"]].add(out["cas_id"])
                    exemples.setdefault(variante, out)
            print(f"== {variante}/{nom} : {sum(compte.values())} exemples")
            for lab in LABELS:
                print(f"   {lab:<17} {compte[lab]:>4} dialogues, {len(cas[lab]):>4} cas distincts")

    # Contrôle anti-fuite : aucun cas_id commun entre train et validation
    def cas_ids(nom):
        with open(sortie / "complet" / nom, encoding="utf-8") as f:
            return {json.loads(l)["cas_id"] for l in f if l.strip()}
    communs = cas_ids("train.jsonl") & cas_ids("validation.jsonl")
    print(f"\nCas communs train / validation : {len(communs)}"
          + (" (FUITE !)" if communs else " (OK)"))

    for variante, ex in exemples.items():
        print(f"\nExemple ({variante}), texte tronqué :")
        print(ex["text"][:300], "...")
        print("Label :", ex["label"], "| label_id :", ex["label_id"])

    if not args.sans_tokenizer:
        mesurer_longueurs(sortie, args.modele)

    print(f"\nErreurs de conversion : {erreurs}")
    sys.exit(1 if erreurs or communs else 0)


if __name__ == "__main__":
    main()