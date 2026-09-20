"""
diag_masquage.py (v2) — Diagnostic SANS GPU : que va-t-on réellement entraîner ?

Vérifie, sur TON dataset et le VRAI tokenizer de Qwen3-1.7B-Base, qu'un template
avec marqueurs {% generation %} (fichier template_triage.jinja) produit bien :
  - un masque non vide sur chaque exemple ;
  - uniquement les tours de l'agent (+ son token de fin de tour <|im_end|>).

Prérequis : pip install transformers   (torch et GPU inutiles ici)

Usage (depuis src/entrainement/) :
    python diag_masquage.py ../../data/sft/train.jsonl
    python diag_masquage.py ../../data/sft/train.jsonl --template template_triage.jinja
"""

import argparse
import json
import re
import sys

from transformers import AutoTokenizer


def a_marqueurs(template: str) -> bool:
    return re.search(r"\{%-?\s*generation\s*-?%\}", template or "") is not None


def segments_entraines(tok, ids, masque):
    segments, groupe = [], []
    for tid, m in zip(ids, masque):
        if m:
            groupe.append(tid)
        elif groupe:
            segments.append(tok.decode(groupe))
            groupe = []
    if groupe:
        segments.append(tok.decode(groupe))
    return segments


parser = argparse.ArgumentParser()
parser.add_argument("train_file")
parser.add_argument("--template", default="template_triage.jinja")
parser.add_argument("--tokenizer", default="Qwen/Qwen3-1.7B-Base")
args = parser.parse_args()

tok = AutoTokenizer.from_pretrained(args.tokenizer)

print("[1] Template NATIF du tokenizer (celui que TRL n'a pas su patcher) :")
print("-" * 60)
print(tok.chat_template)
print("-" * 60)
print("    marqueurs {% generation %} dans le natif :", a_marqueurs(tok.chat_template))

with open(args.template, encoding="utf-8") as f:
    tpl = f.read()
print(f"\n[2] Template testé : {args.template} | marqueurs présents : {a_marqueurs(tpl)}")
if not a_marqueurs(tpl):
    sys.exit("⚠️  Ce template n'a pas de marqueurs {% generation %} : rien à tester.")

id_fin_tour = tok.convert_tokens_to_ids("<|im_end|>")
print(f"    id de <|im_end|> : {id_fin_tour}")

with open(args.train_file, encoding="utf-8") as f:
    exemples = [json.loads(l)["messages"] for l in f if l.strip()]
print(f"\n[3] {len(exemples)} exemples à analyser...")

n_erreurs = n_sans_masque = n_stop_ko = 0
parts = []
premier = None
for i, messages in enumerate(exemples):
    try:
        sortie = tok.apply_chat_template(
            messages, chat_template=tpl, tokenize=True,
            return_dict=True, return_assistant_tokens_mask=True,
        )
    except Exception as e:
        n_erreurs += 1
        if n_erreurs <= 3:
            print(f"    ❌ exemple {i} : {type(e).__name__}: {e}")
        continue
    ids, masque = sortie["input_ids"], sortie["assistant_masks"]
    if sum(masque) == 0:
        n_sans_masque += 1
        continue
    parts.append(sum(masque) / len(ids))
    dernier = [t for t, m in zip(ids, masque) if m and tok.decode([t]).strip() != ""][-1]
    if dernier != id_fin_tour:
        n_stop_ko += 1
    if premier is None:
        premier = (messages, ids, masque)

print(f"    erreurs de rendu            : {n_erreurs}")
print(f"    exemples sans aucun token appris : {n_sans_masque}")
print(f"    exemples où <|im_end|> n'est pas appris : {n_stop_ko}")
if parts:
    print(f"    part moyenne de tokens appris : {100 * sum(parts) / len(parts):.0f} % "
          f"(sans masquage : 100 %)")

if premier:
    messages, ids, masque = premier
    print("\n[4] Premier exemple valide — rendu complet :")
    print("-" * 60)
    print(tok.apply_chat_template(messages, chat_template=tpl, tokenize=False))
    print("-" * 60)
    print("    Passages ENTRAÎNÉS (tout le reste est ignoré par la loss) :")
    for seg in segments_entraines(tok, ids, masque):
        print("   >>>", repr(seg))
    print("\n[5] Prompt d'inférence (fin) après le 1er message patient :")
    idx = next(k for k, m in enumerate(messages) if m["role"] == "user")
    prompt = tok.apply_chat_template(messages[: idx + 1], chat_template=tpl,
                                     tokenize=False, add_generation_prompt=True)
    print("   ...", repr(prompt[-70:]))