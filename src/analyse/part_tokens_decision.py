"""Part des tokens de décision dans les dialogues SFT de l'hôtesse.

Mesure, avec le tokenizer de Qwen3-1.7B-Base :
- le nombre de tokens de l'agent par dialogue (les seuls notés, loss masquée) ;
- le nombre de tokens de la balise [[PRIORITE: ...]] ;
- la part de la balise, et celle du dernier message de l'agent (conclusion + balise) ;
- le découpage exact de chaque balise en tokens.

Lecture seule : ne modifie aucun fichier. Tourne sur CPU.

Usage :
    python src/analyse/part_tokens_decision.py --fichier data/sft/train.jsonl
"""
import argparse
import json
import re
import statistics
from collections import Counter

from transformers import AutoTokenizer

BALISE = re.compile(r"\[\[PRIORITE:\s*([a-z_]+)\s*\]\]")


def messages_du_dialogue(exemple):
    for cle in ("messages", "conversations"):
        if cle in exemple:
            return exemple[cle]
    raise KeyError(f"Format inattendu, clés trouvées : {list(exemple)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fichier", default="data/sft/train.jsonl")
    parser.add_argument("--modele", default="Qwen/Qwen3-1.7B-Base")
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.modele)

    tokens_agent, tokens_balise, tokens_dernier = [], [], []
    decoupages = {}
    sans_balise = 0

    with open(args.fichier, encoding="utf-8") as f:
        for ligne in f:
            messages = messages_du_dialogue(json.loads(ligne))
            agent = [m["content"] for m in messages if m["role"] == "assistant"]
            if not agent:
                continue
            n_agent = sum(len(tok(c, add_special_tokens=False)["input_ids"]) for c in agent)

            dernier = agent[-1]
            enc = tok(dernier, add_special_tokens=False, return_offsets_mapping=True)
            m = BALISE.search(dernier)
            if not m:
                sans_balise += 1
                continue
            # tokens qui chevauchent la balise, dans son contexte réel
            idx = [i for i, (d, fn) in enumerate(enc["offset_mapping"])
                   if d < m.end() and fn > m.start()]
            tokens_agent.append(n_agent)
            tokens_balise.append(len(idx))
            tokens_dernier.append(len(enc["input_ids"]))
            label = m.group(1)
            if label not in decoupages:
                decoupages[label] = tok.convert_ids_to_tokens([enc["input_ids"][i] for i in idx])

    n = len(tokens_agent)
    print(f"Dialogues analysés : {n} (sans balise : {sans_balise})")
    print(f"Tokens de l'agent par dialogue : moyenne {statistics.mean(tokens_agent):.0f}, "
          f"médiane {statistics.median(tokens_agent):.0f}")
    print(f"Tokens de la balise : moyenne {statistics.mean(tokens_balise):.1f}")
    print(f"Part de la balise (global) : {100 * sum(tokens_balise) / sum(tokens_agent):.1f} %")
    print(f"Part du dernier message (conclusion + balise) : "
          f"{100 * sum(tokens_dernier) / sum(tokens_agent):.1f} %")
    print("\nDécoupage de chaque balise :")
    for label, morceaux in sorted(decoupages.items()):
        print(f"  {label:18s} -> {len(morceaux)} tokens : {morceaux}")


if __name__ == "__main__":
    main()
