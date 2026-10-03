"""
preparer_dpo_rejected.py

Génère des paires "rejected" pour le DPO en faisant tourner le modèle SFT
LOCAL (celui qu'on vient de tester dans app_demo.py) en mode plus "libre"
(température élevée, pas de garde-fous) sur des contextes tirés de
train.jsonl -- exactement les dérives (langues étrangères, JSON qui fuite)
qu'on veut apprendre au modèle à éviter.

Le "chosen" de chaque paire est repris tel quel de train.jsonl (déjà bon).
Le "rejected" est la réponse générée ici, sur EXACTEMENT le même contexte.

Usage :
    python preparer_dpo_rejected.py \
        --adapter-dir ../../qwen3-1.7b-triage-sft \
        --train-file ../../data/sft/train.jsonl \
        --n 50 \
        --output ../../data/dpo/rejected_sft_a_relire.jsonl
"""

import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Génère des paires 'rejected' pour le DPO depuis le modèle SFT local")
    parser.add_argument("--adapter-dir", required=True, help="Dossier de l'adaptateur SFT (V1 pour la répétition)")
    parser.add_argument("--modele-base", default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--train-file", required=True, help="train.jsonl (source des contextes ET des 'chosen')")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=1.1,
                         help="Volontairement élevée : on VEUT que le modèle dérive, c'est le but ici")
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--output", default="rejected_sft_a_relire.jsonl")
    args = parser.parse_args()

    adapter_dir = Path(args.adapter_dir)
    assert (adapter_dir / "adapter_config.json").exists(), f"adapter_config.json introuvable dans {adapter_dir}"

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    cuda = torch.cuda.is_available()
    dtype = torch.float32 if not cuda else (torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16)
    print(f"Device : {'cuda' if cuda else 'cpu'} | dtype : {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(args.modele_base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    chemin_template = adapter_dir / "chat_template.jinja"
    if chemin_template.exists():
        tokenizer.chat_template = chemin_template.read_text(encoding="utf-8")
        print(f"Template de chat chargé depuis {chemin_template}")
    else:
        raise FileNotFoundError(
            f"{chemin_template} introuvable -- indispensable pour reproduire exactement le format d'entraînement."
        )

    print(f"Chargement du modèle de base : {args.modele_base}")
    modele_base = AutoModelForCausalLM.from_pretrained(args.modele_base, dtype=dtype, device_map="auto" if cuda else None)
    print(f"Chargement de l'adaptateur : {adapter_dir}")
    modele = PeftModel.from_pretrained(modele_base, str(adapter_dir))
    modele.eval()

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    # Charge train.jsonl et ne garde qu'UN exemple par cas_id (évite de sur-représenter
    # les cas à plusieurs variantes dans les paires rejected).
    exemples_par_cas = {}
    with open(args.train_file, encoding="utf-8") as f:
        for ligne in f:
            d = json.loads(ligne)
            cle = d.get("cas_id", d.get("id"))
            exemples_par_cas.setdefault(cle, d)
    exemples = list(exemples_par_cas.values())
    print(f"{len(exemples)} cas uniques disponibles dans {args.train_file}")

    random.seed(args.seed)
    echantillon = random.sample(exemples, min(args.n, len(exemples)))
    print(f"{len(echantillon)} paires à générer (seed {args.seed}) -> {args.output}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "a", encoding="utf-8") as f_out:
        for i, exemple in enumerate(echantillon, 1):
            messages = exemple["messages"]

            # On ne garde le "chosen" que sur le DERNIER tour assistant : c'est lui qu'on compare
            # au "rejected" qu'on génère sur le même contexte (tout ce qui précède).
            if messages[-1]["role"] != "assistant":
                print(f"  [{i}/{len(echantillon)}] cas ignoré (dernier tour n'est pas l'agent)")
                continue
            chosen = messages[-1]["content"]
            contexte = messages[:-1]

            texte_entree = tokenizer.apply_chat_template(contexte, tokenize=False, add_generation_prompt=True)
            entrees = tokenizer(texte_entree, return_tensors="pt").to(modele.device)

            with torch.no_grad():
                sortie = modele.generate(
                    **entrees,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=0.95,
                    eos_token_id=im_end_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
            nouveaux_tokens = sortie[0][entrees["input_ids"].shape[1]:]
            rejected_brut = tokenizer.decode(nouveaux_tokens, skip_special_tokens=False)
            rejected = rejected_brut.split("<|im_end|>")[0].strip()

            trace = {
                "cas_id": exemple.get("cas_id", exemple.get("id")),
                "messages_contexte": contexte,
                "chosen": chosen,
                "rejected": rejected,
                "source": "derive_sft_locale",
            }
            f_out.write(json.dumps(trace, ensure_ascii=False) + "\n")
            f_out.flush()
            print(f"  [{i}/{len(echantillon)}] {trace['cas_id']} -> rejected généré "
                  f"({'IDENTIQUE au chosen, à surveiller' if rejected.strip() == chosen.strip() else 'OK'})")

    print(f"\nTerminé -> {output_path}")


if __name__ == "__main__":
    main()
