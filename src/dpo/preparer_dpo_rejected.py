"""
preparer_dpo_rejected.py

Génère des réponses candidates pour le DPO avec le modèle SFT v4 (celui de la
démo), sur des contextes tirés de train.jsonl : la conclusion et des répliques
du milieu de l'entretien. Génération alignée sur l'application (gabarit,
pénalité de répétition, tokens d'arrêt, nettoyage), échantillonnée pour
obtenir plusieurs réponses plausibles par contexte.

Le "chosen" est la réplique de référence de train.jsonl, au même tour.
Chaque réponse générée est écrite sur une ligne ; un filtre choisit ensuite,
pour chaque contexte, une réponse avec un défaut comme "rejected".

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
    parser.add_argument("--adapter-dir", required=True, help="Dossier de l'adaptateur SFT v4, avec chat_template.jinja)")
    parser.add_argument("--modele-base", default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--train-file", required=True, help="train.jsonl (source des contextes ET des 'chosen')")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--tours-par-cas", type=int, default=2,
                        help="Répliques du milieu de l'entretien tirées par cas, en plus de la conclusion"")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=1.1,
                         help="0.8 : réponses plausibles, proches de celles du modèle en démo")
    parser.add_argument("--reponses-par-contexte", type=int, default=4,
                        help="Réponses générées par contexte ; une ligne par réponse, le filtre choisira")
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

    # Mêmes tokens d'arrêt que l'application (deploiement/space/app.py)
    fin_ids = [tokenizer.convert_tokens_to_ids(t)
               for t in ("<|im_end|>", "atrigesimal", "<|im_start|>", "<|endoftext|>")]

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
    contextes = []
    for exemple in exemples:
        messages = exemple["messages"]
        # Répliques de l'agent au milieu de l'entretien (hors accueil, imposé par l'application)
        milieu = [i for i, m in enumerate(messages)
                  if m["role"] == "assistant" and "[[PRIORITE:" not in m["content"]][1:]
        tours = random.sample(milieu, min(args.tours_par_cas, len(milieu)))
        # On garde aussi la conclusion (dernier tour), comme dans la version d'origine
        if messages[-1]["role"] == "assistant":
            tours.append(len(messages) - 1)
        for i in tours:
            contextes.append((exemple, i))
    random.shuffle(contextes)
    echantillon = contextes[:args.n]
    print(f"{len(contextes)} contextes disponibles, {len(echantillon)} retenus (seed {args.seed}) -> {args.output}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "a", encoding="utf-8") as f_out:
        for i, (exemple, i_tour) in enumerate(echantillon, 1):
            messages = exemple["messages"]

            # Le "chosen" est la réplique de référence à ce tour ; le "rejected" sera généré
            # sur EXACTEMENT le même contexte (tout ce qui précède).
            chosen = messages[i_tour]["content"]
            contexte = messages[:i_tour]

            texte_entree = tokenizer.apply_chat_template(contexte, tokenize=False, add_generation_prompt=True)
            entrees = tokenizer(texte_entree, return_tensors="pt").to(modele.device)

            with torch.no_grad():
                sortie = modele.generate(
                    **entrees,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=0.95,
                    num_return_sequences=args.reponses_par_contexte,
                    repetition_penalty=1.1,
                    eos_token_id=fin_ids,
                    pad_token_id=tokenizer.pad_token_id,
                )
            debut = entrees["input_ids"].shape[1]
            for n_rep, sequence in enumerate(sortie):
                rejected_brut = tokenizer.decode(sequence[debut:], skip_special_tokens=False)
                rejected = (rejected_brut.split("<|im_end|>")[0].split("<|im_start|>")[0]
                            .split("atrigesimal")[0].strip())

                trace = {
                    "cas_id": exemple.get("cas_id", exemple.get("id")),
                    "tour": i_tour,
                    "reponse_n": n_rep,
                    "messages_contexte": contexte,
                    "chosen": chosen,
                    "rejected": rejected,
                    "rejected_brut": rejected_brut,
                    "source": "derive_sft_locale",
                }
                f_out.write(json.dumps(trace, ensure_ascii=False) + "\n")
                f_out.flush()
            print(f"  [{i}/{len(echantillon)}] {trace['cas_id']} tour {i_tour} -> {len(sortie)} réponses générées")

    print(f"\nTerminé -> {output_path}")


if __name__ == "__main__":
    main()
