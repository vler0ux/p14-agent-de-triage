"""
train_dpo.py

Entraînement DPO (Direct Preference Optimization) par-dessus un modèle déjà
fine-tuné par SFT (Qwen3-1.7B-Base + LoRA).

Approche : on FUSIONNE l'adaptateur SFT dans le modèle de base (merge_and_unload),
puis on attache un NOUVEAU LoRA, aux modules cibles élargis (ajout de
gate_proj/up_proj/down_proj, en plus de q/k/v/o_proj utilisés au SFT) -- pour
toucher indirectement la tête de sortie, jamais adaptée par le LoRA du SFT
seul (cause probable des dérives multilingues observées en inférence).

Le modèle de référence du DPO (nécessaire pour calculer la préférence) est
automatiquement le modèle fusionné SANS le nouveau LoRA -- TRL le fait tout
seul dès qu'on lui donne un PeftModel, pas besoin de charger un second modèle
séparément (ça économiserait de la mémoire, précieux sur un T4).

Usage :
    python train_dpo.py \
        --adapter-sft-dir ../../qwen3-1.7b-triage-sft \
        --train-file ../../data/dpo/train_dpo.jsonl \
        --val-file ../../data/dpo/validation_dpo.jsonl \
        --output-dir ../../qwen3-1.7b-triage-dpo \
        --max-steps 5   # SMOKE TEST d'abord, toujours
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Entraînement DPO (LoRA élargi) par-dessus un modèle SFT")
    parser.add_argument("--adapter-sft-dir", required=True, help="Dossier de l'adaptateur SFT à fusionner (V1 pour la répétition)")
    parser.add_argument("--modele-base", default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--val-file", default=None)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    parser.add_argument("--beta", type=float, default=0.1, help="Température DPO -- force de la préférence apprise")
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--num-epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-prompt-length", type=int, default=768)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--save-steps", type=int, default=20)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", choices=["auto", "bf16", "fp16"], default="auto",
                         help="auto : bf16 si le GPU le gère nativement (Ampere ou plus récent), sinon fp16")
    parser.add_argument("--merge-adapter", action="store_true")
    args = parser.parse_args()

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, PeftModel, get_peft_model
    from trl import DPOConfig, DPOTrainer

    adapter_sft_dir = Path(args.adapter_sft_dir)
    assert (adapter_sft_dir / "adapter_config.json").exists(), f"adapter_config.json introuvable dans {adapter_sft_dir}"

    cuda = torch.cuda.is_available()
    precision = args.precision
    if precision == "auto":
        precision = "bf16" if cuda and torch.cuda.get_device_capability(0)[0] >= 8 else "fp16"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[precision] if cuda else torch.float32
    print(f"Précision d'entraînement : {precision if cuda else 'fp32 (CPU)'}")

    print(f"Chargement du tokenizer et du modele de base : {args.modele_base}")
    tokenizer = AutoTokenizer.from_pretrained(args.modele_base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    chemin_template = adapter_sft_dir / "chat_template.jinja"
    assert chemin_template.exists(), f"{chemin_template} introuvable -- indispensable pour rester cohérent avec le SFT."
    tokenizer.chat_template = chemin_template.read_text(encoding="utf-8")
    print(f"Template de chat chargé depuis {chemin_template}")

    print("Chargement du modele de base...")
    modele_base = AutoModelForCausalLM.from_pretrained(args.modele_base, dtype=dtype, device_map={"": 0} if cuda else None)

    norme_avant = modele_base.get_input_embeddings().weight[151645].float().norm().item()
    print(f"Fusion de l'adaptateur SFT ({adapter_sft_dir}) dans le modele de base...")
    modele_fusionne = PeftModel.from_pretrained(modele_base, str(adapter_sft_dir))
    modele_fusionne = modele_fusionne.merge_and_unload()
    norme_apres = modele_fusionne.get_input_embeddings().weight[151645].float().norm().item()
    print(f"[diag] norme du vecteur <|im_end|> : avant fusion {norme_avant:.4f} -> après {norme_apres:.4f}")
    assert abs(norme_apres - norme_avant) > 1e-3, "Les tokens entraînables de v4 n'ont pas été fusionnés"

    print("Configuration du nouveau LoRA (modules élargis) pour le DPO...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    modele = get_peft_model(modele_fusionne, lora_config)
    modele.print_trainable_parameters()

    print("Chargement du dataset...")
    data_files = {"train": args.train_file}
    if args.val_file:
        data_files["validation"] = args.val_file
    dataset = load_dataset("json", data_files=data_files)

    dpo_config = DPOConfig(
        output_dir=args.output_dir,
        beta=args.beta,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,  # explicite -- cf. le plantage CUDA du SFT sur la valeur par défaut
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        max_length=args.max_length,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps" if args.val_file else "no",
        eval_steps=args.save_steps if args.val_file else None,
        seed=args.seed,
        bf16=cuda and precision == "bf16",
        fp16=cuda and precision == "fp16",
        report_to=[],
    )

    trainer = DPOTrainer(
        model=modele,
        args=dpo_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        processing_class=tokenizer,
        # ref_model=None : TRL détecte que `modele` est un PeftModel et utilise
        # automatiquement le modèle fusionné SANS le nouveau LoRA (adapter désactivé)
        # comme référence -- pas besoin de charger un second modèle séparément.
    )

    print("Debut de l'entrainement DPO...")
    resultat_entrainement = trainer.train()

    print(f"Sauvegarde de l'adaptateur DPO dans {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metriques_train = resultat_entrainement.metrics
    trainer.log_metrics("train", metriques_train)
    trainer.save_metrics("train", metriques_train)
    print(f"Loss finale (train) : {metriques_train.get('train_loss')}")

    if args.val_file:
        metriques_eval = trainer.evaluate()
        trainer.log_metrics("eval", metriques_eval)
        trainer.save_metrics("eval", metriques_eval)
        print(f"Loss finale (eval) : {metriques_eval.get('eval_loss')}")
        print(f"Rewards/accuracies (eval) : {metriques_eval.get('eval_rewards/accuracies')}")

    trainer.save_state()

    if args.merge_adapter:
        print("Fusion de l'adaptateur DPO dans le modele...")
        modele_final = modele.merge_and_unload()
        chemin_fusion = f"{args.output_dir}-merged"
        modele_final.save_pretrained(chemin_fusion)
        tokenizer.save_pretrained(chemin_fusion)
        print(f"Modele fusionne sauvegarde dans {chemin_fusion}")

    print("Termine.")


if __name__ == "__main__":
    main()
