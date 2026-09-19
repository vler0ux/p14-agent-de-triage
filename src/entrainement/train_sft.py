"""
train_sft.py

Entraînement SFT (Supervised Fine-Tuning) de Qwen3-1.7B-Base avec LoRA, à
partir du dataset produit par preparer_dataset_sft.py.
"""

import argparse

MODELE_BASE_PAR_DEFAUT = "Qwen/Qwen3-1.7B-Base"

LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


def main():
    parser = argparse.ArgumentParser(description="Entraînement SFT (LoRA) de Qwen3-1.7B-Base")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--val-file", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--modele-base", default=MODELE_BASE_PAR_DEFAUT)

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--num-epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--merge-adapter", action="store_true")

    args = parser.parse_args()

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from trl import SFTConfig, SFTTrainer

    print(f"Chargement du tokenizer et du modele de base : {args.modele_base}")
    tokenizer = AutoTokenizer.from_pretrained(args.modele_base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    modele = AutoModelForCausalLM.from_pretrained(
        args.modele_base,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    print("Configuration LoRA...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=LORA_TARGET_MODULES,
        task_type="CAUSAL_LM",
    )
    modele = get_peft_model(modele, lora_config)
    modele.print_trainable_parameters()

    print("Chargement du dataset...")
    data_files = {"train": args.train_file}
    if args.val_file:
        data_files["validation"] = args.val_file
    dataset = load_dataset("json", data_files=data_files)

    def formater(exemple):
        return {"text": tokenizer.apply_chat_template(exemple["messages"], tokenize=False)}

    dataset = dataset.map(formater)

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        max_length=args.max_seq_length,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps" if args.val_file else "no",
        eval_steps=args.save_steps if args.val_file else None,
        seed=args.seed,
        bf16=torch.cuda.is_available(),
        report_to=[],
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=modele,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        processing_class=tokenizer,
    )

    print("Debut de l'entrainement...")
    resultat_entrainement = trainer.train()

    print(f"Sauvegarde de l'adaptateur LoRA dans {args.output_dir}")
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

    trainer.save_state()  # écrit trainer_state.json (dont log_history) dans output_dir

    if args.merge_adapter:
        print("Fusion de l'adaptateur dans le modele de base...")
        modele_fusionne = modele.merge_and_unload()
        chemin_fusion = f"{args.output_dir}-merged"
        modele_fusionne.save_pretrained(chemin_fusion)
        tokenizer.save_pretrained(chemin_fusion)
        print(f"Modele fusionne sauvegarde dans {chemin_fusion}")

    print("Termine.")


if __name__ == "__main__":
    main()