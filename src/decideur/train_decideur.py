#!/usr/bin/env python3
"""Entraînement du décideur : classification de la priorité de triage par un encodeur
(ModernCamemBERT-bio-v2-base) à partir des transcriptions produites par
preparer_dataset_decideur.py.

Règles posées AVANT de voir les résultats (pas de réglage sur la validation, faute de
jeu de test) :
  - on garde le modèle de la DERNIÈRE époque, jamais le « meilleur » checkpoint ;
  - pas de pondération des classes ;
  - règle de sécurité : urgence_maximale si P(urgence) >= SEUIL_URGENCE, sinon argmax.
La validation sert uniquement à la mesure (et aux courbes, pour le diagnostic).

Limites de la mesure (à rappeler avec les chiffres) :
  - la validation est petite (~100 dialogues, ~70 cas) : chaque taux est accompagné de son
    intervalle de confiance de Wilson à 95 % et de ses effectifs (clé « intervalles_95 ») ;
  - le choix de la variante d'entrée (complet / patient) a été fait en comparant les deux runs
    SUR CETTE validation : les chiffres de la variante retenue sont donc légèrement optimistes.
    Seul un jeu de test jamais consulté donnerait une estimation sans biais.

Usage (Kaggle, GPU) :
    python train_decideur.py --data-dir /kaggle/input/<dataset>/complet --output-dir decideur-complet
"""
import argparse
import json
from pathlib import Path

# Ordre FIXE, identique à preparer_dataset_decideur.py (0 = le plus urgent).
LABELS = ("urgence_maximale", "moderee", "differee")
LABEL2ID = {lab: i for i, lab in enumerate(LABELS)}
SEUIL_URGENCE = 0.30
MODELE_PAR_DEFAUT = "almanach/ModernCamemBERT-bio-v2-base"


def predire_avec_seuil(probas, seuil=SEUIL_URGENCE):
    """Règle de sécurité : urgence si P(urgence) >= seuil, sinon classe la plus probable."""
    import numpy as np
    return np.where(probas[:, 0] >= seuil, 0, probas.argmax(axis=1))


def intervalle_wilson(succes, n, z=1.96):
    """Intervalle de confiance de Wilson (95 % par défaut) d'une proportion succes / n.
    Reste correct pour les proportions extrêmes (0 % ou 100 %), contrairement à l'intervalle normal."""
    if n == 0:
        return None
    p = succes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    demi = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return [round(max(0.0, centre - demi), 4), round(min(1.0, centre + demi), 4)]


def metriques(vrais, preds):
    """Exactitude, F1 macro, rappel par classe, sous-triage et sur-triage, avec intervalles de confiance."""
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
    vrais, preds = np.asarray(vrais), np.asarray(preds)
    rappels = recall_score(vrais, preds, labels=[0, 1, 2], average=None, zero_division=0)
    n = len(vrais)
    intervalles = {
        "exactitude": {"succes": int(np.sum(preds == vrais)), "n": n,
                       "ic95": intervalle_wilson(int(np.sum(preds == vrais)), n)},
        "taux_sous_triage": {"succes": int(np.sum(preds > vrais)), "n": n,
                             "ic95": intervalle_wilson(int(np.sum(preds > vrais)), n)},
        "taux_sur_triage": {"succes": int(np.sum(preds < vrais)), "n": n,
                            "ic95": intervalle_wilson(int(np.sum(preds < vrais)), n)},
    }
    for i, lab in enumerate(LABELS):
        n_classe, bons = int(np.sum(vrais == i)), int(np.sum((vrais == i) & (preds == i)))
        intervalles[f"rappel_{lab}"] = {"succes": bons, "n": n_classe, "ic95": intervalle_wilson(bons, n_classe)}
    return {
        "exactitude": float(accuracy_score(vrais, preds)),
        "f1_macro": float(f1_score(vrais, preds, average="macro", zero_division=0)),
        **{f"rappel_{lab}": float(r) for lab, r in zip(LABELS, rappels)},
        # id plus grand = moins urgent : prédire moins urgent que la vérité = sous-triage
        "taux_sous_triage": float(np.mean(preds > vrais)),
        "taux_sur_triage": float(np.mean(preds < vrais)),
        "matrice_confusion": confusion_matrix(vrais, preds, labels=[0, 1, 2]).tolist(),
        "intervalles_95": intervalles,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-dir", required=True, help="dossier contenant train.jsonl et validation.jsonl")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--modele", default=MODELE_PAR_DEFAUT)
    p.add_argument("--max-length", type=int, default=768, help="max mesuré : 701 tokens (complet)")
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--num-epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    import numpy as np
    import torch
    from datasets import load_dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)

    set_seed(args.seed)
    data_dir, sortie = Path(args.data_dir), Path(args.output_dir)
    sortie.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("json", data_files={
        "train": str(data_dir / "train.jsonl"),
        "validation": str(data_dir / "validation.jsonl"),
    })
    # Contrôle de cohérence de l'ordre des classes
    for split in dataset:
        for ex in dataset[split]:
            assert LABEL2ID[ex["label"]] == ex["label_id"], f"label_id incohérent : {ex['id']}"

    tokenizer = AutoTokenizer.from_pretrained(args.modele)

    def tokeniser(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    ds = dataset.map(tokeniser, batched=True)
    for split in ds:
        tronques = sum(len(x) >= args.max_length for x in ds[split]["input_ids"])
        print(f"[{split}] {len(ds[split])} exemples, {tronques} tronqués à {args.max_length} tokens")
    ds = ds.rename_column("label_id", "labels")
    colonnes_a_garder = {"input_ids", "attention_mask", "labels"}
    ds = ds.remove_columns([c for c in ds["train"].column_names if c not in colonnes_a_garder])

    modele = AutoModelForSequenceClassification.from_pretrained(
        args.modele,
        num_labels=len(LABELS),
        id2label={i: lab for i, lab in enumerate(LABELS)},
        label2id=LABEL2ID,
    )

    def compute_metrics(eval_pred):
        logits, vrais = eval_pred
        m = metriques(vrais, logits.argmax(axis=-1))
        m.pop("matrice_confusion")
        m.pop("intervalles_95")      # le Trainer n'accepte que des nombres
        return m

    cuda = torch.cuda.is_available()
    bf16 = cuda and torch.cuda.get_device_capability(0)[0] >= 8
    training_args = TrainingArguments(
        output_dir=str(sortie / "checkpoints"),
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",      # courbes pour le diagnostic uniquement
        save_strategy="no",         # on garde le modèle final, pas le « meilleur »
        logging_steps=10,
        seed=args.seed,
        bf16=bf16,
        fp16=cuda and not bf16,
        report_to=[],
    )

    trainer = Trainer(
        model=modele,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    print("Début de l'entraînement...")
    trainer.train()
    trainer.save_model(str(sortie))
    tokenizer.save_pretrained(str(sortie))
    trainer.save_state()  # trainer_state.json (log_history : courbes train / validation)

    # Évaluation finale sur la validation, avec les deux règles de décision
    sortie_pred = trainer.predict(ds["validation"])
    probas = torch.softmax(torch.tensor(sortie_pred.predictions, dtype=torch.float32), dim=-1).numpy()
    vrais = sortie_pred.label_ids
    resultats = {
        "parametres": vars(args) | {"seuil_urgence": SEUIL_URGENCE},
        "argmax": metriques(vrais, probas.argmax(axis=1)),
        f"seuil_{SEUIL_URGENCE}": metriques(vrais, predire_avec_seuil(probas)),
        "reference_toujours_urgence": metriques(vrais, np.zeros_like(vrais)),
    }
    with open(sortie / "resultats_validation.json", "w", encoding="utf-8") as f:
        json.dump(resultats, f, ensure_ascii=False, indent=2)

    # Prédictions détaillées, pour l'analyse des erreurs (cas par cas)
    with open(sortie / "predictions_validation.jsonl", "w", encoding="utf-8") as f:
        for ex, pr in zip(dataset["validation"], probas):
            f.write(json.dumps({
                "id": ex["id"], "cas_id": ex["cas_id"], "label": ex["label"],
                "pred_argmax": LABELS[int(pr.argmax())],
                "pred_seuil": LABELS[int(predire_avec_seuil(pr[None, :])[0])],
                "probas": {lab: round(float(x), 4) for lab, x in zip(LABELS, pr)},
            }, ensure_ascii=False) + "\n")

    for nom, m in resultats.items():
        if nom == "parametres":
            continue
        print(f"\n== {nom}")
        intervalles = m.pop("intervalles_95")
        for k, v in m.items():
            if k == "matrice_confusion":
                print(f"  {k:<26} {v}")
            elif k in intervalles:
                ic = intervalles[k]
                print(f"  {k:<26} {round(v, 3):<6} IC95 {ic['ic95']}  ({ic['succes']}/{ic['n']})")
            else:
                print(f"  {k:<26} {round(v, 3)}")
    print(f"\nModèle, résultats et prédictions dans {sortie}")


if __name__ == "__main__":
    main()
