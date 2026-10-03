"""
fusionner_dpo.py

Fusionne les deux sources de paires DPO en un format unique (prompt / chosen /
rejected) exploitable par train_dpo.py, avec split train/validation SANS fuite
(découpage par identifiant de cas, même logique que preparer_dataset_sft.py).

Sources :
  - rejected_sft_a_relire.jsonl (preparer_dpo_rejected.py) : dérives du modèle
    SFT lui-même, contexte de conversation complet.
  - ultramedical_fr_a_relire.jsonl (preparer_dpo_ultramedical.py) : paires
    traduites d'UltraMedical-Preference. SEULES les lignes avec a_valider=true
    sont utilisées (relecture obligatoire, cf. décision du 24/09) -- si le
    fichier n'a pas encore été relu, ces paires sont simplement ignorées, pas
    d'erreur bloquante.

Les paires UltraMedical, qui n'ont pas de conversation de triage propre, reçoivent un prompt
système DISTINCT du triage (PROMPT_SYSTEME_INFO_MEDICALE) : avec le prompt de triage, le DPO
pousserait le modèle à répondre par un exposé au lieu de mener l'entretien appris au SFT.
Ce prompt vient de src/entrainement/prompt_agent.py (source unique). Les paires
rejected_sft gardent le contexte de conversation de train.jsonl (prompt de triage).

Usage :
    python fusionner_dpo.py \
        --rejected-sft ../../data/dpo/rejected_sft_a_relire.jsonl \
        --ultramedical ../../data/dpo/ultramedical_fr_a_relire.jsonl \
        --output-dir ../../data/dpo \
        --val-ratio 0.1
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "entrainement"))
from prompt_agent import PROMPT_SYSTEME_INFO_MEDICALE  # noqa: E402


def charger_rejected_sft(chemin: str) -> list:
    if not chemin or not Path(chemin).exists():
        print(f"[rejected_sft] fichier absent ou non fourni ({chemin}) -- ignoré")
        return []
    paires = []
    with open(chemin, encoding="utf-8") as f:
        for ligne in f:
            d = json.loads(ligne)
            if d["chosen"].strip() == d["rejected"].strip():
                continue  # aucun signal de préférence, inutile pour le DPO
            paires.append({
                "cas_id": f"rejected_sft_{d['cas_id']}",
                "prompt": d["messages_contexte"],
                "chosen": d["chosen"],
                "rejected": d["rejected"],
                "source": "rejected_sft",
            })
    print(f"[rejected_sft] {len(paires)} paires exploitables sur {chemin}")
    return paires

def charger_ultramedical(chemin: str) -> list:
    if not chemin or not Path(chemin).exists():
        print(f"[ultramedical] fichier absent ou non fourni ({chemin}) -- ignoré")
        return []
    paires = []
    n_total, n_valide = 0, 0
    with open(chemin, encoding="utf-8") as f:
        for ligne in f:
            d = json.loads(ligne)
            if "_erreur" in d:
                continue
            n_total += 1
            if d.get("a_valider") is not True:
                continue  # relecture obligatoire -- pas encore validé ou explicitement rejeté
            n_valide += 1
            paires.append({
                "cas_id": d["id"],  # identifiant produit par preparer_dpo_ultramedical.py (ex. um_xxxx)
                "prompt": [
                    {"role": "system", "content": PROMPT_SYSTEME_INFO_MEDICALE},
                    {"role": "user", "content": d["prompt_fr"]},
                ],
                "chosen": d["chosen_fr"],
                "rejected": d["rejected_fr"],
                "source": "ultramedical",
            })
    print(f"[ultramedical] {n_valide}/{n_total} paires validées (a_valider=true) sur {chemin}")
    return paires


def main():
    parser = argparse.ArgumentParser(description="Fusionne les paires DPO en un dataset train/validation")
    parser.add_argument("--rejected-sft", default=None)
    parser.add_argument("--ultramedical", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paires = charger_rejected_sft(args.rejected_sft) + charger_ultramedical(args.ultramedical)
    assert paires, "Aucune paire chargée -- vérifie les chemins et, pour ultramedical, que a_valider a bien été rempli."
    print(f"\n{len(paires)} paires au total")

    # Découpage par cas_id, PAS ligne à ligne (leçon retenue du SFT) --
    # ici chaque source ne produit qu'une paire par cas_id, mais on garde le
    # même principe de robustesse par précaution.
    cas_ids = sorted(set(p["cas_id"] for p in paires))
    random.seed(args.seed)
    random.shuffle(cas_ids)
    n_val = max(1, int(len(cas_ids) * args.val_ratio))
    cas_val = set(cas_ids[:n_val])

    train = [p for p in paires if p["cas_id"] not in cas_val]
    val = [p for p in paires if p["cas_id"] in cas_val]
    assert not (set(p["cas_id"] for p in train) & set(p["cas_id"] for p in val)), "Fuite train/validation détectée"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for nom, lignes in [("train_dpo.jsonl", train), ("validation_dpo.jsonl", val)]:
        chemin = output_dir / nom
        with open(chemin, "w", encoding="utf-8") as f:
            for p in lignes:
                f.write(json.dumps({
                    "prompt": p["prompt"],
                    "chosen": [{"role": "assistant", "content": p["chosen"]}],
                    "rejected": [{"role": "assistant", "content": p["rejected"]}],
                }, ensure_ascii=False) + "\n")
        print(f"{len(lignes)} paires -> {chemin}")

    from collections import Counter
    print("Répartition par source (train+val) :", dict(Counter(p["source"] for p in paires)))


if __name__ == "__main__":
    main()
