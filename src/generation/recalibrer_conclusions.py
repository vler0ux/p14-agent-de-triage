"""
recalibrer_conclusions.py

Corrige RÉTROACTIVEMENT dialogues_473.jsonl, SANS appel API :
1. Remplace la conclusion de chaque dialogue par celle imposée par le code actuel
   (CONCLUSIONS + choisir_conclusion sont déterministes : id + variante -> même
   résultat, qu'on le calcule maintenant ou au moment de la génération).
2. Retire les cas a_relire=true (jamais filtrés avant ce correctif).

Les dialogues en _erreur ne sont pas concernés : eux ont besoin d'une vraie
régénération (relancer generation_dialogue_etape5.py après ce script).

Usage :
    python recalibrer_conclusions.py --dialogues ../../data/normalized/dialogues_473.jsonl
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

from generation_dialogue_etape5 import choisir_conclusion, balise_categorie


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dialogues", required=True)
    args = parser.parse_args()
    chemin = Path(args.dialogues)

    conservees = []
    n_ok = n_erreur = n_conclusion_maj = n_a_relire_retires = 0

    with open(chemin, encoding="utf-8") as f:
        for ligne in f:
            ligne = ligne.strip()
            if not ligne:
                continue
            e = json.loads(ligne)
            if "_erreur" in e:
                conservees.append(ligne)
                n_erreur += 1
                continue
            n_ok += 1
            if e.get("a_relire"):
                n_a_relire_retires += 1
                continue
            cas_id = e.get("cas_id") or e["id"]
            nouvelle_conclusion = choisir_conclusion(cas_id, e["variante"], e["categorie_finale"])
            if e.get("conclusion", "").strip() != nouvelle_conclusion:
                n_conclusion_maj += 1
            e["conclusion"] = nouvelle_conclusion
            e["balise"] = balise_categorie(e["categorie_finale"])
            e.setdefault("controles", {})["conclusion_coherente"] = True
            e["controles"]["conclusion_recalibree"] = True
            conservees.append(json.dumps(e, ensure_ascii=False))

    sauvegarde = chemin.with_suffix(chemin.suffix + ".avant_recalibrage")
    shutil.copy2(chemin, sauvegarde)
    print(f"Sauvegarde : {sauvegarde}", file=sys.stderr)

    with open(chemin, "w", encoding="utf-8") as f:
        for l in conservees:
            f.write(l + "\n")

    print(f"{n_ok} dialogues OK examinés, {n_erreur} échecs laissés tels quels", file=sys.stderr)
    print(f"{n_conclusion_maj} conclusions mises à jour (0 appel API)", file=sys.stderr)
    print(f"{n_a_relire_retires} cas a_relire retirés", file=sys.stderr)
    print(f"{len(conservees)} lignes conservées -> {chemin}", file=sys.stderr)


if __name__ == "__main__":
    main()