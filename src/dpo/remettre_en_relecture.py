"""
remettre_en_relecture.py

Remet des paires déjà décidées dans la file de relecture de relire_paires_dpo.py.
Les paires inversées par le relecteur sont d'abord ramenées à leur étiquetage d'origine
(chosen/rejected du dataset), pour que la nouvelle relecture reparte de zéro, à l'aveugle.

Usage (depuis la racine du projet) :
    # toutes les paires validées
    python3 src/dpo/remettre_en_relecture.py data/dpo/ultramedical_fr_a_relire.jsonl --validees
    # des paires précises
    python3 src/dpo/remettre_en_relecture.py data/dpo/ultramedical_fr_a_relire.jsonl --ids um_abc um_def
"""
import argparse, json, shutil
from datetime import datetime
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("fichier")
groupe = parser.add_mutually_exclusive_group(required=True)
groupe.add_argument("--validees", action="store_true", help="toutes les paires a_valider=true")
groupe.add_argument("--ids", nargs="+", help="identifiants précis")
args = parser.parse_args()

chemin = Path(args.fichier)
horodatage = datetime.now().strftime("%Y%m%d-%H%M%S")
sauvegarde = chemin.with_suffix(f".jsonl.avant_remise_{horodatage}")
shutil.copy(chemin, sauvegarde)

paires = [json.loads(l) for l in open(chemin, encoding="utf-8") if l.strip()]
cibles = set(args.ids or [])
n, n_desinversees = 0, 0
for p in paires:
    if p.get("a_valider") is None:
        continue
    if args.validees and p["a_valider"] is not True:
        continue
    if args.ids and p["id"] not in cibles:
        continue
    if p.get("inversee_par_relecteur"):
        for langue in ("fr", "en"):
            p[f"chosen_{langue}"], p[f"rejected_{langue}"] = p[f"rejected_{langue}"], p[f"chosen_{langue}"]
        n_desinversees += 1
    p.pop("inversee_par_relecteur", None)
    # l'ancienne décision est conservée pour la traçabilité, la nouvelle l'écrasera dans "relecture"
    historique = p.setdefault("relectures_precedentes", [])
    if p.get("relecture"):
        historique.append(p["relecture"])
    p.pop("relecture", None)
    p["a_valider"] = None
    n += 1

with open(chemin, "w", encoding="utf-8") as f:
    for p in paires:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")
print(f"{n} paires remises en relecture (dont {n_desinversees} ramenées à l'étiquetage d'origine).")
print(f"Copie de l'état précédent : {sauvegarde}")
if args.ids:
    manquants = cibles - {p['id'] for p in paires}
    if manquants: print("Identifiants introuvables :", sorted(manquants))