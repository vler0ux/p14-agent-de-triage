"""
Rattrapage ponctuel : les paires rejetées pour « désaccord avec le dataset » deviennent des paires
validées, avec chosen/rejected échangés (le jugement humain fait foi). À lancer UNE fois.
Usage : python3 src/dpo/rattraper_desaccords.py data/dpo/ultramedical_fr_a_relire.jsonl
"""
import json, shutil, sys
from pathlib import Path

chemin = Path(sys.argv[1])
shutil.copy(chemin, chemin.with_suffix(".jsonl.avant_rattrapage"))
paires = [json.loads(l) for l in open(chemin, encoding="utf-8") if l.strip()]
n = 0
for p in paires:
    r = p.get("relecture") or {}
    if (p.get("a_valider") is False and r.get("motif_rejet") == "desaccord_avec_dataset"
            and not p.get("inversee_par_relecteur")):
        for langue in ("fr", "en"):
            p[f"chosen_{langue}"], p[f"rejected_{langue}"] = p[f"rejected_{langue}"], p[f"chosen_{langue}"]
        p["a_valider"] = True
        p["inversee_par_relecteur"] = True
        r["motif_rejet"] = None
        n += 1
with open(chemin, "w", encoding="utf-8") as f:
    for p in paires:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")
print(f"{n} paires inversées et validées. Copie de l'état précédent : {chemin.with_suffix('.jsonl.avant_rattrapage')}")