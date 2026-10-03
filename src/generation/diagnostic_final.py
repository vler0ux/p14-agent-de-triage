# diagnostic_final.py : sur-triage par rapport à la grille FRENCH + lecture des cas « fievre » (lecture seule, aucun appel API).
# Usage, depuis src/generation :  python diagnostic_final.py
import json, os
from collections import Counter, defaultdict
P = os.environ.get("FICHIER", "../../data/normalized/pipeline_473.jsonl")
REF = os.environ.get("REFERENTIEL", "../referentiel/french_referentiel.json")
T = [json.loads(l) for l in open(P) if l.strip() and "categorie_finale" in json.loads(l)]
ref = json.load(open(REF, encoding="utf-8"))
tri_base = {m["id"]: m["tri_base"] for m in ref["motifs"]}
cat_de_tri = {"1": "urgence_maximale", "2": "urgence_maximale", "3A": "moderee", "3B": "moderee", "4": "differee", "5": "differee"}
ordre = {"urgence_maximale": 1, "moderee": 2, "differee": 3}
abr = {"urgence_maximale": "MAX", "moderee": "MOD", "differee": "DIF", None: "aucun"}
g = lambda d, k: (d.get(k) or {})
n = len(T)
print(f"{n} cas avec label final dans {P}\n")

print("LABEL FINAL  x  PLANCHER DU MOTIF (motif de l'Étape 1 + critères cochés, selon la grille FRENCH)")
croise = Counter((abr[t["categorie_finale"]], abr[g(t, "decision_apres_override_etape2_3").get("categorie_plancher_motif")]) for t in T)
for lab in ("MAX", "MOD", "DIF"):
    print(f"  label {lab} :", {pl: croise[(lab, pl)] for pl in ("MAX", "MOD", "DIF", "aucun") if croise[(lab, pl)]})

def position(t):
    pl = g(t, "decision_apres_override_etape2_3").get("categorie_plancher_motif")
    if not pl: return "sans plancher (motif absent)"
    d = ordre[pl] - ordre[t["categorie_finale"]]
    return "label PLUS urgent que le plancher" if d > 0 else ("label = plancher" if d == 0 else "label MOINS urgent (sous plancher)")
print("\nPOSITION DU LABEL PAR RAPPORT AU PLANCHER :")
for k, v in Counter(position(t) for t in T).most_common(): print(f"  {v:4} ({100 * v / n:.0f} %)  {k}")

print("\nTRI MÉDIAN DE LA GRILLE des motifs retenus (avant critères) :")
med = Counter(abr[cat_de_tri[tri_base[g(t, 'etape1_extraction').get('motif_id')]]] if g(t, 'etape1_extraction').get('motif_id') in tri_base else "aucun" for t in T)
print("  ", dict(med), "  -> à comparer au label final :", dict(Counter(abr[t['categorie_finale']] for t in T)))

print("\nPAR MOTIF (les 10 plus fréquents) : n | labels | plancher | tri médian")
par = defaultdict(list)
for t in T: par[g(t, "etape1_extraction").get("motif_id")].append(t)
for m, l in sorted(par.items(), key=lambda kv: -len(kv[1]))[:10]:
    labs = dict(Counter(abr[t["categorie_finale"]] for t in l))
    pls = dict(Counter(abr[g(t, "decision_apres_override_etape2_3").get("categorie_plancher_motif")] for t in l))
    print(f"  {str(m)[:38]:38} {len(l):3} | {labs} | {pls} | {abr[cat_de_tri[tri_base[m]]] if m in tri_base else '-'}")

print("\nCAS « fievre » CLASSÉS EN URGENCE MAXIMALE (6 premiers) : critères cochés, citations, début de la vignette")
f = [t for t in T if (t.get("etape1_extraction") or {}).get("motif_id") == "fievre" and t.get("categorie_finale") == "urgence_maximale"]
print(f"  ({len(f)} cas au total)")
for t in f[:6]:
    e = t["etape1_extraction"]
    print("\n ", t["id"], "| critères:", e.get("criteres_presents"))
    print("     citations:", e.get("criteres_citations"))
    print("     vignette:", (t.get("vignette_source") or "")[:350].replace("\n", " "))
