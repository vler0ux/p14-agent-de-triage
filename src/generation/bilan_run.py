# bilan_run.py : résumé chiffré d un run du pipeline (lecture seule, aucun appel API).
# Usage, depuis src/generation :  python bilan_run.py   (ou FICHIER=chemin.jsonl python bilan_run.py)
import json, os
from collections import Counter
P = os.environ.get("FICHIER", "../../data/normalized/pipeline_473.jsonl")
T = [json.loads(l) for l in open(P) if l.strip()]
g = lambda d, k: (d.get(k) or {})
n = len(T); abr = {"urgence_maximale": "MAX", "moderee": "MOD", "differee": "DIF", None: "aucun"}
print(f"{n} cas dans {P}\n")
echecs = [t for t in T if "categorie_finale" not in t]
print("ÉCHECS (pas de label final) :", len(echecs), dict(Counter(t.get("_echec") for t in echecs)))
print("échec de l'audit (label conservé) :", sum(1 for t in T if "etape4" in str(t.get("_echec", "")) and "categorie_finale" in t))
print("RÉPARTITION FINALE :", dict(Counter(abr[t.get("categorie_finale")] for t in T)))
print("  Étape 3 seule    :", dict(Counter(abr[g(t, "etape3_resultat").get("categorie")] for t in T)))
print("  plancher constantes appliqué (label relevé) :", sum(1 for t in T if g(t, "decision_apres_override_etape2_3").get("override_applique")))
print("\nFILE DE RELECTURE")
haute = [t for t in T if t.get("a_relire")]
print(f"  priorité haute : {len(haute)} / {n} = {100 * len(haute) / max(n, 1):.0f} %   (budget : 20)")
c = Counter()
for t in T:
    for r in t.get("raisons_relecture", []): c[r.split(" : ")[0]] += 1
for k, v in c.most_common(): print(f"    {v:3}  {k}")
print("  top 5 scores :", sorted(((t.get("score_relecture", 0), t["id"][-5:]) for t in haute), reverse=True)[:5])
print("\nEXTRACTION (Étape 1)")
e = [g(t, "etape1_extraction") for t in T]
print("  motif absent (null) :", sum(1 for x in e if not x.get("motif_id")), "| motif rejeté (inconnu) :", sum(1 for x in e if x.get("motif_rejete")))
print("  Appel B en échec :", sum(1 for x in e if x.get("appel_b_echec")), "| critères retenus :", sum(len(x.get("criteres_presents") or []) for x in e),
      "| critères rejetés :", sum(len(x.get("criteres_rejetes") or []) for x in e))
print("  avertissements âge/motif :", sum(1 for x in e for a in (x.get("avertissements") or []) if "âge" in a))
print("  motifs contestés (Étape 1 ≠ Étape 3) :", sum(1 for t in T if t.get("motif_conteste")))
print("  constantes non retrouvées :", sum(1 for t in T if g(t, "etape2_resultat").get("constantes_non_verifiees")))
print("\nAUDIT (Étape 4)")
print("  signaux :", sum(1 for t in T if g(t, "decision_finale").get("signal_sous_triage")),
      "| dont citation vérifiée :", sum(1 for t in T if g(t, "decision_finale").get("citation_verifiee")))
print("\nMOTIFS LES PLUS FRÉQUENTS :", Counter(x.get("motif_id") for x in e).most_common(6))
print("PROVENANCE :", Counter(json.dumps({k: (v.get("fournisseur"), v.get("modele")) for k, v in g(t, "provenance").items() if isinstance(v, dict) and "modele" in v}, sort_keys=True) for t in T).most_common(3))
