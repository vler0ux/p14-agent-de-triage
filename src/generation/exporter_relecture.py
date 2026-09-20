"""
exporter_relecture.py

Exporte, dans un fichier CSV lisible par Excel ou LibreOffice, les cas à relire par un humain,
TRIÉS par priorité, pour un budget de relectures limité (défaut : 20).

Le tri utilise `score_relecture` (produit par orchestrer_pipeline.py) : un label « différée » sous un
plancher « urgence maximale » de la grille passe avant un label « modérée » sous le même plancher, etc.

Contenu du tableur, une ligne par cas à relire (priorité haute) :
    rang, dans_le_budget, id, cas_id, score, label_actuel, plancher_motif, ecart, motif,
    criteres_coches (avec la citation qui les justifie), raisons, vignette,
    puis trois colonnes VIDES à remplir : ma_decision, mon_label, ma_note.

Les cas au-delà du budget sont listés aussi (dans_le_budget = non) : ils NE SONT PAS relus. Pour l'entraînement
et le test, ils doivent être exclus ou marqués « non relus » (fichier annexe *_non_relus.txt).

Usage :
    python exporter_relecture.py \
        --input ../../data/normalized/mediqal_mcqm_pipeline_complet.jsonl \
        --output ../../data/relecture/relecture.csv --n 20
    # option : --inclure-basses  ajoute aussi les cas de priorité basse à la suite
"""

import argparse
import csv
import json
import sys
from pathlib import Path

COLONNES = ["rang", "dans_le_budget", "id", "cas_id", "score", "label_actuel", "plancher_motif", "ecart",
            "motif", "criteres_coches", "raisons", "vignette", "ma_decision", "mon_label", "ma_note"]


def lire_jsonl(chemin: str) -> list:
    lignes = []
    with open(chemin, encoding="utf-8") as f:
        for ligne in f:
            ligne = ligne.strip()
            if ligne:
                lignes.append(json.loads(ligne))
    return lignes


def criteres_avec_citations(trace: dict) -> str:
    ext = trace.get("etape1_extraction") or {}
    citations = ext.get("criteres_citations") or {}
    if not ext.get("criteres_presents"):
        return "aucun"
    return " || ".join(f"{c} <- « {citations.get(c, '?')} »" for c in ext["criteres_presents"])


def construire_lignes(traces: list, n: int, inclure_basses: bool) -> list:
    retenus = [t for t in traces if t.get("a_relire")]
    retenus.sort(key=lambda t: (-t.get("score_relecture", 0), t["id"]))
    if inclure_basses:
        basses = [t for t in traces if not t.get("a_relire") and t.get("priorite_relecture") == "basse"]
        basses.sort(key=lambda t: t["id"])
        retenus += basses
    lignes = []
    for rang, t in enumerate(retenus, 1):
        d23 = t.get("decision_apres_override_etape2_3") or {}
        lignes.append({
            "rang": rang,
            "dans_le_budget": "oui" if rang <= n and t.get("a_relire") else "non",
            "id": t["id"], "cas_id": t.get("cas_id") or "",
            "score": t.get("score_relecture", 0),
            "label_actuel": t.get("categorie_finale") or "",
            "plancher_motif": d23.get("categorie_plancher_motif") or "",
            "ecart": t.get("ecart_plancher", 0),
            "motif": (t.get("etape1_extraction") or {}).get("motif_id") or "",
            "criteres_coches": criteres_avec_citations(t),
            "raisons": " | ".join(t.get("raisons_relecture") or []),
            "vignette": (t.get("vignette_source") or "").replace("\n", " "),
            "ma_decision": "", "mon_label": "", "ma_note": "",
        })
    return lignes


def main():
    parser = argparse.ArgumentParser(description="Exporte les cas à relire, triés par priorité")
    parser.add_argument("--input", required=True, help="Sortie JSONL de orchestrer_pipeline.py")
    parser.add_argument("--output", required=True, help="Fichier CSV à créer")
    parser.add_argument("--n", type=int, default=20, help="Budget de relectures (défaut : 20)")
    parser.add_argument("--inclure-basses", action="store_true", help="Ajoute aussi les cas de priorité basse (hors budget)")
    args = parser.parse_args()

    traces = lire_jsonl(args.input)
    lignes = construire_lignes(traces, args.n, args.inclure_basses)

    sortie = Path(args.output)
    sortie.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig + séparateur « ; » : s'ouvre correctement dans Excel en français
    with open(sortie, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLONNES, delimiter=";")
        w.writeheader()
        w.writerows(lignes)

    hors_budget = [l["id"] for l in lignes if l["dans_le_budget"] == "non"]
    annexe = sortie.with_name(sortie.stem + "_non_relus.txt")
    annexe.write_text("\n".join(hors_budget) + ("\n" if hors_budget else ""), encoding="utf-8")

    n_haute = sum(1 for t in traces if t.get("a_relire"))
    print(f"{len(traces)} cas lus ; {n_haute} à relire en priorité haute ({100 * n_haute / max(len(traces), 1):.0f} %).", file=sys.stderr)
    print(f"  {min(n_haute, args.n)} dans le budget de {args.n} -> {sortie}", file=sys.stderr)
    print(f"  {len(hors_budget)} hors budget (non relus) -> {annexe}", file=sys.stderr)


if __name__ == "__main__":
    main()
