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

Option --sur-triage N : réserve N places du budget à un ÉCHANTILLON ALÉATOIRE de cas dont le label est PLUS urgent
que le plancher de la grille (hors file prioritaire). Sert à estimer le taux de sur-triage, que la seule file
prioritaire (les labels SOUS le plancher) ne peut pas mesurer. Tirage reproductible (--graine).

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
import random
import sys
from pathlib import Path

ORDRE = {"urgence_maximale": 1, "moderee": 2, "differee": 3}

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


def est_au_dessus_du_plancher(t: dict) -> bool:
    """Vrai si le label final est PLUS urgent que le plancher du motif (grille FRENCH + critères cochés)."""
    plancher = (t.get("decision_apres_override_etape2_3") or {}).get("categorie_plancher_motif")
    label = t.get("categorie_finale")
    return bool(plancher and label and ORDRE[label] < ORDRE[plancher])


def construire_lignes(traces: list, n: int, inclure_basses: bool, sur_triage: int = 0, graine: int = 42) -> list:
    """Cas à relire : la file prioritaire triée par score (n - sur_triage places), puis un échantillon aléatoire
    de `sur_triage` cas au-dessus du plancher (places réservées), puis, en option, les priorités basses (hors budget)."""
    haute = [t for t in traces if t.get("a_relire")]
    haute.sort(key=lambda t: (-t.get("score_relecture", 0), t["id"]))
    places_haute = max(n - sur_triage, 0)
    candidats = sorted((t for t in traces if not t.get("a_relire") and est_au_dessus_du_plancher(t)), key=lambda t: t["id"])
    echantillon = random.Random(graine).sample(candidats, min(sur_triage, len(candidats))) if sur_triage else []
    ids_echantillon = {t["id"] for t in echantillon}
    # ordre du tableur : file prioritaire (score décroissant), puis l'échantillon de sur-triage
    retenus = haute + echantillon
    if inclure_basses:
        basses = [t for t in traces if not t.get("a_relire") and t.get("priorite_relecture") == "basse" and t["id"] not in ids_echantillon]
        basses.sort(key=lambda t: t["id"])
        retenus += basses
    lignes = []
    for rang, t in enumerate(retenus, 1):
        d23 = t.get("decision_apres_override_etape2_3") or {}
        dans = (t["id"] in ids_echantillon) or (t.get("a_relire") and rang <= places_haute)
        lignes.append({
            "rang": rang,
            "dans_le_budget": "oui" if dans else "non",
            "id": t["id"], "cas_id": t.get("cas_id") or "",
            "score": t.get("score_relecture", 0),
            "label_actuel": t.get("categorie_finale") or "",
            "plancher_motif": d23.get("categorie_plancher_motif") or "",
            "ecart": t.get("ecart_plancher", 0),
            "motif": (t.get("etape1_extraction") or {}).get("motif_id") or "",
            "criteres_coches": criteres_avec_citations(t),
            "raisons": ("[échantillon] label plus urgent que le plancher de la grille : tirage pour estimer le sur-triage"
                        if t["id"] in ids_echantillon else " | ".join(t.get("raisons_relecture") or [])),
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
    parser.add_argument("--sur-triage", type=int, default=0, help="Places du budget réservées à un échantillon aléatoire de cas au-dessus du plancher (défaut : 0)")
    parser.add_argument("--graine", type=int, default=42, help="Graine du tirage aléatoire (reproductibilité)")
    args = parser.parse_args()

    traces = lire_jsonl(args.input)
    lignes = construire_lignes(traces, args.n, args.inclure_basses, args.sur_triage, args.graine)

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
    n_budget = sum(1 for l in lignes if l["dans_le_budget"] == "oui")
    print(f"{len(traces)} cas lus ; {n_haute} à relire en priorité haute ({100 * n_haute / max(len(traces), 1):.0f} %).", file=sys.stderr)
    print(f"  {n_budget} dans le budget de {args.n} (dont {args.sur_triage} d'échantillon sur-triage demandés) -> {sortie}", file=sys.stderr)
    print(f"  {len(hors_budget)} hors budget (non relus) -> {annexe}", file=sys.stderr)


if __name__ == "__main__":
    main()
