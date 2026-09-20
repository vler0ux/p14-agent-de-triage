"""
construire_referentiel_depuis_grille.py

Reconstruit les MODULATEURS (critères et niveaux de tri) de french_referentiel.json
à partir de la transcription de la grille FRENCH V1.1 (french_grille_v1_1_transcription.json).

Pourquoi : dans la première version du référentiel, les modulateurs n'existaient que pour 13 motifs
et plusieurs niveaux étaient décalés d'une colonne par rapport à la grille (ex. douleur thoracique :
"ECG anormal typique de SCA" était en tri 2 au lieu du tri 1). La transcription vient d'une extraction
AUTOMATIQUE du tableau du PDF officiel (colonnes détectées par les traits du tableau), donc vérifiable.

Règles de construction :
  - le tri de base de chaque motif est le "Tri M" (tri médian) de la grille ;
  - chaque cellule non vide des colonnes Tri 1 à Tri 5 devient un modulateur (critère = texte de la cellule,
    niveau = colonne) ;
  - les cellules "avis référent (MAO, MCO)" sont omises : ce n'est pas un élément observable dans une vignette ;
  - deux lignes de la grille rattachées au même motif fusionné (ex. hématémèse + méléna) sont combinées
    colonne par colonne ;
  - les fautes de frappe évidentes de la grille sont corrigées dans les critères (la transcription garde
    le texte d'origine).

Usage :
    python construire_referentiel_depuis_grille.py \
        --entree french_referentiel.json --grille french_grille_v1_1_transcription.json \
        --sortie french_referentiel.json
"""

import argparse
import json
import re
from pathlib import Path

COLONNES = [("tri1", "1"), ("tri2", "2"), ("tri3a", "3A"), ("tri3b", "3B"), ("tri4", "4"), ("tri5", "5")]

FAUTES = {
    "régresssive": "régressive",
    "défromation": "déformation",
    "suspiçion": "suspicion",
    "comorbidiés": "comorbidités",
    "60:min": "60/min",
    "abces": "abcès",
}


# Critères dont le texte brut de la grille est ambigu une fois deux lignes fusionnées
SURCHARGES = {
    ("diarrhee_vomissements", "3B"): "Douleur abdominale ou vomissements abondants ; diarrhée abondante et/ou mauvaise tolérance",
}


def nettoyer_critere(texte: str) -> str:
    t = re.sub(r"\s+", " ", texte).strip()
    for faux, juste in FAUTES.items():
        t = t.replace(faux, juste)
    return t[:1].upper() + t[1:] if t else t


def est_avis_referent(texte: str) -> bool:
    """Cellule qui ne contient que « avis référent (MAO, MCO) » : non observable dans une vignette."""
    return re.fullmatch(r"avis référent \(MAO, MCO\)", texte.strip(), flags=re.IGNORECASE) is not None


def modulateurs_depuis_lignes(lignes: list) -> list:
    """Modulateurs d'un motif à partir d'une ou plusieurs lignes de la grille."""
    par_niveau = {}
    for ligne in lignes:
        for col, niveau in COLONNES:
            cellule = ligne.get(col, "").strip()
            if not cellule or est_avis_referent(cellule):
                continue
            critere = nettoyer_critere(cellule)
            liste = par_niveau.setdefault(niveau, [])
            if critere not in liste:
                liste.append(critere)
    return [{"critere": " ; ".join(par_niveau[n]), "tri": n} for _, n in COLONNES if n in par_niveau]


def appliquer_surcharges(motif_id: str, modulateurs: list) -> list:
    for m in modulateurs:
        if (motif_id, m["tri"]) in SURCHARGES:
            m["critere"] = SURCHARGES[(motif_id, m["tri"])]
    return modulateurs


def reconstruire(referentiel: dict, grille: dict) -> dict:
    """Retourne une copie du référentiel dont les motifs présents dans la grille ont leur tri de base
    et leurs modulateurs issus de la grille."""
    lignes_par_id = {}
    for ligne in grille["motifs"]:
        lignes_par_id.setdefault(ligne["id_referentiel"], []).append(ligne)

    sortie = json.loads(json.dumps(referentiel))
    for motif in sortie["motifs"]:
        lignes = lignes_par_id.get(motif["id"])
        if not lignes:
            motif["hors_grille_v1_1"] = True   # motif propre à ce référentiel (absent du PDF)
            continue
        motif["tri_base"] = lignes[0]["tri_median"]
        motif["modulateurs"] = appliquer_surcharges(motif["id"], modulateurs_depuis_lignes(lignes))
        pages = sorted({l["page"] for l in lignes})
        motif["source"] = (f"Grille FRENCH V1.1 (juin 2018), page{'s' if len(pages) > 1 else ''} "
                           f"{', '.join(map(str, pages))} : tri médian et critères extraits automatiquement du PDF.")
        motif.pop("a_valider_par_urgentiste", None)
        motif.pop("hors_grille_v1_1", None)
    return sortie


def main():
    ici = Path(__file__).parent
    parser = argparse.ArgumentParser(description="Reconstruit les modulateurs du référentiel depuis la grille FRENCH")
    parser.add_argument("--entree", default=str(ici / "french_referentiel.json"))
    parser.add_argument("--grille", default=str(ici / "french_grille_v1_1_transcription.json"))
    parser.add_argument("--sortie", default=str(ici / "french_referentiel.json"))
    args = parser.parse_args()

    referentiel = json.load(open(args.entree, encoding="utf-8"))
    grille = json.load(open(args.grille, encoding="utf-8"))
    nouveau = reconstruire(referentiel, grille)
    with open(args.sortie, "w", encoding="utf-8") as f:
        json.dump(nouveau, f, ensure_ascii=False, indent=2)
    n_mod = sum(1 for m in nouveau["motifs"] if m["modulateurs"])
    print(f"{len(nouveau['motifs'])} motifs, {n_mod} avec modulateurs -> {args.sortie}")


if __name__ == "__main__":
    main()
