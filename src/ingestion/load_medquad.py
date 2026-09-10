"""
load_medquad.py

Script d'ingestion du corpus MedQuAD (abachaa/MedQuAD, licence CC BY 4.0),
et normalisation vers le format commun "vignette normalisée" (cf. schema.py).

Structure réelle du corpus (vérifiée directement sur les fichiers, PAS sur
une documentation tierce) :
    - 12 dossiers (un par source NIH : CancerGov, GARD, GHR, MPlus Health
      Topics, NIDDK, NINDS, SeniorHealth, NHLBI, CDC, MPlus ADAM,
      MPlusDrugs, MPlusHerbsSupplements), ~11 274 fichiers XML au total,
      47 457 paires question/réponse au total (toutes langues = anglais
      uniquement, ce corpus n'a pas de version française).
    - Chaque fichier XML = un <Document> avec un <Focus> (sujet médical),
      des métadonnées UMLS, et une liste de <QAPair> (question + réponse).
    - ⚠️ DEUX VARIANTES DE SCHÉMA COEXISTENT selon les dossiers :
        Variante A (ex. 9_CDC_QA) : <UMLS><CUI>...</CUI><SemanticType>...</SemanticType></UMLS>
                                     directement sous <Document>.
        Variante B (ex. 2_GARD_QA, 4_MPlus_Health_Topics_QA) :
                                     <FocusAnnotations><UMLS><CUIs><CUI>...</CUI></CUIs>
                                     <SemanticTypes><SemanticType>...</SemanticType></SemanticTypes>
                                     </UMLS><Synonyms>...</Synonyms></FocusAnnotations>
      Le parseur ci-dessous cherche les balises via XPath relatif ("//") pour
      être robuste aux deux variantes, plutôt que de supposer un chemin fixe.
    - ⚠️ 3 DOSSIERS N'ONT AUCUNE RÉPONSE (retirées pour respecter le
      copyright MedlinePlus, cf. readme officiel du dépôt) :
        10_MPlus_ADAM_QA (4366 documents), 11_MPlusDrugs_QA (1312),
        12_MPlusHerbsSupplements_QA (99). Soit ~5777 documents INUTILISABLES
        pour nos vignettes SFT (question sans réponse). Exclus par défaut.

Usage :
    python load_medquad.py --output ../../data/normalized/medquad.jsonl
    python load_medquad.py --folders 9_CDC_QA 1_CancerGov_QA --limit 200 \
        --output ../../data/normalized/medquad_test.jsonl
"""

import argparse
import json
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from schema import VignetteNormalisee, valider_vignette

ARCHIVE_URL = "https://codeload.github.com/abachaa/MedQuAD/tar.gz/refs/heads/master"

TOUS_LES_DOSSIERS = [
    "1_CancerGov_QA", "2_GARD_QA", "3_GHR_QA", "4_MPlus_Health_Topics_QA",
    "5_NIDDK_QA", "6_NINDS_QA", "7_SeniorHealth_QA", "8_NHLBI_QA_XML",
    "9_CDC_QA", "10_MPlus_ADAM_QA", "11_MPlusDrugs_QA",
    "12_MPlusHerbsSupplements_QA",
]

# Dossiers dont les réponses ont été retirées pour respecter le copyright
# MedlinePlus (vérifié directement sur les données, cf. docstring ci-dessus).
DOSSIERS_SANS_REPONSES = {
    "10_MPlus_ADAM_QA", "11_MPlusDrugs_QA", "12_MPlusHerbsSupplements_QA",
}

DOSSIERS_PAR_DEFAUT = [d for d in TOUS_LES_DOSSIERS if d not in DOSSIERS_SANS_REPONSES]


def telecharger_et_extraire(cache_dir: Path) -> Path:
    """Télécharge et extrait l'archive complète du dépôt si elle n'est pas déjà en cache."""
    racine = cache_dir / "MedQuAD-master"
    if racine.exists():
        print(f"Archive déjà extraite dans {racine} (cache réutilisé)")
        return racine

    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_path = cache_dir / "medquad.tar.gz"
    print("Téléchargement de l'archive complète du corpus (~7 Mo, une seule fois)...")
    urllib.request.urlretrieve(ARCHIVE_URL, archive_path)

    print("Extraction...")
    with tarfile.open(archive_path) as tar:
        tar.extractall(cache_dir)

    archive_path.unlink()  # on garde les XML extraits, pas l'archive compressée
    return racine


def parser_document_xml(chemin_xml: Path, dossier_source: str) -> list:
    """Parse un fichier XML MedQuAD et retourne la liste de ses paires Q/R brutes.

    Utilise des XPath relatifs ("//") pour rester robuste aux deux variantes
    de schéma observées dans le corpus (cf. docstring du module).
    """
    try:
        tree = ET.parse(chemin_xml)
    except ET.ParseError:
        return []  # quelques fichiers du corpus sont mal formés, on les ignore proprement

    root = tree.getroot()
    document_id = root.get("id")

    focus_el = root.find(".//Focus")
    focus = focus_el.text.strip() if focus_el is not None and focus_el.text else None

    cui_el = root.find(".//CUI")
    cui = cui_el.text.strip() if cui_el is not None and cui_el.text else None

    semantic_type_el = root.find(".//SemanticType")
    semantic_type = semantic_type_el.text.strip() if semantic_type_el is not None and semantic_type_el.text else None

    lignes = []
    for qa in root.findall(".//QAPair"):
        question_el = qa.find("Question")
        if question_el is None or not (question_el.text or "").strip():
            continue
        answer_el = qa.find("Answer")
        answer_text = (answer_el.text or "").strip() if answer_el is not None else ""

        lignes.append({
            "id": f"medquad_{document_id}_{qa.get('pid')}",
            "dossier_source": dossier_source,
            "focus": focus,
            "cui": cui,
            "semantic_type": semantic_type,
            "question": question_el.text.strip(),
            "qtype": question_el.get("qtype"),
            "answer": answer_text,
        })
    return lignes


def normaliser_ligne_medquad(ligne: dict) -> dict:
    vignette = VignetteNormalisee(
        id=ligne["id"],
        source="MedQuAD",
        subset=ligne["dossier_source"],
        split=None,  # ce corpus n'a pas de split officiel train/validation/test
        langue="en",
        contexte_clinique=None,  # structure factuelle Focus/Question/Réponse, pas de vignette patient
        question=ligne["question"],
        choix=None,  # pas de QCM dans ce corpus
        reponse_correcte=ligne["answer"] or None,
        specialite=ligne["focus"],  # sujet médical abordé — le champ le plus proche disponible ici
        type_question=ligne["qtype"],  # ex. "information", "symptoms", "treatment", "causes"...
        task="QA",
    )
    return vignette.to_dict()


def main():
    parser = argparse.ArgumentParser(description="Ingestion et normalisation du corpus MedQuAD")
    parser.add_argument(
        "--folders", nargs="+", default=DOSSIERS_PAR_DEFAUT, choices=TOUS_LES_DOSSIERS,
        help="Sous-dossiers à traiter (par défaut : tous sauf les 3 sans réponses)"
    )
    parser.add_argument(
        "--include-empty-answers", action="store_true",
        help="Inclure les vignettes sans réponse (déconseillé pour la génération SFT)"
    )
    parser.add_argument("--cache-dir", default="../../data/raw/medquad_source",
                         help="Dossier de cache pour l'archive téléchargée")
    parser.add_argument("--output", required=True, help="Chemin du fichier JSONL de sortie")
    parser.add_argument("--limit", type=int, default=None, help="Limiter le nombre total de vignettes (debug)")
    args = parser.parse_args()

    racine = telecharger_et_extraire(Path(args.cache_dir))

    lignes_brutes = []
    for dossier in args.folders:
        chemin_dossier = racine / dossier
        fichiers = sorted(chemin_dossier.glob("*.xml"))
        print(f"  {dossier} : {len(fichiers)} fichiers XML")
        for fichier in fichiers:
            lignes_brutes.extend(parser_document_xml(fichier, dossier))
        if args.limit and len(lignes_brutes) >= args.limit:
            break

    if args.limit:
        lignes_brutes = lignes_brutes[: args.limit]

    lignes_normalisees = []
    sans_reponse = 0
    lignes_invalides = 0
    for ligne in lignes_brutes:
        if not ligne["answer"] and not args.include_empty_answers:
            sans_reponse += 1
            continue
        vignette = normaliser_ligne_medquad(ligne)
        manquants = valider_vignette(vignette)
        if manquants:
            lignes_invalides += 1
            continue
        lignes_normalisees.append(vignette)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for vignette in lignes_normalisees:
            f.write(json.dumps(vignette, ensure_ascii=False) + "\n")

    print(f"\n{len(lignes_normalisees)} vignettes normalisées écrites dans {output_path}")
    if sans_reponse:
        print(f"  {sans_reponse} vignettes ignorées (réponse vide — usez --include-empty-answers pour les garder)")
    if lignes_invalides:
        print(f"  {lignes_invalides} lignes ignorées (champs obligatoires manquants)")


if __name__ == "__main__":
    main()