"""
anonymiser_vignettes.py

Anonymisation RGPD des vignettes normalisées, via Presidio (AnalyzerEngine +
AnonymizerEngine) avec le modèle linguistique français spaCy (fr_core_news_md).

Contexte important à documenter dans le rapport : les vignettes de MediQAl,
FrenchMedMCQA et MedQuAD sont des questions d'examen ou des articles de
vulgarisation — PAS des dossiers patients réels. Les noms qui y apparaissent
("Monsieur D.", "Mademoiselle B.") sont déjà des identifiants fictifs par
construction, pas des données personnelles réelles. Ce script reste
néanmoins nécessaire pour :
  1. Respecter l'exigence explicite du brief (anonymisation + traçabilité RGPD) ;
  2. Traiter systématiquement toute donnée d'entrée comme potentiellement
     sensible (principe de précaution), utile si des sources moins "propres"
     étaient ajoutées au pipeline par la suite ;
  3. Documenter un processus reproductible et auditable, plutôt que de se
     reposer sur une confiance non vérifiée dans la nature "déjà anonyme"
     des sources actuelles.

Usage :
    python anonymiser_vignettes.py \
        --input ../../data/normalized/mediqal_mcqm_train.jsonl \
        --output ../../data/anonymized/mediqal_mcqm_train_anonymise.jsonl \
        --audit ../../data/anonymized/mediqal_mcqm_train_audit.jsonl

Prérequis :
    pip install presidio-analyzer presidio-anonymizer spacy
    python -m spacy download fr_core_news_md
    python -m spacy download en_core_web_md   # pour les vignettes MedQuAD (anglais)
"""

import argparse
import json
from pathlib import Path
import re

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

# Champs textuels susceptibles de contenir des données personnelles.
# Les champs "choix" (options de QCM) ne sont volontairement pas inclus :
# leur contenu est structurellement factuel (propositions médicales), le
# risque d'y trouver un nom de patient est nul dans les 3 corpus utilisés.
CHAMPS_A_ANONYMISER = ["contexte_clinique", "question"]

# PERSON couvre l'exigence minimale du brief ("à défaut nom, prénom des
# patients"). D'autres types (LOCATION, DATE_TIME...) peuvent être ajoutés
# via --entites si un usage futur l'exige.
ENTITES_PAR_DEFAUT = ["PERSON"]

MODELES_SPACY = {"fr": "fr_core_news_md", "en": "en_core_web_md"}

# Un nom de patient réel est presque toujours précédé d'un titre de civilité
# ("Monsieur D.", "Mme F. Simone") — alors que les éponymes médicaux (signe
# de Lasègue, maladie de Kahler...), noms de médicaments (Esidrex,
# cyclophosphamide...) et abréviations biologiques (Ht, VS...) ne le sont
# jamais. C'est ce qui permet de distinguer un vrai nom d'un faux positif,
# le modèle spaCy générique ne connaissant pas le vocabulaire médical.
TITRE_CIVILITE_REGEX = re.compile(
    r"(monsieur|madame|mme|m\.|mlle|mademoiselle|dr\.?|docteur)\b",
    re.IGNORECASE,
)


def a_titre_de_civilite(texte_complet: str, debut: int, fin: int) -> bool:
    """Vérifie si l'entité détectée commence par un titre de civilité, ou est
    immédiatement précédée d'un titre (fenêtre de 15 caractères)."""
    entite = texte_complet[debut:fin]
    avant = texte_complet[max(0, debut - 15):debut]
    return bool(TITRE_CIVILITE_REGEX.match(entite)) or bool(TITRE_CIVILITE_REGEX.search(avant))

def construire_moteurs(langue: str):
    modele = MODELES_SPACY.get(langue, "fr_core_news_md")
    config = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": langue, "model_name": modele}],
    }
    provider = NlpEngineProvider(nlp_configuration=config)
    nlp_engine = provider.create_engine()
    analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=[langue])
    anonymizer = AnonymizerEngine()
    return analyzer, anonymizer


def anonymiser_texte(analyzer, anonymizer, texte: str, langue: str, entites: list) -> tuple:
    """Retourne (texte_anonymise, detections_appliquees, detections_ecartees)."""
    if not texte:
        return texte, [], []

    resultats_bruts = analyzer.analyze(text=texte, language=langue, entities=entites)
    if not resultats_bruts:
        return texte, [], []

    # Filtre spécifique aux entités PERSON : on exige un titre de civilité
    # pour éviter de confondre éponymes médicaux / médicaments avec un vrai
    # nom de patient (cf. anomalie constatée sur un premier test manuel).
    resultats_retenus, resultats_ecartes = [], []
    for r in resultats_bruts:
        if r.entity_type == "PERSON" and not a_titre_de_civilite(texte, r.start, r.end):
            resultats_ecartes.append(r)
        else:
            resultats_retenus.append(r)

    detections_ecartees = [
        {"type": r.entity_type, "original": texte[r.start:r.end], "score": round(r.score, 2)}
        for r in resultats_ecartes
    ]

    if not resultats_retenus:
        return texte, [], detections_ecartees

    anonymise = anonymizer.anonymize(
        text=texte,
        analyzer_results=resultats_retenus,
        operators={e: OperatorConfig("replace", {"new_value": f"[{e}]"}) for e in entites},
    )
    detections_appliquees = [
        {"type": r.entity_type, "original": texte[r.start:r.end], "score": round(r.score, 2)}
        for r in resultats_retenus
    ]
    return anonymise.text, detections_appliquees, detections_ecartees


def main():
    parser = argparse.ArgumentParser(description="Anonymisation RGPD des vignettes normalisées (Presidio)")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit", required=True,
                         help="Fichier de traçabilité : ce qui a été détecté/anonymisé, par vignette")
    parser.add_argument("--entites", nargs="+", default=ENTITES_PAR_DEFAUT,
                         help=f"Types d'entités Presidio à anonymiser (défaut : {ENTITES_PAR_DEFAUT})")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    vignettes = []
    with open(args.input, encoding="utf-8") as f:
        for ligne in f:
            ligne = ligne.strip()
            if ligne:
                vignettes.append(json.loads(ligne))
    if args.limit:
        vignettes = vignettes[: args.limit]

    if not vignettes:
        print("Aucune vignette à traiter.")
        return

    langue = vignettes[0].get("langue", "fr")
    print(f"Chargement du moteur Presidio (langue : {langue}, modèle : {MODELES_SPACY.get(langue)})...")
    analyzer, anonymizer = construire_moteurs(langue)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.audit).parent.mkdir(parents=True, exist_ok=True)

    n_avec_detection = 0
    with open(args.output, "w", encoding="utf-8") as f_out, \
         open(args.audit, "w", encoding="utf-8") as f_audit:

        for vignette in vignettes:
            audit_vignette = {"id": vignette["id"], "detections": {}}
            a_une_detection = False

            for champ in CHAMPS_A_ANONYMISER:
                texte_original = vignette.get(champ)
                if not texte_original:
                    continue
                texte_anonymise, detections, ecartees = anonymiser_texte(
                    analyzer, anonymizer, texte_original, vignette.get("langue", langue), args.entites
                )
                vignette[champ] = texte_anonymise
                if detections:
                    audit_vignette.setdefault("detections", {})[champ] = detections
                    a_une_detection = True
                if ecartees:
                    audit_vignette.setdefault("faux_positifs_ecartes", {})[champ] = ecartees
                    a_une_detection = True  # on trace aussi les cas écartés, pour audit
                    
            f_out.write(json.dumps(vignette, ensure_ascii=False) + "\n")
            if a_une_detection:
                f_audit.write(json.dumps(audit_vignette, ensure_ascii=False) + "\n")
                n_avec_detection += 1

    print(f"\n{len(vignettes)} vignettes traitées.")
    print(f"  {n_avec_detection} vignettes avec au moins une entité anonymisée")
    print(f"  Sortie anonymisée -> {args.output}")
    print(f"  Journal d'audit (traçabilité RGPD) -> {args.audit}")


if __name__ == "__main__":
    main()