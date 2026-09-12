#!/usr/bin/env bash
#
# run_pipeline.sh
#
# Script d'orchestration reproductible du pipeline de données du POC
# Agent de Triage Médical (CHSA). Chaîne les étapes déjà validées :
# ingestion des 3 corpus, tests, puis filtre de pertinence (Étape 0).
#
# Prérequis : environnement virtuel activé, dépendances installées
# (pip install -r requirements.txt), fichier .env renseigné (GROQ_API_KEY
# pour l'étape de filtrage).
#
# Usage :
#   ./run_pipeline.sh                # ingestion + tests uniquement (rapide)
#   ./run_pipeline.sh --avec-filtre  # inclut aussi le filtre de pertinence
#                                     # (~1h15 pour mcqm+mcqu à cause du
#                                     # quota Groq, 30 requêtes/minute)

set -euo pipefail  # arrête le script à la moindre erreur, plutôt que de continuer silencieusement

AVEC_FILTRE=false
if [[ "${1:-}" == "--avec-filtre" ]]; then
    AVEC_FILTRE=true
fi

RACINE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="$RACINE/data/normalized"

echo "=== Étape 1/3 : Ingestion des corpus ==="

cd "$RACINE/src/ingestion"

echo "--- MediQAl (mcqm, mcqu) ---"
for subset in mcqm mcqu; do
    for split in train validation test; do
        python3 load_mediqal.py --subset "$subset" --split "$split" \
            --output "$DATA/mediqal_${subset}_${split}.jsonl"
    done
done

echo "--- FrenchMedMCQA ---"
for split in train validation test; do
    python3 load_frenchmedmcqa.py --split "$split" \
        --output "$DATA/frenchmedmcqa_${split}.jsonl"
done

echo "--- MedQuAD (dossiers avec réponses uniquement, par défaut du script) ---"
python3 load_medquad.py --output "$DATA/medquad_full.jsonl"

echo ""
echo "=== Étape 2/3 : Suite de tests ==="
cd "$RACINE"
pytest tests/ -v

if [[ "$AVEC_FILTRE" == true ]]; then
    echo ""
    echo "=== Étape 3/3 : Filtre de pertinence (Étape 0 du mécanisme) ==="
    echo "⚠️  Cette étape appelle l'API Groq, ~30 req/min → prévoir un peu de temps."
    cd "$RACINE/src/generation"

    for subset in mcqm mcqu; do
        echo "--- Filtrage $subset ---"
        python3 filtre_pertinence.py \
            --input "$DATA/mediqal_${subset}_train.jsonl" \
            --output-retenues "$DATA/mediqal_${subset}_urgences.jsonl" \
            --output-exclues "$DATA/mediqal_${subset}_exclues.jsonl"
    done
else
    echo ""
    echo "(Filtre de pertinence non exécuté — relancer avec --avec-filtre pour l'inclure)"
fi

echo ""
echo "=== Pipeline terminé ==="