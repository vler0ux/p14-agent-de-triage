#!/usr/bin/env bash
#
# sauvegarder_donnees.sh
#
# Sauvegarde horodatée du dossier data/normalized/ (vignettes normalisées,
# résultats du filtre de pertinence, échantillons) et data/anonymized/
# (sorties d'anonymisation). Pensé pour être relancé à tout moment, y
# compris pendant que filtre_pertinence.py tourne encore en tâche de fond
# sur plusieurs jours — utile pour ne pas perdre la progression accumulée
# en cas de problème (disque, mauvaise manipulation, etc.).
#
# Le dossier data/raw/ n'est volontairement PAS sauvegardé : il est
# intégralement régénérable via les scripts d'ingestion (voir
# run_pipeline.sh), et contient notamment le cache MedQuAD volumineux.
#
# Usage :
#   ./sauvegarder_donnees.sh
#
# Chaque exécution crée un nouveau dossier horodaté dans backups/ — les
# sauvegardes précédentes ne sont jamais écrasées.

set -euo pipefail

RACINE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HORODATAGE=$(date +%Y%m%d_%H%M%S)
DOSSIER_SAUVEGARDE="$RACINE/backups/$HORODATAGE"

mkdir -p "$DOSSIER_SAUVEGARDE"

if [[ -d "$RACINE/data/normalized" ]]; then
    cp -r "$RACINE/data/normalized" "$DOSSIER_SAUVEGARDE/"
fi

if [[ -d "$RACINE/data/anonymized" ]]; then
    cp -r "$RACINE/data/anonymized" "$DOSSIER_SAUVEGARDE/"
fi

echo "Sauvegarde créée : $DOSSIER_SAUVEGARDE"
du -sh "$DOSSIER_SAUVEGARDE"
echo ""
echo "Contenu :"
find "$DOSSIER_SAUVEGARDE" -type f -name "*.jsonl" -exec wc -l {} \;