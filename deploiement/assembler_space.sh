#!/usr/bin/env bash
# Assemble le dossier à envoyer sur le Space Hugging Face, à partir du code du dépôt.
# Source unique : le plancher, le décideur, le prompt et le chat template sont COPIÉS depuis src/,
# jamais dupliqués à la main. Utilisé à la main (premier déploiement) et par le workflow de CD.
#
# Usage (depuis la racine du projet) :
#   bash deploiement/assembler_space.sh            # -> build/space/
#   bash deploiement/assembler_space.sh /tmp/space # autre dossier de sortie
set -euo pipefail

RACINE="$(cd "$(dirname "$0")/.." && pwd)"
SORTIE="${1:-$RACINE/build/space}"

rm -rf "$SORTIE"
mkdir -p "$SORTIE/src/demo" "$SORTIE/src/decideur" "$SORTIE/src/entrainement"

cp "$RACINE/deploiement/space/app.py" "$RACINE/deploiement/space/requirements.txt" "$SORTIE/"
cp "$RACINE/src/demo/decideur.py" "$SORTIE/src/demo/"
cp "$RACINE/src/decideur/preparer_dataset_decideur.py" "$RACINE/src/decideur/train_decideur.py" "$SORTIE/src/decideur/"
cp "$RACINE/src/entrainement/prompt_agent.py" "$RACINE/src/entrainement/template_triage.jinja" "$SORTIE/src/entrainement/"

echo "Space assemblé dans $SORTIE :"
(cd "$SORTIE" && find . -type f | sort)
