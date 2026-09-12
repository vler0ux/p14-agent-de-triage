# POC Agent de Triage Médical — CHSA

Proof of Concept d'un agent IA de triage médical, développé pour le Centre
Hospitalier Saint-Aurélien (CHSA).

## Structure du projet

```
triage-poc-chsa/
├── data/
│   ├── raw/                # Cache local / échantillons de test des corpus sources
│   ├── normalized/         # Sorties JSONL au format "vignette normalisée" (une par source)
│   └── anonymized/         # Sorties JSONL anonymisées (RGPD) + journaux d'audit
├── src/
│   ├── ingestion/
│   │   ├── schema.py                    # Format commun de vignette normalisée
│   │   ├── load_mediqal.py              # Ingestion MediQAl (FAIT — validé)
│   │   ├── load_frenchmedmcqa.py        # Ingestion FrenchMedMCQA (FAIT — validé)
│   │   ├── load_medquad.py              # Ingestion MedQuAD (FAIT — validé)
│   │   └── echantillonner_vignettes.py  # Échantillonnage aléatoire reproductible (FAIT — validé)
│   ├── referentiel/
│   │   ├── french_referentiel.json  # Référentiel FRENCH structuré (16 motifs, dont la distinction
│   │   │                            # traumatisme_amputation_membre / traumatisme_amputation_digitale
│   │   │                            # et le motif diarrhee_vomissements)
│   │   └── red_flag_detector.py     # Détection déterministe + logique d'override
│   ├── generation/
│   │   ├── filtre_pertinence.py     # Étape 0 : filtre de pertinence clinique (API Groq)
│   │   └── extraction_etape1.py     # Étape 1 : extraction structurée du cas clinique (LLM)
│   └── anonymisation/
│       └── anonymiser_vignettes.py  # Anonymisation RGPD (Presidio + spaCy fr_core_news_md)
├── tests/
│   ├── test_red_flag_detector.py     # Suite de tests unitaires (Étape 2 — 20 tests, FAIT)
│   └── golden_etape1_extraction.json # Jeu de cas de référence pour l'Étape 1
├── run_pipeline.sh         # Script d'orchestration (ingestion + tests, option --avec-filtre)
├── requirements.txt
├── .env.example
└── .gitignore
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # puis renseigner vos clés/tokens si nécessaire
```

## Utilisation — Ingestion des corpus

Les trois scripts suivent la même logique : ils téléchargent leur corpus
source, le normalisent vers le format commun défini dans `schema.py`,
filtrent les lignes dont un champ obligatoire (id, source, langue, question)
est manquant, et affichent des statistiques rapides.

### MediQAl

```bash
cd src/ingestion
python3 load_mediqal.py --subset mcqm --split train \
    --output ../../data/normalized/mediqal_mcqm_train.jsonl
```

Subsets disponibles : `mcqm` (QCM multi-réponses), `mcqu` (QCM réponse
unique). Le subset `oeq` (questions ouvertes) existe mais son schéma exact
n'a pas encore été vérifié — à confirmer avant usage.

Pour tester sans télécharger (ex. avec un export déjà en local) :

```bash
python3 load_mediqal.py --subset mcqm --split train \
    --local-file /chemin/vers/export.jsonl \
    --output ../../data/normalized/mediqal_mcqm_train.jsonl
```

### FrenchMedMCQA

```bash
cd src/ingestion
python3 load_frenchmedmcqa.py --split train \
    --output ../../data/normalized/frenchmedmcqa_train.jsonl
```

⚠️ Ce corpus n'a pas de contexte clinique (questions d'examen de pharmacie
isolées) — c'est normal, pas une erreur d'extraction. Les données sont
téléchargées directement depuis le dépôt GitHub source (pas via
`datasets`/HuggingFace, dont le script de chargement pour ce corpus est
cassé depuis la dépréciation des "dataset scripts").

### MedQuAD

```bash
cd src/ingestion
python3 load_medquad.py --output ../../data/normalized/medquad.jsonl
```

Corpus anglophone uniquement, ~11 000 fichiers XML répartis en 12 dossiers
thématiques. Par défaut, les 3 dossiers dont les réponses ont été retirées
pour respecter le copyright MedlinePlus (`10_MPlus_ADAM_QA`,
`11_MPlusDrugs_QA`, `12_MPlusHerbsSupplements_QA`) sont exclus
automatiquement. Le premier lancement télécharge et met en cache l'archive
complète du dépôt (~7 Mo) ; les lancements suivants réutilisent ce cache.

Options utiles : `--folders` (limiter à certains dossiers), `--limit`
(limiter le nombre de vignettes, pratique pour tester), `--include-empty-answers`
(déconseillé pour la génération SFT).

## Utilisation — Filtrage, extraction et anonymisation

### Étape 0 — Filtre de pertinence clinique (`filtre_pertinence.py`)

Filtre les vignettes normalisées pour ne garder que celles décrivant une
présentation clinique AIGUË plausible pour un passage aux urgences, via
l'API Groq (modèle `openai/gpt-oss-20b`). Écrit ses résultats au fil de
l'eau et reprend automatiquement en cas d'interruption ou de dépassement de
quota (HTTP 429) — relancer exactement la même commande suffit.

```bash
cd src/generation
python3 filtre_pertinence.py \
    --input ../../data/normalized/mediqal_mcqm_train.jsonl \
    --output-retenues ../../data/normalized/mediqal_mcqm_urgences.jsonl \
    --output-exclues ../../data/normalized/mediqal_mcqm_exclues.jsonl
```

Nécessite `GROQ_API_KEY` dans `.env` (clé gratuite sur console.groq.com/keys).

### Étape 1 — Extraction structurée (`extraction_etape1.py`)

Extrait le cas clinique en JSON structuré (motif, constantes vitales,
critères) à partir du texte libre d'une vignette, en s'appuyant
dynamiquement sur le vocabulaire du référentiel FRENCH. Se teste contre le
jeu de cas de référence `tests/golden_etape1_extraction.json`.

```bash
cd src/generation
python3 extraction_etape1.py --golden ../../tests/golden_etape1_extraction.json
```

### Anonymisation RGPD (`anonymiser_vignettes.py`)

Anonymise les champs textuels (`contexte_clinique`, `question`) via Presidio
(AnalyzerEngine + AnonymizerEngine) et spaCy (`fr_core_news_md`), avec un
filtre par titre de civilité pour éviter les faux positifs sur les
éponymes médicaux (ex. « signe de Lasègue ») et les noms de médicaments.
Produit un journal d'audit de traçabilité en plus de la sortie anonymisée.

```bash
cd src/anonymisation
python3 anonymiser_vignettes.py \
    --input ../../data/normalized/mediqal_mcqm_train.jsonl \
    --output ../../data/anonymized/mediqal_mcqm_train_anonymise.jsonl \
    --audit ../../data/anonymized/mediqal_mcqm_train_audit.jsonl
```

Prérequis : `pip install presidio-analyzer presidio-anonymizer spacy` puis
`python -m spacy download fr_core_news_md`.

## Tests

```bash
pytest tests/ -v
```

La suite actuelle couvre le module `red_flag_detector.py` (Étape 2) : seuils
de constantes vitales, règles pédiatriques, modulateurs de motifs,
combinaison de plusieurs red flags, et les 3 scénarios du mécanisme
d'override (LLM sous-estime / LLM déjà prudent / aucun red flag détecté).

## Pour lancer le pipeline

```bash
./run_pipeline.sh                # ingestion des 3 corpus + tests (rapide)
./run_pipeline.sh --avec-filtre  # + le filtre de pertinence complet (~1h15)
```

## Prochaines étapes

1. Vérifier précisément le schéma du subset `oeq` de MediQAl (non encore
   confirmé).
2. Écrire les prompts des Étapes 3 (labellisation), 4 (contre-validation) et
   5 (génération du dialogue) dans `src/generation/`.
3. Brancher `red_flag_detector.py` dans le pipeline complet (croisement
   Étape 2 / Étape 3).
4. Étendre la couverture de tests aux modules `filtre_pertinence.py`,
   `extraction_etape1.py` et `anonymiser_vignettes.py`.
