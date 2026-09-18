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
│   │   ├── llm_client.py                  # Abstraction d'appel LLM (Groq / Gemini, interface commune)
│   │   ├── filtre_pertinence.py           # Étape 0 : filtre de pertinence clinique (API Groq)
│   │   ├── extraction_etape1.py           # Étape 1 : extraction structurée du cas clinique (LLM)
│   │   ├── labellisation_etape3.py        # Étape 3 : proposition du niveau de priorité (LLM)
│   │   ├── contre_validation_etape4.py    # Étape 4 : challenge du label par un LLM "auditeur"
│   │   ├── generation_dialogue_etape5.py  # Étape 5 : génération du dialogue patient/agent
│   │   └── orchestrer_pipeline.py         # Enchaîne réellement les Étapes 1 → 2 → 3 → 4
│   └── anonymisation/
│       └── anonymiser_vignettes.py  # Anonymisation RGPD (Presidio + spaCy fr_core_news_md)
├── tests/
│   ├── test_red_flag_detector.py     # Suite de tests unitaires (Étape 2 — 20 tests, FAIT)
│   └── golden_etape1_extraction.json # Jeu de cas de référence (Étapes 1, 3, 4)
├── run_pipeline.sh         # Script d'orchestration (ingestion + tests, option --avec-filtre)
├── sauvegarder_donnees.sh  # Sauvegarde horodatée de data/normalized/ et data/anonymized/
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

Les Étapes 3, 4 et 5 utilisent `llm_client.py`, une couche d'abstraction
commune à Groq et Google Gemini (`--fournisseur groq|gemini`), afin de
pouvoir varier le modèle d'une étape à l'autre et réduire la corrélation des
erreurs entre les jugements. Selon le fournisseur choisi, prévoir
`GROQ_API_KEY` et/ou `GEMINI_API_KEY` dans `.env` (clé Gemini gratuite sur
Google AI Studio — attention, les quotas du tier gratuit varient beaucoup
selon le modèle).

### Étape 1 — Extraction structurée (`extraction_etape1.py`)

Extrait le cas clinique en JSON structuré (motif, constantes vitales,
critères) à partir du texte libre d'une vignette, en s'appuyant
dynamiquement sur le vocabulaire du référentiel FRENCH. Se teste contre le
jeu de cas de référence `tests/golden_etape1_extraction.json`.

```bash
cd src/generation
python3 extraction_etape1.py --golden ../../tests/golden_etape1_extraction.json
```

### Étape 3 — Labellisation (`labellisation_etape3.py`)

À partir du cas structuré (sortie de l'Étape 1), le LLM propose un niveau de
priorité parmi `urgence_maximale` / `moderee` / `differee`, avec
justification. Ce n'est pas le label final : il est ensuite croisé avec la
détection déterministe de l'Étape 2 (`red_flag_detector.py`), puis challengé
à l'Étape 4. Utilise Gemini par défaut, pour ne pas entrer en compétition
avec le quota Groq du filtre de pertinence.

```bash
cd src/generation
python3 labellisation_etape3.py --golden ../../tests/golden_etape1_extraction.json
```

### Étape 4 — Contre-validation (`contre_validation_etape4.py`)

Un second appel LLM, en posture d'auditeur, vérifie uniquement si le label
retenu après le croisement Étape 2/3 ne SOUS-estime pas la gravité (jamais
l'inverse — le sur-triage n'est pas contesté à cette étape). Utilise Groq
par défaut, volontairement différent du fournisseur de l'Étape 3, pour
réduire la corrélation des erreurs entre les deux jugements.

```bash
cd src/generation
python3 contre_validation_etape4.py --golden ../../tests/golden_etape1_extraction.json
```

### Orchestration complète des Étapes 1 → 4 (`orchestrer_pipeline.py`)

Chaîne réellement les 4 étapes sur un fichier de vignettes (idéalement déjà
filtrées et anonymisées), avec croisement Étape 2/3 et résolution de la
contre-validation à chaque vignette. Conserve la trace complète de chaque
étape dans la sortie, pour l'auditabilité exigée par le brief. Reprend
automatiquement en cas d'interruption (comme `filtre_pertinence.py`).

```bash
cd src/generation
python3 orchestrer_pipeline.py \
    --input ../../data/anonymized/mediqal_mcqm_urgences_anonymise.jsonl \
    --output ../../data/normalized/mediqal_mcqm_pipeline_complet.jsonl \
    --limit 10
```

### Étape 5 — Génération du dialogue (`generation_dialogue_etape5.py`)

Dernière étape : à partir d'un cas structuré et du niveau de priorité déjà
validé par les Étapes 1 à 4 (que le LLM ne doit jamais remettre en cause),
génère un dialogue multi-tours réaliste entre un patient (expression
naturelle et imparfaite — hésitations, ordre désordonné) et un agent de
triage, conclu par une phrase en langage naturel (jamais de jargon type
« priorité modérée »). Prend en entrée la sortie de `orchestrer_pipeline.py`.

```bash
cd src/generation
python3 generation_dialogue_etape5.py \
    --input ../../data/normalized/mediqal_mcqm_pipeline_complet.jsonl \
    --output ../../data/normalized/mediqal_mcqm_dialogues.jsonl
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

## Sauvegarde des données

```bash
./sauvegarder_donnees.sh
```

Copie `data/normalized/` et `data/anonymized/` dans un dossier horodaté sous
`backups/` (jamais écrasé). Peut être relancé à tout moment, y compris
pendant qu'un script de génération tourne encore en tâche de fond.
`data/raw/` n'est volontairement pas sauvegardé : il est régénérable via les
scripts d'ingestion.

## Prochaines étapes

1. Vérifier précisément le schéma du subset `oeq` de MediQAl (non encore
   confirmé).
2. Lancer le pipeline complet (Étapes 1 à 5) à l'échelle sur l'ensemble des
   vignettes filtrées et anonymisées, au-delà des jeux de test actuels.
3. Étendre la couverture de tests aux modules `filtre_pertinence.py`,
   `extraction_etape1.py`, `labellisation_etape3.py`,
   `contre_validation_etape4.py`, `generation_dialogue_etape5.py` et
   `anonymiser_vignettes.py`.
4. Constituer et documenter le jeu de données SFT final à partir des sorties
   de `generation_dialogue_etape5.py`.
