# POC Agent de Triage Médical — CHSA

Proof of Concept d'un agent IA de triage médical, développé pour le Centre
Hospitalier Saint-Aurélien (CHSA).

## Structure du projet

```
triage-poc-chsa/
├── data/
│   ├── raw/                # Cache local / échantillons de test des corpus sources
│   └── normalized/         # Sorties JSONL au format "vignette normalisée" (une par source)
├── src/
│   ├── ingestion/
│   │   ├── schema.py              # Format commun de vignette normalisée
│   │   ├── load_mediqal.py        # Ingestion MediQAl (FAIT — validé)
│   │   ├── load_frenchmedmcqa.py  # Ingestion FrenchMedMCQA (FAIT — validé)
│   │   └── load_medquad.py        # Ingestion MedQuAD (FAIT — validé)
│   ├── referentiel/
│   │   ├── french_referentiel.json  # Référentiel FRENCH structuré (sous-ensemble)
│   │   └── red_flag_detector.py     # Détection déterministe + logique d'override
│   └── generation/         # Prompts et scripts des Étapes 1, 3, 4, 5 du pipeline (à venir)
├── tests/
│   └── test_red_flag_detector.py  # Suite de tests unitaires (Étape 2 — 20 tests, FAIT)
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

## Tests

```bash
pytest tests/ -v
```

La suite actuelle couvre le module `red_flag_detector.py` (Étape 2) : seuils
de constantes vitales, règles pédiatriques, modulateurs de motifs,
combinaison de plusieurs red flags, et les 3 scénarios du mécanisme
d'override (LLM sous-estime / LLM déjà prudent / aucun red flag détecté).

## Prochaines étapes

1. Vérifier précisément le schéma du subset `oeq` de MediQAl (non encore
   confirmé).
2. Construire le jeu de cas de référence (vignette → champs attendus) pour
   servir de critère d'acceptation à l'Étape 1 avant de l'implémenter.
3. Écrire les prompts des Étapes 1 (extraction), 3 (labellisation), 4
   (contre-validation) et 5 (génération du dialogue) dans `src/generation/`.
4. Brancher `red_flag_detector.py` dans le pipeline complet (croisement
   Étape 2 / Étape 3).
5. Étendre la couverture de tests aux futurs modules de `src/generation/`.# p14-agent-de-triage
