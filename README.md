# POC Agent de Triage Médical — CHSA

Preuve de concept d'un agent IA de triage médical pour le Centre Hospitalier
Saint-Aurélien (CHSA, projet fictif — OpenClassrooms, parcours AI Engineer).
L'agent mène un entretien téléphonique en français et attribue un niveau de
priorité : **urgence maximale**, **modérée** ou **différée**.

**Architecture à trois composants :**
- **l'hôtesse** (Qwen3-1.7B + LoRA, fine-tuning SFT) mène l'entretien ;
- **le plancher** (règles déterministes) arrête l'entretien en urgence maximale
  dès qu'un signe d'alerte est décrit ;
- **le décideur** (ModernCamemBERT-bio, classifieur) choisit le niveau de
  priorité. La décision est retirée à l'hôtesse, qui décidait mal.

Un **garde-fou de périmètre** (règles déterministes, après le plancher) répond
par une phrase fixe aux messages hors du champ médical (météo, sport...).

Chaque décision est journalisée (source, probabilités, modèles utilisés).

> ⚠️ **Prototype de recherche, entraîné sur des données synthétiques.
> Ne pas utiliser pour une décision médicale réelle.**

## Livrables

| Livrable | Lien |
|---|---|
| Démo en ligne (Gradio, API) | https://huggingface.co/spaces/vler0ux/triage-chsa-demo |
| Modèle conversationnel (adaptateur LoRA, SFT) | https://huggingface.co/vler0ux/qwen3-1.7b-triage-chsa-hotesse |
| Décideur de priorité | https://huggingface.co/vler0ux/moderncamembert-bio-triage-chsa-decideur |
| Dataset SFT (JSONL, versionné) | [`data/sft/`](data/sft/) : `train.jsonl` (907 dialogues), `validation.jsonl` (107) |
| Pipeline CI/CD | [`.github/workflows/`](.github/workflows/) — voir « CI/CD » |
| Rapport technique | [`docs/rapport_Agent_de_triage_LEROUX.docx`](docs/rapport_Agent_de_triage_LEROUX.docx) |
| Cas de test du triage | [`docs/cas_de_test_triage_CHSA.docx`](docs/cas_de_test_triage_CHSA.docx) |

**Écarts assumés par rapport au cahier des charges** : dataset en français
seulement (non bilingue) ; démo servie par Gradio sur un GPU T4 au lieu de vLLM
(vLLM prévu dans l'architecture cible, il sait charger l'adaptateur LoRA) ;
DPO réalisé mais non retenu (aucun gain mesuré).

## Structure du projet

```text
triage-poc-chsa/
├── data/                      # Sorties de données ignorées par git (régénérables), SAUF data/sft/*.jsonl
│   ├── raw/                   # Cache local / échantillons de test des corpus sources
│   ├── normalized/            # Sorties JSONL : vignettes normalisées, pipeline (Étapes 1-4), dialogues (Étape 5)
│   ├── anonymized/            # Sorties JSONL anonymisées (RGPD) + journaux d'audit
│   ├── relecture/             # Exports CSV de la file de relecture humaine (exporter_relecture.py)
│   ├── sft/                   # Dataset SFT final (train.jsonl / validation.jsonl) — VERSIONNÉ (entrée du notebook Kaggle)
│   ├── dpo/                   # Paires de préférence (prompt / chosen / rejected) pour le DPO
│   └── decideur/              # Dataset du décideur (complet/ et patient/), régénérable depuis data/sft/
├── .github/workflows/         # CI/CD GitHub Actions : tests puis déploiement du Space
├── src/
│   ├── ingestion/
│   │   ├── schema.py                    # Format commun de vignette normalisée
│   │   ├── load_mediqal.py              # Ingestion MediQAl 
│   │   ├── load_frenchmedmcqa.py        # Ingestion FrenchMedMCQA 
│   │   ├── load_medquad.py              # Ingestion MedQuAD 
│   │   └── echantillonner_vignettes.py  # Échantillonnage aléatoire reproductible 
│   ├── referentiel/
│   │   ├── french_referentiel.json                # Référentiel FRENCH structuré (motifs + modulateurs)
│   │   ├── french_grille_v1_1_transcription.json  # Transcription source de la grille FRENCH V1.1 (tableau du PDF officiel)
│   │   ├── construire_referentiel_depuis_grille.py # Reconstruit les modulateurs du référentiel depuis la grille transcrite
│   │   └── red_flag_detector.py                    # Détection déterministe + logique d'override
│   ├── generation/
│   │   ├── llm_client.py                  # Abstraction d'appel LLM (Groq / Gemini / Anthropic, interface commune)
│   │   ├── filtre_pertinence.py           # Étape 0 : filtre de pertinence clinique (API Groq)
│   │   ├── extraction_etape1.py           # Étape 1 : extraction structurée du cas clinique (LLM)
│   │   ├── dedoublonner_vignettes.py      # Regroupe les vignettes qui décrivent le même cas clinique
│   │   ├── labellisation_etape3.py        # Étape 3 : proposition du niveau de priorité (LLM)
│   │   ├── contre_validation_etape4.py    # Étape 4 : audit du label retenu (signalement seul, LLM "auditeur")
│   │   ├── generation_dialogue_etape5.py  # Étape 5 : dialogues patient/agent (variantes, juge de fidélité)
│   │   ├── orchestrer_pipeline.py         # Enchaîne les Étapes 1 → 2 → 3 → 4 + file de relecture
│   │   ├── exporter_relecture.py          # Exporte les cas à relire en CSV, triés par priorité
│   │   ├── bilan_run.py                   # Résumé chiffré d'un run du pipeline (lecture seule)
│   │   ├── diagnostic_final.py            # Sur-triage par rapport à la grille FRENCH (lecture seule)
│   │   ├── nettoyer_labels_perimes.py     # Retire les dialogues dont le label ne correspond plus au pipeline
│   │   ├── nettoyer_conclusions_ambigues.py # Retire les dialogues dont la conclusion mélange plusieurs niveaux
│   │   ├── recalibrer_conclusions.py      # Réapplique les conclusions imposées par le code (sans appel API)
│   │   └── preparer_dpo_ultramedical.py   # Traduit un échantillon d'UltraMedical-Preference (paires DPO, à relire)
│   ├── anonymisation/
│   │   └── anonymiser_vignettes.py  # Anonymisation RGPD (Presidio + spaCy fr_core_news_md)
│   ├── entrainement/
│   │   ├── prompt_agent.py          # Prompt système de l'agent : SOURCE UNIQUE (SFT, DPO, démo)
│   │   ├── preparer_dataset_sft.py  # Dialogues (Étape 5) -> dataset conversationnel (format TRL / chat template Qwen3)
│   │   ├── template_triage.jinja    # Chat template Qwen3 avec marqueurs {% generation %} (masquage de l'entraînement)
│   │   ├── diag_masquage.py         # Diagnostic sans GPU : vérifie le masquage produit par le chat template
│   │   └── train_sft.py             # Entraînement SFT (LoRA) de Qwen3-1.7B-Base
│   ├── dpo/
│   │   ├── preparer_dpo_rejected.py # Génère des réponses "rejected" avec le modèle SFT (température élevée)
│   │   ├── defauts.py               # Détecteurs de défauts des répliques de l'hôtesse (règles simples, DPO v6)
│   │   ├── filtrer_rejected.py      # DPO v6 : paires formées par les détecteurs de défauts
│   │   ├── juger_paires_dpo.py      # DPO v6 : paires formées par un juge LLM (notes sur 5 critères, à l'aveugle)
│   │   ├── relire_paires_dpo.py     # Relecture humaine à l'aveugle des paires, en terminal
│   │   ├── remettre_en_relecture.py # Remet des paires déjà décidées dans la file de relecture
│   │   ├── rattraper_desaccords.py  # Rattrapage ponctuel : paires en désaccord inversées (jugement humain)
│   │   ├── fusionner_dpo.py         # Fusionne les sources de paires -> train_dpo.jsonl / validation_dpo.jsonl
│   │   └── train_dpo.py             # DPO : fusion de l'adaptateur SFT + nouveau LoRA élargi
│   ├── decideur/
│   │   ├── preparer_dataset_decideur.py # Dialogues SFT -> transcriptions + label (variantes complet / patient)
│   │   ├── train_decideur.py            # Entraînement + évaluation du classifieur (ModernCamemBERT-bio), IC 95 %
│   │   └── evaluer_raccourci_style.py   # Test par paires minimales du biais « ton du patient » (jeux dev / contrôle)
│   ├── analyse/
│   │   └── part_tokens_decision.py  # Part des tokens de la balise de priorité dans les dialogues SFT (lecture seule)
│   └── demo/
│       ├── app_demo.py              # Interface Gradio : l'hôtesse mène l'entretien, le décideur tranche
│       ├── decideur.py              # Plancher de sécurité + garde-fou hors sujet + décideur encodeur + journal d'audit
│       └── decideur-patient/        # Modèle du décideur téléchargé depuis Kaggle (poids non versionnés)
├── tests/
│   ├── test_red_flag_detector.py     # Étape 2 : seuils, pédiatrie, modulateurs, override
│   ├── test_lot1.py                  # Lot 1 sans appel API (faux LLM) : croisement, planchers, file de relecture
│   ├── test_etape5.py                # Étape 5 sans appel API (faux LLM) : contrôles, juge, reprise, ouverture/conclusion
│   ├── test_prompt_agent.py          # Le prompt système est identique dans les datasets et à l'inférence
│   ├── test_plancher.py              # Plancher de la démo : signes d'alerte, négations, formes atténuées, cas A/B/C
│   ├── test_hors_sujet.py            # Garde-fou de périmètre : hors sujet bloqués, mots pièges, recadrage/orientation
│   ├── test_decideur_metriques.py    # Métriques du décideur : intervalles de Wilson, effectifs
│   ├── test_defauts_dpo.py           # Détecteurs de défauts du DPO v6 (src/dpo/defauts.py)
│   ├── test_juge_dpo.py              # Logique du juge LLM du DPO v6, sans appel API
│   └── golden_etape1_extraction.json # Jeu de cas de référence (Étapes 1, 3, 4)
├── entrainement-sft-kaggle.ipynb  # Notebook d'entraînement SFT sur Kaggle (GPU T4)
├── notebook-decideur.ipynb        # Notebook d'entraînement du décideur sur Kaggle (variantes complet + patient)
├── entrainement_dpo_kaggle.ipynb  # Notebook d'entraînement DPO sur Kaggle
├── entrainement_dpo_v6_kaggle.ipynb  # Notebook d'entraînement DPO sur Kaggle (version 6)
├── deploiement/
│   ├── assembler_space.sh  # Assemble le code du Space Hugging Face à partir des sources testées
│   └── space/              # app.py et requirements.txt propres au Space
├── run_pipeline.sh         # Script d'orchestration (ingestion + tests, option --avec-filtre)
├── sauvegarder_donnees.sh  # Sauvegarde horodatée de data/normalized/ et data/anonymized/
├── pytest.ini
├── requirements.txt
├── requirements-ci.txt     # Dépendances minimales des tests (CI)
├── .env.example
├── docs/                      # Rapport technique, cas de test, article Taboulet 2019 (échelle FRENCH, PDF)
└── .gitignore
```

Ne sont **pas** versionnés (voir `.gitignore`) : les données générées (JSONL hors
`data/sft/`), les exports HTML de relecture, les modèles entraînés
(`qwen3-1.7b-triage-*`, checkpoints, `*.safetensors`, archives `.zip`), le
dossier `kaggle_upload/` (copie des fichiers envoyés sur Kaggle) et les
sauvegardes locales.

## Sources et licences

| Ressource | Usage dans le projet | Licence |
|---|---|---|
| [MediQAl](https://huggingface.co/datasets/ANR-MALADES/MediQAl) | Vignettes cliniques du corpus final (726 vignettes labellisées, ~486 cas distincts) | CC-BY-4.0 |
| FrenchMedMCQA | Ingéré, non retenu (questions sans contexte clinique) | [à vérifier] |
| MedQuAD | Ingéré, non retenu (corpus anglophone) | [à vérifier] |
| [UltraMedical-Preference](https://huggingface.co/datasets/TsinghuaC3I/UltraMedical-Preference) | Paires DPO (premier essai, traduites en français) | MIT |
| Grille FRENCH V1.1 (SFMU, 2018) | Référentiel de triage (planchers, modulateurs) | Référentiel publié par la SFMU, cité comme source |
| [Qwen3-1.7B-Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base) | Modèle de base de l'hôtesse | Apache 2.0 |
| [ModernCamemBERT-bio-v2-base](https://huggingface.co/almanach/ModernCamemBERT-bio-v2-base) | Modèle de base du décideur | MIT |

**RGPD** : aucune donnée de patient réel n'est utilisée. Les vignettes sont des
cas d'examen ; elles ont tout de même été anonymisées (Presidio + spaCy) et
chaque transformation est tracée dans un journal d'audit. Avec des données
réelles, l'hébergement devrait être certifié HDS.

**Corpus final** : MediQAl uniquement, en français. Le dataset n'est donc pas
bilingue, contrairement au cahier des charges (écart documenté dans le rapport).

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # puis renseigner vos clés/tokens si nécessaire
```

Les modèles spaCy de l'anonymisation s'installent ensuite à part :

```bash
python -m spacy download fr_core_news_md
python -m spacy download en_core_web_md   # uniquement pour les vignettes MedQuAD (anglais)
```

`requirements.txt` contient aussi les dépendances d'entraînement (`torch`,
`transformers`, `trl`...), lourdes et inutiles sans GPU : pour le seul
pipeline de données, on peut commenter la section « Phase 2 ».

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
unique).

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
isolées). Les données sont
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

Les Étapes 1, 3, 4 et 5 utilisent `llm_client.py`, une couche d'abstraction
commune à Groq, Google Gemini et Anthropic (`--fournisseur
groq|gemini|anthropic`), afin de pouvoir varier le modèle d'une étape à
l'autre et réduire la corrélation des erreurs entre les jugements. Selon le
fournisseur choisi, prévoir `GROQ_API_KEY`, `GEMINI_API_KEY` et/ou
`ANTHROPIC_API_KEY` dans `.env` (clé Gemini gratuite sur Google AI Studio —
attention, les quotas du tier gratuit varient beaucoup selon le modèle ;
l'API Anthropic est payante, facturée à l'usage).

### Étape 1 — Extraction structurée (`extraction_etape1.py`)

Extrait le cas clinique en JSON structuré (motif, constantes vitales,
critères) à partir du texte libre d'une vignette, en s'appuyant
dynamiquement sur le vocabulaire du référentiel FRENCH. Se teste contre le
jeu de cas de référence `tests/golden_etape1_extraction.json`. Utilise
Anthropic (Claude Haiku) par défaut.

```bash
cd src/generation
python3 extraction_etape1.py --golden ../../tests/golden_etape1_extraction.json
```

### Dédoublonnage des vignettes (`dedoublonner_vignettes.py`)

MediQAl est un jeu de *questions* d'examen : un même dossier clinique donne
souvent plusieurs questions, donc plusieurs vignettes qui répètent le même
texte de cas. Ce script regroupe les vignettes par cas clinique (une seule
conservée par cas), ajoute un `cas_id` partagé (utile pour un futur découpage
train/validation/test par cas plutôt que par question) et marque les cas où
le filtre de pertinence (Étape 0) s'est contredit d'une question à l'autre.
Ne modifie jamais le fichier d'entrée ; à lancer après l'anonymisation.

```bash
cd src/generation
python3 dedoublonner_vignettes.py --dry-run   # simulation, rapport seul
python3 dedoublonner_vignettes.py             # écrit *_dedup.jsonl
```

### Étape 3 — Labellisation (`labellisation_etape3.py`)

À partir du cas structuré (sortie de l'Étape 1), le LLM propose un niveau de
priorité parmi `urgence_maximale` / `moderee` / `differee`, avec
justification. Ce n'est pas le label final : il est ensuite croisé avec la
détection déterministe de l'Étape 2 (`red_flag_detector.py`), puis audité à
l'Étape 4. Utilise Anthropic (Claude Haiku) par défaut, retenu après
comparaison avec Gemini sur un premier lot de vignettes.

```bash
cd src/generation
python3 labellisation_etape3.py --golden ../../tests/golden_etape1_extraction.json
```

### Étape 4 — Audit du label (`contre_validation_etape4.py`)

Un second appel LLM, en posture d'auditeur avec un fournisseur différent
(Groq par défaut, pour réduire la corrélation des erreurs entre les deux
jugements), signale si le label retenu après le croisement Étape 2/3
SOUS-estime la gravité. Ce signalement n'écrase plus automatiquement le label : il
alimente la file de relecture humaine (voir `orchestrer_pipeline.py`
ci-dessous).

```bash
cd src/generation
python3 contre_validation_etape4.py --golden ../../tests/golden_etape1_extraction.json
```

### Orchestration complète des Étapes 1 → 4 (`orchestrer_pipeline.py`)

Chaîne les 4 étapes sur un fichier de vignettes filtrées, anonymisées et
dédoublonnées, avec croisement Étape 2/3 (plancher des constantes vitales
inviolable, plancher du motif révisable) et calcul d'un `score_relecture`
qui priorise les cas dont le label est SOUS un plancher de la grille FRENCH
(le vrai risque : le sous-triage). Conserve la trace complète de chaque
étape — y compris fournisseur, modèle et empreinte du prompt utilisés — dans
la sortie, pour l'auditabilité exigée par le brief. Reprend automatiquement
en cas d'interruption (comme `filtre_pertinence.py`).

```bash
cd src/generation
python3 orchestrer_pipeline.py \
    --input ../../data/anonymized/mediqal_mcqm_urgences_complet_dedup.jsonl \
    --output ../../data/normalized/mediqal_mcqm_pipeline_complet.jsonl \
    --limit 10
```

Les fournisseurs/modèles par défaut de chaque étape (Étapes 1 et 3 =
Anthropic/Haiku, Étape 4 = Groq) peuvent être surchargés en ligne de
commande (`--etape1-fournisseur`, `--etape1-modele`, etc.), via le `.env`
(`ETAPE1_FOURNISSEUR`, `ETAPE1_MODELE`, ...), ou tous basculés sur Haiku
avec `--tout-haiku` si un quota bloque. Priorité : ligne de commande > `.env`
> valeurs par défaut du script.

### Export de la file de relecture (`exporter_relecture.py`)

Exporte en CSV (lisible Excel/LibreOffice) les cas à relire par un humain,
triés par `score_relecture` et limités à un budget de relectures (défaut :
20). Les cas au-delà du budget sont listés à part (`dans_le_budget = non`,
fichier annexe `*_non_relus.txt`) et ne doivent pas être utilisés pour
l'entraînement ou le test tant qu'ils n'ont pas été relus.

Dans ce POC, la relecture a été partielle (20 cas relus sur 47 proposés) : voir
les limites dans le rapport.

```bash
cd src/generation
python3 exporter_relecture.py \
    --input ../../data/normalized/mediqal_mcqm_pipeline_complet.jsonl \
    --output ../../data/relecture/relecture.csv --n 20
```

### Étape 5 — Génération du dialogue (`generation_dialogue_etape5.py`)

Dernière étape : à partir d'un cas structuré et du niveau de priorité déjà
validé par les Étapes 1 à 4 (que le LLM ne doit jamais remettre en cause),
génère un dialogue multi-tours réaliste entre un patient (expression
naturelle et imparfaite — hésitations, ordre désordonné) et un agent de
triage. Prend en entrée la sortie de `orchestrer_pipeline.py` ; les cas
marqués `a_relire` sont exclus. Anthropic (Claude Haiku) par défaut,
température 0,8.

- **Variantes** : plusieurs dialogues par cas selon le label (défaut : 1
  urgence maximale, 2 modérée, 3 différée), chacun avec un style de patient
  différent, pour rééquilibrer les classes sans toucher aux labels. Les
  variantes d'un même cas partagent `cas_id`.
- **Phrases imposées par le code** : la première phrase de l'agent (selon
  `--contexte appel|accueil`) et la conclusion sont tirées de listes fixes, de
  façon reproductible (id du cas + variante). La balise
  `[[PRIORITE: <categorie>]]` est calculée depuis le label validé, jamais par
  le LLM.
- **Contrôles automatiques** avec nouvelle tentative (`--essais`, défaut 3) :
  alternance des tours, 4 à 8 échanges, aucun nom propre ni marqueur
  d'anonymisation, aucune valeur chiffrée de constante vitale, aucun jargon.
- **Juge de fidélité** : un second appel LLM repère les éléments cliniques
  inventés par le patient ; un élément d'importance « médicale » fait refuser
  le dialogue (`--sans-juge` pour le désactiver, déconseillé).
- Reprise automatique : les couples (id, variante) réussis sont sautés, les
  échecs (`_erreur`) sont retentés.

```bash
cd src/generation
python3 generation_dialogue_etape5.py \
    --input ../../data/normalized/pipeline_473.jsonl \
    --output ../../data/normalized/dialogues_473.jsonl
# essai : --limit 3 ; moins de variantes : --variantes urgence_maximale=1,moderee=1,differee=2
```

### Maintenance des dialogues déjà générés

Scripts sans appel API, à lancer avant de relancer l'Étape 5 (qui régénère
alors les entrées retirées grâce à la reprise) :

```bash
cd src/generation
# dialogues dont le label ne correspond plus au pipeline actuel (mis de côté dans *_perimes.jsonl)
python3 nettoyer_labels_perimes.py \
    --dialogues ../../data/normalized/dialogues_473.jsonl \
    --pipeline ../../data/normalized/pipeline_473.jsonl
# conclusions qui mélangent le vocabulaire de plusieurs niveaux
python3 nettoyer_conclusions_ambigues.py --dialogues ../../data/normalized/dialogues_473.jsonl
# réapplique les conclusions imposées par le code et retire les cas a_relire
python3 recalibrer_conclusions.py --dialogues ../../data/normalized/dialogues_473.jsonl
```

### Diagnostics d'un run (lecture seule)

```bash
cd src/generation
python3 bilan_run.py          # échecs, répartition des labels, file de relecture, extraction, audit
python3 diagnostic_final.py   # position des labels par rapport aux planchers de la grille (sur-triage)
# autre fichier : FICHIER=../../data/normalized/autre_pipeline.jsonl python3 bilan_run.py
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

Prérequis : Presidio et spaCy sont dans `requirements.txt` ; le modèle
français s'installe avec `python -m spacy download fr_core_news_md`.

## Référentiel FRENCH

### Reconstruction des modulateurs depuis la grille (`construire_referentiel_depuis_grille.py`)

Reconstruit les modulateurs (critères et niveaux de tri) de
`french_referentiel.json` à partir de `french_grille_v1_1_transcription.json`,
une transcription vérifiable du tableau de la grille FRENCH V1.1 extraite
automatiquement du PDF officiel. Le tri de base de chaque motif devient le
« Tri M » (tri médian) de la grille, et chaque cellule non vide des colonnes
Tri 1 à Tri 5 devient un modulateur.

```bash
cd src/referentiel
python3 construire_referentiel_depuis_grille.py \
    --entree french_referentiel.json \
    --grille french_grille_v1_1_transcription.json \
    --sortie french_referentiel.json
```

## Entraînement SFT

### Préparation du dataset (`preparer_dataset_sft.py`)

Transforme les dialogues générés (sortie de `generation_dialogue_etape5.py`)
en dataset conversationnel `{"messages": [...]}` compatible avec le chat
template de Qwen3 et le format attendu par TRL (`SFTTrainer`). Le prompt
système vient de `prompt_agent.py` (source unique, aussi utilisée par le DPO
et la démo). Le message final de l'agent se termine par la balise
`[[PRIORITE: <categorie>]]` sur sa propre ligne. Le découpage
train/validation se fait **par cas** (`cas_id`) : toutes les variantes d'un
même cas vont du même côté, sans fuite (graine fixée).

```bash
cd src/entrainement
python3 preparer_dataset_sft.py \
    --input ../../data/normalized/dialogues_473.jsonl \
    --output-dir ../../data/sft --val-ratio 0.1
```

Modifier `prompt_agent.py` impose de régénérer les datasets et de réentraîner
(`test_prompt_agent.py` le vérifie).

### Diagnostic du masquage (`diag_masquage.py`)

Vérifie sans GPU, sur le vrai tokenizer de Qwen3-1.7B-Base, que le chat
template `template_triage.jinja` (marqueurs `{% generation %}`) masque bien
uniquement les tours de l'agent lors de l'entraînement — utile pour détecter
une erreur de template avant de lancer un entraînement GPU coûteux.

```bash
cd src/entrainement
python3 diag_masquage.py ../../data/sft/train.jsonl
```

### Entraînement (`train_sft.py`)

Entraînement SFT (LoRA) de `Qwen/Qwen3-1.7B-Base` à partir du dataset produit
par `preparer_dataset_sft.py`. Nécessite un GPU (voir la section Phase 2 de
`requirements.txt` : `torch`, `transformers`, `accelerate`, `peft`, `trl`).

```bash
cd src/entrainement
python3 train_sft.py \
    --train-file ../../data/sft/train.jsonl \
    --val-file ../../data/sft/validation.jsonl \
    --output-dir ../../qwen3-1.7b-triage-sft \
    --max-steps 5   # smoke test d'abord, puis sans --max-steps
```

La précision (`--precision auto`) choisit bf16 sur GPU Ampere ou plus récent,
fp16 sinon (ex. T4). Le script affiche au démarrage le nombre de tokens
appris (`[masquage] tokens appris : N/M`) : seuls les tours de l'agent doivent
l'être.

### Entraînement sur Kaggle (`entrainement-sft-kaggle.ipynb`)

Le notebook lance `train_sft.py` sur un GPU T4 Kaggle. Il attend un Dataset
Kaggle contenant à sa racine `train_sft.py`, `template_triage.jinja`,
`train.jsonl` et `validation.jsonl` (copies préparées dans `kaggle_upload/`,
non versionné). L'adaptateur entraîné est à télécharger puis dézipper en local
(ex. `qwen3-1.7b-triage-sft/`).

## Alignement DPO (expérimental)

Le DPO (Direct Preference Optimization) aligne le modèle SFT à partir de paires
« chosen » (réponse attendue) / « rejected » (réponse à éviter).

**Résultat : non retenu.** Deux essais ont été menés : paires UltraMedical
traduites et relues (aucun effet observable), puis 1 844 paires issues des
dialogues, notées par un LLM (bonnes métriques, mais conversation dégradée :
le modèle ne conclut plus). L'hôtesse en service est le SFT seul.

### Premier essai : paires UltraMedical + rejected du SFT

```bash
# 1. Réponses "rejected" générées par le modèle SFT sur des contextes de train.jsonl
cd src/dpo
python3 preparer_dpo_rejected.py --adapter-dir ../../qwen3-1.7b-triage-sft \
    --train-file ../../data/sft/train.jsonl --n 50 \
    --output ../../data/dpo/rejected_sft_a_relire.jsonl

# 2. (optionnel) paires UltraMedical-Preference traduites en français, à relire
cd ../generation
python3 preparer_dpo_ultramedical.py --inspecter-seulement
python3 preparer_dpo_ultramedical.py --n 180 --output ../../data/dpo/ultramedical_fr_a_relire.jsonl

# 3. Fusion + découpage train/validation par cas
cd ../dpo
python3 fusionner_dpo.py \
    --rejected-sft ../../data/dpo/rejected_sft_a_relire.jsonl \
    --ultramedical ../../data/dpo/ultramedical_fr_a_relire.jsonl \
    --output-dir ../../data/dpo

# 4. Entraînement (smoke test d'abord)
python3 train_dpo.py --adapter-sft-dir ../../qwen3-1.7b-triage-sft \
    --train-file ../../data/dpo/train_dpo.jsonl \
    --val-file ../../data/dpo/validation_dpo.jsonl \
    --output-dir ../../qwen3-1.7b-triage-dpo --max-steps 5
```

Seules les paires UltraMedical marquées `a_valider=true` après relecture
humaine sont retenues par `fusionner_dpo.py`. `train_dpo.py` fusionne
l'adaptateur SFT dans le modèle de base avant d'ajouter un nouveau LoRA :
l'adaptateur DPO enregistré ne s'utilise donc **pas** seul sur le modèle de
base (utiliser `--merge-adapter` pour obtenir un modèle complet).

La relecture humaine des paires se fait à l'aveugle (`relire_paires_dpo.py`) :
les deux réponses sont affichées dans un ordre aléatoire, et la paire n'est
validée que si le choix du relecteur coïncide avec le « chosen ».
`remettre_en_relecture.py` remet des paires dans la file ; `rattraper_desaccords.py`
est un correctif ponctuel (à lancer une seule fois).

### DPO v6 : paires issues des dialogues

Pour chaque contexte de `train.jsonl`, le SFT génère plusieurs réponses
(`preparer_dpo_rejected.py`), puis les paires sont formées de l'une des deux
façons suivantes :

- **règles** (`filtrer_rejected.py`, sans GPU ni API) : le « rejected » est une
  réponse générée qui présente un défaut détecté par `defauts.py` (vide,
  artefact, langue, rassurance ou hypothèse diagnostique, plusieurs questions,
  conclusion prématurée ou manquée...) ; le contexte est écarté si la réplique
  de référence a elle-même un défaut ;
- **juge LLM** (`juger_paires_dpo.py`, Gemini par défaut, ni Claude ni Qwen pour
  limiter l'auto-préférence) : le juge reçoit l'appel complet et les réponses
  candidates mélangées et anonymes (la référence peut donc être rejetée), les
  note sur 5 critères et dit si elles sont sûres. Paire gardée seulement si
  l'écart de notes est net (`--ecart-min`, 3 points sur 25). Reprise
  automatique en cas d'interruption.

```bash
# paires par le juge LLM (essai sur 20 contextes, puis lot complet)
python3 src/dpo/juger_paires_dpo.py --entree data/dpo/rejected_v6.jsonl \
    --jugements data/dpo/jugements_v6_essai.jsonl --n 20
python3 src/dpo/juger_paires_dpo.py --entree data/dpo/rejected_v6.jsonl \
    --jugements data/dpo/jugements_v6.jsonl \
    --paires data/dpo/paires_v6.jsonl --relecture data/dpo/paires_v6_a_relire.jsonl

# ou paires par les règles
python3 src/dpo/filtrer_rejected.py --entree data/dpo/rejected_v6.jsonl \
    --sortie data/dpo/paires_v6.jsonl --relecture data/dpo/paires_v6_a_relire.jsonl

# vérification à l'aveugle d'un échantillon
python3 src/dpo/relire_paires_dpo.py data/dpo/paires_v6_a_relire.jsonl
```

Les paires obtenues passent ensuite par `fusionner_dpo.py` (`--rejected-sft`)
et `train_dpo.py` ; l'entraînement a été fait sur Kaggle
(`entrainement_dpo_v6_kaggle.ipynb`).

## Décideur de priorité

### Pourquoi un décideur séparé

Testée en démo, l'hôtesse fine-tunée (SFT) mène correctement l'entretien mais choisit mal la priorité : sur 138 tokens appris par dialogue, la décision ne se joue que sur 1 ou 2 (mesure : `python src/analyse/part_tokens_decision.py --fichier data/sft/train.jsonl`, CPU, lecture seule). La décision est donc confiée à un composant dédié :

- l'hôtesse (Qwen3-1.7B + LoRA) mène l'entretien, sa proposition de priorité est ignorée ;
- le plancher de sécurité (`src/demo/decideur.py`, règles par mots-clés, sans apprentissage) force l'urgence maximale dès qu'un signe d'alerte est décrit par le patient. Il est évalué à chaque message : un signe d'alerte arrête l'entretien ;
- le décideur (encodeur `almanach/ModernCamemBERT-bio-v2-base`, licence MIT, environ 136 M de paramètres, exécutable sur CPU) classe la conversation. Règle de prudence fixée avant la mesure : urgence maximale si P(urgence) ≥ 0,30, sinon la classe la plus probable ;
- la conclusion est choisie par le code et chaque décision est écrite dans le journal d'audit (source, signes détectés, probabilités).

### 1. Préparer le dataset (local, sans GPU)

```bash
python src/decideur/preparer_dataset_decideur.py
```

Lit `data/sft/train.jsonl` et `validation.jsonl`, retire la conclusion de l'agent (qui contient la réponse), et écrit `data/decideur/complet/` (agent + patient) et `data/decideur/patient/` (messages du patient seuls). Contrôles : balise présente et cohérente, aucune fuite du label, aucun cas commun entre train et validation. Mesure aussi la longueur des transcriptions (maximum 701 tokens, d'où `--max-length 768` à l'entraînement).

### 2. Entraîner sur Kaggle (`notebook-decideur.ipynb`)

Créer un Dataset Kaggle contenant `complet/`, `patient/` et `train_decideur.py` (le plus sûr : déposer une archive zip pour conserver les sous-dossiers) :

```bash
cp src/decideur/train_decideur.py data/decideur/
cd data/decideur && zip -r ../decideur.zip complet patient train_decideur.py
```

Dans le notebook : accélérateur GPU, Internet activé. Avec un accélérateur « T4 x2 », forcer un seul GPU (`CUDA_VISIBLE_DEVICES=0`, déjà dans le notebook) : ModernBERT plante en mode multi-GPU (`StopIteration`).

Règles posées avant la mesure : modèle de la dernière époque (pas de sélection du « meilleur » checkpoint), pas de pondération des classes. Chaque run écrit `resultats_validation.json` (métriques avec intervalles de confiance de Wilson à 95 %) et `predictions_validation.jsonl`. Récupérer l'archive `decideur_resultats.zip` (cellule 7) et dézipper `decideur-patient/` dans `src/demo/`.

### 3. Résultats sur la validation (107 dialogues, ~72 cas)

Variante retenue : patient seul, avec le seuil de 0,30.

| Métrique | Résultat | IC 95 % |
|---|---|---|
| Rappel urgence maximale | 45/45 | 92 % – 100 % |
| Rappel modérée | 23/33 | 53 % – 83 % |
| Rappel différée | 24/29 | 65 % – 92 % |
| Sous-triage | 5/107 | 2 % – 10 % |
| Exactitude | 86 % | — |

Référence « toujours urgence » : 42 % d'exactitude. Limites : petit échantillon, pas de jeu de test indépendant, et choix de la variante fait sur cette même validation (chiffres légèrement optimistes).

**En conditions réelles** (11 essais en démo) : 5/11 corrects, urgences 2/2,
différées 3/4, **modérées 0/5** — le décideur tranche presque toujours entre
urgence et différée.

### 4. Biais connu : le ton du patient

```bash
python src/decideur/evaluer_raccourci_style.py --jeu dev
python src/decideur/evaluer_raccourci_style.py --jeu controle
```

Test par paires minimales : chaque cas clinique est écrit deux fois, avec les mêmes faits, par un patient stressé puis par un patient qui minimise. Résultat : le décideur seul change de décision avec le ton sur 4 paires sur 10 (ex. déficit neurologique brutal minimisé classé « différée »). Cause : à l'Étape 5, le style du patient dépendait du niveau (`STYLES[variante % 6]` avec 1 / 2 / 3 variantes selon le niveau) — 100 % des urgences sont jouées par un patient stressé, et le style « minimise » n'existe qu'en « différée ».

Sur le jeu de contrôle (jamais utilisé pour régler le plancher), la décision finale (plancher + décideur) reconnaît 11 urgences sur 12, sans déclencher le plancher sur aucun cas bénin. Urgence restante : une détresse respiratoire minimisée (« je respire un peu mal »). Le jeu dev a servi à corriger le plancher : son score (12/12) est optimiste par construction.

Ce test n'est pas un test pytest : il charge le modèle du décideur et sert au diagnostic.

> ⚠️ En l'état, une décision « différée » ne doit jamais être définitive sans validation humaine.

## Démonstration (`app_demo.py`)

En ligne : https://huggingface.co/spaces/vler0ux/triage-chsa-demo

Interface Gradio : l'hôtesse (base Qwen3-1.7B + adaptateur LoRA SFT) mène l'entretien avec le même prompt système et le même chat template qu'à l'entraînement ; la priorité est décidée par le plancher puis le décideur (voir « Décideur de priorité »).

```bash
cd src/demo
python3 app_demo.py --adapter-dir v4-qwen3-1.7b-triage-sft --decideur-dir decideur-patient

# puis ouvrir http://127.0.0.1:7860
```

Sans `--decideur-dir`, la démo revient à l'ancien fonctionnement (décision de l'hôtesse seule) : à réserver à la comparaison.

**Garde-fou hors sujet** (`hors_sujet()` dans `decideur.py`) : un message hors du champ médical (météo, sport, cuisine, actualité, loisirs, informatique) reçoit une phrase fixe choisie par le code, sans appel à l'hôtesse (qui n'a pas accès à Internet et inventerait sa réponse). Double condition pour limiter les faux positifs : un mot hors domaine **et** aucun mot médical (« il neige, j'ai glissé et j'ai mal au dos » passe). Premier message hors sujet : recadrage ; au deuxième d'affilée : orientation (« rappelez-nous ou composez le 15 »). Le garde-fou est évalué **après** le plancher : une urgence passe toujours en premier. En cas de doute, le message va à l'hôtesse. Pour l'instant, il n'est branché que dans `app_demo.py` (pas encore dans `deploiement/space/app.py`).

Ce qui s'affiche dans le terminal : `[triage] plancher déclenché : …` quand un signe d'alerte arrête l'entretien, `[perimetre] message hors sujet -> …` pour un message hors sujet, sinon `[triage] catégorie proposée par l'hôtesse : …` puis `[triage] décision retenue : <niveau> (<source>) | <probabilités>`. Sources possibles : `plancher_immediat`, `plancher`, `decideur_seuil`, `decideur`, `hors_sujet`.

**Journal d'audit :** chaque décision est ajoutée à `src/demo/logs/decisions.jsonl` (conversation, proposition de l'hôtesse, niveau retenu, source, signes détectés, probabilités, modèles utilisés). Ignoré par git : il contient les conversations. Avec des données réelles, ce journal relèverait d'un hébergement certifié HDS.

**Mémoire :** sans GPU, Qwen est chargé en float32 (≈ 7 Go) en plus du décideur (≈ 0,6 Go). Prévoir au moins 8 à 10 Go de RAM libres (`free -h`), sinon le processus est tué par Linux (`Killed`). Fermer les applications lourdes (navigateur, VS Code / Pylance) avant de lancer.

`--share` crée un lien public temporaire : à éviter pour une démo médicale.

> ⚠️ POC de démonstration : ne constitue pas une décision médicale validée.

## CI/CD (GitHub Actions)

Chaque push sur `main` déclenche le workflow [`ci.yml`](.github/workflows/ci.yml) :

1. **Job « Tests »** : installe des dépendances allégées (`requirements-ci.txt`) et lance la suite pytest (225 tests : données, référentiel FRENCH, plancher, garde-fou hors sujet, métriques du décideur, détecteurs et juge du DPO, prompts). Aucun appel API, aucun modèle chargé.
2. **Job « Déploiement »** : ne démarre que si tous les tests passent. Le script [`deploiement/assembler_space.sh`](deploiement/assembler_space.sh) assemble le code du Space à partir des sources testées (aucune copie manuelle), puis le pousse sur Hugging Face avec la référence du commit d'origine.

**Sécurité et robustesse :**
- le jeton Hugging Face est stocké en secret GitHub (`HF_TOKEN`), avec
  un droit d'écriture limité à ce seul Space ;
- deux déploiements ne tournent jamais en même temps (`concurrency`) ;
- une procédure manuelle documentée sert de secours.

**Hébergement** : le Space tourne sur un GPU T4 dédié (l'offre gratuite
ZeroGPU échouait régulièrement à attribuer un GPU). L'hôtesse utilise le GPU ;
le plancher et le décideur tournent sur CPU et continuent de répondre si le GPU
est indisponible. Le Space se met en veille après inactivité : le réveiller
quelques minutes avant une démonstration.

## Tests

```bash
pytest -v        # depuis la racine (pytest.ini : testpaths = tests)
```

Aucun test ne fait d'appel API.

- `test_red_flag_detector.py` couvre le module `red_flag_detector.py`
  (Étape 2) : seuils de constantes vitales adulte et pédiatriques par tranche
  d'âge, règles pédiatriques, modulateurs de motifs (y compris à la baisse),
  combinaison de plusieurs red flags, et les scénarios du mécanisme
  d'override.
- `test_lot1.py` utilise un faux LLM qui renvoie des réponses prévues et
  couvre le code du lot 1 : planchers de la grille, contrôle de présence des
  constantes, lecture des réponses LLM, et file de relecture
  (`orchestrer_pipeline.py`, `construire_referentiel_depuis_grille.py`).
- `test_etape5.py` couvre l'Étape 5 avec un faux LLM : contrôles de
  structure et de contenu, ouverture et conclusion imposées, juge de
  fidélité, nouvelles tentatives, reprise et options de la ligne de commande.
- `test_prompt_agent.py` vérifie que les datasets SFT/DPO ont été produits
  avec le prompt système actuel de `prompt_agent.py`.
- `test_plancher.py` couvre le plancher de la démo (`src/demo/decideur.py`) :
  signes d'alerte francs et atténués, négations limitées au membre de phrase et
  au message, exception traumatique, combinaisons sur plusieurs messages, et les
  cas de référence A, B et C. Aucun modèle n'est chargé.
- `test_hors_sujet.py` couvre le garde-fou de périmètre : vrais hors sujet
  bloqués, messages médicaux contenant un mot piège (« match », « recette »,
  « neige ») qui passent, recadrage puis orientation, priorité du plancher.
- `test_defauts_dpo.py` couvre les détecteurs de défauts du DPO v6
  (`src/dpo/defauts.py`) : questions multiples, conclusion prématurée ou
  manquée, rassurance et diagnostic, langue, artefacts.
- `test_juge_dpo.py` couvre la logique du juge LLM (`juger_paires_dpo.py`) avec
  des réponses prévues : doublons, mélange anonyme et reproductible, validation
  de la réponse du juge, choix de la paire.
- `test_decideur_metriques.py` vérifie les intervalles de confiance de Wilson et
  les effectifs de `train_decideur.py`. Nécessite scikit-learn (le test est
  ignoré s'il n'est pas installé).

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
