"""
test_prompt_agent.py

Le prompt système de l'agent doit être identique partout où le modèle le voit :
datasets SFT / DPO (entraînement) et démo (inférence). Les scripts l'importent de
src/entrainement/prompt_agent.py ; ce test vérifie en plus que les datasets déjà
générés ont bien été produits avec ce texte (sinon : régénérer puis réentraîner).

Exception voulue : les paires DPO UltraMedical (questions médicales générales) utilisent
PROMPT_SYSTEME_INFO_MEDICALE, lui aussi défini dans prompt_agent.py.

Lancer avec : pytest tests/test_prompt_agent.py -v
"""

import json
import sys
from pathlib import Path

import pytest

RACINE = Path(__file__).parent.parent
sys.path.insert(0, str(RACINE / "src" / "entrainement"))

from prompt_agent import PROMPT_SYSTEME_AGENT, PROMPT_SYSTEME_INFO_MEDICALE  # noqa: E402

PROMPTS_SFT = {PROMPT_SYSTEME_AGENT}
PROMPTS_DPO = {PROMPT_SYSTEME_AGENT, PROMPT_SYSTEME_INFO_MEDICALE}

DATASETS = [
    ("data/sft/train.jsonl", "messages", PROMPTS_SFT),
    ("data/sft/validation.jsonl", "messages", PROMPTS_SFT),
    ("data/dpo/train_dpo.jsonl", "prompt", PROMPTS_DPO),
    ("data/dpo/validation_dpo.jsonl", "prompt", PROMPTS_DPO),
]


def test_les_deux_prompts_sont_distincts():
    assert PROMPT_SYSTEME_AGENT != PROMPT_SYSTEME_INFO_MEDICALE


@pytest.mark.parametrize("chemin, champ, prompts_admis", DATASETS)
def test_dataset_utilise_le_prompt_centralise(chemin, champ, prompts_admis):
    fichier = RACINE / chemin
    if not fichier.exists():
        pytest.skip(f"{chemin} absent (régénérable)")
    with open(fichier, encoding="utf-8") as f:
        for n, ligne in enumerate(f, 1):
            if not ligne.strip():
                continue
            messages = json.loads(ligne)[champ]
            assert messages[0]["role"] == "system", f"{chemin}:{n} : pas de message système en tête"
            assert messages[0]["content"] in prompts_admis, f"{chemin}:{n} : prompt système inconnu"
