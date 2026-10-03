"""
Métriques du décideur (train_decideur.py) : intervalles de confiance de Wilson et effectifs.

Lancer avec : pytest tests/test_decideur_metriques.py -v
"""

import sys
from pathlib import Path

import pytest

RACINE = Path(__file__).parent.parent
sys.path.insert(0, str(RACINE / "src" / "decideur"))

import train_decideur as td  # noqa: E402


def test_wilson_valeurs_de_reference():
    # 45/45 : l'intervalle ne descend pas à 100 % (ce qu'afficherait un intervalle normal)
    bas, haut = td.intervalle_wilson(45, 45)
    assert haut == 1.0 and bas == pytest.approx(0.9213, abs=1e-3)
    bas, haut = td.intervalle_wilson(5, 107)
    assert bas == pytest.approx(0.0201, abs=1e-3) and haut == pytest.approx(0.1046, abs=1e-3)


def test_wilson_effectif_nul():
    assert td.intervalle_wilson(0, 0) is None


def test_metriques_donne_effectifs_et_intervalles():
    pytest.importorskip("sklearn")
    vrais = [0, 0, 0, 1, 1, 2]
    preds = [0, 0, 1, 1, 2, 2]
    m = td.metriques(vrais, preds)
    ic = m["intervalles_95"]
    assert ic["rappel_urgence_maximale"]["succes"] == 2 and ic["rappel_urgence_maximale"]["n"] == 3
    assert ic["taux_sous_triage"]["succes"] == 2 and ic["taux_sous_triage"]["n"] == 6
    assert m["taux_sous_triage"] == pytest.approx(2 / 6)
    for entree in ic.values():
        bas, haut = entree["ic95"]
        assert 0.0 <= bas <= entree["succes"] / entree["n"] <= haut <= 1.0
