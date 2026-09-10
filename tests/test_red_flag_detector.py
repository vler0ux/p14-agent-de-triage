"""
test_red_flag_detector.py

Suite de tests unitaires pour le module de détection déterministe des red
flags (Étape 2 du pipeline) et de la logique d'override (croisement
Étape 2 / Étape 3).

Lancer avec : pytest tests/test_red_flag_detector.py -v
"""

import sys
from pathlib import Path

# Le module à tester vit dans src/referentiel/, pas dans tests/ — on l'ajoute
# au chemin de recherche des imports plutôt que de dupliquer le code.
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "referentiel"))

import pytest
from red_flag_detector import (
    CasClinique,
    charger_referentiel,
    detecter_red_flags,
    appliquer_override,
)


@pytest.fixture(scope="module")
def referentiel():
    chemin = Path(__file__).parent.parent / "src" / "referentiel" / "french_referentiel.json"
    return charger_referentiel(str(chemin))


# ---------------------------------------------------------------------------
# Constantes vitales adulte
# ---------------------------------------------------------------------------

def test_pas_effondree_force_tri1(referentiel):
    cas = CasClinique(pas_mmhg=65)
    resultat = detecter_red_flags(cas, referentiel)
    assert resultat.tri_minimal_force == "1"
    assert resultat.categorie_brief_minimale == "urgence_maximale"


def test_pas_basse_avec_tachycardie_force_tri2(referentiel):
    cas = CasClinique(pas_mmhg=95, fc_min=110)
    resultat = detecter_red_flags(cas, referentiel)
    assert resultat.tri_minimal_force == "2"


def test_fc_extreme_force_tri1(referentiel):
    cas = CasClinique(fc_min=190)
    resultat = detecter_red_flags(cas, referentiel)
    assert resultat.tri_minimal_force == "1"


def test_spo2_effondree_force_tri1(referentiel):
    cas = CasClinique(spo2_pct=82)
    resultat = detecter_red_flags(cas, referentiel)
    assert resultat.tri_minimal_force == "1"


def test_spo2_moderement_basse_force_tri2(referentiel):
    cas = CasClinique(spo2_pct=88)
    resultat = detecter_red_flags(cas, referentiel)
    assert resultat.tri_minimal_force == "2"


def test_gcs_coma_force_tri1(referentiel):
    cas = CasClinique(gcs=6)
    resultat = detecter_red_flags(cas, referentiel)
    assert resultat.tri_minimal_force == "1"


def test_gcs_intermediaire_force_tri2(referentiel):
    cas = CasClinique(gcs=11)
    resultat = detecter_red_flags(cas, referentiel)
    assert resultat.tri_minimal_force == "2"


def test_aucune_donnee_ne_force_rien(referentiel):
    """Un cas sans aucune constante vitale renseignée ne doit déclencher aucun red flag."""
    cas = CasClinique()
    resultat = detecter_red_flags(cas, referentiel)
    assert resultat.tri_minimal_force is None
    assert resultat.categorie_brief_minimale is None
    assert resultat.flags_declenches == []


def test_constantes_normales_ne_forcent_rien(referentiel):
    """Des constantes dans les bornes normales ne doivent déclencher aucune règle."""
    cas = CasClinique(pas_mmhg=120, fc_min=75, spo2_pct=98, fr_min=16, gcs=15)
    resultat = detecter_red_flags(cas, referentiel)
    assert resultat.tri_minimal_force is None


# ---------------------------------------------------------------------------
# Règles pédiatriques
# ---------------------------------------------------------------------------

def test_fievre_nourrisson_moins_3_mois_force_tri2(referentiel):
    cas = CasClinique(age_annees=0.1, fievre=True)
    resultat = detecter_red_flags(cas, referentiel)
    assert resultat.tri_minimal_force == "2"
    assert any("fievre_nourrisson_3_mois" in f for f in resultat.flags_declenches)


def test_fievre_nourrisson_plus_3_mois_ne_force_pas_la_regle_specifique(referentiel):
    """Passé 3 mois, cette règle spécifique ne doit plus se déclencher (elle
    peut en revanche être couverte par d'autres règles non testées ici)."""
    cas = CasClinique(age_annees=1.5, fievre=True)
    resultat = detecter_red_flags(cas, referentiel)
    assert not any("fievre_nourrisson_3_mois" in f for f in resultat.flags_declenches)


# ---------------------------------------------------------------------------
# Motifs de recours et modulateurs
# ---------------------------------------------------------------------------

def test_motif_sca_avec_ecg_typique_force_tri2(referentiel):
    cas = CasClinique(
        motif_id="douleur_thoracique_sca",
        criteres_presents=["ECG anormal typique de SCA"],
    )
    resultat = detecter_red_flags(cas, referentiel)
    assert resultat.tri_minimal_force == "2"
    assert resultat.categorie_brief_minimale == "urgence_maximale"


def test_motif_sca_sans_critere_utilise_le_tri_de_base(referentiel):
    """Sans modulateur déclenché, on retombe sur le tri_base du motif (3B)."""
    cas = CasClinique(motif_id="douleur_thoracique_sca", criteres_presents=[])
    resultat = detecter_red_flags(cas, referentiel)
    assert resultat.tri_minimal_force == "3B"
    assert resultat.categorie_brief_minimale == "moderee"


def test_motif_inconnu_est_ignore_sans_erreur(referentiel):
    """Un motif_id qui n'existe pas dans le référentiel ne doit pas faire planter la détection."""
    cas = CasClinique(motif_id="motif_qui_nexiste_pas", criteres_presents=["quoi que ce soit"])
    resultat = detecter_red_flags(cas, referentiel)
    assert resultat.tri_minimal_force is None


def test_arret_cardiorespiratoire_force_tri1_sans_modulateur(referentiel):
    """Un motif sans aucun modulateur (comme l'ACR) doit directement appliquer son tri_base."""
    cas = CasClinique(motif_id="arret_cardiorespiratoire")
    resultat = detecter_red_flags(cas, referentiel)
    assert resultat.tri_minimal_force == "1"


# ---------------------------------------------------------------------------
# Combinaison de plusieurs sources de red flags : le plus urgent doit gagner
# ---------------------------------------------------------------------------

def test_combinaison_constantes_et_motif_retient_le_plus_urgent(referentiel):
    """PAS effondrée (tri 1) + motif SCA sans modulateur (tri 3B) → le tri 1
    (plus urgent) doit l'emporter, pas une moyenne ni le dernier calculé."""
    cas = CasClinique(
        pas_mmhg=60,
        motif_id="douleur_thoracique_sca",
        criteres_presents=[],
    )
    resultat = detecter_red_flags(cas, referentiel)
    assert resultat.tri_minimal_force == "1"


# ---------------------------------------------------------------------------
# Mécanisme d'override (croisement Étape 2 / Étape 3)
# ---------------------------------------------------------------------------

def test_override_declenche_si_llm_sous_estime(referentiel):
    cas = CasClinique(
        motif_id="douleur_thoracique_sca",
        criteres_presents=["ECG anormal typique de SCA"],
    )
    resultat = detecter_red_flags(cas, referentiel)
    decision = appliquer_override(categorie_llm="differee", resultat_deterministe=resultat)

    assert decision["override_applique"] is True
    assert decision["label_final"] == "urgence_maximale"
    assert decision["label_llm_initial"] == "differee"


def test_pas_override_si_llm_deja_assez_prudent(referentiel):
    """Si le LLM propose déjà un niveau au moins aussi urgent que le
    référentiel, on ne doit PAS écraser son jugement."""
    cas = CasClinique(
        motif_id="douleur_thoracique_sca",
        criteres_presents=["ECG anormal typique de SCA"],  # impose "urgence_maximale"
    )
    resultat = detecter_red_flags(cas, referentiel)
    decision = appliquer_override(categorie_llm="urgence_maximale", resultat_deterministe=resultat)

    assert decision["override_applique"] is False
    assert decision["label_final"] == "urgence_maximale"


def test_pas_override_si_aucun_red_flag_deterministe(referentiel):
    """Si l'Étape 2 ne détecte rien, le label du LLM est conservé tel quel,
    quel qu'il soit — il n'y a aucune référence à laquelle le comparer."""
    cas = CasClinique()  # aucune donnée
    resultat = detecter_red_flags(cas, referentiel)
    decision = appliquer_override(categorie_llm="differee", resultat_deterministe=resultat)

    assert decision["override_applique"] is False
    assert decision["label_final"] == "differee"


def test_override_conserve_la_tracabilite(referentiel):
    """Les champs de traçabilité (flags déclenchés, tri français minimal)
    doivent être présents dans la décision finale — exigence d'audit du brief."""
    cas = CasClinique(pas_mmhg=65)
    resultat = detecter_red_flags(cas, referentiel)
    decision = appliquer_override(categorie_llm="moderee", resultat_deterministe=resultat)

    assert "red_flags_detectes" in decision
    assert len(decision["red_flags_detectes"]) > 0
    assert decision["tri_french_minimal_force"] == "1"