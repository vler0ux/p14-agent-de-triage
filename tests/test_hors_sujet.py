"""Tests du garde-fou de périmètre (src/demo/decideur.py) : messages hors sujet.

Aucun modèle n'est chargé : seules les règles déterministes sont testées.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "demo"))
from decideur import (  # noqa: E402
    ORIENTATION, RECADRAGE, hors_sujet, plancher, reponse_hors_sujet,
)


# --- 1. Vrais hors sujet : bloqués -------------------------------------------------
@pytest.mark.parametrize("message", [
    "Quel temps fera-t-il demain ?",
    "Tu sais s'il va pleuvoir ce week-end ?",
    "Qui a gagné le match de foot hier soir ?",
    "Tu peux me donner une recette de gâteau ?",
    "Raconte-moi une blague.",
    "Comment on écrit une boucle en Python ?",
])
def test_hors_sujet_bloque(message):
    assert hors_sujet(message)


# --- 2. Messages médicaux contenant un mot piège : ils passent ----------------------
@pytest.mark.parametrize("message", [
    "Il fait très chaud et j'ai la tête qui tourne.",
    "Je me suis fait mal au genou pendant le match.",
    "Je me suis coupé en préparant une recette, ça saigne.",
    "Il neige, j'ai glissé et j'ai mal au dos.",
    "J'ai de la fièvre depuis que j'ai regardé le film hier soir.",
])
def test_message_medical_avec_mot_piege_passe(message):
    assert not hors_sujet(message)


# --- 3. Messages ordinaires : ils passent ---------------------------------------------
@pytest.mark.parametrize("message", [
    "Bonjour",
    "Oui",
    "non, rien de particulier",
    "J'ai mal au ventre depuis ce matin.",
])
def test_message_ordinaire_passe(message):
    assert not hors_sujet(message)


# --- 4. Une urgence passe toujours en premier ----------------------------------------
def test_urgence_prioritaire_sur_hors_sujet():
    message = "Quel temps fera-t-il ? Et j'ai très mal à la poitrine."
    assert "douleur_thoracique" in plancher([{"role": "user", "content": message}])
    assert not hors_sujet(message)


# --- 5. Recadrage, puis orientation si le patient insiste ---------------------------
def _conversation(*tours):
    messages = [{"role": "system", "content": "prompt"},
                {"role": "assistant", "content": "Centre Hospitalier Saint-Aurélien, bonjour."}]
    for role, contenu in tours:
        messages.append({"role": role, "content": contenu})
    return messages


def test_premier_hors_sujet_recadre():
    messages = _conversation(("user", "Quel temps fera-t-il demain ?"))
    assert reponse_hors_sujet(messages) == RECADRAGE


def test_second_hors_sujet_consecutif_oriente():
    messages = _conversation(
        ("user", "Quel temps fera-t-il demain ?"),
        ("assistant", RECADRAGE),
        ("user", "Et qui a gagné le match ?"),
    )
    assert reponse_hors_sujet(messages) == ORIENTATION


def test_compteur_remis_a_zero_apres_echange_medical():
    messages = _conversation(
        ("user", "Quel temps fera-t-il demain ?"),
        ("assistant", RECADRAGE),
        ("user", "J'ai mal au ventre."),
        ("assistant", "Depuis quand avez-vous mal ?"),
        ("user", "Et qui a gagné le match ?"),
    )
    assert reponse_hors_sujet(messages) == RECADRAGE


def test_orientation_mentionne_le_15():
    assert "15" in ORIENTATION
