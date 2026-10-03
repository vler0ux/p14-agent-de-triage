"""
Tests du plancher de sécurité de la démo (src/demo/decideur.py : plancher()).

Le plancher force l'urgence maximale dès qu'un signe d'alerte est décrit par le patient, quel que soit
son ton. Ces tests fixent les comportements voulus, y compris les deux erreurs relevées par l'audit
(négation qui débordait sur un autre membre de phrase, « AVC » non reconnu).
Aucun modèle n'est chargé : tests rapides, exécutables en CI.

Lancer depuis la racine du projet :
    python -m pytest tests/test_plancher.py -v
"""
import sys
from pathlib import Path

import pytest

RACINE = Path(__file__).parent.parent
sys.path.insert(0, str(RACINE / "src" / "demo"))

from decideur import plancher  # noqa: E402


def patient(*messages):
    return [{"role": "user", "content": m} for m in messages]


@pytest.mark.parametrize("texte, signe", [
    # erreurs relevées par l'audit
    ("il respire pas bien et a perdu connaissance", "trouble_conscience"),
    ("il fait un AVC je crois", "deficit_moteur"),
    # formulations franches
    ("Elle peut plus bouger son bras droit, c'est arrivé d'un coup.", "deficit_moteur"),
    ("elle arrive pas à trouver ses mots", "trouble_parole"),
    ("j'ai mal à la poitrine, ça serre fort", "douleur_thoracique"),
    ("il fait une crise cardiaque", "douleur_thoracique"),
    ("je n'arrive plus à respirer", "detresse_respiratoire"),
    ("il a vomi du sang", "hemorragie"),
    ("j'ai la gorge qui serre après avoir mangé des cacahuètes", "reaction_allergique_grave"),
    ("il a la nuque raide", "syndrome_meninge"),
    # formulations atténuées : un patient qui minimise reste à risque
    ("c'est sûrement rien, juste une petite gêne dans la poitrine", "douleur_thoracique"),
    ("je cherche un peu mes mots depuis ce matin", "trouble_parole"),
    ("j'ai la main droite un peu engourdie", "deficit_moteur"),
    ("c'est pas grave mais j'ai une gêne dans la poitrine", "douleur_thoracique"),
    # incertitude du patient, à ne pas prendre pour une négation
    ("je sais pas trop, il a du mal à respirer", "detresse_respiratoire"),
])
def test_signe_detecte(texte, signe):
    assert signe in plancher(patient(texte))


@pytest.mark.parametrize("textes, signe", [
    (("J'ai des petites taches rouges sur les jambes.", "Et de la fièvre depuis ce matin."), "purpura_febrile"),
    (("J'ai deux semaines de retard de règles.", "Et quelques petits saignements."), "grossesse_douleur_saignement"),
])
def test_combinaison_sur_plusieurs_messages(textes, signe):
    assert signe in plancher(patient(*textes))


@pytest.mark.parametrize("texte", [
    "Non, pas de douleur à la poitrine, et pas de mal à respirer.",
    "j'ai pas perdu connaissance",
    "Je respire bien, rien d'autre, pas de gonflement ailleurs.",
    "aucune douleur thoracique",
    # cas bénins
    "je me suis coupé le doigt en cuisinant, ça saigne un peu",
    "j'ai le nez qui coule et mal à la gorge depuis trois jours",
    "j'ai mal à la gorge et un peu de mal à avaler, j'ai 38 de fièvre",
    "je respire bien par le nez maintenant",
    # impotence après un traumatisme : douleur qui bloque, pas un déficit neurologique
    "je suis tombé sur l'épaule, je peux pas lever le bras",
    "j'ai pris un coup de sabot, je peux pas bouger la jambe",
    # grossesse sans douleur ni saignement, plaie qui saigne sans grossesse en jeu
    "je suis enceinte de 5 mois, tout va bien, c'est pour une question",
])
def test_pas_de_declenchement(texte):
    assert plancher(patient(texte)) == []


def test_traumatisme_n_efface_pas_les_signes_neurologiques():
    # « d'un coup » = début brutal, pas un traumatisme ; et un engourdissement après une chute reste un signe d'alerte
    assert "deficit_moteur" in plancher(patient("tout à coup il peut plus bouger la jambe"))
    assert "deficit_moteur" in plancher(patient("il est tombé et depuis il a le bras tout engourdi"))


def test_seuls_les_messages_du_patient_sont_lus():
    messages = [{"role": "assistant", "content": "Avez-vous mal à la poitrine ?"},
                {"role": "user", "content": "Non, pas du tout."}]
    assert plancher(messages) == []


def test_cas_de_reference_de_la_demo():
    cas_a = patient("Je suis tombée dans ma salle de bain, mon poignet me fait très mal.",
                    "C'est le droit. Ça fait vraiment mal, et j'arrive à peine à bouger ma main.",
                    "Non, non... c'est juste le poignet.")
    cas_b = patient("Elle peut plus bouger son bras droit, et elle parle pas bien. C'est arrivé d'un coup.",
                    "Elle arrive pas à trouver ses mots.")
    cas_c = patient("Il a une grosse bosse sur le front.", "Il est conscient, il parle, il est actif.")
    assert plancher(cas_a) == []
    assert {"deficit_moteur", "trouble_parole"} <= set(plancher(cas_b))
    assert plancher(cas_c) == []

def test_la_negation_ne_deborde_pas_d_un_message_sur_le_suivant():
    # Messages sans ponctuation finale : la négation du premier ne doit pas annuler le second
    assert "trouble_conscience" in plancher(patient("Non je n'ai pas mal à la tête", "il a perdu connaissance"))