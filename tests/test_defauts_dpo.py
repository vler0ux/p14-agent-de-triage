"""
test_defauts_dpo.py — détecteurs de défauts de l'hôtesse (src/dpo/defauts.py).

Lancer avec : pytest tests/test_defauts_dpo.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "dpo"))
from defauts import evaluer, nettoyer, nombre_questions, non_francais, rassurance_ou_diagnostic  # noqa: E402


def _eval(texte, reference_conclut=False, brute=None, tronquee=False):
    return evaluer(texte, reference_conclut=reference_conclut, seuil_mots=26,
                   reponse_brute=brute, tronquee=tronquee)


def test_une_question_n_est_pas_un_defaut():
    assert not _eval("Depuis quand avez-vous mal ?")["plusieurs_questions"]


def test_plusieurs_questions():
    assert nombre_questions("Depuis quand ? Et ça s'aggrave ??") == 2
    assert _eval("Depuis quand ? Et ça s'aggrave ?")["plusieurs_questions"]


def test_conclusion_prematuree_et_manquee():
    conclusion = "On s'occupe de vous rapidement.\n[[PRIORITE: moderee]]"
    assert _eval(conclusion, reference_conclut=False)["conclusion_prematuree"]
    assert not _eval(conclusion, reference_conclut=True)["conclusion_prematuree"]
    assert _eval("Avez-vous de la fièvre ?", reference_conclut=True)["conclusion_manquee"]


def test_une_conclusion_n_est_pas_comptee_comme_plusieurs_questions():
    assert not _eval("Des questions ? Non ? Alors on vous attend.\n[[PRIORITE: differee]]",
                     reference_conclut=True)["plusieurs_questions"]


def test_rassurance_et_diagnostic():
    assert rassurance_ou_diagnostic("Ne vous inquiétez pas, ce n'est rien.") == [
        "ce n'est rien", "ne vous inquiétez pas"]
    assert rassurance_ou_diagnostic("C'est sûrement une entorse.") == ["c'est sûrement / probablement"]
    assert rassurance_ou_diagnostic("Avez-vous déjà eu ce genre de douleur ?") == []


def test_langue():
    assert not non_francais("Avez-vous de la fièvre depuis hier ?")
    assert non_francais("Do you have any pain in your chest and is it worse ?")
    assert non_francais("您好，请描述症状")


def test_artefacts_et_nettoyage():
    brute = "Avez-vous mal ?atrigesimal blabla<|im_end|>"
    assert nettoyer(brute) == "Avez-vous mal ?"
    d = _eval(nettoyer(brute), brute=brute)
    assert d["artefact"] and d["_artefacts"] == ["atrigesimal"]
    assert not _eval("Avez-vous mal ?", brute="Avez-vous mal ?<|im_end|>")["artefact"]
    assert _eval("Avez-vous", brute="Avez-vous", tronquee=True)["artefact"]


def test_reference_sans_sortie_brute_n_a_pas_d_artefact():
    assert not _eval("Avez-vous mal ?")["artefact"]


def test_une_conclusion_n_est_jugee_ni_sur_la_rassurance_ni_sur_la_longueur():
    conclusion = "Ne vous inquiétez pas, " + "mot " * 30 + "\n[[PRIORITE: urgence_maximale]]"
    d = _eval(conclusion, reference_conclut=True)
    assert not d["rassurance_ou_diagnostic"] and not d["trop_longue"]


def test_trop_longue():
    assert _eval("mot " * 27)["trop_longue"]


# ---------------------------------------------------------------- filtre (filtrer_rejected.py)
from filtrer_rejected import choisir_paire, format_relecture  # noqa: E402

CONTEXTE = [{"role": "system", "content": "s"}, {"role": "assistant", "content": "Urgences, bonjour."},
            {"role": "user", "content": "J'ai mal au ventre depuis ce matin."}]


def _ligne(rejected, chosen="Depuis quelle heure exactement ?", n=0):
    return {"cas_id": "cas_x", "tour": 2, "reponse_n": n, "messages_contexte": CONTEXTE,
            "chosen": chosen, "rejected": rejected, "rejected_brut": rejected + "<|im_end|>"}


def test_pas_de_paire_si_toutes_les_reponses_sont_correctes():
    paire, raison = choisir_paire([_ligne("La douleur est-elle continue ?")], 26, 42)
    assert paire is None and raison == "aucune_reponse_fautive"


def test_le_rejected_le_plus_grave_l_emporte():
    groupe = [_ligne("Avez-vous de la fièvre ? Des vomissements ?", n=0),
              _ligne("Ce n'est rien, avez-vous mangé ?", n=1),
              _ligne("Où avez-vous mal ?", n=2)]
    paire, _ = choisir_paire(groupe, 26, 42)
    assert paire["raison_rejet"] == "rassurance_ou_diagnostic" and paire["reponse_n"] == 1


def test_chosen_avec_defaut_ecarte_le_contexte():
    groupe = [_ligne("Ce n'est rien.", chosen="Ne vous inquiétez pas, depuis quand ?")]
    assert choisir_paire(groupe, 26, 42) == (None, "chosen_avec_defaut")


def test_reponse_tronquee_est_un_artefact():
    ligne = _ligne("Avez-vous")
    ligne["rejected_brut"] = "Avez-vous"          # aucun marqueur de fin : coupé par max_new_tokens
    paire, _ = choisir_paire([ligne], 26, 42)
    assert paire["raison_rejet"] == "artefact"


def test_plusieurs_questions_ne_fait_plus_un_rejected():
    paire, raison = choisir_paire([_ligne("Avez-vous de la fièvre ? Des vomissements ?")], 26, 42)
    assert paire is None and raison == "aucune_reponse_fautive"


def test_conclusion_prematuree():
    paire, _ = choisir_paire([_ligne("On s'occupe de vous.\n[[PRIORITE: moderee]]")], 26, 42)
    assert paire["raison_rejet"] == "conclusion_prematuree"


def test_conclusion_manquee():
    conclusion = "On s'occupe de vous immédiatement.\n[[PRIORITE: urgence_maximale]]"
    paire, _ = choisir_paire([_ligne("Est-ce qu'il mange bien ?", chosen=conclusion)], 26, 42)
    assert paire["raison_rejet"] == "conclusion_manquee" and paire["chosen_est_conclusion"]


def test_choix_reproductible_a_gravite_egale():
    groupe = [_ligne("On s'en occupe.\n[[PRIORITE: moderee]]", n=0),
              _ligne("Nous allons vous prendre en charge.\n[[PRIORITE: differee]]", n=1)]
    assert choisir_paire(groupe, 26, 42)[0]["reponse_n"] == choisir_paire(groupe, 26, 42)[0]["reponse_n"]


def test_format_relecture_compatible_avec_relire_paires_dpo():
    paire, _ = choisir_paire([_ligne("Ce n'est rien.")], 26, 42)
    r = format_relecture(paire, 1)
    for cle in ("id", "prompt_fr", "chosen_fr", "rejected_fr", "prompt_en", "chosen_en", "rejected_en"):
        assert cle in r
    assert r["a_valider"] is None and "J'ai mal au ventre" in r["prompt_fr"]
