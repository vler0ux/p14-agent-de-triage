"""
test_juge_dpo.py — logique du juge LLM du DPO v6 (src/dpo/juger_paires_dpo.py), sans appel API.

Lancer avec : pytest tests/test_juge_dpo.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "dpo"))
from juger_paires_dpo import (candidats_du_contexte, choisir_paire, construire_demande,  # noqa: E402
                              raison_du_rejet, valider_reponse_juge)

CONTEXTE = [{"role": "system", "content": "s"}, {"role": "assistant", "content": "Urgences, bonjour."},
            {"role": "user", "content": "Mon bébé de 6 mois a de la fièvre."}]


def _groupe(reponses, chosen="Depuis quand a-t-il de la fièvre ?"):
    return [{"cas_id": "cas_x", "tour": 2, "reponse_n": i, "messages_contexte": CONTEXTE,
             "chosen": chosen, "rejected": r} for i, r in enumerate(reponses)]


def _cand(source, total, sur=True):
    notes = {"clarte": 5, "adaptation": 5, "utilite": 5, "securite": 5, "ton": 5}
    notes["adaptation"] -= 25 - total
    return {"source": source, "texte": source, "notes": notes, "total": total,
            "securite_ok": sur, "justification": ""}


def test_doublons_retires_et_reference_prioritaire():
    c = candidats_du_contexte(_groupe(["Depuis quand a-t-il de la fièvre ?", "Il se plaint de maux de tête ?"]))
    assert [x["source"] for x in c] == ["reference", "modele_1"]


def test_melange_reproductible_et_anonyme():
    g = _groupe(["A ?", "B ?", "C ?", "D ?"])
    c1, texte = construire_demande(g, 42)
    c2, _ = construire_demande(g, 42)
    assert [x["source"] for x in c1] == [x["source"] for x in c2]
    assert "reference" not in texte and "modele" not in texte
    assert "PATIENT : Mon bébé de 6 mois" in texte


def test_validation_reponse_juge():
    bonne = {"evaluations": [{"lettre": l, "clarte": 3, "adaptation": 3, "utilite": 3, "securite": 3,
                              "ton": 3, "securite_ok": True} for l in "AB"]}
    assert set(valider_reponse_juge(bonne, ["A", "B"])) == {"A", "B"}
    with pytest.raises(ValueError):
        valider_reponse_juge({"evaluations": bonne["evaluations"][:1]}, ["A", "B"])
    with pytest.raises(ValueError):
        valider_reponse_juge({"_erreur_parsing": "texte"}, ["A"])
    mauvaise = {"evaluations": [dict(bonne["evaluations"][0], clarte=7), bonne["evaluations"][1]]}
    with pytest.raises(ValueError):
        valider_reponse_juge(mauvaise, ["A", "B"])


def test_reference_peut_etre_rejetee():
    (chosen, rejected), _ = choisir_paire([_cand("reference", 12), _cand("modele_0", 23)], 3)
    assert chosen["source"] == "modele_0" and rejected["source"] == "reference"


def test_reponse_non_sure_rejetee_en_priorite_et_jamais_choisie():
    (chosen, rejected), _ = choisir_paire([_cand("reference", 20), _cand("modele_0", 24, sur=False),
                                           _cand("modele_1", 15)], 3)
    assert chosen["source"] == "reference" and rejected["source"] == "modele_0"
    assert raison_du_rejet(rejected) == "securite"


def test_ecart_trop_faible():
    assert choisir_paire([_cand("reference", 20), _cand("modele_0", 19)], 3) == (None, "ecart_trop_faible")


def test_aucune_reponse_sure():
    assert choisir_paire([_cand("reference", 20, sur=False)], 3) == (None, "aucune_reponse_sure")
