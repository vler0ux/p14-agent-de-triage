"""
Tests de l'Étape 5 (génération des dialogues) — AUCUN appel API : un faux LLM renvoie des dialogues prévus.

Lancer depuis la racine du projet :
    python -m pytest tests/test_etape5.py -v
"""

import copy
import json
import re
import sys
from pathlib import Path

import pytest

RACINE = Path(__file__).parent.parent
sys.path.insert(0, str(RACINE / "src" / "generation"))

import generation_dialogue_etape5 as e5  # noqa: E402


def dialogue_ok(n=4, conclusion="On va vous prendre en charge tout de suite, madame."):
    tours = []
    for i in range(n):
        tours.append({"role": "agent", "content": f"Question {i + 1} : depuis quand avez-vous mal ?"})
        tours.append({"role": "patient", "content": f"Ben je sais plus trop... depuis hier soir je crois ({i})."})
    return {"dialogue": tours, "conclusion": conclusion}


CONCLUSIONS = {
    "urgence_maximale": "On vous prend en charge tout de suite, sans attendre.",
    "moderee": "On va vous voir rapidement, dans la journée, mais ce n'est pas une urgence vitale.",
    "differee": "Vous pouvez patienter, ce n'est pas urgent, mais on va s'occuper de vous.",
}


def dialogue_avec_constante(conclusion=CONCLUSIONS["moderee"]):
    """Dialogue refusé par les contrôles (valeur de tension chiffrée dans la bouche du patient)."""
    r = dialogue_ok(conclusion=conclusion)
    r["dialogue"][1] = {"role": "patient", "content": "Ma tension était à 120/80 ce matin."}
    return r


def trace(categorie="moderee", **kw):
    t = {"id": "id_1", "cas_id": "cas_1", "categorie_finale": categorie, "vignette_source": "Femme de 50 ans, fatigue et amaigrissement.",
         "etape1_extraction": {"age_annees": 50, "pas_mmhg": 120, "fc_min": 100, "motif_id": "aeg_asthenie"},
         "a_relire": False, "priorite_relecture": None}
    t.update(kw)
    return t


# ---------------------------------------------------------------- validation

def test_dialogue_valide():
    assert e5.valider_dialogue(dialogue_ok()) == []


def test_balise():
    assert e5.balise_categorie("moderee") == "[[PRIORITE: moderee]]"


@pytest.mark.parametrize("mutation, morceau", [
    (lambda r: r["dialogue"].__setitem__(1, {"role": "patient", "content": "Je m'appelle [PERSON] et j'ai mal."}), "anonymisation"),
    (lambda r: r["dialogue"].__setitem__(1, {"role": "patient", "content": "Monsieur L. a mal au ventre."}), "initiale"),
    (lambda r: r["dialogue"].__setitem__(1, {"role": "patient", "content": "Ma tension était à 120/80 ce matin."}), "constante"),
    (lambda r: r["dialogue"].__setitem__(1, {"role": "patient", "content": "L'infirmière a dit que mon pouls était à 110."}), "constante"),
    (lambda r: r["dialogue"].__setitem__(1, {"role": "patient", "content": "Ma tension est à 12/8, ça va."}), "constante"),
    (lambda r: r["dialogue"].pop(), "finir par une réponse du patient"),
    (lambda r: r.__setitem__("dialogue", r["dialogue"][:4]), "échanges"),
    (lambda r: r["dialogue"].insert(0, {"role": "patient", "content": "Bonjour"}), "invalide"),
    (lambda r: r.__setitem__("dialogue", []), "absent ou vide"),
])
def test_dialogue_refuse(mutation, morceau):
    r = dialogue_ok(conclusion=CONCLUSIONS["urgence_maximale"])
    mutation(r)
    problemes = e5.valider_dialogue(r)
    assert any(morceau in p for p in problemes), problemes


@pytest.mark.parametrize("conclusion_du_modele", ["Vous êtes classé en catégorie 2.", "", None, "on va vous faire un scanner"])
def test_la_conclusion_du_modele_est_toujours_remplacee_par_celle_du_code(faux, conclusion_du_modele):
    r = dialogue_ok()
    if conclusion_du_modele is None:
        del r["conclusion"]          # champ omis par le modèle : ne doit pas faire planter
    else:
        r["conclusion"] = conclusion_du_modele
    faux([r])
    ligne = e5.generer_variante(trace("moderee"), 0, "anthropic", "m", 0.8, essais=1)
    assert "_erreur" not in ligne
    assert ligne["conclusion"] == e5.choisir_conclusion("id_1", 0, "moderee")
    assert ligne["controles"]["conclusion_corrigee"] is True and ligne["controles"]["conclusion_coherente"] is True


def test_une_temperature_dite_par_le_patient_est_acceptee():
    r = dialogue_ok(conclusion=CONCLUSIONS["moderee"])
    r["dialogue"][1] = {"role": "patient", "content": "J'ai pris ma température hier soir, j'avais 39 de fièvre depuis 2 jours."}
    assert e5.valider_dialogue(r) == []


# ---------------------------------------------------------------- génération

AUCUN_INVENTE = {"elements_inventes": []}


class FauxLLM:
    """Répond aux appels de GÉNÉRATION (file `reponses`) et aux appels du JUGE (file `juges`, défaut : rien d'inventé)."""

    def __init__(self, reponses, juges=()):
        self.reponses, self.juges = list(reponses), list(juges)
        self.appels = []

    @property
    def appels_generation(self):
        return [a for a in self.appels if "juge de fidélité" not in a["prompt"]]

    @property
    def appels_juge(self):
        return [a for a in self.appels if "juge de fidélité" in a["prompt"]]

    def __call__(self, system_prompt, user_content, fournisseur, modele, max_tokens=400, max_tentatives=5, temperature=0.0):
        self.appels.append({"prompt": system_prompt, "contenu": user_content, "fournisseur": fournisseur, "modele": modele,
                            "temperature": temperature, "max_tokens": max_tokens})
        if "juge de fidélité" in system_prompt:
            return copy.deepcopy(self.juges.pop(0) if self.juges else AUCUN_INVENTE)
        return copy.deepcopy(self.reponses.pop(0) if self.reponses else dialogue_ok(conclusion=CONCLUSIONS["moderee"]))


@pytest.fixture
def faux(monkeypatch):
    def installer(reponses=(), juges=()):
        f = FauxLLM(reponses, juges)
        monkeypatch.setattr(e5, "completer", f)
        return f
    return installer


def test_generer_variante_succes_et_champs(faux):
    f = faux([dialogue_ok(conclusion=CONCLUSIONS["moderee"])])
    ligne = e5.generer_variante(trace(), 1, "anthropic", "claude-x", 0.8, essais=2)
    assert "_erreur" not in ligne
    assert ligne["balise"] == "[[PRIORITE: moderee]]" and ligne["variante"] == 1 and ligne["cas_id"] == "cas_1"
    assert ligne["style"] == e5.STYLES[1] and ligne["controles"] == {"conclusion_coherente": True, "tentatives": 1, "fidelite": "non_jugee", "elements_mineurs": [], "refus_precedents": [],
                                  "ouverture_corrigee": True, "conclusion_corrigee": True}
    assert ligne["provenance"]["temperature"] == 0.8 and ligne["provenance"]["modele"] == "claude-x"
    assert f.appels_generation[0]["temperature"] == 0.8 and e5.STYLES[1] in f.appels_generation[0]["prompt"]


def test_le_llm_ne_recoit_ni_constantes_ni_motif_code(faux):
    f = faux()
    e5.generer_variante(trace(), 0, "anthropic", "m", 0.8, essais=1)
    contenu = json.loads(f.appels_generation[0]["contenu"])
    assert set(contenu) == {"description_clinique", "age_annees", "niveau_de_priorite_valide"}
    assert "120" not in f.appels_generation[0]["contenu"] and "aeg_asthenie" not in f.appels_generation[0]["contenu"]
    assert contenu["niveau_de_priorite_valide"] == "moderee"


def test_nouvelle_tentative_puis_succes(faux):
    f = faux([dialogue_avec_constante(), dialogue_ok(conclusion=CONCLUSIONS["moderee"])])
    ligne = e5.generer_variante(trace(), 0, "anthropic", "m", 0.8, essais=2)
    assert "_erreur" not in ligne and ligne["controles"]["tentatives"] == 2 and len(f.appels_generation) == 2


def test_echec_apres_les_essais_est_trace_avec_le_probleme(faux):
    faux([dialogue_avec_constante(), dialogue_avec_constante()])
    ligne = e5.generer_variante(trace(), 0, "anthropic", "m", 0.8, essais=2)
    assert "_erreur" in ligne and any("constante" in p for p in ligne["_erreur"]["problemes"]) and "dialogue" not in ligne


def test_reponse_illisible_du_llm(faux):
    faux([{"_erreur_parsing": "pas du json"}])
    ligne = e5.generer_variante(trace(), 0, "anthropic", "m", 0.8, essais=1)
    assert "_erreur_parsing" in ligne["_erreur"]


def test_lire_variantes():
    assert e5.lire_variantes("urgence_maximale=1,moderee=2,differee=3") == {"urgence_maximale": 1, "moderee": 2, "differee": 3}
    assert e5.lire_variantes("differee=5") == {"urgence_maximale": 1, "moderee": 2, "differee": 5}   # le reste : défauts
    with pytest.raises(SystemExit):
        e5.lire_variantes("inconnue=2")
    with pytest.raises(SystemExit):
        e5.lire_variantes("moderee=beaucoup")


# ---------------------------------------------------------------- programme complet

def test_main_variantes_reprise_et_reessai_des_echecs(faux, tmp_path, monkeypatch):
    traces = [trace("urgence_maximale", id="A", cas_id="cA"), trace("differee", id="B", cas_id="cB"),
              {"id": "C", "cas_id": "cC", "vignette_source": "x"}]                  # sans label : ignoré
    entree, sortie = tmp_path / "pipeline.jsonl", tmp_path / "out" / "dialogues.jsonl"
    entree.write_text("\n".join(json.dumps(t, ensure_ascii=False) for t in traces) + "\n", encoding="utf-8")
    monkeypatch.setattr(e5.time, "sleep", lambda s: None)

    def reponses_selon_appel():
        return [dialogue_ok(conclusion=CONCLUSIONS["urgence_maximale"]), dialogue_ok(conclusion=CONCLUSIONS["differee"]),
                dialogue_ok(conclusion=CONCLUSIONS["differee"]), dialogue_ok(conclusion=CONCLUSIONS["differee"])]

    f = faux(reponses_selon_appel())
    monkeypatch.setattr(sys, "argv", ["e5", "--input", str(entree), "--output", str(sortie), "--pause", "0"])
    e5.main()
    lignes = [json.loads(l) for l in sortie.read_text(encoding="utf-8").splitlines()]
    assert [(l["id"], l["variante"]) for l in lignes] == [("A", 0), ("B", 0), ("B", 1), ("B", 2)]     # 1 + 3 variantes
    assert all("_erreur" not in l for l in lignes) and len(f.appels_generation) == 4
    # reprise : rien de nouveau
    f2 = faux()
    e5.main()
    assert len(f2.appels) == 0 and len(sortie.read_text(encoding="utf-8").splitlines()) == 4   # ni génération ni juge


def test_main_les_echecs_sont_retentes_a_la_reprise(faux, tmp_path, monkeypatch):
    entree, sortie = tmp_path / "p.jsonl", tmp_path / "d.jsonl"
    entree.write_text(json.dumps(trace("urgence_maximale", id="A"), ensure_ascii=False) + "\n", encoding="utf-8")
    monkeypatch.setattr(e5.time, "sleep", lambda s: None)
    faux([dialogue_avec_constante(), dialogue_avec_constante()])
    monkeypatch.setattr(sys, "argv", ["e5", "--input", str(entree), "--output", str(sortie), "--pause", "0", "--essais", "2"])
    e5.main()
    assert "_erreur" in json.loads(sortie.read_text(encoding="utf-8").splitlines()[0])
    faux([dialogue_ok(conclusion=CONCLUSIONS["urgence_maximale"])])
    e5.main()
    lignes = [json.loads(l) for l in sortie.read_text(encoding="utf-8").splitlines()]
    assert len(lignes) == 2 and "_erreur" in lignes[0] and "_erreur" not in lignes[1]
    assert e5.deja_traites(sortie) == {("A", 0)}


def test_fournisseur_autre_que_anthropic_exige_un_modele(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["e5", "--input", "x", "--output", str(tmp_path / "o"), "--fournisseur", "groq"])
    with pytest.raises(SystemExit):
        e5.main()


# ---------------------------------------------------------------- correctif de fidélité (juge, prompt, actes)

def test_le_prompt_interdit_de_confirmer_un_symptome_absent_et_les_horaires_inventes():
    p = e5.construire_prompt(e5.STYLES[0], conclusion_imposee="Venez nous voir d'ici peu dans la journée.")
    assert "ne CONFIRME JAMAIS" in p and "reste vague" in p and "à recopier MOT POUR MOT" in p
    assert "« Venez nous voir d'ici peu dans la journée. »" in p and "{conclusion_imposee}" not in p


ACTES_ET_JARGON = re.compile(r"\b(monitoring|scanner|irm|perfusion|prise de sang|analyses?|radio|ecg|examens|"
                             r"cat[ée]gorie|priorit[ée]|niveau|mod[ée]r[ée]e|diff[ée]r[ée]e|tri)\b", re.IGNORECASE)


@pytest.mark.parametrize("categorie", e5.CATEGORIES)
def test_les_conclusions_imposees_ne_promettent_aucun_acte_et_sans_jargon(categorie):
    # La conclusion n'est plus contrôlée après génération : c'est la liste du code qui doit être irréprochable.
    assert e5.CONCLUSIONS[categorie]
    for conclusion in e5.CONCLUSIONS[categorie]:
        assert not ACTES_ET_JARGON.search(conclusion), conclusion


def test_le_juge_recoit_la_description_et_le_dialogue_et_un_dialogue_fidele_est_verifie(faux):
    f = faux([dialogue_ok(conclusion=CONCLUSIONS["moderee"])], juges=[AUCUN_INVENTE])
    ligne = e5.generer_variante(trace(), 0, "anthropic", "m", 0.8, essais=1, juge=("anthropic", "juge-x"))
    assert ligne["controles"]["fidelite"] == "verifiee" and ligne["provenance"]["juge"]["modele"] == "juge-x"
    appel = f.appels_juge[0]
    assert appel["modele"] == "juge-x" and appel["temperature"] == 0.0
    contenu = json.loads(appel["contenu"])
    assert "fatigue et amaigrissement" in contenu["description_clinique"] and contenu["dialogue"][-1]["role"] == "agent"   # conclusion incluse


def test_un_element_medical_invente_fait_refuser_le_dialogue_puis_nouvelle_tentative(faux):
    invente = {"elements_inventes": [{"element": "douleur au cou confirmée par le patient", "importance": "medicale"}]}
    f = faux([dialogue_ok(conclusion=CONCLUSIONS["moderee"]), dialogue_ok(conclusion=CONCLUSIONS["moderee"])], juges=[invente, AUCUN_INVENTE])
    ligne = e5.generer_variante(trace(), 0, "anthropic", "m", 0.8, essais=2, juge=("anthropic", "m"))
    assert "_erreur" not in ligne and ligne["controles"]["tentatives"] == 2
    assert len(f.appels_generation) == 2 and len(f.appels_juge) == 2


def test_echec_final_si_le_juge_refuse_a_chaque_essai_et_garde_la_raison(faux):
    invente = {"elements_inventes": [{"element": "raideur de nuque", "importance": "medicale"}]}
    faux([dialogue_ok(conclusion=CONCLUSIONS["moderee"])] * 2, juges=[invente, invente])
    ligne = e5.generer_variante(trace(), 0, "anthropic", "m", 0.8, essais=2, juge=("anthropic", "m"))
    assert "_erreur" in ligne and "raideur de nuque" in ligne["_erreur"]["problemes"][0] and ligne["_erreur"]["elements_inventes"]


def test_un_detail_mineur_est_garde_dans_les_controles_sans_refuser(faux):
    mineur = {"elements_inventes": [{"element": "dit « hier soir » sans que la description le donne", "importance": "mineure"}]}
    faux([dialogue_ok(conclusion=CONCLUSIONS["moderee"])], juges=[mineur])
    ligne = e5.generer_variante(trace(), 0, "anthropic", "m", 0.8, essais=1, juge=("anthropic", "m"))
    assert "_erreur" not in ligne and ligne["controles"]["elements_mineurs"] == ["dit « hier soir » sans que la description le donne"]


def test_juge_illisible_le_dialogue_est_garde_mais_marque_non_verifie(faux):
    faux([dialogue_ok(conclusion=CONCLUSIONS["moderee"])], juges=[{"_erreur_parsing": "illisible"}])
    ligne = e5.generer_variante(trace(), 0, "anthropic", "m", 0.8, essais=1, juge=("anthropic", "m"))
    assert "_erreur" not in ligne and ligne["controles"]["fidelite"] == "non_verifiee"


def test_main_juge_actif_par_defaut_et_desactivable(faux, tmp_path, monkeypatch):
    entree = tmp_path / "p.jsonl"
    entree.write_text(json.dumps(trace("urgence_maximale", id="A"), ensure_ascii=False) + "\n", encoding="utf-8")
    monkeypatch.setattr(e5.time, "sleep", lambda s: None)
    f = faux([dialogue_ok(conclusion=CONCLUSIONS["urgence_maximale"])])
    monkeypatch.setattr(sys, "argv", ["e5", "--input", str(entree), "--output", str(tmp_path / "o1.jsonl"), "--pause", "0"])
    e5.main()
    assert len(f.appels_juge) == 1 and f.appels_juge[0]["modele"] == e5.MODELE_DEFAUT
    f2 = faux([dialogue_ok(conclusion=CONCLUSIONS["urgence_maximale"])])
    monkeypatch.setattr(sys, "argv", ["e5", "--input", str(entree), "--output", str(tmp_path / "o2.jsonl"), "--pause", "0", "--sans-juge"])
    e5.main()
    assert len(f2.appels_juge) == 0
    ligne = json.loads((tmp_path / "o2.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert ligne["controles"]["fidelite"] == "non_jugee" and ligne["provenance"]["juge"] is None


def test_fournisseur_juge_different_exige_un_modele(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["e5", "--input", "x", "--output", str(tmp_path / "o"), "--fournisseur-juge", "groq"])
    with pytest.raises(SystemExit):
        e5.main()


def test_les_refus_sont_traces_pour_comprendre_les_nouvelles_tentatives(faux):
    invente = {"elements_inventes": [{"element": "elle semble chaude", "importance": "medicale"}]}
    faux([dialogue_avec_constante(), dialogue_ok(conclusion=CONCLUSIONS["moderee"]), dialogue_ok(conclusion=CONCLUSIONS["moderee"])],
         juges=[invente, AUCUN_INVENTE])
    ligne = e5.generer_variante(trace(), 0, "anthropic", "m", 0.8, essais=3, juge=("anthropic", "m"))
    refus = ligne["controles"]["refus_precedents"]
    assert ligne["controles"]["tentatives"] == 3 and [r["tentative"] for r in refus] == [1, 2]
    assert any("constante" in p for p in refus[0]["problemes"]) and "elle semble chaude" in refus[1]["problemes"][0]


def test_echec_final_garde_tous_les_refus(faux):
    faux([dialogue_avec_constante()] * 2)
    ligne = e5.generer_variante(trace(), 0, "anthropic", "m", 0.8, essais=2)
    assert len(ligne["_erreur"]["refus"]) == 2 and ligne["_erreur"]["problemes"]


def test_le_prompt_du_juge_couvre_les_formulations_prudentes_et_la_chronologie():
    assert "même formulée avec prudence" in e5.PROMPT_JUGE and "Dater un début" in e5.PROMPT_JUGE


# ---------------------------------------------------------------- première phrase imposée (appel / accueil)

def test_aucune_ouverture_ne_reprend_la_formule_de_commerce():
    for contexte, liste in e5.OUVERTURES.items():
        assert liste and all("vous amène" not in o.lower() for o in liste), contexte
    assert any("Quel est l'objet de votre appel" in o for o in e5.OUVERTURES["appel"])
    assert any("Que puis-je faire pour vous" in o for o in e5.OUVERTURES["accueil"])


def test_choisir_ouverture_est_reproductible_et_variee():
    a = e5.choisir_ouverture("cas_1", 0, "appel")
    assert a == e5.choisir_ouverture("cas_1", 0, "appel") and a in e5.OUVERTURES["appel"]
    utilisees = {e5.choisir_ouverture(f"cas_{i}", 0, "appel") for i in range(60)}
    assert len(utilisees) == len(e5.OUVERTURES["appel"])            # toutes les formules servent
    assert e5.choisir_ouverture("cas_1", 0, "accueil") in e5.OUVERTURES["accueil"]


def test_le_prompt_donne_le_contexte_et_la_phrase_a_recopier():
    p = e5.construire_prompt(e5.STYLES[0], "appel", "Urgences, bonjour. Quel est le motif de votre appel ?")
    assert "APPEL TÉLÉPHONIQUE" in p and "ne décrit jamais son apparence" in p
    assert "« Urgences, bonjour. Quel est le motif de votre appel ? »" in p and "{contexte}" not in p and "{ouverture}" not in p
    assert "ACCUEIL SUR PLACE" in e5.construire_prompt(e5.STYLES[0], "accueil")


def test_la_premiere_phrase_de_l_agent_est_toujours_celle_du_code(faux):
    ouverture = e5.choisir_ouverture("id_1", 0, "appel")
    bon = dialogue_ok(conclusion=CONCLUSIONS["moderee"]); bon["dialogue"][0]["content"] = ouverture
    faux([bon])
    ligne = e5.generer_variante(trace(), 0, "anthropic", "m", 0.8, essais=1)
    assert ligne["dialogue"][0]["content"] == ouverture and ligne["controles"]["ouverture_corrigee"] is False
    assert ligne["contexte"] == "appel" and ligne["ouverture"] == ouverture
    # le modèle a écrit une autre phrase (la formule à éviter) : le code la rétablit et le signale
    mauvais = dialogue_ok(conclusion=CONCLUSIONS["moderee"]); mauvais["dialogue"][0]["content"] = "Qu'est-ce qui vous amène aujourd'hui ?"
    faux([mauvais])
    ligne2 = e5.generer_variante(trace(), 0, "anthropic", "m", 0.8, essais=1)
    assert ligne2["dialogue"][0]["content"] == ouverture and ligne2["controles"]["ouverture_corrigee"] is True
    assert "amène" not in json.dumps(ligne2["dialogue"][0], ensure_ascii=False)


def test_contexte_accueil_change_l_ouverture_et_le_prompt(faux):
    f = faux([dialogue_ok(conclusion=CONCLUSIONS["moderee"])])
    ligne = e5.generer_variante(trace(), 0, "anthropic", "m", 0.8, essais=1, contexte="accueil")
    assert ligne["contexte"] == "accueil" and ligne["dialogue"][0]["content"] in e5.OUVERTURES["accueil"]
    assert "ACCUEIL SUR PLACE" in f.appels_generation[0]["prompt"]


def test_main_option_contexte(faux, tmp_path, monkeypatch):
    entree, sortie = tmp_path / "p.jsonl", tmp_path / "d.jsonl"
    entree.write_text(json.dumps(trace("urgence_maximale", id="A"), ensure_ascii=False) + "\n", encoding="utf-8")
    monkeypatch.setattr(e5.time, "sleep", lambda s: None)
    faux([dialogue_ok(conclusion=CONCLUSIONS["urgence_maximale"])])
    monkeypatch.setattr(sys, "argv", ["e5", "--input", str(entree), "--output", str(sortie), "--pause", "0", "--sans-juge", "--contexte", "accueil"])
    e5.main()
    assert json.loads(sortie.read_text(encoding="utf-8").splitlines()[0])["contexte"] == "accueil"
    faux([dialogue_ok(conclusion=CONCLUSIONS["urgence_maximale"])])
    monkeypatch.setattr(sys, "argv", ["e5", "--input", str(entree), "--output", str(tmp_path / "d2.jsonl"), "--pause", "0", "--sans-juge"])
    e5.main()
    assert json.loads((tmp_path / "d2.jsonl").read_text(encoding="utf-8").splitlines()[0])["contexte"] == "appel"   # défaut
