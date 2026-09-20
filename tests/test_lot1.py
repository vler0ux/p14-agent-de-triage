"""
Tests du lot 1 — AUCUN appel API : un faux LLM renvoie des réponses prévues.

Lancer depuis la racine du projet :
    python -m pytest tests/test_lot1.py -v

Ces tests vérifient le code (planchers, contrôle de présence, unités, lecture
des réponses, file de relecture), pas la qualité clinique des réponses d'un vrai LLM.
"""

import argparse
import copy
import inspect
import json
import sys
from pathlib import Path

import pytest

RACINE = Path(__file__).parent.parent
sys.path.insert(0, str(RACINE / "src" / "generation"))
sys.path.insert(0, str(RACINE / "src" / "referentiel"))

import llm_client  # noqa: E402
import red_flag_detector as rfd  # noqa: E402
import extraction_etape1 as e1  # noqa: E402
import labellisation_etape3 as e3  # noqa: E402
import contre_validation_etape4 as e4  # noqa: E402
import orchestrer_pipeline as orch  # noqa: E402
import construire_referentiel_depuis_grille as cg  # noqa: E402
import exporter_relecture as exp  # noqa: E402


@pytest.fixture(scope="module")
def ref():
    return rfd.charger_referentiel(str(RACINE / "src" / "referentiel" / "french_referentiel.json"))


def motif_de(ref, motif_id):
    return next(m for m in ref["motifs"] if m["id"] == motif_id)


# ----------------------------------------------------------------------------
# Référentiel
# ----------------------------------------------------------------------------

def test_referentiel_coherent(ref):
    ids = [m["id"] for m in ref["motifs"]]
    assert len(ids) == len(set(ids)), "identifiants de motifs dupliqués"
    assert len(ids) == 114
    for m in ref["motifs"]:
        assert m["tri_base"] in rfd.ORDRE_URGENCE
        for mod in m["modulateurs"]:
            assert mod["tri"] in rfd.ORDRE_URGENCE
            assert mod["critere"].strip()


def test_referentiel_motif_ajoute_et_grossesse(ref):
    m = motif_de(ref, "dyspnee_insuffisance_cardiaque")
    assert m["tri_base"] == "3B" and len(m["modulateurs"]) == 2
    assert [x["tri"] for x in m["modulateurs"]] == ["1", "2"]
    assert motif_de(ref, "problemes_de_grossesse_3e_trimestre")["tri_base"] == "3A"
    assert motif_de(ref, "problemes_de_grossesse_1er_et_2e_trimestres")["tri_base"] == "3A"
    assert "validation" in ref["meta"] and "NON VALIDÉ" in ref["meta"]["validation"]["statut"]


def test_prompts_contiennent_la_liste_groupee(ref):
    p1 = e1.construire_prompt_etape1a(ref)
    p3 = e3.construire_prompt_systeme(ref)
    for p in (p1, p3):
        assert "## Cardio-circulatoire" in p
        assert "dyspnee_insuffisance_cardiaque : Dyspnée/insuffisance cardiaque" in p
        assert "{vocabulaire}" not in p
    assert '"motif_suggere"' in p3 and '"citation"' in e4.PROMPT_SYSTEME
    # tous les motifs sont dans la liste
    for m in ref["motifs"]:
        assert f"- {m['id']} :" in p1


def test_temperature_zero_par_defaut():
    assert inspect.signature(llm_client.completer).parameters["temperature"].default == 0.0


# ----------------------------------------------------------------------------
# Étape 2 : planchers
# ----------------------------------------------------------------------------

def test_plancher_constantes_pas_basse(ref):
    r = rfd.detecter_red_flags(rfd.CasClinique(pas_mmhg=65), ref)
    assert r.tri_constantes == "1" and r.categorie_constantes == "urgence_maximale"


def test_ecg_typique_reste_urgence_maximale(ref):
    cas = rfd.CasClinique(motif_id="douleur_thoracique_sca", age_annees=68, criteres_presents=["ECG anormal: typique de SCA"])
    r = rfd.detecter_red_flags(cas, ref)
    assert r.tri_motif == "1" and r.categorie_motif == "urgence_maximale" and r.tri_constantes is None


def test_modulateur_a_la_baisse_est_applique_et_trace(ref):
    critere = next(x["critere"] for x in motif_de(ref, "hypotension_arterielle")["modulateurs"] if x["tri"] == "3B")
    cas = rfd.CasClinique(motif_id="hypotension_arterielle", criteres_presents=[critere])
    r = rfd.detecter_red_flags(cas, ref)
    assert r.tri_motif == "3B" and r.categorie_motif == "moderee"
    assert len(r.modulateurs_a_la_baisse) == 1 and "à valider par un urgentiste" in r.modulateurs_a_la_baisse[0]
    # sans modulateur déclenché : tri de base
    r2 = rfd.detecter_red_flags(rfd.CasClinique(motif_id="hypotension_arterielle"), ref)
    assert r2.tri_motif == "2" and r2.modulateurs_a_la_baisse == []


def test_controle_de_presence_constante_inventee(ref):
    cas = rfd.CasClinique(fc_min=140)
    sans_valeur = rfd.detecter_red_flags(cas, ref, texte_source="Patient de 50 ans, douleur thoracique depuis ce matin.")
    assert sans_valeur.tri_constantes is None and sans_valeur.constantes_non_verifiees == ["fc_min=140"]
    avec_valeur = rfd.detecter_red_flags(cas, ref, texte_source="Patient tachycarde, pouls à 140/min.")
    assert avec_valeur.tri_constantes == "2" and avec_valeur.constantes_non_verifiees == []
    # sans texte fourni : ancien comportement (pas de contrôle)
    assert rfd.detecter_red_flags(cas, ref).tri_constantes == "2"


def test_controle_de_presence_cmhg_et_conversion_du_llm(ref):
    # valeur convertie par le code : la valeur brute (6,5) est dans le texte
    cas = rfd.CasClinique(pas_mmhg=65, valeurs_brutes={"pas_mmhg": 6.5})
    r = rfd.detecter_red_flags(cas, ref, texte_source="Elle est pâle, PA à 6,5/4 cm Hg.")
    assert r.tri_constantes == "1" and r.constantes_non_verifiees == []
    # le LLM a converti lui-même (65 pour "6,5") : toléré
    r2 = rfd.detecter_red_flags(rfd.CasClinique(pas_mmhg=65), ref, texte_source="PA à 6,5/4 cm Hg.")
    assert r2.tri_constantes == "1"
    # valeur vraiment absente
    r3 = rfd.detecter_red_flags(rfd.CasClinique(pas_mmhg=65), ref, texte_source="Pas de constantes notées.")
    assert r3.tri_constantes is None and r3.constantes_non_verifiees


def test_glycemie_signalee_seulement_au_dessus_de_20(ref):
    bas = rfd.detecter_red_flags(rfd.CasClinique(glycemie_mmol_l=8), ref)
    haut = rfd.detecter_red_flags(rfd.CasClinique(glycemie_mmol_l=25), ref)
    assert not any("Glycémie" in f for f in bas.flags_declenches)
    assert any("Glycémie > 20" in f for f in haut.flags_declenches) and haut.tri_constantes is None


def test_plancher_constantes_inviolable_plancher_motif_revisable(ref):
    # constantes : le LLM ne peut pas descendre en dessous
    r = rfd.detecter_red_flags(rfd.CasClinique(pas_mmhg=60), ref, texte_source="PA 60/30")
    d = rfd.appliquer_override("differee", r)
    assert d["label_final"] == "urgence_maximale" and d["override_applique"] is True
    # motif : le LLM peut descendre, le cas est signalé
    r2 = rfd.detecter_red_flags(rfd.CasClinique(motif_id="deficit_neurologique_avc"), ref)
    assert r2.categorie_motif == "urgence_maximale"
    d2 = rfd.appliquer_override("moderee", r2)
    assert d2["label_final"] == "moderee" and d2["override_applique"] is False and d2["sous_plancher_motif"] is True
    # le LLM peut monter au-dessus des deux planchers
    d3 = rfd.appliquer_override("urgence_maximale", r2)
    assert d3["label_final"] == "urgence_maximale" and d3["sous_plancher_motif"] is False


def test_ancien_format_de_resultat_reste_compatible():
    ancien = rfd.ResultatDetection("2", "urgence_maximale", ["x"])
    d = rfd.appliquer_override("moderee", ancien)
    assert d["label_final"] == "urgence_maximale" and d["override_applique"] is True


# ----------------------------------------------------------------------------
# Faux LLM
# ----------------------------------------------------------------------------

class FauxLLM:
    """Répond selon l'étape (reconnue au prompt) et la vignette (reconnue par un marqueur)."""

    def __init__(self, scenario):
        self.scenario = scenario
        self.appels = []

    def __call__(self, system_prompt, user_content, fournisseur, modele, max_tokens=400, max_tentatives=5, temperature=0.0):
        if "Motifs disponibles, regroupés par section" in system_prompt:
            etape = "1A"
        elif "Critères possibles" in system_prompt:
            etape = "1B"
        elif "auditeur clinique" in system_prompt:
            etape = "4"
        elif "classification de triage" in system_prompt:
            etape = "3"
        else:
            raise AssertionError("étape non reconnue")
        self.appels.append((etape, fournisseur, modele, temperature))
        for marqueur, reponses in self.scenario.items():
            if marqueur in user_content:
                rep = reponses.get(etape, {"criteres_presents": []} if etape == "1B" else None)
                assert rep is not None, f"réponse manquante pour l'étape {etape}"
                if callable(rep):
                    rep = rep()
                return copy.deepcopy(rep)
        raise AssertionError(f"aucun scénario pour l'étape {etape}")


@pytest.fixture
def faux(monkeypatch):
    def installer(scenario):
        f = FauxLLM(scenario)
        for module in (e1, e3, e4):
            monkeypatch.setattr(module, "completer", f)
        return f
    return installer


def extraction_a(**champs):
    base = {"motif_id": None, "age_annees": 40, "pas": None, "pas_unite": None, "fc_min": None, "spo2_pct": None,
            "fr_min": None, "glycemie": None, "glycemie_unite": None, "gcs": None, "fievre": "non_mentionne"}
    base.update(champs)
    return base


# ----------------------------------------------------------------------------
# Étape 1
# ----------------------------------------------------------------------------

def test_etape1_motif_inconnu_rejete(ref, faux):
    faux({"VIGN": {"1A": extraction_a(motif_id="motif_invente")}})
    r = e1.extraire_deux_etapes(ref, "VIGN", "anthropic", "m")
    assert r["motif_id"] is None and r["motif_rejete"] == "motif_invente"
    assert any("motif inconnu rejeté" in a for a in r["avertissements"])


def test_etape1_conversion_des_unites_par_le_code(ref, faux):
    f = faux({"A": {"1A": extraction_a(pas=16.5, pas_unite="cmHg")},
              "B": {"1A": extraction_a(pas=12, pas_unite=None)},
              "C": {"1A": extraction_a(pas=120, pas_unite="mmHg", glycemie=1.8, glycemie_unite="g/l")}})
    a = e1.extraire_deux_etapes(ref, "A", "anthropic", "m")
    assert a["pas_mmhg"] == 165 and a["conversions"][0]["brut"] == 16.5 and a["conversions"][0]["champ"] == "pas_mmhg"
    b = e1.extraire_deux_etapes(ref, "B", "anthropic", "m")
    assert b["pas_mmhg"] == 120 and "< 30" in b["conversions"][0]["raison"]
    c = e1.extraire_deux_etapes(ref, "C", "anthropic", "m")
    assert c["pas_mmhg"] == 120 and c["glycemie_mmol_l"] == 10.0
    assert [x["champ"] for x in c["conversions"]] == ["glycemie_mmol_l"]


def test_etape1_fievre_a_trois_valeurs(ref, faux):
    faux({"A": {"1A": extraction_a(fievre="oui")}, "B": {"1A": extraction_a(fievre="non_mentionne")},
          "C": {"1A": extraction_a(fievre="non")}, "D": {"1A": extraction_a(fievre="peut-être")}})
    res = {k: e1.extraire_deux_etapes(ref, k, "anthropic", "m") for k in "ABCD"}
    assert (res["A"]["fievre"], res["A"]["fievre_statut"]) == (True, "oui")
    assert (res["B"]["fievre"], res["B"]["fievre_statut"]) == (False, "non_mentionne")
    assert (res["C"]["fievre"], res["C"]["fievre_statut"]) == (False, "non")
    assert res["D"]["fievre_statut"] == "non_mentionne"


def test_etape1_echec_appel_b_est_trace(ref, faux):
    faux({"VIGN_A": {"1A": extraction_a(motif_id="douleur_thoracique_sca"), "1B": {"_erreur_parsing": "bla"}}})
    a = e1.extraire_deux_etapes(ref, "VIGN_A douleur thoracique", "anthropic", "m")
    assert a["appel_b_echec"] is True and a["criteres_presents"] == [] and any("Appel B" in x for x in a["avertissements"])


def test_etape1_appel_b_exige_une_citation_litterale_de_la_vignette(ref, faux):
    c0, c1, c2 = [x["critere"] for x in motif_de(ref, "douleur_thoracique_sca")["modulateurs"][:3]]
    texte = "VIGN_C Patient de 50 ans, douleur thoracique avec ECG sus-décalage du segment ST depuis une heure."
    faux({"VIGN_C": {"1A": extraction_a(motif_id="douleur_thoracique_sca"), "1B": {"criteres_presents": [
        {"critere": c0, "citation": "ECG sus-décalage du segment ST"},          # citation présente : retenu
        {"critere": c1, "citation": "ECG franchement anormal et atypique"},      # citation absente de la vignette : rejeté
        {"critere": c2},                                                         # aucune citation : rejeté
        c0,                                                                      # doublon sans citation : ignoré
        {"critere": "critère inventé par le LLM", "citation": "douleur thoracique avec ECG"},   # inconnu : ignoré
    ]}}})
    r = e1.extraire_deux_etapes(ref, texte, "anthropic", "m")
    assert r["criteres_presents"] == [c0] and r["criteres_citations"] == {c0: "ECG sus-décalage du segment ST"}
    assert [(x["critere"], x["raison"]) for x in r["criteres_rejetes"]] == [(c1, "citation absente de la vignette"), (c2, "aucune citation fournie")]
    assert any("critère inconnu" in x for x in r["avertissements"]) and any("2 critère(s)" in x for x in r["avertissements"])


def test_etape1_citation_trop_courte_refusee_et_ancien_format_sans_citation_refuse(ref, faux):
    c0 = motif_de(ref, "douleur_thoracique_sca")["modulateurs"][0]["critere"]
    texte = "VIGN_D douleur thoracique avec ECG sus-décalage du segment ST."
    faux({"VIGN_D": {"1A": extraction_a(motif_id="douleur_thoracique_sca"),
                     "1B": {"criteres_presents": [{"critere": c0, "citation": "ECG sus"}]}}})
    r = e1.extraire_deux_etapes(ref, texte, "anthropic", "m")
    assert r["criteres_presents"] == [] and r["criteres_rejetes"][0]["raison"] == "citation absente de la vignette"
    faux({"VIGN_E": {"1A": extraction_a(motif_id="douleur_thoracique_sca"), "1B": {"criteres_presents": [c0]}}})
    r2 = e1.extraire_deux_etapes(ref, "VIGN_E texte quelconque", "anthropic", "m")
    assert r2["criteres_presents"] == [] and r2["criteres_rejetes"][0]["raison"] == "aucune citation fournie"


def test_verifier_citation_seuils_reduits_pour_les_criteres():
    assert e4.verifier_citation("amaigrissement de 2 kg", VIGN, min_mots=3, min_mots_fragment=2)
    assert e4.verifier_citation("amaigrissement de ... 15 jours", VIGN, min_mots=3, min_mots_fragment=2)
    assert not e4.verifier_citation("de 2", VIGN, min_mots=3, min_mots_fragment=2)          # 2 mots seulement au total
    assert e4.verifier_citation("amaigrissement de 2 kg", VIGN)                              # 4 mots : accepté même avec les seuils par défaut



def test_etape1_controle_age_motif(ref, faux):
    faux({"A": {"1A": extraction_a(motif_id="fievre_3_mois", age_annees=30)},
          "B": {"1A": extraction_a(motif_id="fievre_3_mois", age_annees=0.2)}})
    a = e1.extraire_deux_etapes(ref, "A", "anthropic", "m")
    b = e1.extraire_deux_etapes(ref, "B", "anthropic", "m")
    assert any("incompatible" in x for x in a["avertissements"]) and b["avertissements"] == []


def test_etape1_erreur_de_lecture_est_renvoyee(ref, faux):
    faux({"A": {"1A": {"_erreur_parsing": "réponse illisible"}}})
    assert "_erreur_parsing" in e1.extraire_deux_etapes(ref, "A", "anthropic", "m")


# ----------------------------------------------------------------------------
# Étape 4 : citation et signalement
# ----------------------------------------------------------------------------

VIGN = "Femme de 50 ans consulte pour une altération de l'état général, avec asthénie et amaigrissement de 2 kg en 15 jours."


def test_verifier_citation():
    assert e4.verifier_citation("altération de l'état général, avec asthénie", VIGN)
    assert e4.verifier_citation("ALTERATION DE L'ETAT GENERAL avec asthenie", VIGN)          # casse, accents, ponctuation
    assert e4.verifier_citation("amaigrissement de 2 kg ... en 15 jours", VIGN)               # fragments
    assert not e4.verifier_citation("risque de septicémie sévère", VIGN)                      # absente
    assert not e4.verifier_citation("asthénie", VIGN)                                         # trop courte
    assert not e4.verifier_citation(None, VIGN) and not e4.verifier_citation("", VIGN)


def test_etape4_signale_sans_modifier_le_label():
    v = {"risque_sous_triage": True, "niveau_suggere": "urgence_maximale",
         "citation": "amaigrissement de 2 kg en 15 jours", "justification": "x"}
    d = e4.resoudre_contre_validation("moderee", v, VIGN)
    assert d["categorie_finale"] == "moderee" and d["contre_validation_a_modifie"] is False
    assert d["signal_sous_triage"] is True and d["citation_verifiee"] is True and d["priorite_relecture"] == "haute"
    v2 = dict(v, citation="fièvre élevée avec frissons")
    d2 = e4.resoudre_contre_validation("moderee", v2, VIGN)
    assert d2["citation_verifiee"] is False and d2["priorite_relecture"] == "basse" and d2["categorie_finale"] == "moderee"


def test_etape4_suggestion_incoherente_ou_sans_risque():
    assert e4.resoudre_contre_validation("moderee", {"risque_sous_triage": False}, VIGN)["signal_sous_triage"] is False
    d = e4.resoudre_contre_validation("moderee", {"risque_sous_triage": True, "niveau_suggere": "differee", "citation": "x y z w"}, VIGN)
    assert d["signal_sous_triage"] is False and d["_incoherence_auditeur"] is True
    d2 = e4.resoudre_contre_validation("moderee", {"risque_sous_triage": True, "niveau_suggere": None}, VIGN)
    assert d2["signal_sous_triage"] is False


# ----------------------------------------------------------------------------
# Pipeline complet (faux LLM)
# ----------------------------------------------------------------------------

def scenario_pipeline(ref):
    critere_ecg = motif_de(ref, "douleur_thoracique_sca")["modulateurs"][0]["critere"]
    e3_ok = lambda cat, motif, **k: {"categorie": cat, "justification": "j", "motif_suggere": motif,
                                       "extraction_incoherente": False, "detail_extraction": None, **k}
    e4_rien = {"risque_sous_triage": False, "niveau_suggere": None, "citation": None, "justification": None,
               "extraction_incoherente": False, "detail_extraction": None}
    return {
        "V1_SCA": {"1A": extraction_a(motif_id="douleur_thoracique_sca", age_annees=58),
                   "1B": {"criteres_presents": [{"critere": critere_ecg, "citation": "ECG avec sus-décalage du segment ST"}]},
                   "3": e3_ok("urgence_maximale", "douleur_thoracique_sca"), "4": e4_rien},
        "V2_AEG": {"1A": extraction_a(motif_id="aeg_asthenie", age_annees=50), "3": e3_ok("moderee", "aeg_asthenie"),
                   "4": {**e4_rien, "risque_sous_triage": True, "niveau_suggere": "urgence_maximale",
                         "citation": "amaigrissement de 2 kg en 15 jours", "justification": "scénario"}},
        "V3_PAS": {"1A": extraction_a(motif_id="malaise", pas=6.5, pas_unite="cmHg"), "3": e3_ok("differee", "malaise"), "4": e4_rien},
        "V4_MOTIF": {"1A": extraction_a(motif_id="douleur_abdominale"), "3": e3_ok("moderee", "vomissements_sans_autre_cas"), "4": e4_rien},
        "V5_FC": {"1A": extraction_a(motif_id="malaise", fc_min=150), "3": e3_ok("moderee", "malaise"), "4": e4_rien},
        "V6_AVC": {"1A": extraction_a(motif_id="deficit_neurologique_avc"), "1B": {"criteres_presents": []},
                   "3": e3_ok("moderee", "deficit_neurologique_avc"), "4": e4_rien},
        "V7_NONVERIF": {"1A": extraction_a(motif_id="aeg_asthenie"), "3": e3_ok("moderee", "aeg_asthenie"),
                        "4": {**e4_rien, "risque_sous_triage": True, "niveau_suggere": "urgence_maximale",
                              "citation": "phrase qui ne figure pas dans la vignette", "justification": "x"}},
        "V8_E3KO": {"1A": extraction_a(motif_id="malaise"), "3": {"_erreur_parsing": "illisible"}, "4": e4_rien},
        # motif contesté : l'Étape 3 propose un motif au tri médian « maximale », le label est « modérée »
        "V9_CONF": {"1A": extraction_a(motif_id="malaise"), "3": e3_ok("moderee", "alteration_conscience_coma"), "4": e4_rien},
        # label « différée » sous un plancher « maximale » : écart de 2 catégories
        "V10_ECART2": {"1A": extraction_a(motif_id="deficit_neurologique_avc"), "3": e3_ok("differee", "deficit_neurologique_avc"), "4": e4_rien},
        # critère coché sans citation vérifiable : rejeté, pas de plancher indu
        "V11_REJET": {"1A": extraction_a(motif_id="douleur_abdominale"), "3": e3_ok("moderee", "douleur_abdominale"), "4": e4_rien,
                      "1B": {"criteres_presents": [{"critere": "Douleur sévère et/ou mauvaise tolérance", "citation": "douleur atroce depuis hier"}]}},
    }


VIGNETTES = {
    "V1_SCA": "Homme de 58 ans, douleur thoracique constrictive depuis 40 minutes, ECG avec sus-décalage du segment ST. V1_SCA",
    "V2_AEG": VIGN + " V2_AEG",
    "V3_PAS": "Patiente confuse, pâle, PA à 6,5/4 cm Hg. V3_PAS",
    "V4_MOTIF": "Adulte, douleur abdominale et vomissements depuis hier. V4_MOTIF",
    "V5_FC": "Malaise bref à domicile, sans autre précision. V5_FC",
    "V6_AVC": "Patient de 70 ans, gêne à la parole depuis ce matin. V6_AVC",
    "V7_NONVERIF": VIGN + " V7_NONVERIF",
    "V8_E3KO": "Malaise. V8_E3KO",
    "V9_CONF": "Adulte confus après un malaise. V9_CONF",
    "V10_ECART2": "Patient de 70 ans, gêne à la parole depuis ce matin. V10_ECART2",
    "V11_REJET": "Adulte, gêne abdominale modérée depuis hier. V11_REJET",
}


def config_test(ref, **kw):
    args = argparse.Namespace(tout_haiku=kw.get("tout_haiku", False), pause_interne=0,
                              etape1_fournisseur=None, etape1_modele=None, etape3_fournisseur=None,
                              etape3_modele=None, etape4_fournisseur=None, etape4_modele=None)
    return orch.construire_config(args, ref)


@pytest.fixture
def traces(ref, faux, monkeypatch):
    monkeypatch.setattr(orch.time, "sleep", lambda s: None)
    f = faux(scenario_pipeline(ref))
    cfg = config_test(ref)
    sortie = {}
    for k, texte in VIGNETTES.items():
        vign = {"id": f"id_{k}", "cas_id": f"cas_{k}", "contexte_clinique": texte}
        sortie[k] = orch.traiter_une_vignette(vign, ref, cfg)
    sortie["_faux"] = f
    return sortie


def test_pipeline_nominal_et_tracabilite(traces):
    t = traces["V1_SCA"]
    assert t["categorie_finale"] == "urgence_maximale" and t["a_relire"] is False and t["raisons_relecture"] == []
    assert t["cas_id"] == "cas_V1_SCA"
    p = t["provenance"]
    assert p["version"] == "lot1" and p["temperature"] == 0 and set(["etape1", "etape3", "etape4"]) <= set(p)
    assert t["etape1_extraction"]["fievre_statut"] == "non_mentionne"


def test_pipeline_fournisseurs_par_etape(traces):
    appels = traces["_faux"].appels
    par_etape = {}
    for etape, fournisseur, modele, temperature in appels:
        par_etape.setdefault(etape[0], set()).add(fournisseur)
        assert temperature == 0.0
    assert par_etape["1"] == {"anthropic"} and par_etape["3"] == {"anthropic"} and par_etape["4"] == {"groq"}


def test_pipeline_etape4_signale_sans_changer_le_label_et_sans_priorite_haute(traces):
    t = traces["V2_AEG"]
    assert t["categorie_finale"] == "moderee"
    assert t["decision_finale"]["contre_validation_a_modifie"] is False and t["decision_finale"]["signal_sous_triage"] is True
    # règle du lot 1 ter : un signal de l'Étape 4 seul n'est PAS une relecture prioritaire (label déjà au niveau des planchers)
    assert t["a_relire"] is False and t["priorite_relecture"] == "basse" and t["score_relecture"] == 0
    assert any("[basse] etape4_signal_sous_triage" in r for r in t["raisons_relecture"])


def test_pipeline_citation_non_verifiee_sort_de_la_file_prioritaire(traces):
    t = traces["V7_NONVERIF"]
    assert t["categorie_finale"] == "moderee" and t["a_relire"] is False and t["priorite_relecture"] == "basse"
    assert any("etape4_signal_non_verifie" in r for r in t["raisons_relecture"])


def test_pipeline_plancher_de_constantes_inviolable(traces):
    t = traces["V3_PAS"]
    assert t["etape1_extraction"]["pas_mmhg"] == 65 and t["etape1_extraction"]["conversions"][0]["brut"] == 6.5
    assert t["decision_apres_override_etape2_3"]["override_applique"] is True
    assert t["categorie_finale"] == "urgence_maximale"        # le LLM disait "differee"
    assert t["decision_apres_override_etape2_3"]["label_llm_initial"] == "differee"


def test_pipeline_constante_inventee_ignoree_et_signalee(traces):
    t = traces["V5_FC"]
    assert t["decision_apres_override_etape2_3"]["override_applique"] is False
    assert t["etape2_resultat"]["constantes_non_verifiees"] == ["fc_min=150"]
    assert t["categorie_finale"] == "moderee" and t["a_relire"] is True
    assert any("constante_non_retrouvee" in r for r in t["raisons_relecture"])


def test_pipeline_motif_conteste_sans_risque_est_en_priorite_basse(traces):
    t = traces["V4_MOTIF"]
    assert t["motif_conteste"] is True and t["a_relire"] is False and t["priorite_relecture"] == "basse"
    assert any("[basse] motif_conteste" in r for r in t["raisons_relecture"])
    assert traces["V1_SCA"]["motif_conteste"] is False


def test_pipeline_motif_conteste_sous_le_tri_median_du_motif_de_letape_3_est_prioritaire(traces):
    t = traces["V9_CONF"]
    assert t["motif_conteste"] is True and t["a_relire"] is True and t["categorie_finale"] == "moderee"
    assert any("[haute] motif_conteste_sous_plancher" in r for r in t["raisons_relecture"])
    assert t["score_relecture"] == 70


def test_pipeline_le_score_classe_les_ecarts_au_plancher(traces):
    assert traces["V10_ECART2"]["ecart_plancher"] == 2 and traces["V10_ECART2"]["score_relecture"] == 120
    assert traces["V6_AVC"]["ecart_plancher"] == 1 and traces["V6_AVC"]["score_relecture"] == 110
    ordre = sorted((k for k in traces if k != "_faux" and traces[k].get("a_relire")), key=lambda k: -traces[k]["score_relecture"])
    assert ordre[:4] == ["V10_ECART2", "V6_AVC", "V9_CONF", "V5_FC"]   # écart 2 > écart 1 > motif contesté > constante non retrouvée


def test_pipeline_critere_rejete_ne_cree_pas_de_plancher_indu(traces):
    t = traces["V11_REJET"]
    assert t["etape1_extraction"]["criteres_presents"] == [] and t["etape1_extraction"]["criteres_rejetes"]
    assert t["decision_apres_override_etape2_3"]["sous_plancher_motif"] is False
    assert t["a_relire"] is False and any("[basse] criteres_rejetes" in r for r in t["raisons_relecture"])


def test_pipeline_sous_plancher_du_motif(traces):
    t = traces["V6_AVC"]
    d = t["decision_apres_override_etape2_3"]
    assert d["categorie_plancher_motif"] == "urgence_maximale" and d["sous_plancher_motif"] is True
    assert t["categorie_finale"] == "moderee" and t["a_relire"] is True


def test_pipeline_echec_etape3(traces):
    t = traces["V8_E3KO"]
    assert t["_echec"] == "etape3" and "categorie_finale" not in t


def test_pipeline_ne_montre_pas_les_champs_internes_aux_llm(ref):
    contenu = e3.construire_contenu({"motif_id": "malaise", "conversions": [{"x": 1}], "avertissements": ["a"],
                                      "fievre_statut": "non_mentionne"}, "VIGNETTE")
    assert "conversions" not in contenu and "avertissements" not in contenu and "non_mentionne" in contenu


# ----------------------------------------------------------------------------
# Configuration des fournisseurs
# ----------------------------------------------------------------------------

def test_config_par_defaut_et_tout_haiku(ref):
    cfg = config_test(ref)
    assert (cfg["etape1_fournisseur"], cfg["etape3_fournisseur"], cfg["etape4_fournisseur"]) == ("anthropic", "anthropic", "groq")
    haiku = config_test(ref, tout_haiku=True)
    assert {haiku[f"etape{n}_fournisseur"] for n in (1, 3, 4)} == {"anthropic"}
    assert {haiku[f"etape{n}_modele"] for n in (1, 3, 4)} == {orch.MODELE_HAIKU}


def test_fournisseur_impose_sans_modele_est_refuse(ref):
    args = argparse.Namespace(tout_haiku=False, pause_interne=0, etape1_fournisseur=None, etape1_modele=None,
                              etape3_fournisseur="gemini", etape3_modele=None, etape4_fournisseur=None, etape4_modele=None)
    with pytest.raises(SystemExit):
        orch.construire_config(args, ref)


# ----------------------------------------------------------------------------
# Appel Anthropic : température envoyée sans dépendre de la signature du SDK
# ----------------------------------------------------------------------------

def _httpx():
    """La bibliothèque HTTP du SDK Anthropic : 'httpx2' dans les versions récentes, 'httpx' avant."""
    import importlib
    for nom in ("httpx2", "httpx"):
        try:
            return importlib.import_module(nom)
        except ImportError:
            continue
    pytest.skip("bibliothèque HTTP du SDK Anthropic introuvable")


def _faux_client_anthropic(monkeypatch, gestionnaire):
    """Un vrai client Anthropic dont le réseau est remplacé par un faux serveur (aucun appel réel)."""
    anthropic = pytest.importorskip("anthropic")
    httpx = _httpx()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "cle-de-test")
    transport = httpx.MockTransport(gestionnaire)
    vraie_classe = anthropic.Anthropic   # gardée avant le remplacement (sinon la fabrique s'appellerait elle-même)
    fabrique = lambda api_key=None, **kw: vraie_classe(api_key="cle-de-test", http_client=httpx.Client(transport=transport),
                                                       max_retries=0)
    monkeypatch.setattr(anthropic, "Anthropic", fabrique)


def _reponse_message(texte):
    httpx = _httpx()
    return httpx.Response(200, json={
        "id": "msg_test", "type": "message", "role": "assistant", "model": "test", "stop_reason": "end_turn",
        "stop_sequence": None, "content": [{"type": "text", "text": texte}],
        "usage": {"input_tokens": 1, "output_tokens": 1}})


def test_anthropic_envoie_la_temperature_sans_typeerror(monkeypatch):
    vus = []

    def gestionnaire(request):
        vus.append(json.loads(request.content))
        return _reponse_message('{"ok": true}')

    _faux_client_anthropic(monkeypatch, gestionnaire)
    assert llm_client.completer("sys", "user", "anthropic", "claude-haiku-4-5-20251001", max_tokens=50) == {"ok": True}
    assert vus[0]["temperature"] == 0.0 and vus[0]["max_tokens"] == 50 and vus[0]["system"] == "sys"


def test_anthropic_reessaie_sans_temperature_si_le_modele_la_refuse(monkeypatch, capsys):
    pytest.importorskip("anthropic")
    httpx = _httpx()
    vus = []

    def gestionnaire(request):
        corps = json.loads(request.content)
        vus.append(corps)
        if "temperature" in corps:
            return httpx.Response(400, json={"type": "error", "error": {
                "type": "invalid_request_error", "message": "`temperature` is not supported for this model"}})
        return _reponse_message('{"ok": true}')

    _faux_client_anthropic(monkeypatch, gestionnaire)
    assert llm_client.completer("sys", "user", "anthropic", "un-modele-recent") == {"ok": True}
    assert len(vus) == 2 and "temperature" in vus[0] and "temperature" not in vus[1]
    assert "refuse le paramètre temperature" in capsys.readouterr().err


# ----------------------------------------------------------------------------
# Référentiel reconstruit depuis la grille FRENCH V1.1
# ----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def grille():
    chemin = RACINE / "src" / "referentiel" / "french_grille_v1_1_transcription.json"
    return json.load(open(chemin, encoding="utf-8"))


def test_referentiel_est_exactement_celui_que_produit_la_grille(ref, grille):
    """Le fichier est reproductible : le reconstruire depuis la transcription de la grille ne change rien."""
    assert len(grille["motifs"]) == 115
    assert cg.reconstruire(ref, grille)["motifs"] == ref["motifs"]


def test_niveaux_des_criteres_cles_selon_la_grille(ref):
    def niveau(motif_id, debut):
        return next(x["tri"] for x in motif_de(ref, motif_id)["modulateurs"] if x["critere"].lower().startswith(debut.lower()))
    assert niveau("douleur_thoracique_sca", "ECG anormal: typique") == "1"
    assert niveau("douleur_thoracique_sca", "ECG anormal: non typique") == "2"
    assert niveau("douleur_thoracique_sca", "ECG normal et douleur atypique") == "4"
    assert niveau("cephalee", "Inhabituelle") == "2"
    assert niveau("douleur_rachidienne_cervicale_dorsale_lombaire", "Déficit sensitif ou moteur") == "2"
    assert niveau("trouble_visuel_il_douloureux_cecite", "Début brutal") == "2"
    assert niveau("douleur_abdominale", "Douleur sévère") == "2"
    assert niveau("hemoptysie", "Hémoptysie répétée") == "2"
    assert niveau("detresse_respiratoire_dyspnee", "Détresse respiratoire") == "1"
    assert niveau("deficit_neurologique_avc", "Délai ≤ 4h") == "1"


def test_asthenie_ne_peut_jamais_atteindre_le_tri_1_ou_2(ref):
    m = motif_de(ref, "aeg_asthenie")
    assert m["tri_base"] == "3B" and all(x["tri"] in ("3B", "5") for x in m["modulateurs"])


def test_cellules_avis_referent_omises(ref):
    for m in ref["motifs"]:
        for x in m["modulateurs"]:
            assert x["critere"].strip().lower() != "avis référent (mao, mco)"


def test_deficit_moteur_dans_une_douleur_rachidienne_donne_un_plancher_de_motif_maximal(ref):
    """Cas type 18765 : sciatique avec déficit moteur, classée 'modérée' par le LLM -> le cas doit être signalé."""
    critere = "Déficit sensitif ou moteur associé"
    cas = rfd.CasClinique(motif_id="douleur_rachidienne_cervicale_dorsale_lombaire", age_annees=49, criteres_presents=[critere])
    r = rfd.detecter_red_flags(cas, ref)
    assert r.tri_motif == "2" and r.categorie_motif == "urgence_maximale" and r.modulateurs_a_la_baisse == []
    d = rfd.appliquer_override("moderee", r)
    assert d["label_final"] == "moderee" and d["sous_plancher_motif"] is True


def test_modulateur_a_la_baisse_reel_de_la_grille(ref):
    cas = rfd.CasClinique(motif_id="douleur_abdominale", criteres_presents=["Douleur régressive / indolore"])
    r = rfd.detecter_red_flags(cas, ref)
    assert r.tri_motif == "5" and r.categorie_motif == "differee" and len(r.modulateurs_a_la_baisse) == 1


def test_prompt_appel_b_propose_les_criteres_reels(ref):
    p = e1.construire_prompt_etape1b(motif_de(ref, "douleur_rachidienne_cervicale_dorsale_lombaire"))
    assert "Déficit sensitif ou moteur associé" in p and "Fièvre ou paresthésies" in p


# ----------------------------------------------------------------------------
# Fournisseurs réglés par le fichier .env (variables d'environnement)
# ----------------------------------------------------------------------------

VARIABLES = [f"ETAPE{n}_{k}" for n in (1, 3, 4) for k in ("FOURNISSEUR", "MODELE")]


@pytest.fixture
def env_propre(monkeypatch):
    for v in VARIABLES:
        monkeypatch.delenv(v, raising=False)
    return monkeypatch


def _args(**kw):
    base = dict(tout_haiku=False, pause_interne=0, etape1_fournisseur=None, etape1_modele=None,
                etape3_fournisseur=None, etape3_modele=None, etape4_fournisseur=None, etape4_modele=None)
    base.update(kw)
    return argparse.Namespace(**base)


def test_env_change_le_modele_sans_toucher_au_code(ref, env_propre):
    env_propre.setenv("ETAPE3_FOURNISSEUR", "gemini")
    env_propre.setenv("ETAPE3_MODELE", "un-modele-gemini")
    cfg = orch.construire_config(_args(), ref)
    assert (cfg["etape3_fournisseur"], cfg["etape3_modele"]) == ("gemini", "un-modele-gemini")
    assert (cfg["etape1_fournisseur"], cfg["etape4_fournisseur"]) == ("anthropic", "groq")   # les autres étapes ne bougent pas
    assert cfg["provenance"]["etape3"]["fournisseur"] == "gemini"                            # tracé dans la provenance


def test_env_seul_le_modele(ref, env_propre):
    env_propre.setenv("ETAPE4_MODELE", "openai/un-autre-modele")
    cfg = orch.construire_config(_args(), ref)
    assert (cfg["etape4_fournisseur"], cfg["etape4_modele"]) == ("groq", "openai/un-autre-modele")


def test_la_ligne_de_commande_prime_sur_env(ref, env_propre):
    env_propre.setenv("ETAPE3_FOURNISSEUR", "gemini")
    env_propre.setenv("ETAPE3_MODELE", "un-modele-gemini")
    cfg = orch.construire_config(_args(etape3_fournisseur="anthropic", etape3_modele="claude-x"), ref)
    assert (cfg["etape3_fournisseur"], cfg["etape3_modele"]) == ("anthropic", "claude-x")


def test_tout_haiku_ignore_env(ref, env_propre):
    env_propre.setenv("ETAPE3_FOURNISSEUR", "gemini")
    env_propre.setenv("ETAPE3_MODELE", "un-modele-gemini")
    cfg = orch.construire_config(_args(tout_haiku=True), ref)
    assert {cfg[f"etape{n}_fournisseur"] for n in (1, 3, 4)} == {"anthropic"}


def test_env_fournisseur_sans_modele_est_refuse_sauf_anthropic(ref, env_propre):
    env_propre.setenv("ETAPE3_FOURNISSEUR", "gemini")
    with pytest.raises(SystemExit):
        orch.construire_config(_args(), ref)
    env_propre.setenv("ETAPE3_FOURNISSEUR", "anthropic")
    assert orch.construire_config(_args(), ref)["etape3_modele"] == orch.MODELE_HAIKU


def test_variable_vide_est_ignoree(ref, env_propre):
    env_propre.setenv("ETAPE3_FOURNISSEUR", "  ")
    env_propre.setenv("ETAPE3_MODELE", "")
    assert orch.construire_config(_args(), ref)["etape3_fournisseur"] == orch.DEFAUTS["etape3"][0]


def test_le_modele_du_env_nest_pas_reutilise_avec_un_autre_fournisseur_de_la_ligne_de_commande(ref, env_propre):
    env_propre.setenv("ETAPE3_FOURNISSEUR", "gemini")
    env_propre.setenv("ETAPE3_MODELE", "un-modele-gemini")
    cfg = orch.construire_config(_args(etape3_fournisseur="anthropic"), ref)
    assert (cfg["etape3_fournisseur"], cfg["etape3_modele"]) == ("anthropic", orch.MODELE_HAIKU)


# ----------------------------------------------------------------------------
# Export des cas à relire (budget limité, tri par priorité)
# ----------------------------------------------------------------------------

def _liste_traces(traces):
    return [t for k, t in traces.items() if k != "_faux"]


def test_export_relecture_trie_par_score_et_applique_le_budget(traces):
    lignes = exp.construire_lignes(_liste_traces(traces), n=2, inclure_basses=False)
    assert [l["id"] for l in lignes[:3]] == ["id_V10_ECART2", "id_V6_AVC", "id_V9_CONF"]
    assert [l["dans_le_budget"] for l in lignes[:3]] == ["oui", "oui", "non"]
    assert all(l["ma_decision"] == "" and l["mon_label"] == "" and l["ma_note"] == "" for l in lignes)
    assert all(l["id"] in {t["id"] for t in _liste_traces(traces) if t.get("a_relire")} for l in lignes)   # priorité haute seulement


def test_export_relecture_inclure_basses_ne_les_met_jamais_dans_le_budget(traces):
    lignes = exp.construire_lignes(_liste_traces(traces), n=99, inclure_basses=True)
    basses = [l for l in lignes if l["id"] in {"id_V2_AEG", "id_V7_NONVERIF", "id_V4_MOTIF", "id_V11_REJET"}]
    assert basses and all(l["dans_le_budget"] == "non" for l in basses)


def test_export_relecture_ecrit_un_csv_pour_excel(traces, tmp_path, monkeypatch):
    entree, sortie = tmp_path / "pipeline.jsonl", tmp_path / "sous_dossier" / "relecture.csv"
    entree.write_text("\n".join(json.dumps(t, ensure_ascii=False) for t in _liste_traces(traces)) + "\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["exporter_relecture.py", "--input", str(entree), "--output", str(sortie), "--n", "2"])
    exp.main()
    brut = sortie.read_bytes()
    assert brut.startswith(b"\xef\xbb\xbf")                                  # BOM utf-8 : Excel reconnaît les accents
    tete = brut.decode("utf-8-sig").splitlines()[0]
    assert tete.startswith("rang;dans_le_budget;id;cas_id;score") and tete.endswith("ma_decision;mon_label;ma_note")
    assert (tmp_path / "sous_dossier" / "relecture_non_relus.txt").read_text(encoding="utf-8").strip() != ""
