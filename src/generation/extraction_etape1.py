"""
extraction_etape1.py

Étape 1 du pipeline : extraction structurée du cas clinique à partir d'une
vignette source en texte libre, au format CasClinique (cf.
src/referentiel/red_flag_detector.py) — c'est cette structure qui alimente
ensuite l'Étape 2 (détection déterministe des red flags).

--- Ce qui a changé au lot 1 (sept. 2026) ---
- Prompt réécrit d'après le référentiel IOA 2020 de la SFMU (pages 21-22) :
    * le motif de recours est la PLAINTE à l'accueil, jamais un diagnostic ;
    * s'il y en a plusieurs, on retient la plus complexe / la plus sévère ;
    * on ne choisit pas le motif d'après le diagnostic final ni les examens ;
    * null si aucun motif de la liste ne correspond vraiment.
- Liste des motifs regroupée par section (17 sections), plus lisible pour le LLM.
- Les UNITÉS sont converties par le CODE, pas par le LLM : le LLM recopie le
  nombre tel qu'écrit avec son unité (cmHg / mmHg, g/l / mmol/l), le code
  convertit et garde la trace de la valeur brute (champ "conversions").
- Fièvre à trois valeurs : "oui" / "non" / "non_mentionne" (champ
  "fievre_statut") ; le champ booléen "fievre" ne vaut True que si "oui".
- Tout motif_id absent du référentiel est REJETÉ (motif_id = null + trace).
- Un critère absent de la liste des critères du motif est ignoré (+ trace).
- L'échec de l'Appel B (critères) est tracé (appel_b_echec) au lieu d'être silencieux.
- Appel B STRICT (lot 1 ter) : pour chaque critère coché, le LLM doit recopier MOT POUR MOT un
  passage de la vignette (>= 3 mots) ; le code vérifie que ce passage figure dans la vignette
  (verifier_citation). Un critère sans citation, ou dont la citation est absente de la vignette,
  est REJETÉ (champ "criteres_rejetes"). Motif : sur 19 vignettes, l'Appel B cochait « hémoptysie
  répétée ou abondante » alors que le texte disait seulement « consulte pour hémoptysie ».
  Limite : une citation littérale prouve que le passage existe, pas qu'il soutient le critère.
  Les citations retenues sont gardées dans "criteres_citations" (audit).
- Contrôle âge / motif : avertissements pour les incohérences évidentes
  (ex. motif pédiatrique pour un adulte).
- Le test golden et le mode réel utilisent maintenant la vraie fonction
  extraire_deux_etapes (avant : un ancien prompt en un seul appel).

Le prompt intègre DYNAMIQUEMENT le vocabulaire du référentiel FRENCH pour que
le LLM ne produise que des motif_id et criteres_presents réellement
exploitables par l'Étape 2.

Usage (mode test, contre le jeu de cas de référence) :
    python extraction_etape1.py --golden ../../tests/golden_etape1_extraction.json

Usage (sur de vraies vignettes) :
    python extraction_etape1.py --input ../../data/anonymized/mediqal_mcqm_urgences_complet_dedup.jsonl \
        --output ../../data/normalized/mediqal_mcqm_extraction.jsonl --limit 20
"""

import argparse
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "referentiel"))
from llm_client import completer  # noqa: E402
from red_flag_detector import charger_referentiel  # noqa: E402
from contre_validation_etape4 import plus_long_passage_litteral, verifier_citation_critere  # noqa: E402

FOURNISSEUR_DEFAUT = "anthropic"
MODELE_DEFAUT = "claude-haiku-4-5-20251001"

SECTIONS = {
    "cardio_circulatoire": "Cardio-circulatoire",
    "respiratoire": "Respiratoire",
    "neurologie": "Neurologie",
    "psychiatrie": "Psychiatrie",
    "abdominal": "Abdominal",
    "digestif": "Digestif",
    "genito_urinaire": "Génito-urinaire",
    "gyneco_obstetrique": "Gynéco-obstétrique",
    "traumatologie": "Traumatologie",
    "infectiologie": "Infectiologie",
    "intoxication": "Intoxication",
    "orl_stomatologie": "ORL / stomatologie",
    "ophtalmologie": "Ophtalmologie",
    "peau": "Peau",
    "rhumatologie": "Rhumatologie",
    "pediatrie": "Pédiatrie (pathologies spécifiques, enfants de 2 ans ou moins)",
    "divers": "Divers",
}


def construire_vocabulaire_motifs_leger(referentiel: dict) -> str:
    """Liste des motifs (id : libellé), REGROUPÉE PAR SECTION. Les critères
    détaillés ne sont envoyés qu'à l'Appel B, pour le motif déjà choisi.
    Utilisée aussi par l'Étape 3 (motif_suggere) : les deux étapes voient
    exactement la même liste."""
    par_section = {}
    for m in referentiel["motifs"]:
        par_section.setdefault(m["specialite"], []).append(m)
    blocs = []
    ordre = [s for s in SECTIONS if s in par_section] + [s for s in par_section if s not in SECTIONS]
    for section in ordre:
        titre = SECTIONS.get(section, section)
        lignes = [f"- {m['id']} : {m['libelle']}" for m in par_section[section]]
        blocs.append(f"## {titre}\n" + "\n".join(lignes))
    return "\n\n".join(blocs)


def construire_prompt_etape1a(referentiel: dict) -> str:
    vocabulaire = construire_vocabulaire_motifs_leger(referentiel)
    return f"""Tu es un assistant d'extraction de données cliniques structurées, pour un projet de triage médical aux urgences.

À partir du texte d'une vignette clinique, extrais UNIQUEMENT les informations explicitement présentes dans le texte, au format JSON suivant :

{{
  "motif_id": "un identifiant EXACT de la liste ci-dessous, ou null si aucun ne correspond vraiment",
  "age_annees": nombre ou null (décimal pour un nourrisson : 6 mois = 0.5),
  "pas": nombre ou null (pression artérielle SYSTOLIQUE, recopiée EXACTEMENT comme écrite, sans la convertir),
  "pas_unite": "mmHg" ou "cmHg" ou null,
  "fc_min": nombre ou null,
  "spo2_pct": nombre ou null,
  "fr_min": nombre ou null,
  "glycemie": nombre ou null (recopiée EXACTEMENT comme écrite, sans la convertir),
  "glycemie_unite": "mmol/l" ou "g/l" ou null,
  "gcs": nombre ou null,
  "fievre": "oui" ou "non" ou "non_mentionne"
}}

COMMENT CHOISIR LE MOTIF (règles du référentiel de l'infirmier organisateur de l'accueil, SFMU 2020) :
1. Le motif de recours est ce qui amène le patient aux urgences : sa PLAINTE principale, exprimée en symptôme (ex. essoufflement, douleur thoracique, fièvre, vomissements). Ce n'est JAMAIS un diagnostic.
2. S'il y a plusieurs plaintes, retiens celle qui semble la plus complexe ou la plus sévère.
3. Ne choisis PAS le motif d'après le diagnostic final, les résultats d'examens ou l'évolution sous traitement : base-toi sur ce que le patient présente à son arrivée. Si le texte ne décrit aucun symptôme d'arrivée, mets null.
4. Si aucun motif de la liste ne correspond vraiment, mets null. Ne choisis pas le motif "le moins mauvais" par défaut.

AUTRES RÈGLES IMPÉRATIVES :
- N'invente JAMAIS une donnée absente du texte. Si une information n'est pas mentionnée, mets null. Ne devine JAMAIS une constante vitale.
- Recopie les nombres tels qu'ils sont écrits et donne l'unité écrite ; ne fais AUCUNE conversion (le programme s'en charge).
- "fievre" vaut "oui" seulement si une fièvre ou une température élevée est écrite ; "non" seulement si le texte dit explicitement qu'il n'y a pas de fièvre ; sinon "non_mentionne" (ne pas en parler n'est PAS dire qu'il n'y en a pas).

Motifs disponibles, regroupés par section (choisis l'id EXACT, ou null) :
{vocabulaire}

Réponds UNIQUEMENT avec le JSON, sans texte avant ni après, sans balises markdown.
"""


def construire_prompt_etape1b(motif: dict) -> str:
    criteres = [mod["critere"] for mod in motif["modulateurs"]]
    criteres_txt = "\n".join(f"- {c}" for c in criteres)
    return f"""Tu es un assistant d'extraction clinique. Pour le motif "{motif['libelle']}", détermine lesquels des critères suivants sont EXPLICITEMENT vérifiés dans le texte de la vignette fournie.

Critères possibles :
{criteres_txt}

RÈGLES IMPÉRATIVES :
1. Ne coche un critère que si un passage du texte le confirme clairement. Ne devine jamais, n'interprète pas : dans le doute, ne coche pas.
2. Pour chaque critère coché, recopie MOT POUR MOT (entre 3 et 25 mots) le passage de la vignette qui le prouve. Ce passage doit contenir l'élément précis du critère (par exemple, pour un critère « abondante », un passage qui décrit l'abondance). Le seul fait que le motif soit mentionné ne prouve AUCUN critère.
3. Deux critères qui s'excluent (par exemple « ECG normal » et « ECG anormal ») ne peuvent pas être cochés ensemble.

Réponds UNIQUEMENT avec ce JSON, sans texte avant ni après :
{{"criteres_presents": [{{"critere": "critère recopié EXACTEMENT depuis la liste ci-dessus", "citation": "passage copié mot pour mot de la vignette"}}]}}
Si aucun critère n'est vérifié : {{"criteres_presents": []}}
"""


# --------------------------------------------------------------------------
# Conversions d'unités, faites par le CODE (le LLM ne convertit rien)
# --------------------------------------------------------------------------

def _nombre(valeur):
    """Convertit une valeur renvoyée par le LLM en nombre, ou None."""
    if valeur is None or isinstance(valeur, bool):
        return None
    if isinstance(valeur, (int, float)):
        return valeur
    if isinstance(valeur, str):
        try:
            return float(valeur.strip().replace(",", "."))
        except ValueError:
            return None
    return None


def _sans_decimale_inutile(x):
    return int(x) if isinstance(x, float) and x == int(x) else x


def convertir_unites(res_a: dict):
    """Retourne (pas_mmhg, glycemie_mmol_l, conversions, avertissements).

    - PAS : convertie en mmHg. Une valeur < 30 est toujours interprétée comme
      des cmHg (une systolique < 30 mmHg n'est pas compatible avec un patient
      qui se présente : c'est "12/8" ou "16,5/10" écrit en cmHg).
    - Glycémie : g/l -> mmol/l (x 5,55)."""
    conversions, avertissements = [], []

    pas_mmhg = None
    pas = _nombre(res_a.get("pas"))
    unite = (res_a.get("pas_unite") or "").strip().lower() if isinstance(res_a.get("pas_unite"), str) else ""
    if pas is not None:
        if unite == "cmhg" or pas < 30:
            pas_mmhg = _sans_decimale_inutile(round(pas * 10, 1))
            raison = "unité cmHg déclarée" if unite == "cmhg" else "valeur < 30 interprétée en cmHg"
            conversions.append({"champ": "pas_mmhg", "brut": pas, "unite": "cmHg", "converti": pas_mmhg, "raison": raison})
        else:
            pas_mmhg = _sans_decimale_inutile(pas)

    glycemie_mmol = None
    gly = _nombre(res_a.get("glycemie"))
    unite_g = (res_a.get("glycemie_unite") or "").strip().lower() if isinstance(res_a.get("glycemie_unite"), str) else ""
    if gly is not None:
        if unite_g == "g/l":
            glycemie_mmol = round(gly * 5.55, 1)
            conversions.append({"champ": "glycemie_mmol_l", "brut": gly, "unite": "g/l", "converti": glycemie_mmol,
                                "raison": "unité g/l déclarée"})
        else:
            glycemie_mmol = gly
    return pas_mmhg, glycemie_mmol, conversions, avertissements


# --------------------------------------------------------------------------
# Contrôle âge / motif : incohérences évidentes seulement
# --------------------------------------------------------------------------

def controle_age_motif(extraction: dict, referentiel: dict) -> list:
    """Avertissements (liste de textes) quand l'âge est incompatible avec le motif."""
    age = extraction.get("age_annees")
    motif_id = extraction.get("motif_id")
    if age is None or not motif_id:
        return []
    motif = next((m for m in referentiel["motifs"] if m["id"] == motif_id), None)
    if motif is None:
        return []
    avert = []
    if motif["specialite"] == "pediatrie":
        limite = 0.5 if motif_id == "fievre_3_mois" else (6 if motif_id == "convulsion_hyperthermique" else 2)
        if age > limite:
            avert.append(f"âge {age} ans incompatible avec le motif pédiatrique '{motif_id}' (limite : {limite} ans)")
    if motif["specialite"] == "gyneco_obstetrique" and any(k in motif_id for k in ("grossesse", "accouchement", "post_partum")):
        if age < 12 or age > 55:
            avert.append(f"âge {age} ans peu compatible avec le motif obstétrical '{motif_id}'")
    return avert


# --------------------------------------------------------------------------
# Extraction en deux appels
# --------------------------------------------------------------------------

def extraire_deux_etapes(referentiel: dict, texte_source: str, fournisseur: str, modele: str) -> dict:
    """Extraction en 2 appels légers :
    Appel A (vocabulaire léger, regroupé par section) choisit le motif et
    extrait les champs simples ; Appel B (minuscule, seulement les critères
    du motif choisi) vérifie les critères — seulement si le motif en a.

    Résultat : dict avec les champs de CasClinique (motif_id, age_annees,
    pas_mmhg, fc_min, spo2_pct, fr_min, glycemie_mmol_l, gcs, fievre,
    criteres_presents) plus des champs de traçabilité (fievre_statut,
    conversions, avertissements, motif_rejete, appel_b_echec)."""
    prompt_a = construire_prompt_etape1a(referentiel)
    res_a = completer(prompt_a, f"Vignette :\n{texte_source}", fournisseur, modele, max_tokens=600)
    if "_erreur_parsing" in res_a:
        return res_a

    ids_valides = {m["id"] for m in referentiel["motifs"]}
    avertissements = []

    motif_id = res_a.get("motif_id")
    if isinstance(motif_id, str):
        motif_id = motif_id.strip()
        if motif_id.lower() in ("", "null", "none"):
            motif_id = None
    elif motif_id is not None:
        motif_id = None
    motif_rejete = None
    if motif_id is not None and motif_id not in ids_valides:
        motif_rejete = motif_id
        avertissements.append(f"motif inconnu rejeté : {motif_id!r}")
        motif_id = None

    pas_mmhg, glycemie_mmol, conversions, av_conv = convertir_unites(res_a)
    avertissements += av_conv

    statut_fievre = res_a.get("fievre")
    if statut_fievre is True:
        statut_fievre = "oui"
    elif statut_fievre is False:
        statut_fievre = "non_mentionne"   # ancien format booléen : on ne peut pas savoir si c'était un "non" explicite
    if statut_fievre not in ("oui", "non", "non_mentionne"):
        statut_fievre = "non_mentionne"

    age = _nombre(res_a.get("age_annees"))
    resultat = {
        "motif_id": motif_id,
        "age_annees": _sans_decimale_inutile(age) if isinstance(age, float) else age,
        "pas_mmhg": pas_mmhg,
        "fc_min": _nombre(res_a.get("fc_min")),
        "spo2_pct": _nombre(res_a.get("spo2_pct")),
        "fr_min": _nombre(res_a.get("fr_min")),
        "glycemie_mmol_l": glycemie_mmol,
        "gcs": _nombre(res_a.get("gcs")),
        "fievre": statut_fievre == "oui",
        "fievre_statut": statut_fievre,
        "criteres_presents": [],
        "conversions": conversions,
    }
    if motif_rejete is not None:
        resultat["motif_rejete"] = motif_rejete

    if motif_id:
        motif = next((m for m in referentiel["motifs"] if m["id"] == motif_id), None)
        if motif and motif["modulateurs"]:
            prompt_b = construire_prompt_etape1b(motif)
            res_b = completer(prompt_b, f"Vignette :\n{texte_source}", fournisseur, modele, max_tokens=700)
            if "_erreur_parsing" in res_b:
                # On garde le reste du résultat plutôt que de tout perdre ; criteres_presents
                # reste vide (l'Étape 2 retombe sur le tri de base du motif), mais l'échec
                # est TRACÉ, plus silencieux.
                resultat["appel_b_echec"] = True
                avertissements.append("Appel B (critères du motif) en échec : critères non évalués")
            else:
                criteres_valides = {mod["critere"] for mod in motif["modulateurs"]}
                citations, rejetes, couvertures = {}, [], {}
                for item in (res_b.get("criteres_presents") or []):
                    # format attendu : {"critere": ..., "citation": ...} ; une simple chaîne (ancien format) n'a pas de citation
                    critere = item.get("critere") if isinstance(item, dict) else item
                    citation = item.get("citation") if isinstance(item, dict) else None
                    if critere not in criteres_valides:
                        avertissements.append(f"critère inconnu ignoré : {str(critere)[:80]!r}")
                        continue
                    if critere in citations or any(r["critere"] == critere for r in rejetes):
                        continue
                    if not isinstance(citation, str) or not citation.strip():
                        rejetes.append({"critere": critere, "raison": "aucune citation fournie"})
                    # APRÈS
                    elif not verifier_citation_critere(citation, texte_source):
                        n_mots, total = plus_long_passage_litteral(citation, texte_source)
                        rejetes.append({"critere": critere, "raison": "citation absente de la vignette",
                                        "couverture": f"{n_mots}/{total} mots", "citation": citation[:400]})
                    else:
                        n_mots, total = plus_long_passage_litteral(citation, texte_source)
                        citations[critere] = citation
                        couvertures[critere] = f"{n_mots}/{total} mots"
                resultat["criteres_presents"] = list(citations)
                resultat["criteres_citations"] = citations
                resultat["criteres_couverture"] = couvertures
                if rejetes:
                    resultat["criteres_rejetes"] = rejetes
                    avertissements.append(f"{len(rejetes)} critère(s) coché(s) rejeté(s) faute de citation vérifiable")
        # motif sans aucun modulateur : rien à vérifier, Appel B inutile

    avertissements += controle_age_motif(resultat, referentiel)
    resultat["avertissements"] = avertissements
    return resultat


def comparer(attendu: dict, obtenu: dict) -> dict:
    champs = ["motif_id", "age_annees", "fievre"]
    resultat = {}
    for champ in champs:
        resultat[champ] = {
            "attendu": attendu.get(champ),
            "obtenu": obtenu.get(champ),
            "ok": attendu.get(champ) == obtenu.get(champ),
        }
    resultat["criteres_ok"] = set(attendu.get("criteres_presents", [])) == set(obtenu.get("criteres_presents", []))
    return resultat


def executer_mode_test(golden_path: str, referentiel: dict, fournisseur: str, modele: str, pause: float):
    with open(golden_path, encoding="utf-8") as f:
        golden = json.load(f)

    n_ok, n_total = 0, 0
    for cas in golden["cas"]:
        n_total += 1
        obtenu = extraire_deux_etapes(referentiel, cas["vignette_source"], fournisseur, modele)
        attendu = cas["extraction_attendue"]

        print(f"\n=== {cas['id']} ===")
        if "_erreur_parsing" in obtenu:
            print(f"❌ ERREUR DE PARSING JSON : {obtenu['_erreur_parsing']}")
            time.sleep(pause)
            continue

        diff = comparer(attendu, obtenu)
        tout_ok = all(v["ok"] for v in diff.values() if isinstance(v, dict)) and diff["criteres_ok"]

        for champ, detail in diff.items():
            if champ == "criteres_ok":
                continue
            marqueur = "✅" if detail["ok"] else "❌"
            print(f"  {marqueur} {champ}: attendu={detail['attendu']!r}, obtenu={detail['obtenu']!r}")
        marqueur_criteres = "✅" if diff["criteres_ok"] else "❌"
        print(f"  {marqueur_criteres} criteres_presents: attendu={attendu.get('criteres_presents')!r}, "
              f"obtenu={obtenu.get('criteres_presents')!r}")
        if obtenu.get("avertissements"):
            print(f"  ⚠️  avertissements : {obtenu['avertissements']}")

        if tout_ok:
            n_ok += 1
            print("  => CAS ENTIÈREMENT CORRECT")

        time.sleep(pause)

    print(f"\n{'=' * 50}\n{n_ok}/{n_total} cas entièrement corrects (motif + âge + fièvre + critères)")


def executer_mode_reel(input_path: str, output_path: str, referentiel: dict, fournisseur: str,
                        modele: str, pause: float, limit: int):
    vignettes = []
    with open(input_path, encoding="utf-8") as f:
        for ligne in f:
            ligne = ligne.strip()
            if ligne:
                vignettes.append(json.loads(ligne))
    if limit:
        vignettes = vignettes[:limit]

    print(f"{len(vignettes)} vignettes à traiter...", file=sys.stderr)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    n_ok, n_erreurs = 0, 0
    with open(output_file, "a", encoding="utf-8") as f_out:
        for i, vignette in enumerate(vignettes, 1):
            texte_source = vignette.get("contexte_clinique") or vignette.get("question", "")
            extraction = extraire_deux_etapes(referentiel, texte_source, fournisseur, modele)

            if "_erreur_parsing" in extraction:
                n_erreurs += 1
            else:
                n_ok += 1

            resultat = {"id": vignette["id"], "cas_id": vignette.get("cas_id"),
                        "vignette_source": texte_source, "extraction": extraction}
            f_out.write(json.dumps(resultat, ensure_ascii=False) + "\n")
            f_out.flush()

            if i % 10 == 0 or i == len(vignettes):
                print(f"  {i}/{len(vignettes)} traitées — {n_ok} ok, {n_erreurs} erreurs de parsing", file=sys.stderr)

            time.sleep(pause)

    print(f"\nTerminé : {n_ok} extractions réussies, {n_erreurs} erreurs -> {output_file}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Étape 1 : extraction structurée du cas clinique")
    parser.add_argument("--golden", default=None,
                         help="Mode test : chemin du jeu de cas de référence")
    parser.add_argument("--input", default=None,
                         help="Mode réel : fichier JSONL de vignettes")
    parser.add_argument("--output", default=None, help="Mode réel : fichier de sortie JSONL")
    parser.add_argument("--limit", type=int, default=None, help="Mode réel : limiter le nombre de vignettes")
    parser.add_argument("--fournisseur", choices=["groq", "gemini", "anthropic"], default=FOURNISSEUR_DEFAUT)
    parser.add_argument("--modele", default=None, help=f"Défaut : {MODELE_DEFAUT} (Anthropic)")
    parser.add_argument("--pause", type=float, default=3.0)
    args = parser.parse_args()

    modele = args.modele or MODELE_DEFAUT

    ref_path = Path(__file__).parent.parent / "referentiel" / "french_referentiel.json"
    referentiel = charger_referentiel(str(ref_path))

    if args.input:
        if not args.output:
            print("--output est requis avec --input", file=sys.stderr)
            sys.exit(1)
        executer_mode_reel(args.input, args.output, referentiel, args.fournisseur, modele, args.pause, args.limit)
    else:
        golden_path = args.golden or str(Path(__file__).parent.parent.parent / "tests" / "golden_etape1_extraction.json")
        executer_mode_test(golden_path, referentiel, args.fournisseur, modele, args.pause)


if __name__ == "__main__":
    main()
