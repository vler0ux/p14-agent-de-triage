"""
labellisation_etape3.py

Étape 3 du pipeline : à partir de la VIGNETTE SOURCE et du cas clinique
structuré qui en a été extrait (sortie de l'Étape 1), le LLM propose un
niveau de priorité parmi les 3 catégories du brief (urgence_maximale /
moderee / differee), avec justification.

La vignette est fournie EN PLUS du cas structuré : l'extraction de l'Étape 1
est un résumé partiel qui peut se tromper de motif ou omettre des signes
graves ; sans la vignette, une erreur d'extraction traverse toute la chaîne
sans que personne puisse la voir.

--- Ce qui a changé au lot 1 (sept. 2026) ---
- Nouveau champ "motif_suggere" : le motif de recours que l'Étape 3 retient
  elle-même (même liste et mêmes règles que l'Étape 1). S'il diffère de celui
  de l'Étape 1, l'orchestrateur marque le cas "motif_conteste" pour relecture.
  Ce n'est PAS un avis indépendant (mêmes modèles possibles) : un filet, pas
  une garantie. À retirer si le re-test montre qu'il confirme presque toujours
  l'Étape 1, même quand elle se trompe.
- "extraction_incoherente" est redéfini : vrai seulement si le MOTIF ou une
  VALEUR CHIFFRÉE du cas structuré contredit la vignette. L'absence
  d'antécédents, d'examens ou de symptômes secondaires n'en est pas une (le cas
  structuré ne les contient pas volontairement).
- Le prompt reçoit maintenant la liste des motifs : il se construit avec
  construire_prompt_systeme(referentiel).
- La fièvre est à trois valeurs (fievre_statut) : "non_mentionne" ne veut pas
  dire "pas de fièvre".

Cette proposition n'est PAS le label final : elle est ensuite croisée avec
le résultat de l'Étape 2 (détection déterministe, red_flag_detector.py) via
appliquer_override(), puis l'Étape 4 peut seulement SIGNALER un risque.

Usage (mode test, contre le jeu de cas de référence) :
    python labellisation_etape3.py --golden ../../tests/golden_etape1_extraction.json

Usage (mode réel, sur la sortie de extraction_etape1.py) :
    python labellisation_etape3.py \
        --input ../../data/normalized/mediqal_mcqm_extraction_test.jsonl \
        --output ../../data/normalized/mediqal_mcqm_labellisation_test.jsonl
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
from extraction_etape1 import construire_vocabulaire_motifs_leger  # noqa: E402

CATEGORIES_VALIDES = {"urgence_maximale", "moderee", "differee"}

FOURNISSEUR_DEFAUT = "gemini"
MODELE_DEFAUT = "gemini-3.6-flash"   # à vérifier dans ta console avant un gros lot

# Champs du cas structuré montrés au LLM (les champs de traçabilité internes restent cachés)
CHAMPS_VISIBLES = ["motif_id", "age_annees", "pas_mmhg", "fc_min", "spo2_pct", "fr_min",
                   "glycemie_mmol_l", "gcs", "fievre_statut", "criteres_presents"]

PROMPT_SYSTEME_BASE = """Tu es un assistant d'aide à la classification de triage clinique, pour un projet d'agent de triage médical.

On te fournit deux éléments : la VIGNETTE SOURCE (texte clinique original, qui fait foi) et un CAS STRUCTURÉ extrait automatiquement de cette vignette (résumé partiel, qui peut être incomplet ou comporter des erreurs, notamment sur le motif de recours).

Détermine le niveau de priorité de la présentation AIGUË INITIALE du patient (à son arrivée) parmi exactement ces 3 catégories :
- "urgence_maximale" : pronostic vital ou fonctionnel engagé à court terme, prise en charge sans délai ou très rapide nécessaire
- "moderee" : atteinte potentielle nécessitant une évaluation médicale dans la journée, sans urgence vitale immédiate
- "differee" : motif de recours légitime aux urgences mais sans risque de dégradation à court terme

RÈGLES IMPÉRATIVES :
1. En cas de doute entre deux niveaux, choisis TOUJOURS le niveau le plus élevé (principe de précaution).
2. Justifie ta réponse en citant explicitement les éléments cliniques qui motivent ce choix.
3. Base-toi sur ce qui est écrit dans la vignette. N'invente AUCUN élément clinique absent de la vignette. Si le cas structuré contredit la vignette (motif erroné, signe grave manquant), c'est la vignette qui fait foi.
4. Juge la gravité de la présentation INITIALE, pas l'état après traitement ni les résultats d'examens réalisés ensuite (imagerie, ECG, bilans) : un patient décrit comme stabilisé ou hospitalisé a pu arriver dans un état grave.
5. Un motif de recours plausible mais bénin (ex. traumatisme mineur, symptôme chronique stable) doit être classé "differee", pas "moderee" par excès de prudence non justifié.
6. Dans le cas structuré, "fievre_statut" vaut "oui", "non" ou "non_mentionne". "non_mentionne" signifie que la vignette n'en parle pas : ce n'est PAS l'absence de fièvre.

MOTIF DE RECOURS (champ "motif_suggere") : indique le motif que TU retiens pour cette vignette, dans la liste ci-dessous, en suivant les règles du référentiel de l'infirmier organisateur de l'accueil : le motif est la PLAINTE principale à l'arrivée, exprimée en symptôme, jamais un diagnostic ; s'il y en a plusieurs, la plus complexe ou la plus sévère ; sans te baser sur le diagnostic final ni sur les examens. Mets null si aucun motif de la liste ne correspond vraiment. Donne l'identifiant EXACT.

Motifs disponibles :
{vocabulaire}

Réponds UNIQUEMENT avec un objet JSON, sans texte avant ni après :
{{"categorie": "urgence_maximale" ou "moderee" ou "differee", "justification": "une ou deux phrases citant les éléments qui motivent ce choix", "motif_suggere": "identifiant exact de la liste, ou null", "extraction_incoherente": true ou false, "detail_extraction": "une phrase décrivant la contradiction, ou null si extraction_incoherente vaut false"}}
où "extraction_incoherente" vaut true UNIQUEMENT si le motif du cas structuré est manifestement différent de la plainte décrite dans la vignette, ou si une valeur chiffrée du cas structuré contredit la vignette. Ne le mets PAS à true pour l'absence d'antécédents, d'examens complémentaires ou de symptômes secondaires : le cas structuré ne les contient pas volontairement. Toute explication doit figurer DANS le champ "detail_extraction" : n'écris AUCUN texte en dehors de l'objet JSON.
"""


def construire_prompt_systeme(referentiel: dict) -> str:
    return PROMPT_SYSTEME_BASE.format(vocabulaire=construire_vocabulaire_motifs_leger(referentiel))


def construire_contenu(cas_structure: dict, vignette_source: str) -> str:
    """Contenu envoyé au LLM : vignette source + cas structuré (champs
    cliniques uniquement). Partagé avec orchestrer_pipeline.py pour que les
    deux ne divergent jamais."""
    visible = {k: cas_structure.get(k) for k in CHAMPS_VISIBLES if k in cas_structure}
    cas_json = json.dumps(visible, ensure_ascii=False, indent=2)
    return (f"Vignette source :\n{vignette_source}\n\n"
            f"Cas structuré extrait (peut être incomplet ou erroné) :\n{cas_json}")


def labelliser(cas_structure: dict, vignette_source: str, fournisseur: str, modele: str,
               prompt_systeme: str, ids_motifs: set) -> dict:
    resultat = completer(prompt_systeme, construire_contenu(cas_structure, vignette_source),
                         fournisseur, modele, max_tokens=2048)

    if "_erreur_parsing" in resultat:
        return resultat

    if resultat.get("categorie") not in CATEGORIES_VALIDES:
        resultat["_erreur_categorie_invalide"] = resultat.get("categorie")

    suggere = resultat.get("motif_suggere")
    if isinstance(suggere, str):
        suggere = suggere.strip()
        if suggere.lower() in ("", "null", "none"):
            suggere = None
    elif suggere is not None:
        suggere = None
    if suggere is not None and suggere not in ids_motifs:
        resultat["_motif_suggere_invalide"] = suggere
        suggere = None
    resultat["motif_suggere"] = suggere

    return resultat


def _contexte_reference(referentiel_path: str = None):
    ref_path = referentiel_path or str(Path(__file__).parent.parent / "referentiel" / "french_referentiel.json")
    referentiel = charger_referentiel(ref_path)
    return referentiel, construire_prompt_systeme(referentiel), {m["id"] for m in referentiel["motifs"]}


def executer_mode_test(golden_path: str, fournisseur: str, modele: str, pause: float):
    with open(golden_path, encoding="utf-8") as f:
        golden = json.load(f)
    _, prompt, ids = _contexte_reference()

    n_ok, n_total = 0, 0
    for cas in golden["cas"]:
        n_total += 1
        resultat = labelliser(cas["extraction_attendue"], cas["vignette_source"], fournisseur, modele, prompt, ids)

        print(f"\n=== {cas['id']} ===")
        if "_erreur_parsing" in resultat:
            print(f"❌ ERREUR DE PARSING : {resultat['_erreur_parsing']}")
            time.sleep(pause)
            continue

        categorie_attendue = cas.get("categorie_attendue", "(non définie)")
        categorie_obtenue = resultat.get("categorie")
        ok = categorie_obtenue == categorie_attendue

        marqueur = "✅" if ok else "❌"
        print(f"  {marqueur} attendu={categorie_attendue!r}, obtenu={categorie_obtenue!r}")
        print(f"  Justification LLM : {resultat.get('justification')}")
        attendu_motif = cas["extraction_attendue"].get("motif_id")
        print(f"  Motif : attendu (extraction de référence)={attendu_motif!r}, suggéré={resultat.get('motif_suggere')!r}")
        if "_erreur_categorie_invalide" in resultat:
            print(f"  ⚠️  GARDE-FOU DÉCLENCHÉ : catégorie hors des 3 valeurs autorisées "
                  f"({resultat['_erreur_categorie_invalide']!r})")

        if ok:
            n_ok += 1
        time.sleep(pause)

    print(f"\n{'=' * 50}\n{n_ok}/{n_total} catégories correctes")


def charger_ids_deja_traites(output_file: Path) -> set:
    """Lit le fichier de sortie déjà existant pour connaître les ids déjà
    traités lors d'une exécution précédente (mécanisme de reprise)."""
    ids = set()
    if output_file.exists():
        with open(output_file, encoding="utf-8") as f:
            for ligne in f:
                ligne = ligne.strip()
                if ligne:
                    ids.add(json.loads(ligne)["id"])
    return ids


def executer_mode_reel(input_path: str, output_path: str, fournisseur: str, modele: str, pause: float):
    lignes = []
    with open(input_path, encoding="utf-8") as f:
        for ligne in f:
            ligne = ligne.strip()
            if ligne:
                lignes.append(json.loads(ligne))

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    _, prompt, ids = _contexte_reference()

    ids_deja_traites = charger_ids_deja_traites(output_file)
    if ids_deja_traites:
        print(f"Reprise détectée : {len(ids_deja_traites)} vignettes déjà traitées, elles seront sautées.",
              file=sys.stderr)
    lignes = [l for l in lignes if l["id"] not in ids_deja_traites]

    n_ok, n_erreurs, n_ignorees = 0, 0, 0
    with open(output_file, "a", encoding="utf-8") as f_out:
        for i, entree in enumerate(lignes, 1):
            cas_structure = entree.get("extraction", {})
            vignette_source = entree.get("vignette_source")
            if "_erreur_parsing" in cas_structure or not vignette_source:
                n_ignorees += 1
                continue

            resultat = labelliser(cas_structure, vignette_source, fournisseur, modele, prompt, ids)
            if "_erreur_parsing" in resultat:
                n_erreurs += 1
            else:
                n_ok += 1

            sortie = {"id": entree["id"], "cas_id": entree.get("cas_id"), "vignette_source": vignette_source,
                      "cas_structure": cas_structure, "label_etape3": resultat}
            f_out.write(json.dumps(sortie, ensure_ascii=False) + "\n")
            f_out.flush()

            if i % 10 == 0 or i == len(lignes):
                print(f"  {i}/{len(lignes)} traitées — {n_ok} ok, {n_erreurs} erreurs, "
                      f"{n_ignorees} ignorées (Étape 1 en échec)", file=sys.stderr)

            time.sleep(pause)

    print(f"\nTerminé -> {output_file}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Étape 3 : labellisation du niveau de priorité")
    parser.add_argument("--golden", default=None, help="Mode test : chemin du jeu de cas de référence")
    parser.add_argument("--input", default=None, help="Mode réel : sortie JSONL de extraction_etape1.py")
    parser.add_argument("--output", default=None, help="Mode réel : fichier de sortie JSONL")
    parser.add_argument("--fournisseur", choices=["groq", "gemini", "anthropic"], default=FOURNISSEUR_DEFAUT)
    parser.add_argument("--modele", default=None,
                         help=f"Défaut : {MODELE_DEFAUT} (à vérifier dans ta console)")
    parser.add_argument("--pause", type=float, default=3.0)
    args = parser.parse_args()

    modele = args.modele or MODELE_DEFAUT

    if args.input:
        if not args.output:
            print("--output est requis avec --input", file=sys.stderr)
            sys.exit(1)
        executer_mode_reel(args.input, args.output, args.fournisseur, modele, args.pause)
    else:
        golden_path = args.golden or str(Path(__file__).parent.parent.parent / "tests" / "golden_etape1_extraction.json")
        executer_mode_test(golden_path, args.fournisseur, modele, args.pause)


if __name__ == "__main__":
    main()
