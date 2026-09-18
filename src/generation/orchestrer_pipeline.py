"""
orchestrer_pipeline.py

Chaîne RÉELLEMENT les 4 étapes du mécanisme de labellisation, dans l'ordre
conçu depuis le début du projet.

Déroulé pour chaque vignette :
    1. Extraction structurée (LLM) -> cas_structure
    2. Détection déterministe (red_flag_detector.py, SANS LLM) -> resultat_deterministe
    3. Labellisation (LLM) -> categorie_llm
    -> Croisement 2/3 via appliquer_override() -> label après override
    4. Contre-validation (LLM, un troisième appel/modèle) -> verdict
    -> Résolution finale via resoudre_contre_validation() -> categorie_finale

Chaque étape garde sa trace complète dans la sortie, pour l'auditabilité
exigée par le brief.

Usage :
    python orchestrer_pipeline.py \
        --input ../../data/anonymized/mediqal_mcqm_urgences_anonymise.jsonl \
        --output ../../data/normalized/mediqal_mcqm_pipeline_complet.jsonl \
        --limit 10
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
from red_flag_detector import (  # noqa: E402
    CasClinique, charger_referentiel, detecter_red_flags, appliquer_override,
)
from extraction_etape1 import extraire_deux_etapes  # noqa: E402
from labellisation_etape3 import PROMPT_SYSTEME as PROMPT_ETAPE3, CATEGORIES_VALIDES  # noqa: E402
from contre_validation_etape4 import (  # noqa: E402
    PROMPT_SYSTEME as PROMPT_ETAPE4, resoudre_contre_validation,
)

CHAMPS_CAS_CLINIQUE = {
    "motif_id", "age_annees", "pas_mmhg", "fc_min", "spo2_pct", "fr_min",
    "glycemie_mmol_l", "gcs", "fievre", "criteres_presents",
}


def dict_vers_cas_clinique(d: dict) -> CasClinique:
    """Construit un CasClinique en ne gardant que les champs connus — évite
    de planter si l'extraction contient un champ inattendu."""
    filtre = {k: v for k, v in d.items() if k in CHAMPS_CAS_CLINIQUE}
    return CasClinique(**filtre)


def traiter_une_vignette(vignette: dict, referentiel: dict, config: dict) -> dict:
    """Fait passer une vignette par les 4 étapes. Retourne la trace complète."""
    texte_source = vignette.get("contexte_clinique") or vignette.get("question", "")
    trace = {"id": vignette["id"], "vignette_source": texte_source}

    # --- Étape 1 : extraction ---
    extraction = extraire_deux_etapes(
        referentiel, texte_source, config["etape1_fournisseur"], config["etape1_modele"],
    )
    trace["etape1_extraction"] = extraction
    if "_erreur_parsing" in extraction:
        trace["_echec"] = "etape1"
        return trace
    time.sleep(config["pause_interne"])

    # --- Étape 2 : détection déterministe (pas de LLM) ---
    cas = dict_vers_cas_clinique(extraction)
    resultat_etape2 = detecter_red_flags(cas, referentiel)
    trace["etape2_resultat"] = {
        "tri_minimal_force": resultat_etape2.tri_minimal_force,
        "categorie_brief_minimale": resultat_etape2.categorie_brief_minimale,
        "flags_declenches": resultat_etape2.flags_declenches,
    }

    # --- Étape 3 : labellisation ---
    cas_json = json.dumps(extraction, ensure_ascii=False, indent=2)
    resultat_etape3 = completer(
        PROMPT_ETAPE3, f"Cas structuré :\n{cas_json}",
        config["etape3_fournisseur"], config["etape3_modele"], max_tokens=2048,
    )
    trace["etape3_resultat"] = resultat_etape3
    if "_erreur_parsing" in resultat_etape3 or resultat_etape3.get("categorie") not in CATEGORIES_VALIDES:
        trace["_echec"] = "etape3"
        return trace
    time.sleep(config["pause_interne"])

    # --- Croisement Étape 2 / Étape 3 ---
    decision_23 = appliquer_override(resultat_etape3["categorie"], resultat_etape2)
    trace["decision_apres_override_etape2_3"] = decision_23

    # --- Étape 4 : contre-validation ---
    contenu_etape4 = json.dumps(
        {"cas_structure": extraction, "niveau_deja_attribue": decision_23["label_final"]},
        ensure_ascii=False, indent=2,
    )
    verdict_etape4 = completer(
        PROMPT_ETAPE4, contenu_etape4,
        config["etape4_fournisseur"], config["etape4_modele"], max_tokens=2048,
    )
    trace["etape4_verdict"] = verdict_etape4
    if "_erreur_parsing" in verdict_etape4:
        # L'Étape 4 échoue : on garde le label d'après l'Étape 2/3 comme
        # final, plutôt que de perdre toute la vignette pour un seul échec
        # sur l'étape la moins critique (garde-fou en plus, pas le seul).
        trace["categorie_finale"] = decision_23["label_final"]
        trace["_echec"] = "etape4 (label conservé depuis l'Étape 2/3)"
        return trace

    decision_finale = resoudre_contre_validation(decision_23["label_final"], verdict_etape4)
    trace["decision_finale"] = decision_finale
    trace["categorie_finale"] = decision_finale["categorie_finale"]

    return trace


def charger_ids_deja_traites(output_file: Path) -> set:
    ids = set()
    if output_file.exists():
        with open(output_file, encoding="utf-8") as f:
            for ligne in f:
                ligne = ligne.strip()
                if ligne:
                    ids.add(json.loads(ligne)["id"])
    return ids


def main():
    parser = argparse.ArgumentParser(description="Orchestration complète du pipeline (Étapes 1 à 4)")
    parser.add_argument("--input", required=True, help="Vignettes filtrées et anonymisées (JSONL)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--pause", type=float, default=3.0, help="Pause entre deux vignettes")
    parser.add_argument("--pause-interne", type=float, default=2.0, help="Pause entre deux appels LLM d'une même vignette")
    parser.add_argument("--etape1-fournisseur", default="anthropic")
    parser.add_argument("--etape1-modele", default="claude-haiku-4-5-20251001")
    parser.add_argument("--etape3-fournisseur", default="anthropic")
    parser.add_argument("--etape3-modele", default="claude-haiku-4-5-20251001")
    parser.add_argument("--etape4-fournisseur", default="anthropic")
    parser.add_argument("--etape4-modele", default="claude-haiku-4-5-20251001")
    args = parser.parse_args()

    ref_path = Path(__file__).parent.parent / "referentiel" / "french_referentiel.json"
    referentiel = charger_referentiel(str(ref_path))

    config = {
        "pause_interne": args.pause_interne,
        "etape1_fournisseur": args.etape1_fournisseur, "etape1_modele": args.etape1_modele,
        "etape3_fournisseur": args.etape3_fournisseur, "etape3_modele": args.etape3_modele,
        "etape4_fournisseur": args.etape4_fournisseur, "etape4_modele": args.etape4_modele,
    }

    vignettes = []
    with open(args.input, encoding="utf-8") as f:
        for ligne in f:
            ligne = ligne.strip()
            if ligne:
                vignettes.append(json.loads(ligne))
    if args.limit:
        vignettes = vignettes[: args.limit]

    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    ids_deja_traites = charger_ids_deja_traites(output_file)
    if ids_deja_traites:
        print(f"Reprise détectée : {len(ids_deja_traites)} vignettes déjà traitées, elles seront sautées.",
              file=sys.stderr)
    vignettes = [v for v in vignettes if v["id"] not in ids_deja_traites]

    print(f"{len(vignettes)} vignettes à traiter dans le pipeline complet...", file=sys.stderr)

    n_succes, n_echecs = 0, 0
    with open(output_file, "a", encoding="utf-8") as f_out:
        for i, vignette in enumerate(vignettes, 1):
            try:
                trace = traiter_une_vignette(vignette, referentiel, config)
            except RuntimeError as e:
                print(f"\n⚠️  Arrêt propre : {e}", file=sys.stderr)
                print(f"  {i - 1}/{len(vignettes)} vignettes traitées avant l'arrêt — "
                      f"le reste est sauvegardé, relancez la même commande plus tard pour reprendre.",
                      file=sys.stderr)
                sys.exit(0)

            if "_echec" in trace and "categorie_finale" not in trace:
                n_echecs += 1
            else:
                n_succes += 1

            f_out.write(json.dumps(trace, ensure_ascii=False) + "\n")
            f_out.flush()

            statut = trace.get("categorie_finale", f"ÉCHEC ({trace.get('_echec')})")
            print(f"  [{i}/{len(vignettes)}] {vignette['id']} -> {statut}", file=sys.stderr)

            time.sleep(args.pause)

    print(f"\nTerminé : {n_succes} succès, {n_echecs} échecs -> {output_file}", file=sys.stderr)


if __name__ == "__main__":
    main()