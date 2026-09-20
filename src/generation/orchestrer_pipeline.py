"""
orchestrer_pipeline.py

Chaîne RÉELLEMENT les 4 étapes du mécanisme de labellisation, dans l'ordre
conçu depuis le début du projet.

Déroulé pour chaque vignette :
    1. Extraction structurée (LLM) -> cas_structure
    2. Détection déterministe (red_flag_detector.py, SANS LLM) -> resultat_deterministe
    3. Labellisation (LLM) -> categorie_llm + motif_suggere
    -> Croisement 2/3 via appliquer_override() : plancher des constantes
       inviolable, plancher du motif révisable -> label après croisement
    4. Audit (LLM, un autre fournisseur) -> signalement éventuel d'un risque
       de sous-triage. L'Étape 4 NE MODIFIE PLUS le label.
    -> categorie_finale = label après croisement Étape 2 / Étape 3

--- Ce qui a changé au lot 1 (sept. 2026) ---
- La trace reprend `cas_id` (regroupement des questions d'un même cas : sert au
  découpage train / validation / test par cas).
- La trace contient une FILE DE RELECTURE : `a_relire` (booléen, vrai si au
  moins une raison de priorité haute), `priorite_relecture`, `raisons_relecture`
  (liste de textes), `motif_conteste`, `ecart_plancher` et `score_relecture`
  (plus il est élevé, plus le cas est à relire en premier ; sert au tri du budget de relectures).
  Règle du lot 1 ter : la relecture PRIORITAIRE concerne les cas où le label est SOUS un plancher de
  la grille FRENCH (le vrai risque : le sous-triage), plus les données douteuses.
  Raisons de priorité HAUTE : label moins urgent que le plancher du motif retenu par l'Étape 1
  (score 100 + 10 par catégorie d'écart) ; label moins urgent que le tri médian du motif suggéré par
  l'Étape 3 (60 + 10 par catégorie d'écart) ; constante non retrouvée dans la vignette (50) ;
  incohérence âge / motif (40).
  Raisons de priorité BASSE (à relire seulement s'il reste du budget) : signalement de l'Étape 4
  (avec ou sans citation vérifiée : sur 19 cas, 1 seul sur 5 était soutenu par la grille) ; motif
  contesté quand le label est déjà au moins aussi urgent que les planchers ; extraction signalée
  incohérente ; critères de l'Appel B rejetés ; critères non évalués ; échec de l'Étape 4.
- `notes_validation` liste ce qui doit être validé par un urgentiste
  (modulateurs appliqués à la baisse).
- Fournisseurs par défaut : Étapes 1 et 3 = Anthropic (Haiku), Étape 4 = Groq (auditeur
  différent, pour réduire la corrélation des erreurs). `--tout-haiku` met aussi l'Étape 4 sur
  Haiku (si un quota bloque). Le nom du modèle Groq est à vérifier dans ta console.
  Étape 3 : Haiku retenu après comparaison avec Gemini sur 19 vignettes : les deux modèles
  ne s'accordaient que sur 12 cas, et sur 5 des 6 cas où Gemini était moins urgent, la grille
  FRENCH (début brutal, céphalée inhabituelle, douleur sévère) donne le tri 2, comme Haiku.
- Les fournisseurs et modèles se règlent aussi dans le fichier .env, sans toucher au code :
      ETAPE1_FOURNISSEUR=anthropic   ETAPE1_MODELE=claude-haiku-4-5-20251001
      ETAPE3_FOURNISSEUR=anthropic   ETAPE3_MODELE=claude-haiku-4-5-20251001
      ETAPE4_FOURNISSEUR=groq        ETAPE4_MODELE=openai/gpt-oss-120b
  Priorité : options de la ligne de commande > fichier .env > valeurs de DEFAUTS ci-dessous.
  `--tout-haiku` ignore le .env.
- La provenance enregistre aussi la température (0) et la version du lot.

Chaque étape garde sa trace complète dans la sortie, pour l'auditabilité
exigée par le brief.

Usage :
    python orchestrer_pipeline.py \
        --input ../../data/anonymized/mediqal_mcqm_urgences_complet_dedup.jsonl \
        --output ../../data/normalized/mediqal_mcqm_pipeline_complet.jsonl \
        --limit 3
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "referentiel"))

from red_flag_detector import (  # noqa: E402
    CasClinique, charger_referentiel, detecter_red_flags, appliquer_override,
    MAPPING_CATEGORIES_BRIEF, ORDRE_CATEGORIE_BRIEF,
)
from extraction_etape1 import extraire_deux_etapes, construire_prompt_etape1a  # noqa: E402
from labellisation_etape3 import (  # noqa: E402
    CATEGORIES_VALIDES, construire_prompt_systeme as construire_prompt_etape3, labelliser,
)
from contre_validation_etape4 import (  # noqa: E402
    PROMPT_SYSTEME as PROMPT_ETAPE4, contre_valider, resoudre_contre_validation,
)

VERSION_LOT = "lot1"
MODELE_HAIKU = "claude-haiku-4-5-20251001"

# Nom du modèle Groq : À VÉRIFIER dans ta console avant un gros lot.
DEFAUTS = {
    "etape1": ("anthropic", MODELE_HAIKU),
    "etape3": ("anthropic", MODELE_HAIKU),
    "etape4": ("groq", "openai/gpt-oss-120b"),
}

CHAMPS_CAS_CLINIQUE = {
    "motif_id", "age_annees", "pas_mmhg", "fc_min", "spo2_pct", "fr_min",
    "glycemie_mmol_l", "gcs", "fievre", "criteres_presents",
}


def empreinte(texte: str) -> str:
    """Empreinte courte (8 caractères) d'un prompt : permet de savoir, longtemps
    après, avec quelle version exacte d'un prompt un label a été produit."""
    return hashlib.sha256(texte.encode("utf-8")).hexdigest()[:8]


def construire_provenance(referentiel: dict, config: dict) -> dict:
    """Fournisseur, modèle et empreinte du prompt de chaque étape LLM, à recopier
    dans la trace de chaque vignette (auditabilité exigée par le brief).
    Pour l'Étape 1, seule l'empreinte du prompt de choix du motif est enregistrée."""
    return {
        "version": VERSION_LOT,
        "temperature": 0,
        "etape1": {"fournisseur": config["etape1_fournisseur"], "modele": config["etape1_modele"],
                   "prompt": empreinte(construire_prompt_etape1a(referentiel))},
        "etape3": {"fournisseur": config["etape3_fournisseur"], "modele": config["etape3_modele"],
                   "prompt": empreinte(config["prompt_etape3"])},
        "etape4": {"fournisseur": config["etape4_fournisseur"], "modele": config["etape4_modele"],
                   "prompt": empreinte(PROMPT_ETAPE4)},
    }


def dict_vers_cas_clinique(d: dict) -> CasClinique:
    """Construit un CasClinique en ne gardant que les champs connus — évite
    de planter si l'extraction contient un champ inattendu. Les valeurs telles
    qu'écrites dans la vignette (avant conversion d'unité) sont reprises pour
    le contrôle de présence des constantes."""
    filtre = {k: v for k, v in d.items() if k in CHAMPS_CAS_CLINIQUE}
    cas = CasClinique(**filtre)
    cas.valeurs_brutes = {c["champ"]: c["brut"] for c in (d.get("conversions") or []) if "champ" in c and "brut" in c}
    return cas


def evaluer_relecture(trace: dict, referentiel: dict) -> None:
    """Remplit dans la trace : motif_conteste, a_relire, priorite_relecture, raisons_relecture,
    ecart_plancher, score_relecture et notes_validation (règle du lot 1 ter : voir l'en-tête)."""
    ordre = ORDRE_CATEGORIE_BRIEF
    raisons = []   # (priorité, code, texte, score)
    ext = trace.get("etape1_extraction") or {}
    e3 = trace.get("etape3_resultat") or {}
    d23 = trace.get("decision_apres_override_etape2_3") or {}
    dfin = trace.get("decision_finale") or {}
    e4 = trace.get("etape4_verdict") or {}
    label = trace.get("categorie_finale")

    # Label sous le plancher du motif retenu par l'Étape 1 (avec ses critères cochés) : le risque de sous-triage
    ecart_plancher = 0
    plancher = d23.get("categorie_plancher_motif")
    if d23.get("sous_plancher_motif") and label and plancher:
        ecart_plancher = ordre[label] - ordre[plancher]
        raisons.append(("haute", "sous_plancher_motif",
                        f"le label ({label}) est moins urgent que le plancher du motif ({plancher}) ; "
                        f"critères cochés : {ext.get('criteres_presents') or 'aucun'}", 100 + 10 * ecart_plancher))
    trace["ecart_plancher"] = ecart_plancher

    # Motif de l'Étape 1 contesté par l'Étape 3 (seulement si l'Étape 3 a répondu proprement)
    motif_conteste = False
    if "motif_suggere" in e3 and "_erreur_parsing" not in e3 and e3.get("motif_suggere") != ext.get("motif_id"):
        motif_conteste = True
        detail = f"Étape 1 = {ext.get('motif_id')!r}, Étape 3 = {e3.get('motif_suggere')!r}"
        motif3 = next((m for m in referentiel["motifs"] if m["id"] == e3.get("motif_suggere")), None)
        plancher3 = MAPPING_CATEGORIES_BRIEF.get(motif3["tri_base"]) if motif3 else None
        if label and plancher3 and ordre[label] > ordre[plancher3]:
            ecart = ordre[label] - ordre[plancher3]
            raisons.append(("haute", "motif_conteste_sous_plancher",
                            f"{detail} ; le label ({label}) est moins urgent que le tri médian du motif de l'Étape 3 ({plancher3})",
                            60 + 10 * ecart))
        else:
            raisons.append(("basse", "motif_conteste", f"{detail} (label déjà au moins aussi urgent que le plancher)", 10))
    trace["motif_conteste"] = motif_conteste

    if d23.get("constantes_non_verifiees"):
        raisons.append(("haute", "constante_non_retrouvee",
                        "valeur(s) extraite(s) absente(s) de la vignette et ignorée(s) : " + ", ".join(d23["constantes_non_verifiees"]), 50))
    for av in ext.get("avertissements") or []:
        if "âge" in av:
            raisons.append(("haute", "age_motif", av, 40))

    if dfin.get("signal_sous_triage"):
        if dfin.get("citation_verifiee"):
            raisons.append(("basse", "etape4_signal_sous_triage",
                            f"l'auditeur suggère {dfin.get('niveau_suggere')} (citation retrouvée dans la vignette ; non prioritaire "
                            "sauf si le label est aussi sous un plancher de la grille)", 5))
        else:
            raisons.append(("basse", "etape4_signal_non_verifie",
                            f"l'auditeur suggère {dfin.get('niveau_suggere')} mais sa citation n'est pas retrouvée dans la vignette", 2))

    if e3.get("extraction_incoherente") is True or e4.get("extraction_incoherente") is True:
        raisons.append(("basse", "extraction_incoherente", "motif ou valeur chiffrée signalé comme contredisant la vignette", 3))
    if ext.get("criteres_rejetes"):
        raisons.append(("basse", "criteres_rejetes",
                        "critère(s) coché(s) sans citation vérifiable : " + ", ".join(r["critere"][:40] for r in ext["criteres_rejetes"]), 3))
    if ext.get("motif_rejete"):
        raisons.append(("basse", "motif_rejete", f"l'Étape 1 avait proposé un motif absent du référentiel ({ext['motif_rejete']!r})", 3))
    if ext.get("appel_b_echec"):
        raisons.append(("basse", "criteres_non_evalues", "l'Appel B de l'Étape 1 a échoué", 3))
    if "etape4" in str(trace.get("_echec", "")):
        raisons.append(("basse", "etape4_echec", "l'audit n'a pas pu être fait", 1))

    trace["a_relire"] = any(p == "haute" for p, _, _, _ in raisons)
    trace["priorite_relecture"] = "haute" if trace["a_relire"] else ("basse" if raisons else None)
    trace["score_relecture"] = max((sc for p, _, _, sc in raisons if p == "haute"), default=0)
    trace["raisons_relecture"] = [f"[{p}] {code} : {texte}" for p, code, texte, _ in raisons]

    notes = []
    for n in d23.get("modulateurs_a_la_baisse") or []:
        notes.append(f"modulateur à la baisse appliqué : {n}")
    trace["notes_validation"] = notes


def traiter_une_vignette(vignette: dict, referentiel: dict, config: dict) -> dict:
    """Fait passer une vignette par les 4 étapes. Retourne la trace complète."""
    texte_source = vignette.get("contexte_clinique") or vignette.get("question", "")
    trace = {"id": vignette["id"], "cas_id": vignette.get("cas_id"), "vignette_source": texte_source}
    # Présent dès le départ, y compris pour les vignettes qui échouent en cours de route.
    trace["provenance"] = {**config["provenance"],
                           "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}

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
    resultat_etape2 = detecter_red_flags(cas, referentiel, texte_source=texte_source)
    trace["etape2_resultat"] = {
        "tri_minimal_force": resultat_etape2.tri_minimal_force,
        "categorie_brief_minimale": resultat_etape2.categorie_brief_minimale,
        "categorie_constantes": resultat_etape2.categorie_constantes,
        "categorie_motif": resultat_etape2.categorie_motif,
        "flags_declenches": resultat_etape2.flags_declenches,
        "modulateurs_a_la_baisse": resultat_etape2.modulateurs_a_la_baisse,
        "constantes_non_verifiees": resultat_etape2.constantes_non_verifiees,
    }

    # --- Étape 3 : labellisation ---
    resultat_etape3 = labelliser(
        extraction, texte_source, config["etape3_fournisseur"], config["etape3_modele"],
        config["prompt_etape3"], config["ids_motifs"],
    )
    trace["etape3_resultat"] = resultat_etape3
    if "_erreur_parsing" in resultat_etape3 or resultat_etape3.get("categorie") not in CATEGORIES_VALIDES:
        trace["_echec"] = "etape3"
        evaluer_relecture(trace, referentiel)
        return trace
    time.sleep(config["pause_interne"])

    # --- Croisement Étape 2 / Étape 3 ---
    decision_23 = appliquer_override(resultat_etape3["categorie"], resultat_etape2)
    trace["decision_apres_override_etape2_3"] = decision_23
    trace["categorie_finale"] = decision_23["label_final"]

    # --- Étape 4 : audit (signalement seul, ne modifie pas le label) ---
    verdict_etape4 = contre_valider(
        extraction, texte_source, decision_23["label_final"],
        config["etape4_fournisseur"], config["etape4_modele"],
    )
    trace["etape4_verdict"] = verdict_etape4
    if "_erreur_parsing" in verdict_etape4:
        # L'Étape 4 échoue : on garde le label d'après l'Étape 2/3 (c'est de toute
        # façon le label final), et on le trace.
        trace["_echec"] = "etape4 (label conservé depuis l'Étape 2/3)"
        evaluer_relecture(trace, referentiel)
        return trace

    decision_finale = resoudre_contre_validation(decision_23["label_final"], verdict_etape4, texte_source)
    trace["decision_finale"] = decision_finale
    trace["categorie_finale"] = decision_finale["categorie_finale"]   # == decision_23["label_final"]

    evaluer_relecture(trace, referentiel)
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


def construire_config(args, referentiel: dict) -> dict:
    if args.tout_haiku:
        choix = {k: ("anthropic", MODELE_HAIKU) for k in DEFAUTS}
    else:
        choix = dict(DEFAUTS)
    config = {"pause_interne": args.pause_interne, "ids_motifs": {m["id"] for m in referentiel["motifs"]}}
    for etape in ("etape1", "etape3", "etape4"):
        # Priorité : option de la ligne de commande > variable du .env (ETAPE3_FOURNISSEUR...) > DEFAUTS.
        # --tout-haiku impose Haiku partout et ignore donc le .env.
        depuis_env = (lambda nom: None) if args.tout_haiku else (lambda nom: (os.environ.get(nom) or "").strip() or None)
        fournisseur_cli = getattr(args, f"{etape}_fournisseur")
        fournisseur = fournisseur_cli or depuis_env(f"{etape.upper()}_FOURNISSEUR") or choix[etape][0]
        # Un fournisseur imposé sans modèle : modèle par défaut de ce fournisseur pour l'étape si le
        # fournisseur est celui du défaut, sinon Haiku pour Anthropic ; les autres exigent --<etape>-modele.
        # Si le fournisseur est imposé en ligne de commande, le modèle du .env (écrit pour un autre
        # fournisseur) n'est pas repris : on l'associerait à un fournisseur qui ne le connaît pas.
        modele = getattr(args, f"{etape}_modele") or (None if fournisseur_cli else depuis_env(f"{etape.upper()}_MODELE"))
        if modele is None:
            if fournisseur == choix[etape][0]:
                modele = choix[etape][1]
            elif fournisseur == "anthropic":
                modele = MODELE_HAIKU
            else:
                sys.exit(f"--{etape}-fournisseur {fournisseur} demande aussi --{etape}-modele "
                         f"(nom de modèle à prendre dans ta console).")
        config[f"{etape}_fournisseur"] = fournisseur
        config[f"{etape}_modele"] = modele
    config["prompt_etape3"] = construire_prompt_etape3(referentiel)
    config["provenance"] = construire_provenance(referentiel, config)
    return config


def main():
    parser = argparse.ArgumentParser(description="Orchestration complète du pipeline (Étapes 1 à 4)")
    parser.add_argument("--input", required=True, help="Vignettes filtrées, anonymisées et dédoublonnées (JSONL)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--pause", type=float, default=3.0, help="Pause entre deux vignettes")
    parser.add_argument("--pause-interne", type=float, default=2.0, help="Pause entre deux appels LLM d'une même vignette")
    parser.add_argument("--tout-haiku", action="store_true",
                        help="Toutes les étapes sur Anthropic Haiku, Étape 4 comprise (repli si un quota gratuit bloque)")
    parser.add_argument("--etape1-fournisseur", default=None)
    parser.add_argument("--etape1-modele", default=None)
    parser.add_argument("--etape3-fournisseur", default=None)
    parser.add_argument("--etape3-modele", default=None)
    parser.add_argument("--etape4-fournisseur", default=None)
    parser.add_argument("--etape4-modele", default=None)
    args = parser.parse_args()

    ref_path = Path(__file__).parent.parent / "referentiel" / "french_referentiel.json"
    referentiel = charger_referentiel(str(ref_path))
    config = construire_config(args, referentiel)

    print("Fournisseurs : "
          + " | ".join(f"{e} = {config[e + '_fournisseur']}/{config[e + '_modele']}" for e in ("etape1", "etape3", "etape4")),
          file=sys.stderr)

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

    n_succes, n_echecs, n_relire = 0, 0, 0
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
            if trace.get("a_relire"):
                n_relire += 1

            f_out.write(json.dumps(trace, ensure_ascii=False) + "\n")
            f_out.flush()

            statut = trace.get("categorie_finale", f"ÉCHEC ({trace.get('_echec')})")
            marque = "  [à relire]" if trace.get("a_relire") else ""
            print(f"  [{i}/{len(vignettes)}] {vignette['id']} -> {statut}{marque}", file=sys.stderr)

            time.sleep(args.pause)

    print(f"\nTerminé : {n_succes} succès, {n_echecs} échecs, {n_relire} à relire -> {output_file}", file=sys.stderr)


if __name__ == "__main__":
    main()
