"""
preparer_dpo_ultramedical.py

Prépare un sous-échantillon traduit en français de TsinghuaC3I/UltraMedical-Preference
(HF, ~175k paires, anglais, licence MIT) pour l'entraînement DPO de l'agent de triage.

Pourquoi traduire : injecter des paires anglaises dans le DPO risquerait d'apprendre
au modèle que répondre en anglais est parfois "la bonne réponse", alors qu'il dérive
déjà vers d'autres langues en génération libre.

Pourquoi filtrer : le dataset contient ~75k paires hors domaine médical et beaucoup
de QCM (biochimie, pharmaco...) sans rapport avec le triage. On ne garde que les
questions dont le texte contient au moins un mot-clé orienté urgence. Filtre
grossier (mots-clés anglais) : il réduit l'inadéquation au triage, il ne la supprime
pas -> limite à documenter dans le rapport.

À PLACER dans src/generation/, à côté de llm_client.py (pas de copie ailleurs).

Usage, en 2 temps :

    # 1. Inspection gratuite (aucun appel LLM) : schéma réel + exemple + taux de filtrage
    python preparer_dpo_ultramedical.py --inspecter-seulement

    # 2. Traduction réelle (reprise automatique si interrompue)
    python preparer_dpo_ultramedical.py --n 150 \
        --output ../../data/dpo/ultramedical_fr_a_relire.jsonl

Sortie : une ligne JSON par paire, anglais et français côte à côte
(prompt_en/prompt_fr, chosen_en/chosen_fr, rejected_en/rejected_fr),
plus a_valider=null à passer à true/false pendant la relecture.
fusionner_dpo.py n'utilise QUE les lignes a_valider=true.
"""

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

DATASET_HF = "TsinghuaC3I/UltraMedical-Preference"
FOURNISSEUR_DEFAUT = "anthropic"
MODELE_DEFAUT = "claude-haiku-4-5-20251001"

# Mots-clés orientés triage / urgence, cherchés dans la QUESTION uniquement.
MOTS_CLES_URGENCE = [
    "emergency", "emergency department", "chest pain", "shortness of breath",
    "dyspnea", "fever", "trauma", "injury", "fall", "syncope", "fainting",
    "unconscious", "loss of consciousness", "seizure", "bleeding", "hemorrhage",
    "stroke", "headache", "abdominal pain", "vomiting", "palpitations",
    "burn", "fracture", "confusion", "allergic reaction", "anaphylaxis",
    "hypotension", "tachycardia", "sepsis", "poisoning", "overdose",
]
MOTIF_URGENCE = re.compile(
    r"\b(" + "|".join(re.escape(m) for m in MOTS_CLES_URGENCE) + r")\b",
    re.IGNORECASE,
)

# Au-delà, la paire coûte cher à traduire, se relit mal et dépasse
# la longueur utile pour un DPO sur T4.
LONGUEUR_MAX_CARACTERES = 4000

PROMPT_SYSTEME_TRADUCTION = (
    "Tu es un traducteur médical anglais -> français, précis et fidèle. "
    "Traduis intégralement la question et les deux réponses fournies, sans rien "
    "résumer, ajouter, corriger ni omettre, y compris si une réponse est fausse : "
    "ses erreurs doivent être conservées telles quelles. Utilise la terminologie "
    "médicale française usuelle et les unités françaises uniquement si elles figurent "
    "déjà dans le texte. Réponds UNIQUEMENT avec un objet JSON de la forme "
    '{"prompt_fr": "...", "chosen_fr": "...", "rejected_fr": "..."}, sans texte autour.'
)


# ---------------------------------------------------------------------------
# Extraction des champs (robuste aux deux formats courants sur HF)
# ---------------------------------------------------------------------------

def texte_depuis(valeur) -> str:
    """Accepte une chaîne, ou une liste de messages {role, content} :
    dans ce cas on garde le dernier message assistant (la réponse)."""
    if valeur is None:
        return ""
    if isinstance(valeur, str):
        return valeur.strip()
    if isinstance(valeur, list):
        assistants = [m.get("content", "") for m in valeur
                      if isinstance(m, dict) and m.get("role") == "assistant"]
        if assistants:
            return assistants[-1].strip()
        contenus = [m.get("content", "") for m in valeur if isinstance(m, dict)]
        return (contenus[-1] if contenus else "").strip()
    return str(valeur).strip()


def extraire_champs(ligne: dict, col_prompt: str, col_chosen: str, col_rejected: str) -> dict:
    return {
        "prompt_en": texte_depuis(ligne.get(col_prompt)),
        "chosen_en": texte_depuis(ligne.get(col_chosen)),
        "rejected_en": texte_depuis(ligne.get(col_rejected)),
    }


def identifiant(prompt_en: str) -> str:
    return "um_" + hashlib.md5(prompt_en.encode("utf-8")).hexdigest()[:12]


def est_candidat(paire: dict) -> bool:
    if not (paire["prompt_en"] and paire["chosen_en"] and paire["rejected_en"]):
        return False
    if paire["chosen_en"] == paire["rejected_en"]:
        return False
    longueur = len(paire["prompt_en"]) + len(paire["chosen_en"]) + len(paire["rejected_en"])
    if longueur > LONGUEUR_MAX_CARACTERES:
        return False
    return bool(MOTIF_URGENCE.search(paire["prompt_en"]))


def passe_filtre_source(ligne: dict, champ_source, valeurs_source) -> bool:
    if not champ_source:
        return True
    return str(ligne.get(champ_source, "")) in valeurs_source


# ---------------------------------------------------------------------------
# Chargement (streaming : évite de télécharger les 175k lignes)
# ---------------------------------------------------------------------------

def charger_flux(seed: int, buffer: int):
    from datasets import load_dataset
    flux = load_dataset(DATASET_HF, split="train", streaming=True)
    # Mélange approximatif : le dataset est probablement ordonné par source,
    # prendre les N premières lignes biaiserait l'échantillon.
    return flux.shuffle(seed=seed, buffer_size=buffer)


# ---------------------------------------------------------------------------
# Mode inspection (gratuit)
# ---------------------------------------------------------------------------

def inspecter(args):
    flux = charger_flux(args.seed, args.buffer)
    n_vus, n_candidats, exemple, colonnes = 0, 0, None, None
    valeurs_par_colonne = {}

    for ligne in flux:
        if colonnes is None:
            colonnes = list(ligne.keys())
            exemple = ligne
        # Colonnes courtes à faible cardinalité = candidates pour filtrer la source
        for cle, val in ligne.items():
            if isinstance(val, (str, int)) and len(str(val)) < 60:
                valeurs_par_colonne.setdefault(cle, {})
                if len(valeurs_par_colonne[cle]) < 30:
                    valeurs_par_colonne[cle][str(val)] = valeurs_par_colonne[cle].get(str(val), 0) + 1
        paire = extraire_champs(ligne, args.col_prompt, args.col_chosen, args.col_rejected)
        if est_candidat(paire):
            n_candidats += 1
        n_vus += 1
        if n_vus >= args.echantillon_inspection:
            break

    print("=== Colonnes ===")
    print(colonnes)
    print("\n=== Exemple (tronqué à 400 caractères par champ) ===")
    for cle, val in (exemple or {}).items():
        print(f"- {cle} ({type(val).__name__}) : {str(val)[:400]}")
    print("\n=== Colonnes à faible cardinalité (utiles pour --champ-source) ===")
    for cle, valeurs in valeurs_par_colonne.items():
        if 1 < len(valeurs) < 30:
            print(f"- {cle} : {valeurs}")
    print(f"\n=== Filtre urgence : {n_candidats}/{n_vus} lignes candidates "
          f"({100 * n_candidats / max(n_vus, 1):.1f} %) ===")
    print("\nSi des champs prompt/chosen/rejected sont vides ci-dessus, relance avec "
          "--col-prompt / --col-chosen / --col-rejected adaptés.")


# ---------------------------------------------------------------------------
# Mode traduction
# ---------------------------------------------------------------------------

def charger_ids_deja_traites(sortie: Path) -> set:
    ids = set()
    if sortie.exists():
        with open(sortie, encoding="utf-8") as f:
            for ligne in f:
                ligne = ligne.strip()
                if ligne:
                    ids.add(json.loads(ligne)["id"])
    return ids


def traduire(paire: dict, fournisseur: str, modele: str) -> dict:
    from llm_client import completer
    contenu = json.dumps(
        {"question": paire["prompt_en"], "reponse_A": paire["chosen_en"], "reponse_B": paire["rejected_en"]},
        ensure_ascii=False,
    )
    resultat = completer(PROMPT_SYSTEME_TRADUCTION, contenu,
                         fournisseur=fournisseur, modele=modele, max_tokens=4000)
    if not isinstance(resultat, dict) or "_erreur_parsing" in resultat:
        raise ValueError(f"réponse non exploitable : {str(resultat)[:200]}")
    for cle in ("prompt_fr", "chosen_fr", "rejected_fr"):
        if not str(resultat.get(cle, "")).strip():
            raise ValueError(f"champ {cle} vide dans la traduction")
    return resultat


def executer(args):
    sortie = Path(args.output)
    sortie.parent.mkdir(parents=True, exist_ok=True)
    deja = charger_ids_deja_traites(sortie)
    if deja:
        print(f"Reprise détectée : {len(deja)} paires déjà traduites, elles seront sautées.")
    if len(deja) >= args.n:
        print("Objectif déjà atteint, rien à faire.")
        return

    valeurs_source = set(args.valeurs_source.split(",")) if args.valeurs_source else set()
    n_ok, n_erreurs, n_vus = len(deja), 0, 0

    with open(sortie, "a", encoding="utf-8") as f_out:
        for ligne in charger_flux(args.seed, args.buffer):
            n_vus += 1
            if n_vus > args.max_lignes_parcourues:
                print(f"Arrêt : {args.max_lignes_parcourues} lignes parcourues sans atteindre --n.")
                break
            if not passe_filtre_source(ligne, args.champ_source, valeurs_source):
                continue
            paire = extraire_champs(ligne, args.col_prompt, args.col_chosen, args.col_rejected)
            if not est_candidat(paire):
                continue
            pid = identifiant(paire["prompt_en"])
            if pid in deja:
                continue

            try:
                trad = traduire(paire, args.fournisseur, args.modele)
            except Exception as e:  # une paire ratée ne doit pas arrêter le lot
                n_erreurs += 1
                print(f"  [échec] {pid} : {e}", file=sys.stderr)
                continue

            enregistrement = {
                "id": pid,
                "source": DATASET_HF,
                "prompt_id": ligne.get("prompt_id"),
                "label_type": ligne.get("label_type"),
                **paire,
                "prompt_fr": trad["prompt_fr"].strip(),
                "chosen_fr": trad["chosen_fr"].strip(),
                "rejected_fr": trad["rejected_fr"].strip(),
                "mot_cle_detecte": MOTIF_URGENCE.search(paire["prompt_en"]).group(0).lower(),
                "traduit_par": f"{args.fournisseur}/{args.modele}",
                "a_valider": None,
            }
            # Écriture immédiate + flush : rien n'est perdu si le script est interrompu
            f_out.write(json.dumps(enregistrement, ensure_ascii=False) + "\n")
            f_out.flush()
            deja.add(pid)
            n_ok += 1
            print(f"[{n_ok}/{args.n}] {pid} ({enregistrement['mot_cle_detecte']})")
            if n_ok >= args.n:
                break
            time.sleep(args.pause)

    print(f"\nTerminé : {n_ok} paires dans {sortie}, {n_erreurs} échecs, {n_vus} lignes parcourues.")
    print("Relecture : passer a_valider à true ou false sur chaque ligne.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inspecter-seulement", action="store_true")
    p.add_argument("--n", type=int, default=150, help="nombre de paires traduites visé")
    p.add_argument("--output", default="../../data/dpo/ultramedical_fr_a_relire.jsonl")
    p.add_argument("--col-prompt", default="prompt")
    p.add_argument("--col-chosen", default="chosen")
    p.add_argument("--col-rejected", default="rejected")
    p.add_argument("--champ-source", default=None,
                   help="colonne permettant de ne garder que le médical (voir l'inspection)")
    p.add_argument("--valeurs-source", default=None,
                   help="valeurs acceptées pour --champ-source, séparées par des virgules")
    p.add_argument("--fournisseur", default=FOURNISSEUR_DEFAUT)
    p.add_argument("--modele", default=MODELE_DEFAUT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--buffer", type=int, default=10000, help="taille du buffer de mélange")
    p.add_argument("--pause", type=float, default=0.5)
    p.add_argument("--echantillon-inspection", type=int, default=2000)
    p.add_argument("--max-lignes-parcourues", type=int, default=50000)
    args = p.parse_args()

    if args.inspecter_seulement:
        inspecter(args)
    else:
        executer(args)


if __name__ == "__main__":
    main()
