"""
filtre_pertinence.py

Étape 0 du pipeline : filtre les vignettes normalisées pour ne garder que
celles qui décrivent une présentation clinique AIGUË, plausible pour un
passage aux urgences — par opposition aux consultations programmées, aux
suivis chroniques, ou aux questions sans patient (méthodologie, épidémio...).

Contexte : sur un échantillon aléatoire de 15 vignettes MediQAl avec
contexte clinique, seules 1 à 3 décrivaient une vraie urgence (cf. analyse
manuelle documentée dans le projet). Un filtre par mots-clés a été écarté
au profit d'une classification par LLM (cf. justification dans le README).

⚠️ ROBUSTESSE FACE AU QUOTA GRATUIT :
Les quotas gratuits (Groq ou autre) peuvent être atteints en cours de route
sur un volume de plusieurs milliers de vignettes. Ce script :
  - écrit chaque résultat au fil de l'eau (pas seulement à la fin), pour ne
    jamais perdre le travail déjà fait ni les appels déjà payés en tokens ;
  - reprend automatiquement là où il s'était arrêté si on le relance sur
    les mêmes fichiers de sortie (--resume, activé par défaut) ;
  - retente automatiquement avec un délai d'attente en cas de dépassement
    de quota (HTTP 429), au lieu de planter tout le batch.

Usage :
    python filtre_pertinence.py \
        --input ../../data/normalized/mediqal_mcqm_train.jsonl \
        --output-retenues ../../data/normalized/mediqal_mcqm_urgences.jsonl \
        --output-exclues ../../data/normalized/mediqal_mcqm_exclues.jsonl

    # Si le quota est atteint en cours de route, relancez EXACTEMENT la même
    # commande : les vignettes déjà traitées seront automatiquement sautées.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROMPT_SYSTEME = """Tu es un assistant qui aide à filtrer un corpus de questions médicales d'examen pour un projet de triage aux urgences.

Pour chaque cas clinique fourni, détermine s'il décrit une présentation AIGUË plausible pour un passage aux urgences — c'est-à-dire une situation où un patient se présenterait dans les heures suivant l'apparition ou l'aggravation de symptômes, et non une consultation programmée, un suivi de maladie chronique, un bilan de routine, ou une question sans patient réel (méthodologie, épidémiologie...).

Réponds UNIQUEMENT avec un objet JSON, sans aucun texte avant ou après, au format exact :
{"urgence": true ou false, "justification": "une phrase courte expliquant pourquoi"}

Exemples de décisions, tirés de vraies vignettes déjà analysées manuellement pour ce projet :
- "admise en urgence pour altération de l'état général avec obnubilation, nausées, dyspnée et rachialgies" → urgence: true (présentation aiguë explicite, motif de recours clair)
- "présente pour la première fois de son existence une crise dyspnéique avec sensation d'étouffement qui le réveille" → urgence: true (épisode aigu inaugural, même sans le mot "urgence")
- "gonalgie interne s'accompagnant de gonflement, douleur vive à la pression, léger flessum irréductible" → urgence: true (motif de recours aigu plausible aux urgences, MÊME SI la gravité finale sera probablement faible — ce filtre ne juge PAS la gravité, seulement la plausibilité d'un passage aux urgences)
- "consulte pour que lui soit prescrit un traitement oestroprogestatif anti-conceptionnel" → urgence: false (consultation programmée, pas de symptôme aigu)
- "consulte pour des troubles de mémoire, installés progressivement depuis quelques mois" → urgence: false (suivi chronique)
- question sur une méthode d'étude en double aveugle, sans patient décrit → urgence: false (pas un cas clinique)
- "revient consulter pour bilan annuel systématique" → urgence: false (consultation programmée, même après un antécédent chirurgical lourd)
"""


def classifier_vignette(client, vignette: dict, modele: str) -> dict:
    """Appelle le LLM pour classer une vignette. Retourne {"urgence": bool, "justification": str}."""
    texte_cas = vignette.get("contexte_clinique") or ""

    reponse = client.chat.completions.create(
        model=modele,
        max_tokens=200,
        messages=[
            {"role": "system", "content": PROMPT_SYSTEME},
            {"role": "user", "content": f"Cas clinique :\n{texte_cas}"},
        ],
    )
    texte_reponse = reponse.choices[0].message.content.strip()

    try:
        resultat = json.loads(texte_reponse)
    except json.JSONDecodeError:
        resultat = {"urgence": False, "justification": f"ERREUR DE PARSING, réponse brute : {texte_reponse[:200]}"}

    return resultat


def classifier_avec_retry(client, vignette: dict, modele: str, max_tentatives: int = 5) -> dict:
    """Enveloppe classifier_vignette() avec une reprise automatique en cas de
    dépassement de quota (HTTP 429), au lieu de laisser planter tout le batch."""
    import groq

    for tentative in range(1, max_tentatives + 1):
        try:
            return classifier_vignette(client, vignette, modele)
        except groq.RateLimitError as e:
            attente = _extraire_delai_attente(str(e))
            print(f"  Quota atteint (tentative {tentative}/{max_tentatives}), "
                  f"attente de {attente}s avant de réessayer...", file=sys.stderr)
            time.sleep(attente)

    raise RuntimeError(
        f"Quota toujours dépassé après {max_tentatives} tentatives. "
        f"Relancez le script plus tard avec la même commande : la reprise "
        f"automatique repartira de là où on s'est arrêté."
    )


def _extraire_delai_attente(message_erreur: str, defaut: float = 70.0) -> float:
    """Extrait le délai d'attente suggéré par l'API (ex. 'try again in 1m16.464s'),
    avec une marge de sécurité. Retombe sur une valeur par défaut si non trouvé."""
    match = re.search(r"try again in (\d+)m([\d.]+)s", message_erreur)
    if match:
        minutes, secondes = int(match.group(1)), float(match.group(2))
        return minutes * 60 + secondes + 5  # marge de sécurité de 5s
    match = re.search(r"try again in ([\d.]+)s", message_erreur)
    if match:
        return float(match.group(1)) + 5
    return defaut


def charger_ids_deja_traites(chemin_retenues: Path, chemin_exclues: Path) -> set:
    """Lit les fichiers de sortie déjà existants pour connaître les vignettes
    déjà traitées lors d'une exécution précédente (mécanisme de reprise)."""
    ids = set()
    for chemin in (chemin_retenues, chemin_exclues):
        if chemin.exists():
            with open(chemin, encoding="utf-8") as f:
                for ligne in f:
                    ligne = ligne.strip()
                    if ligne:
                        ids.add(json.loads(ligne)["id"])
    return ids


def main():
    parser = argparse.ArgumentParser(description="Filtre de pertinence clinique (Étape 0 du pipeline)")
    parser.add_argument("--input", required=True, help="Fichier JSONL de vignettes normalisées")
    parser.add_argument("--output-retenues", required=True, help="Fichier de sortie : vignettes jugées pertinentes")
    parser.add_argument("--output-exclues", required=True, help="Fichier de sortie : vignettes exclues, avec justification (pour audit)")
    parser.add_argument("--limit", type=int, default=None, help="Limiter le nombre de vignettes traitées (debug/coût)")
    parser.add_argument("--modele", default="openai/gpt-oss-20b", help="Modèle Groq à utiliser")
    parser.add_argument("--pause", type=float, default=2.0, help="Pause en secondes entre deux appels (défaut : 2s, reste sous 30 req/min)")
    parser.add_argument("--no-resume", action="store_true", help="Ignorer la reprise automatique et tout retraiter depuis le début")
    args = parser.parse_args()

    try:
        from groq import Groq
    except ImportError:
        print("La librairie 'groq' n'est pas installée. Lancez : pip install groq", file=sys.stderr)
        sys.exit(1)

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("GROQ_API_KEY n'est pas défini. Renseignez-le dans votre fichier .env "
              "(clé gratuite sur console.groq.com/keys).", file=sys.stderr)
        sys.exit(1)

    client = Groq(api_key=api_key)

    vignettes = []
    with open(args.input, encoding="utf-8") as f:
        for ligne in f:
            ligne = ligne.strip()
            if ligne:
                v = json.loads(ligne)
                if v.get("contexte_clinique"):
                    vignettes.append(v)

    if args.limit:
        vignettes = vignettes[: args.limit]

    output_retenues = Path(args.output_retenues)
    output_exclues = Path(args.output_exclues)
    output_retenues.parent.mkdir(parents=True, exist_ok=True)
    output_exclues.parent.mkdir(parents=True, exist_ok=True)

    ids_deja_traites = set()
    if not args.no_resume:
        ids_deja_traites = charger_ids_deja_traites(output_retenues, output_exclues)
        if ids_deja_traites:
            print(f"Reprise détectée : {len(ids_deja_traites)} vignettes déjà traitées lors d'un "
                  f"lancement précédent, elles seront sautées.", file=sys.stderr)

    a_traiter = [v for v in vignettes if v["id"] not in ids_deja_traites]
    print(f"{len(vignettes)} vignettes avec contexte clinique au total, "
          f"{len(a_traiter)} restant à classer...", file=sys.stderr)

    n_retenues, n_exclues = 0, 0
    # Ouverture en mode "append" : chaque résultat est écrit immédiatement,
    # pas seulement à la fin — c'est ce qui évite de perdre le travail déjà
    # fait (et déjà payé en tokens) en cas d'interruption ou de quota atteint.
    with open(output_retenues, "a", encoding="utf-8") as f_retenues, \
         open(output_exclues, "a", encoding="utf-8") as f_exclues:

        for i, vignette in enumerate(a_traiter, 1):
            resultat = classifier_avec_retry(client, vignette, modele=args.modele)
            vignette["_filtre_urgence"] = resultat["urgence"]
            vignette["_filtre_justification"] = resultat["justification"]

            if resultat["urgence"]:
                f_retenues.write(json.dumps(vignette, ensure_ascii=False) + "\n")
                f_retenues.flush()
                n_retenues += 1
            else:
                f_exclues.write(json.dumps(vignette, ensure_ascii=False) + "\n")
                f_exclues.flush()
                n_exclues += 1

            if i % 20 == 0 or i == len(a_traiter):
                print(f"  {i}/{len(a_traiter)} traitées cette session — "
                      f"{n_retenues} retenues, {n_exclues} exclues", file=sys.stderr)

            if args.pause:
                time.sleep(args.pause)

    print(f"\nTerminé pour cette session : {n_retenues} retenues, {n_exclues} exclues.", file=sys.stderr)
    print(f"  Retenues (cumul) -> {output_retenues}", file=sys.stderr)
    print(f"  Exclues (cumul, avec justification) -> {output_exclues}", file=sys.stderr)


if __name__ == "__main__":
    main()