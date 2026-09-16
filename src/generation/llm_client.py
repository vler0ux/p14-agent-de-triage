"""
llm_client.py

Petite couche d'abstraction pour appeler soit Groq soit Google Gemini avec
la même interface — permet d'utiliser un fournisseur différent d'une étape
à l'autre du pipeline (ex. Étape 3 sur Groq pendant que le filtre tourne
sur Gemini, ou l'inverse), et prépare le terrain pour l'Étape 4 où l'on
veut un modèle DIFFÉRENT de celui de l'Étape 3 (réduction de la corrélation
des erreurs entre les deux jugements).

⚠️ Les limites du tier gratuit de Gemini ne sont plus publiées de façon
fiable par Google (chiffres contradictoires selon les sources en ligne,
variables par projet) — vérifiez vos propres limites dans Google AI Studio
après création de votre clé, plutôt que de vous fier à un chiffre trouvé
sur le web.

⚠️ Le package `google-generativeai` est officiellement déprécié depuis peu.
Ce module utilise le nouveau SDK `google-genai` (import `from google import
genai`) — installez bien `pip install google-genai`, PAS
`google-generativeai`.

Usage :
    from llm_client import completer
    resultat = completer(prompt_systeme, contenu_utilisateur,
                          fournisseur="gemini", modele="gemini-2.5-flash")
    # résultat est un dict (JSON déjà parsé), ou {"_erreur_parsing": "..."}
"""

import json
import os
import re
import sys
import time

FOURNISSEURS_VALIDES = {"groq", "gemini"}


def _extraire_delai_attente(message_erreur: str, defaut: float = 70.0) -> float:
    """Tente d'extraire un délai d'attente suggéré dans le message d'erreur
    (formats variables selon le fournisseur) ; retombe sur une valeur par
    défaut prudente si rien n'est trouvé."""
    match = re.search(r"try again in (\d+)m([\d.]+)s", message_erreur)
    if match:
        minutes, secondes = int(match.group(1)), float(match.group(2))
        return minutes * 60 + secondes + 5
    match = re.search(r"try again in ([\d.]+)s", message_erreur)
    if match:
        return float(match.group(1)) + 5
    match = re.search(r"retry.{0,20}?(\d+(?:\.\d+)?)\s*second", message_erreur, re.IGNORECASE)
    if match:
        return float(match.group(1)) + 5
    return defaut


def _appeler_groq(system_prompt: str, user_content: str, modele: str, max_tokens: int, max_tentatives: int) -> str:
    import groq

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY manquant dans .env")
    client = groq.Groq(api_key=api_key)

    for tentative in range(1, max_tentatives + 1):
        try:
            reponse = client.chat.completions.create(
                model=modele,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            )
            return reponse.choices[0].message.content.strip()
        except groq.RateLimitError as e:
            attente = _extraire_delai_attente(str(e))
            print(f"  [Groq] Quota atteint (tentative {tentative}/{max_tentatives}), attente {attente}s...",
                  file=sys.stderr)
            time.sleep(attente)

    raise RuntimeError("Quota Groq toujours dépassé après plusieurs tentatives")


def _appeler_gemini(system_prompt: str, user_content: str, modele: str, max_tokens: int, max_tentatives: int) -> str:
    from google import genai
    from google.genai import errors, types

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY manquant dans .env")
    client = genai.Client(api_key=api_key)

    for tentative in range(1, max_tentatives + 1):
            try:
                config = types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    max_output_tokens=max_tokens,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                )
            except (AttributeError, TypeError):
                # Selon la version du SDK/modèle, ThinkingConfig peut ne pas
                # être disponible — on retombe sur la config sans, plutôt
                # que de planter.
                config = types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    max_output_tokens=max_tokens,
                )
            reponse = client.models.generate_content(
                model=modele,
                contents=user_content,
                config=config,
            )
            return reponse.text.strip()

    raise RuntimeError("Quota Gemini toujours dépassé après plusieurs tentatives")


def completer(system_prompt: str, user_content: str, fournisseur: str, modele: str,
              max_tokens: int = 400, max_tentatives: int = 5) -> dict:
    """Point d'entrée unique : appelle le fournisseur demandé, nettoie et
    parse la réponse en JSON. Retourne {"_erreur_parsing": "..."} si le
    modèle n'a pas renvoyé un JSON exploitable."""
    if fournisseur not in FOURNISSEURS_VALIDES:
        raise ValueError(f"Fournisseur inconnu : {fournisseur!r} (attendu : {FOURNISSEURS_VALIDES})")

    if fournisseur == "groq":
        texte = _appeler_groq(system_prompt, user_content, modele, max_tokens, max_tentatives)
    else:
        texte = _appeler_gemini(system_prompt, user_content, modele, max_tokens, max_tentatives)

    texte_nettoye = texte.strip()
    if texte_nettoye.startswith("```"):
        texte_nettoye = re.sub(r"^```(?:json)?\s*|\s*```$", "", texte_nettoye).strip()

    try:
        return json.loads(texte_nettoye)
    except json.JSONDecodeError:
        return {"_erreur_parsing": texte[:300]}