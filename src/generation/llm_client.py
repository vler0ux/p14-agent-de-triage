"""
llm_client.py

Petite couche d'abstraction pour appeler Groq, Google Gemini, ou l'API
Anthropic (Claude) avec la même interface — permet d'utiliser un
fournisseur différent d'une étape à l'autre du pipeline.

⚠️ Les limites du tier gratuit de Gemini varient énormément selon le modèle
(certains modèles récents n'ont que 20 requêtes/jour sur le tier gratuit) —
vérifiez vos propres limites dans Google AI Studio plutôt que de vous fier
à un chiffre trouvé en ligne.

⚠️ Le package `google-generativeai` est officiellement déprécié. Ce module
utilise le nouveau SDK `google-genai` (import `from google import genai`).

⚠️ L'API Anthropic est PAYANTE (compte séparé sur console.anthropic.com,
différent d'un abonnement claude.ai) — facturation à l'usage.

Usage :
    from llm_client import completer
    resultat = completer(prompt_systeme, contenu_utilisateur,
                          fournisseur="anthropic", modele="claude-haiku-4-5-20251001")
    # résultat est un dict (JSON déjà parsé), ou {"_erreur_parsing": "..."}
"""

import json
import os
import re
import sys
import time

FOURNISSEURS_VALIDES = {"groq", "gemini", "anthropic"}


def _extraire_delai_attente(message_erreur: str, defaut: float = 70.0) -> float:
    """Tente d'extraire un délai d'attente suggéré dans le message d'erreur
    (formats variables selon le fournisseur) ; retombe sur une valeur par
    défaut prudente si rien n'est trouvé."""
    match = re.search(r"try again in (\d+)m([\d.]+)s", message_erreur)
    if match:
        minutes, secondes = int(match.group(1)), float(match.group(2))
        return minutes * 60 + secondes + 5
    match = re.search(r"(?:try again|retry) in ([\d.]+)s", message_erreur, re.IGNORECASE)
    if match:
        return float(match.group(1)) + 5
    match = re.search(r"retryDelay.{0,10}?(\d+(?:\.\d+)?)\s*s", message_erreur, re.IGNORECASE)
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


def _construire_config_gemini(types_module, system_prompt: str, max_tokens: int):
    """Construit la config Gemini, avec repli si thinking_config n'est pas
    disponible sur cette version du SDK/modèle."""
    try:
        return types_module.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=max_tokens,
            thinking_config=types_module.ThinkingConfig(thinking_budget=0),
        )
    except (AttributeError, TypeError):
        return types_module.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=max_tokens,
        )


def _appeler_gemini(system_prompt: str, user_content: str, modele: str, max_tokens: int, max_tentatives: int) -> str:
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY manquant dans .env")
    client = genai.Client(api_key=api_key)

    config = _construire_config_gemini(types, system_prompt, max_tokens)

    for tentative in range(1, max_tentatives + 1):
        try:
            reponse = client.models.generate_content(
                model=modele,
                contents=user_content,
                config=config,
            )
            return reponse.text.strip()
        except Exception as e:
            message = str(e)
            est_transitoire = any(motif in message for motif in [
                "429", "RESOURCE_EXHAUSTED", "quota",
                "503", "UNAVAILABLE", "500", "INTERNAL", "high demand", "overloaded",
            ]) or "quota" in message.lower()
            if not est_transitoire:
                raise
            attente = _extraire_delai_attente(message, defaut=60.0)
            print(f"  [Gemini] Quota atteint (tentative {tentative}/{max_tentatives}), attente {attente}s...",
                  file=sys.stderr)
            time.sleep(attente)

    raise RuntimeError("Quota Gemini toujours dépassé après plusieurs tentatives")


def _appeler_anthropic(system_prompt: str, user_content: str, modele: str, max_tokens: int, max_tentatives: int) -> str:
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY manquant dans .env")
    client = anthropic.Anthropic(api_key=api_key)

    for tentative in range(1, max_tentatives + 1):
        try:
            reponse = client.messages.create(
                model=modele,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_content}],
            )
            return reponse.content[0].text.strip()
        except Exception as e:
            message = str(e)
            # 429 = quota/rate limit ; 529 = surcharge momentanée des serveurs
            # Anthropic (code spécifique à leur API, différent du 503 usuel).
            est_transitoire = any(motif in message for motif in [
                "429", "rate_limit", "529", "overloaded", "503", "internal_server",
            ])
            if not est_transitoire:
                raise
            attente = _extraire_delai_attente(message, defaut=30.0)
            print(f"  [Anthropic] Quota/surcharge (tentative {tentative}/{max_tentatives}), attente {attente}s...",
                  file=sys.stderr)
            time.sleep(attente)

    raise RuntimeError("Quota/surcharge Anthropic toujours présent après plusieurs tentatives")


def completer(system_prompt: str, user_content: str, fournisseur: str, modele: str,
              max_tokens: int = 400, max_tentatives: int = 5) -> dict:
    """Point d'entrée unique : appelle le fournisseur demandé, nettoie et
    parse la réponse en JSON. Retourne {"_erreur_parsing": "..."} si le
    modèle n'a pas renvoyé un JSON exploitable."""
    if fournisseur not in FOURNISSEURS_VALIDES:
        raise ValueError(f"Fournisseur inconnu : {fournisseur!r} (attendu : {FOURNISSEURS_VALIDES})")

    if fournisseur == "groq":
        texte = _appeler_groq(system_prompt, user_content, modele, max_tokens, max_tentatives)
    elif fournisseur == "anthropic":
        texte = _appeler_anthropic(system_prompt, user_content, modele, max_tokens, max_tentatives)
    else:
        texte = _appeler_gemini(system_prompt, user_content, modele, max_tokens, max_tentatives)

    texte_nettoye = texte.strip()
    if texte_nettoye.startswith("```"):
        texte_nettoye = re.sub(r"^```(?:json)?\s*|\s*```$", "", texte_nettoye).strip()

    try:
        return json.loads(texte_nettoye)
    except json.JSONDecodeError:
        return {"_erreur_parsing": texte[:300]}