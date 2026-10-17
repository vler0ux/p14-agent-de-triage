"""
defauts.py

Détecteurs de défauts des répliques de l'hôtesse, pour le DPO v6 :
filtrer_rejected.py s'en sert pour choisir, parmi les réponses générées par
preparer_dpo_rejected.py, celles qui deviennent des "rejected".

Fonctions pures, sans dépendance lourde : testées dans la CI (tests/test_defauts_dpo.py).

Ce sont des RÈGLES SIMPLES, donc imparfaites : chaque détection garde le motif qui l'a
déclenchée, pour qu'une relecture humaine puisse calibrer les règles avant de s'en servir.
Chaque taux mesuré sur le modèle se lit À CÔTÉ du même taux mesuré sur les répliques de
référence du dataset SFT : un défaut aussi fréquent dans la référence est un style appris
des données, pas une dérive du modèle, et le corriger par DPO contredirait le SFT.
"""

import re

BALISE_CONCLUSION = "[[PRIORITE:"

# Fragments indésirables observés ou plausibles en sortie de Qwen3-Base.
ARTEFACTS = ("atrigesimal", "<|im_start|>", "<|endoftext|>", "<think>", "</think>")

# Rassurance qui minimise, ou hypothèse diagnostique : l'agent ne doit faire ni l'un ni
# l'autre (il recueille, il ne juge pas). Liste volontairement courte, à calibrer.
MOTIFS_RASSURANCE_DIAGNOSTIC = {
    "ce n'est rien": r"\bce n'est (?:rien|pas grave)\b",
    "pas de quoi s'inquiéter": r"\bpas de quoi (?:vous |s')inqui[eé]t",
    "ne vous inquiétez pas": r"\bne vous inqui[eé]tez pas\b",
    "c'est sûrement / probablement": r"\bc'est (?:s[uû]rement|probablement|certainement|s[uû]r que)\b",
    "ça ressemble à": r"\b[çc]a (?:ressemble|fait penser) [àa]\b",
    "vous avez sûrement": r"\bvous (?:avez|faites) (?:s[uû]rement|probablement|certainement)\b",
    "diagnostic": r"\bdiagnostic\b",
}
_MOTIFS_COMPILES = {nom: re.compile(motif, re.IGNORECASE) for nom, motif in MOTIFS_RASSURANCE_DIAGNOSTIC.items()}

# Langue : comptage de mots-outils, suffisant pour repérer une réplique qui bascule en anglais.
MOTS_FR = {"le", "la", "les", "de", "des", "du", "un", "une", "et", "est", "vous", "je", "pas",
           "que", "qui", "avez", "votre", "pour", "dans", "en", "il", "elle", "ce", "ça"}
MOTS_EN = {"the", "and", "is", "you", "your", "are", "have", "to", "of", "in", "it", "do",
           "what", "how", "with", "for", "this", "that", "please"}
_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")


def nettoyer(reponse_brute: str) -> str:
    """Même nettoyage que deploiement/space/app.py : coupe au premier marqueur de fin."""
    return (reponse_brute.split("<|im_end|>")[0].split("<|im_start|>")[0]
            .split("atrigesimal")[0].strip())


def conclut(texte: str) -> bool:
    return BALISE_CONCLUSION in texte


def nombre_questions(texte: str) -> int:
    """Nombre de phrases interrogatives (« ?? » compte pour une)."""
    return len(re.findall(r"\?+", texte))


def rassurance_ou_diagnostic(texte: str) -> list:
    """Noms des motifs trouvés (liste vide si aucun)."""
    return [nom for nom, motif in _MOTIFS_COMPILES.items() if motif.search(texte)]


def non_francais(texte: str) -> bool:
    if _CJK.search(texte):
        return True
    mots = re.findall(r"[a-zàâçéèêëîïôûùüÿœ']+", texte.lower())
    n_fr = sum(m in MOTS_FR for m in mots)
    n_en = sum(m in MOTS_EN for m in mots)
    return n_en >= 3 and n_en > n_fr


def artefacts(reponse_brute: str) -> list:
    """Fragments indésirables présents dans la sortie brute, avant le premier <|im_end|>."""
    avant_fin = reponse_brute.split("<|im_end|>")[0]
    return [a for a in ARTEFACTS if a in avant_fin]


def nombre_mots(texte: str) -> int:
    return len(texte.split())


def evaluer(reponse: str, *, reference_conclut: bool, seuil_mots: int,
            reponse_brute: str = None, tronquee: bool = False) -> dict:
    """Défauts d'une réplique.

    reference_conclut : la réplique de référence, au même tour, conclut-elle ?
    seuil_mots        : au-delà, réplique « trop longue » (p95 des répliques de référence).
    reponse_brute     : sortie avant nettoyage (pour les artefacts) ; None pour une référence.
    tronquee          : génération arrêtée par max_new_tokens, sans marqueur de fin.
    """
    conclusion = conclut(reponse)
    # Une conclusion de l'hôtesse est remplacée par celle du décideur dans l'application :
    # son contenu (rassurance, longueur) n'atteint jamais le patient, on ne le juge pas.
    motifs = [] if conclusion else rassurance_ou_diagnostic(reponse)
    trouves = artefacts(reponse_brute) if reponse_brute is not None else []
    return {
        "vide": not reponse.strip(),
        "plusieurs_questions": (not conclusion) and nombre_questions(reponse) >= 2,
        "conclusion_prematuree": conclusion and not reference_conclut,
        "conclusion_manquee": reference_conclut and not conclusion,
        "rassurance_ou_diagnostic": bool(motifs),
        "non_francais": non_francais(reponse),
        "artefact": bool(trouves) or tronquee,
        "trop_longue": (not conclusion) and nombre_mots(reponse) > seuil_mots,
        "_motifs_rassurance": motifs,
        "_artefacts": trouves + (["tronquée (max_new_tokens)"] if tronquee else []),
    }


DEFAUTS = ["vide", "plusieurs_questions", "conclusion_prematuree", "conclusion_manquee",
           "rassurance_ou_diagnostic", "non_francais", "artefact", "trop_longue"]
