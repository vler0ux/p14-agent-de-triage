"""
decideur.py

Décision de priorité de la démo, séparée de l'hôtesse (Qwen) :
  1. plancher de sécurité par mots-clés sur les messages du patient (déterministe, local) ;
  2. sinon, décideur encodeur (ModernCamemBERT-bio-v2-base fine-tuné, variante « patient »)
     avec la règle de sécurité : urgence si P(urgence) >= SEUIL_URGENCE, sinon argmax.
La phrase de conclusion est choisie par le code, jamais générée.

Les constantes (LABELS, SEUIL_URGENCE) et la mise en forme des transcriptions sont importées
des scripts d'entraînement (src/decideur/) : source unique, pas de copie qui pourrait diverger.
"""
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "decideur"))
from preparer_dataset_decideur import formater_transcription  # noqa: E402
from train_decideur import LABELS, SEUIL_URGENCE  # noqa: E402

# Variante retenue en comparant resultats_validation.json des deux runs (complet / patient).
# Ce choix a été fait SUR la validation : les métriques de validation de cette variante sont donc
# légèrement optimistes (pas de jeu de test indépendant). Voir les intervalles de confiance (intervalles_95).
VARIANTE = "patient"

# Phrases de conclusion du dataset (étape 5) : on prend la première de chaque niveau,
# pour une sortie déterministe et traçable.
CONCLUSIONS = {
    "urgence_maximale": "Vous devez venir aux urgences tout de suite.",
    "moderee": "Venez nous voir d'ici peu dans la journée.",
    "differee": "Vous pouvez patienter, mais rappelez-nous si ça s'aggrave.",
}

# Plancher simplifié : signes d'alerte majeurs, formulés comme un patient les dit.
# Plus pauvre que la grille FRENCH du pipeline (limite assumée et documentée).
# Les textes sont comparés sans accents et en minuscules.
SIGNES_ALERTE = {
    "deficit_moteur": r"(peu[tx]|arrive) (plus|pas) (a )?(bouger|lever) (son|sa|ses|le|la|les|mon|ma|mes) (bras|jambe|main|cote)|paralys|bouche de travers|visage (de travers|deforme)",
    "trouble_parole": r"(arrive|peu[tx]) (plus|pas) a parler|trouver (ses|mes) mots|parle (plus|pas) (bien|correctement)|ne parle plus",
    "trouble_conscience": r"perdu connaissance|perte de connaissance|inconscient|evanoui|ne repond plus|convuls",
    "douleur_thoracique": r"(douleur|mal|serre|oppression)\w* (a|dans|sur) (la|le) (poitrine|thorax)|douleur thoracique",
    "detresse_respiratoire": r"((du )?mal a respirer|(arrive|peu[tx]) (plus|pas) a respirer)(?! par le nez)|etouff|levres bleues",
    "hemorragie_abondante": r"saigne (beaucoup|enormement|abondamment|sans arret)|hemorragie",
}
# Négation juste avant l'expression (« pas de douleur à la poitrine », « j'ai pas perdu connaissance ») :
# on ne déclenche pas. « je sais pas » n'est pas une négation du symptôme.
NEGATION_AVANT = re.compile(r"(?<!sais )\b(pas|ni|aucune?|sans|jamais)\b(\W+\w+){0,3}\W*$")


def normaliser(texte):
    texte = unicodedata.normalize("NFD", texte.lower().replace("’", "'"))
    return "".join(c for c in texte if unicodedata.category(c) != "Mn")


def plancher(messages):
    """Signes d'alerte détectés dans les messages du patient (liste vide si aucun)."""
    texte = normaliser(" ".join(m["content"] for m in messages if m["role"] == "user"))
    signes = []
    for nom, motif in SIGNES_ALERTE.items():
        for trouve in re.finditer(motif, texte):
            if not NEGATION_AVANT.search(texte[max(0, trouve.start() - 30):trouve.start()]):
                signes.append(nom)
                break
    return signes


class Decideur:
    def __init__(self, dossier_modele):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(dossier_modele)
        self.modele = AutoModelForSequenceClassification.from_pretrained(dossier_modele).eval()
        # Garde-fou : l'ordre des classes du modèle doit être celui du code
        ordre = [self.modele.config.id2label[i] for i in range(len(LABELS))]
        assert tuple(ordre) == LABELS, f"ordre des classes incohérent : {ordre} != {LABELS}"
        self.dossier = str(dossier_modele)

    def decider(self, messages):
        """Retourne niveau, conclusion, source (plancher / decideur), signes, probabilités."""
        texte = formater_transcription(messages, VARIANTE)
        entrees = self.tokenizer(texte, truncation=True, max_length=768, return_tensors="pt")
        with self.torch.no_grad():
            probas = self.torch.softmax(self.modele(**entrees).logits[0], dim=-1).tolist()
        probas = {lab: round(p, 4) for lab, p in zip(LABELS, probas)}

        signes = plancher(messages)
        if signes:
            niveau, source = "urgence_maximale", "plancher"
        elif probas["urgence_maximale"] >= SEUIL_URGENCE:
            niveau, source = "urgence_maximale", "decideur_seuil"
        else:
            niveau, source = max(probas, key=probas.get), "decideur"
        return {
            "niveau": niveau,
            "conclusion": CONCLUSIONS[niveau],
            "source": source,
            "signes_alerte": signes,
            "probas": probas,
            "seuil_urgence": SEUIL_URGENCE,
            "modele_decideur": self.dossier,
        }