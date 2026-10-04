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
# Biais connu : le décideur se fie en partie au ton du patient (artefact de génération des données,
# mesuré par src/decideur/evaluer_raccourci_style.py) ; le plancher ci-dessous sert de contrepoids.
VARIANTE = "patient"

# Phrases de conclusion du dataset (étape 5) : la plus fréquente de chaque niveau dans train.jsonl,
# toujours la même pour une sortie déterministe et traçable.
CONCLUSIONS = {
    "urgence_maximale": "Vous devez venir aux urgences tout de suite.",
    "moderee": "Venez nous voir d'ici peu dans la journée.",
    "differee": "Vous pouvez patienter, mais rappelez-nous si ça s'aggrave.",
}

# Plancher de sécurité : signes d'alerte majeurs, cherchés dans les messages du PATIENT uniquement.
# Il ne lit que des symptômes, jamais le ton : c'est le contrepoids du décideur, qui a appris en
# partie à se fier au ton du patient (biais des données, mesuré par evaluer_raccourci_style.py).
# Plus pauvre que la grille FRENCH du pipeline : limite assumée et documentée.
# Les textes sont comparés en minuscules, sans accents. Formulations atténuées incluses
# (« une gêne dans la poitrine », « je cherche mes mots »), car un patient qui minimise reste à risque.
_MOTS = r"(?: \w+){0,3}"   # jusqu'à 3 mots intercalés
 
SIGNES_ALERTE = {
    "deficit_moteur": [
        r"(peu[tx]|arrive|arrivent) (plus|pas) (a )?(bouger|lever|serrer)",
        r"(bras|jambe|main|visage|cote|pied)" + _MOTS + r" (engourdi|paralyse|tout mou|ne repond plus|lache)",
        r"paralys", r"(bouche|visage)" + _MOTS + r" (de travers|tordue?|deforme|tombe)",
        r"\bavc\b", r"attaque cerebrale"],
    "trouble_parole": [
        # « n'arrive pas à trouver ses mots » : la tournure négative EST le symptôme, le motif
        # commence donc avant le « pas » pour qu'il ne soit pas pris pour une négation.
        r"(arrive|peu[tx]) (plus|pas)(?: \w+){0,2} (trouver|chercher) (ses|mes|les) mots",
        r"(trouve|trouver|cherche|chercher)(?: \w+){0,2} (ses|mes|les) mots",
        r"(arrive|peu[tx]) (plus|pas) a parler", r"ne parle plus", r"parle (plus|mal|pas) (bien|correctement)",
        r"articule (plus|mal|pas)", r"bafouill", r"mots (bizarres|melanges|sans sens)"],
    "trouble_conscience": [
        r"perdu connaissance", r"perte de connaissance", r"inconscien", r"evanoui", r"ne repond plus",
        r"ne se reveille (pas|plus)", r"convuls", r"s'est effondre", r"sens partir", r"tombe dans les pommes"],
    "douleur_thoracique": [
        r"(douleur|mal|gene|serre|serrement|oppression|poids|brulure)" + _MOTS
        + r" (a|dans|sur|au milieu de) (la|le) (poitrine|thorax|sternum)",
        r"(poitrine|thorax)" + _MOTS + r" (serre|ecrase|brule|fait mal|comprime)", r"douleur thoracique",
        r"infarctus", r"crise cardiaque", r"arret cardiaque"],
    "detresse_respiratoire": [
        r"(du )?mal a respirer(?! par le nez)", r"(arrive|peu[tx]) (plus|pas) a respirer", r"ne respire plus",
        r"respire (plus|tres mal|difficilement|pas bien)", r"etouff", r"levres bleues",
        r"(arrive|peu[tx]) (plus|pas) a finir (ses|mes) phrases"],
    "reaction_allergique_grave": [
        r"(gorge|langue)" + _MOTS + r" (serre|gonfl|enfl)", r"(levres|visage|paupieres)" + _MOTS + r" (gonfl|enfl)"],
    "hemorragie": [
        r"saigne (beaucoup|enormement|abondamment|sans arret)", r"hemorragie",
        r"(vomi|vomit|vomir|crache|cracher)" + _MOTS + r" (du )?sang", r"selles" + _MOTS + r" (noires|avec du sang)",
        r"sang dans (les|ses|mes) selles"],
    "syndrome_meninge": [
        r"nuque" + _MOTS + r" (raide|bloquee)", r"raideur (de la |de )?nuque",
        r"(arrive|peu[tx]) (plus|pas)" + _MOTS + r" (baisser|plier|pencher) la tete"],
}
# Combinaisons : les deux éléments doivent être présents (non niés), dans n'importe quels messages.
COMBINAISONS = {
    "purpura_febrile": (r"taches? (rouges?|violettes?|violacees?)", r"fievre|\b(38|39|40)\b"),
    "grossesse_douleur_saignement": (r"retard de regles|enceinte|grossesse",
                                     r"saignements?|je saigne|pertes de sang|mal au ventre|douleur" + _MOTS
                                     + r" (au|du|dans le) (ventre|bas[- ]ventre)"),
}
# La négation ne vaut que dans le MÊME membre de phrase, avant le symptôme : on coupe sur la
# ponctuation et les conjonctions (« il respire pas bien ET a perdu connaissance » : le « pas »
# ne nie pas la perte de connaissance). « je sais pas » n'est pas une négation du symptôme.
_COUPURES = re.compile(r"[.;:!?,…]|\b(?:et|mais|puis|donc|alors|sauf que|par contre)\b")
_NEGATION = re.compile(r"(?<!sais )\b(pas|ni|aucune?|sans|jamais)\b")


def normaliser(texte):
    texte = unicodedata.normalize("NFD", texte.lower().replace("’", "'"))
    return "".join(c for c in texte if unicodedata.category(c) != "Mn")

def _present(motif, texte):
    """Vrai si le motif apparaît dans un membre de phrase sans négation qui le précède."""
    for membre in _COUPURES.split(texte):
        for trouve in re.finditer(motif, membre):
            if not _NEGATION.search(membre[:trouve.start()]):
                return True
    return False
 
 
# « Je peux pas bouger le bras » après une chute est le plus souvent une douleur qui bloque le
# mouvement (entorse, fracture), pas un déficit neurologique : cette seule règle est ignorée en
# contexte de traumatisme. Les autres signes moteurs (engourdi, paralysé, visage de travers) restent actifs.
_REGLE_IMPOTENCE = SIGNES_ALERTE["deficit_moteur"][0]
# « d'un coup » / « tout à coup » (début brutal, typique de l'AVC) ne doivent PAS compter comme un traumatisme.
_TRAUMATISME = re.compile(r"\b(tombe|tombee|chute|accident|choc|cogne|glisse|tordu)|(?<!d'un )(?<!tout a )\bcoup (de|sur|au|dans)\b")
 
 
def plancher(messages):
    """Signes d'alerte détectés dans les messages du patient (liste vide si aucun)."""
    texte = normaliser(" . ".join(m["content"] for m in messages if m["role"] == "user"))
    trauma = bool(_TRAUMATISME.search(texte))
    signes = [nom for nom, motifs in SIGNES_ALERTE.items()
              if any(_present(m, texte) for m in motifs if not (trauma and m == _REGLE_IMPOTENCE))]
    signes += [nom for nom, (a, b) in COMBINAISONS.items() if _present(a, texte) and _present(b, texte)]
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