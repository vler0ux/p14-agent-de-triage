#!/usr/bin/env python3
"""
Mesure du raccourci « style du patient → niveau de priorité » (audit, point 1).

Dans les dialogues d'entraînement, le style du patient dépend du niveau : 100 % des urgences sont
jouées par un patient « stressé », et le style « minimise ses symptômes » n'existe qu'en « différée ».
Le décideur a donc pu apprendre le style plutôt que la clinique.

Test par PAIRES MINIMALES : chaque cas clinique est écrit deux fois, avec les mêmes informations
cliniques, une fois par un patient stressé, une fois par un patient qui minimise. Si le décideur
comprend la clinique, les deux versions reçoivent le même niveau. S'il lit le style, la version
« minimise » d'une urgence glisse vers différée, et la version « stressé » d'un cas bénin vers l'urgence.

Cas écrits à la main pour ce test (jamais vus à l'entraînement). Niveaux attendus fixés à l'avance,
sur des tableaux cliniques sans ambiguïté ; à faire relire par un urgentiste pour un usage au-delà du POC.

Le décideur est évalué SEUL (seuil 0,30, sans plancher), puis on indique si le plancher aurait rattrapé
le cas. Aucun réseau, aucun GPU.

Usage (depuis la racine du projet) :
    python src/decideur/evaluer_raccourci_style.py --decideur-dir src/demo/decideur-patient
"""
import argparse
import json
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(RACINE / "src" / "demo"))
sys.path.insert(0, str(RACINE / "src" / "decideur"))
from decideur import Decideur, plancher  # noqa: E402
from train_decideur import LABELS, SEUIL_URGENCE  # noqa: E402

# (identifiant, niveau attendu, {style: [messages du patient]}) — mêmes faits cliniques dans les deux styles
CAS = [
    ("douleur_thoracique", "urgence_maximale", {
        "stresse": ["Bonjour, j'ai mal à la poitrine, ça serre fort depuis une heure, j'ai vraiment peur !",
                    "Ça descend dans le bras gauche, et je transpire beaucoup.",
                    "J'ai 58 ans, je fume, j'ai de la tension."],
        "minimise": ["Bonjour, bon, c'est sûrement rien, mais j'ai une petite gêne dans la poitrine depuis une heure, ça serre un peu.",
                     "Ça descend un peu dans le bras gauche, et je transpire un peu, mais ça doit être musculaire.",
                     "J'ai 58 ans, je fume, un peu de tension, rien de bien méchant."]}),
    ("deficit_neurologique", "urgence_maximale", {
        "stresse": ["Bonjour, c'est affreux, depuis ce matin ma main droite est engourdie et je trouve plus mes mots !",
                    "Ça a commencé d'un coup, vers 8 heures.",
                    "Non, je n'ai pas mal à la tête."],
        "minimise": ["Bonjour, c'est pas grand-chose, mais depuis ce matin j'ai la main droite un peu engourdie et je cherche un peu mes mots.",
                     "Ça a commencé d'un coup vers 8 heures, ça va sûrement passer.",
                     "Non, je n'ai pas mal à la tête."]}),
    ("raideur_nuque_purpura", "urgence_maximale", {
        "stresse": ["J'ai très peur, j'ai de la fièvre et un mal de tête horrible depuis ce matin !",
                    "J'arrive pas à baisser la tête, la nuque est raide.",
                    "Et j'ai des petites taches rouges sur les jambes qui sont apparues."],
        "minimise": ["Bonjour, j'ai un peu de fièvre et mal à la tête depuis ce matin, rien de méchant.",
                     "J'ai un peu de mal à baisser la tête, la nuque est raide, sûrement une mauvaise position.",
                     "Et j'ai des petites taches rouges sur les jambes, ça doit être une allergie."]}),
    ("hematemese", "urgence_maximale", {
        "stresse": ["Au secours, j'ai vomi du sang tout à l'heure, et mes selles sont noires depuis hier !",
                    "Je prends de l'aspirine tous les jours.",
                    "J'ai la tête qui tourne dès que je me lève."],
        "minimise": ["Bonjour, c'est sûrement rien, mais j'ai vomi un peu de sang tout à l'heure, et mes selles sont noires depuis hier.",
                     "Je prends de l'aspirine tous les jours, comme d'habitude.",
                     "J'ai un peu la tête qui tourne quand je me lève, mais ça va."]}),
    ("allergie_gorge", "urgence_maximale", {
        "stresse": ["Vite, j'ai mangé des cacahuètes, je suis allergique, mes lèvres gonflent !",
                    "J'ai la gorge qui serre et j'ai du mal à avaler.",
                    "Ma voix change, elle est bizarre."],
        "minimise": ["Bonjour, j'ai mangé des cacahuètes, je suis un peu allergique, mes lèvres ont un peu gonflé, rien de grave.",
                     "J'ai la gorge qui serre un peu et j'ai du mal à avaler, mais ça va aller je pense.",
                     "Ma voix est un peu bizarre, c'est tout."]}),
    ("douleur_pelvienne_grossesse", "urgence_maximale", {
        "stresse": ["J'ai très mal au ventre, à droite, et je saigne, je panique !",
                    "J'ai deux semaines de retard de règles.",
                    "J'ai la tête qui tourne et je me sens partir."],
        "minimise": ["Bonjour, j'ai un peu mal au ventre, plutôt à droite, et quelques petits saignements, rien d'inquiétant.",
                     "J'ai deux semaines de retard de règles, mais ça m'arrive.",
                     "J'ai un peu la tête qui tourne, sûrement la fatigue."]}),
    ("coupure_doigt", "differee", {
        "stresse": ["Au secours, je me suis coupé le doigt en cuisinant, ça saigne, j'ai super peur !",
                    "C'est une petite coupure, ça s'arrête quand j'appuie.",
                    "Je bouge le doigt normalement."],
        "minimise": ["Bonjour, rien de grave, je me suis coupé le doigt en cuisinant, ça saigne un peu.",
                     "C'est une petite coupure, ça s'arrête quand j'appuie.",
                     "Je bouge le doigt normalement."]}),
    ("rhume", "differee", {
        "stresse": ["Je suis vraiment inquiète, j'ai le nez qui coule et mal à la gorge depuis trois jours, j'en peux plus !",
                    "Je n'ai pas de fièvre, je respire bien.",
                    "Je mange et je bois normalement."],
        "minimise": ["Bonjour, rien de grave, j'ai juste le nez qui coule et un peu mal à la gorge depuis trois jours.",
                     "Je n'ai pas de fièvre, je respire bien.",
                     "Je mange et je bois normalement."]}),
    ("entorse_cheville", "differee", {
        "stresse": ["C'est horrible, je me suis tordu la cheville dans l'escalier, j'ai trop mal !",
                    "C'est un peu gonflé, mais j'arrive à marcher.",
                    "Non, je ne me suis pas cogné ailleurs."],
        "minimise": ["Bonjour, je me suis tordu la cheville dans l'escalier, ça fait un peu mal.",
                     "C'est un peu gonflé, mais j'arrive à marcher.",
                     "Non, je ne me suis pas cogné ailleurs."]}),
    ("piqure_guepe", "differee", {
        "stresse": ["Je panique, je me suis fait piquer par une guêpe au bras, ça brûle !",
                    "C'est rouge et un peu gonflé autour de la piqûre.",
                    "Je respire bien, rien d'autre, pas de gonflement ailleurs."],
        "minimise": ["Bonjour, petite chose, je me suis fait piquer par une guêpe au bras.",
                     "C'est rouge et un peu gonflé autour de la piqûre.",
                     "Je respire bien, rien d'autre, pas de gonflement ailleurs."]}),
]

# Jeu de CONTRÔLE, écrit avant de mesurer le plancher corrigé et jamais utilisé pour le régler :
# autres motifs et autres formulations que le jeu de développement ci-dessus (qui, lui, a servi
# à corriger le plancher et donne donc un score optimiste). C'est ce jeu qui mesure la généralisation.
CAS_CONTROLE = [
    ("infarctus_atypique", "urgence_maximale", {
        "stresse": ["J'ai une brûlure horrible en haut du ventre qui remonte dans la mâchoire, et je suis tout moite, j'ai peur !",
                    "Ça a commencé il y a quarante minutes, au repos.",
                    "Je suis diabétique, j'ai 67 ans."],
        "minimise": ["Bonjour, j'ai une petite brûlure en haut du ventre qui remonte un peu dans la mâchoire, je suis un peu moite, sûrement une indigestion.",
                     "Ça a commencé il y a quarante minutes, au repos, rien de bien grave.",
                     "Je suis diabétique, j'ai 67 ans."]}),
    ("avc_visage", "urgence_maximale", {
        "stresse": ["Vite, ma femme a la bouche qui tombe d'un côté depuis une demi-heure !",
                    "Elle dit des mots qui n'ont aucun sens, c'est terrifiant.",
                    "Son bras gauche est tout faible."],
        "minimise": ["Bonjour, ma femme a la bouche qui tombe un peu d'un côté depuis une demi-heure, elle est sûrement fatiguée.",
                     "Elle dit des mots qui n'ont pas trop de sens, mais bon.",
                     "Son bras gauche est un peu faible."]}),
    ("detresse_respiratoire", "urgence_maximale", {
        "stresse": ["Je respire très mal, j'arrive pas à finir mes phrases, aidez-moi !",
                    "Mes lèvres deviennent violettes.",
                    "J'ai de l'asthme, la ventoline ne fait rien."],
        "minimise": ["Bonjour, je respire un peu mal, j'arrive pas trop à finir mes phrases, ça va passer.",
                     "Mes lèvres sont un peu violettes, c'est sûrement le froid.",
                     "J'ai de l'asthme, la ventoline ne fait pas grand-chose."]}),
    ("hemoptysie_anticoagulant", "urgence_maximale", {
        "stresse": ["Mon père crache du sang depuis ce matin, beaucoup de sang, je panique !",
                    "Il est très pâle.",
                    "Il prend un anticoagulant pour le cœur."],
        "minimise": ["Bonjour, mon père crache un peu de sang depuis ce matin, rien d'alarmant je pense.",
                     "Il est un peu pâle.",
                     "Il prend un anticoagulant pour le cœur."]}),
    ("convulsion_enfant", "urgence_maximale", {
        "stresse": ["Ma fille de 3 ans a fait une crise, elle tremblait de partout et ne répondait plus, c'est affreux !",
                    "Maintenant elle dort, on n'arrive pas à la réveiller.",
                    "Elle a beaucoup de fièvre."],
        "minimise": ["Bonjour, ma fille de 3 ans a un peu tremblé de partout tout à l'heure et ne répondait plus, mais c'est passé.",
                     "Maintenant elle dort, on n'arrive pas trop à la réveiller, elle doit être fatiguée.",
                     "Elle a un peu de fièvre."]}),
    ("anaphylaxie_piqure", "urgence_maximale", {
        "stresse": ["Au secours, une abeille m'a piqué, j'ai des plaques partout et la langue qui gonfle !",
                    "J'ai la tête qui tourne beaucoup.",
                    "Je n'ai jamais réagi comme ça."],
        "minimise": ["Bonjour, une abeille m'a piqué, j'ai quelques plaques et la langue qui gonfle un peu, rien de grave.",
                     "J'ai un peu la tête qui tourne.",
                     "Je n'ai jamais réagi comme ça."]}),
    ("lombalgie", "differee", {
        "stresse": ["J'ai un mal de dos terrible depuis que j'ai porté des cartons, je n'en peux plus !",
                    "Je n'ai pas de douleur dans les jambes, je marche.",
                    "Pas de fièvre, rien d'autre."],
        "minimise": ["Bonjour, j'ai un peu mal au dos depuis que j'ai porté des cartons.",
                     "Je n'ai pas de douleur dans les jambes, je marche.",
                     "Pas de fièvre, rien d'autre."]}),
    ("conjonctivite", "differee", {
        "stresse": ["Je m'inquiète beaucoup, j'ai l'œil rouge qui colle le matin depuis deux jours !",
                    "Je vois bien, je n'ai pas de forte douleur.",
                    "Personne ne m'a rien jeté dans l'œil."],
        "minimise": ["Bonjour, j'ai juste l'œil un peu rouge qui colle le matin depuis deux jours.",
                     "Je vois bien, je n'ai pas de forte douleur.",
                     "Personne ne m'a rien jeté dans l'œil."]}),
    ("brulure_legere", "differee", {
        "stresse": ["Je me suis brûlé la main avec la casserole, ça brûle, je panique !",
                    "C'est rouge sur une petite surface, sans cloque.",
                    "Je l'ai passée sous l'eau froide."],
        "minimise": ["Bonjour, je me suis un peu brûlé la main avec la casserole.",
                     "C'est rouge sur une petite surface, sans cloque.",
                     "Je l'ai passée sous l'eau froide."]}),
    ("mal_de_dents", "differee", {
        "stresse": ["J'ai une rage de dents insupportable depuis deux jours, je deviens fou !",
                    "Je n'ai pas de fièvre, pas de gonflement de la joue.",
                    "J'arrive à manger de l'autre côté."],
        "minimise": ["Bonjour, j'ai un peu mal à une dent depuis deux jours.",
                     "Je n'ai pas de fièvre, pas de gonflement de la joue.",
                     "J'arrive à manger de l'autre côté."]}),
]

JEUX = {"dev": CAS, "controle": CAS_CONTROLE}

ORDRE = {lab: i for i, lab in enumerate(LABELS)}  # 0 = le plus urgent


def decision_seule(probas):
    """Règle du décideur sans plancher : urgence si P(urgence) >= seuil, sinon la plus probable."""
    if probas["urgence_maximale"] >= SEUIL_URGENCE:
        return "urgence_maximale"
    return max(probas, key=probas.get)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--decideur-dir", default=str(RACINE / "src" / "demo" / "decideur-patient"))
    p.add_argument("--jeu", choices=["dev", "controle"], default="dev",
                   help="dev : cas ayant servi à corriger le plancher ; controle : cas jamais utilisés pour le régler")
    p.add_argument("--sortie", default=None, help="fichier JSON des résultats détaillés (facultatif)")
    args = p.parse_args()

    decideur = Decideur(args.decideur_dir)
    resultats, abr = [], {"urgence_maximale": "URG", "moderee": "MOD", "differee": "DIF"}
    cas = JEUX[args.jeu]
    print(f"Jeu : {args.jeu} ({len(cas)} paires)\n")
    print(f"{'cas':<28}{'attendu':<9}{'style':<10}{'P(urg)':>7}{'P(mod)':>7}{'P(dif)':>7}  décideur  plancher  final")
    for nom, attendu, versions in cas:
        for style, textes in versions.items():
            messages = [{"role": "user", "content": t} for t in textes]
            probas = decideur.decider(messages)["probas"]
            niveau = decision_seule(probas)
            signes = plancher(messages)
            final = "urgence_maximale" if signes else niveau      # décision de la démo : plancher, puis décideur
            ecart = ORDRE[niveau] - ORDRE[attendu]                 # décideur seul : > 0 sous-triage, < 0 sur-triage
            ecart_final = ORDRE[final] - ORDRE[attendu]
            resultats.append({"cas": nom, "attendu": attendu, "style": style, "probas": probas, "decideur": niveau,
                              "ecart": ecart, "plancher": signes, "final": final, "ecart_final": ecart_final})
            marque = "  SOUS-TRIAGE" if ecart_final > 0 else ("  sur-triage" if ecart_final < 0 else "")
            print(f"{nom:<28}{abr[attendu]:<9}{style:<10}{probas['urgence_maximale']:>7.2f}{probas['moderee']:>7.2f}"
                  f"{probas['differee']:>7.2f}  {abr[niveau]:<8}  {'oui' if signes else 'non':<8}  {abr[final]}{marque}")

    print("\nBilan par style (décideur seul) :")
    for style in ("stresse", "minimise"):
        lignes = [r for r in resultats if r["style"] == style]
        urg = [r for r in lignes if r["attendu"] == "urgence_maximale"]
        ben = [r for r in lignes if r["attendu"] == "differee"]
        print(f"  {style:<9} urgences reconnues {sum(r['ecart'] == 0 for r in urg)}/{len(urg)} | "
              f"bénins bien classés {sum(r['ecart'] == 0 for r in ben)}/{len(ben)} | "
              f"P(urg) moyenne : urgences {sum(r['probas']['urgence_maximale'] for r in urg) / len(urg):.2f}, "
              f"bénins {sum(r['probas']['urgence_maximale'] for r in ben) / len(ben):.2f}")
    changes = sum(1 for nom, _, _ in cas
                  if len({r["decideur"] for r in resultats if r["cas"] == nom}) > 1)
    print(f"\nPaires dont la décision du décideur change avec le style seul : {changes}/{len(cas)}")
    rattrapes = [r for r in resultats if r["ecart"] > 0 and r["plancher"]]
    print(f"Sous-triages du décideur rattrapés par le plancher : {len(rattrapes)}/"
          f"{sum(r['ecart'] > 0 for r in resultats)}")
    urg = [r for r in resultats if r["attendu"] == "urgence_maximale"]
    ben = [r for r in resultats if r["attendu"] != "urgence_maximale"]
    print(f"\nDécision finale (plancher + décideur) : urgences reconnues {sum(r['ecart_final'] == 0 for r in urg)}/{len(urg)}"
          f" | sous-triages {sum(r['ecart_final'] > 0 for r in resultats)}"
          f" | bénins surclassés {sum(r['ecart_final'] < 0 for r in ben)}/{len(ben)}"
          f" (dont par le plancher : {sum(bool(r['plancher']) for r in ben)})")

    if args.sortie:
        Path(args.sortie).write_text(json.dumps(resultats, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nRésultats détaillés : {args.sortie}")


if __name__ == "__main__":
    main()
