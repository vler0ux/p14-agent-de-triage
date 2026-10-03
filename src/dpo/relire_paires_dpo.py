"""
relire_paires_dpo.py

Relecture en terminal des paires DPO traduites (sortie de preparer_dpo_ultramedical.py).

Relecture À L'AVEUGLE : les deux réponses sont affichées dans un ordre tiré au hasard,
sans dire laquelle est "chosen". Tu indiques celle que tu préfères ; la paire est validée
(a_valider = true) seulement si ton choix coïncide avec le "chosen" du dataset.
Intérêt : évite de valider par complaisance une paire parce qu'on sait déjà laquelle
est censée être la bonne -> argument méthodologique pour le rapport.

Touches :
    1 / 2  réponse préférée
    =      les deux se valent / impossible de trancher   -> a_valider = false
    t      traduction fautive (sens altéré)               -> a_valider = false
    e      afficher la version anglaise originale
    s      passer (la paire reste à relire)
    q      quitter (tout ce qui est déjà décidé est enregistré)

Chaque décision est écrite immédiatement dans le fichier (réécriture atomique) :
on peut quitter et reprendre à tout moment, seules les paires a_valider=null sont proposées.

Usage (depuis la racine du projet) :
    python3 src/dpo/relire_paires_dpo.py data/dpo/ultramedical_fr_a_relire.jsonl
"""

import argparse
import json
import os
import random
import shutil
import tempfile
import textwrap
from datetime import datetime
from pathlib import Path


def charger(chemin: Path) -> list:
    with open(chemin, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def sauvegarder(chemin: Path, paires: list) -> None:
    # Réécriture atomique : un Ctrl+C pendant l'écriture ne corrompt pas le fichier.
    fd, tmp = tempfile.mkstemp(dir=chemin.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        for p in paires:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    os.replace(tmp, chemin)


def afficher_bloc(titre: str, texte: str, largeur: int) -> None:
    print(f"\n── {titre} " + "─" * max(0, largeur - len(titre) - 4))
    for paragraphe in str(texte).split("\n"):
        print(textwrap.fill(paragraphe, width=largeur) if paragraphe.strip() else "")


def bilan(paires: list) -> str:
    oui = sum(1 for p in paires if p.get("a_valider") is True)
    non = sum(1 for p in paires if p.get("a_valider") is False)
    reste = sum(1 for p in paires if p.get("a_valider") is None)
    return f"validées : {oui} | rejetées : {non} | restantes : {reste}"


def main():
    parser = argparse.ArgumentParser(description="Relecture à l'aveugle des paires DPO traduites")
    parser.add_argument("fichier")
    parser.add_argument("--seed", type=int, default=0, help="graine de l'ordre d'affichage (reproductible)")
    parser.add_argument("--largeur", type=int, default=100)
    args = parser.parse_args()

    chemin = Path(args.fichier)
    paires = charger(chemin)
    sauvegarde = chemin.with_suffix(chemin.suffix + ".bak")
    if not sauvegarde.exists():
        shutil.copy(chemin, sauvegarde)  # copie de l'état initial, une seule fois
        print(f"Copie de sécurité : {sauvegarde}")

    a_relire = [i for i, p in enumerate(paires) if p.get("a_valider") is None]
    print(f"{len(paires)} paires | {bilan(paires)}")

    for rang, i in enumerate(a_relire, 1):
        p = paires[i]
        # Ordre d'affichage déterministe par paire (graine + id) : relancer le script
        # remontre la même paire dans le même ordre.
        rng = random.Random(f"{args.seed}-{p['id']}")
        chosen_en_premier = rng.random() < 0.5
        rep = {
            "1": ("chosen" if chosen_en_premier else "rejected"),
            "2": ("rejected" if chosen_en_premier else "chosen"),
        }

        while True:
            os.system("clear")
            print(f"Paire {rang}/{len(a_relire)}  ({p['id']}, mot-clé : {p.get('mot_cle_detecte', '?')})"
                  f"   —   {bilan(paires)}")
            afficher_bloc("QUESTION", p["prompt_fr"], args.largeur)
            afficher_bloc("RÉPONSE 1", p[f"{rep['1']}_fr"], args.largeur)
            afficher_bloc("RÉPONSE 2", p[f"{rep['2']}_fr"], args.largeur)
            choix = input("\n[1] [2] meilleure  [=] égales  [t] traduction fautive  "
                          "[e] anglais  [s] passer  [q] quitter > ").strip().lower()

            if choix == "e":
                os.system("clear")
                afficher_bloc("QUESTION (EN)", p["prompt_en"], args.largeur)
                afficher_bloc("RÉPONSE 1 (EN)", p[f"{rep['1']}_en"], args.largeur)
                afficher_bloc("RÉPONSE 2 (EN)", p[f"{rep['2']}_en"], args.largeur)
                input("\nEntrée pour revenir > ")
                continue
            if choix == "q":
                print(f"\nArrêt. {bilan(paires)}")
                return
            if choix == "s":
                break
            if choix in ("1", "2", "=", "t"):
                if choix in ("1", "2"):
                    p["a_valider"] = True
                    if rep[choix] == "rejected":
                        # Désaccord avec l'annotation du dataset : c'est le jugement humain qui fait foi,
                        # on échange chosen/rejected (versions FR et EN) et on le trace.
                        for langue in ("fr", "en"):
                            p[f"chosen_{langue}"], p[f"rejected_{langue}"] = p[f"rejected_{langue}"], p[f"chosen_{langue}"]
                        p["inversee_par_relecteur"] = True
                else:
                    p["a_valider"] = False
                p["relecture"] = {
                    "choix": choix,
                    "prefere": rep.get(choix),  # "chosen" / "rejected" / None
                    "motif_rejet": {"=": "egalite", "t": "traduction"}.get(choix),
                    "date": datetime.now().isoformat(timespec="seconds"),
                }
                sauvegarder(chemin, paires)
                break
            # touche inconnue : on réaffiche la paire

    print(f"\nFin de la liste. {bilan(paires)}")


if __name__ == "__main__":
    main()