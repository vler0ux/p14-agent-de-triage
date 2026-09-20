"""
dedoublonner_vignettes.py

Regroupe les vignettes qui décrivent le MÊME cas clinique.

Contexte : MediQAl est un jeu de QUESTIONS d'examen. Un dossier clinique donne
souvent plusieurs questions, et le chargement (load_mediqal.py) crée une
vignette par question en recopiant le texte du cas. Or le pipeline de triage
n'utilise que le texte du cas, jamais la question : ces lignes sont pour lui
un seul patient répété. Ce script en garde UNE par texte de cas.

Ce que fait le script :
  - garde une seule ligne par texte de cas exactement identique (la première
    rencontrée dans le fichier) ;
  - ajoute à chaque ligne conservée :
        cas_id        empreinte des 200 premiers caractères du texte : les
                      versions "enrichies" d'un même cas (même début, fin
                      différente) partagent le même cas_id ;
        ids_questions liste des identifiants de toutes les lignes regroupées ;
        n_questions   nombre de lignes regroupées ;
  - marque les cas sur lesquels le FILTRE DE PERTINENCE s'est contredit : si des
    lignes portant le même cas ont été les unes retenues et les autres exclues,
    la ligne conservée reçoit filtre_retenues, filtre_exclues et
    filtre_instable=true (cas limites, à passer en relecture) ;
  - écrit un fichier de correspondance : pour chaque ligne supprimée, quelle
    ligne a été conservée à sa place (auditabilité).

Ce qu'il ne fait pas :
  - il ne modifie JAMAIS le fichier d'entrée (il écrit un nouveau fichier et
    refuse d'écraser un fichier existant sans --force) ;
  - il n'appelle aucune API ;
  - par défaut, il garde toutes les versions différentes d'un même cas
    (--variantes garder). Avec --variantes plus-courte, il ne garde que la
    version la plus courte de chaque cas : à n'utiliser qu'après avoir lu les
    exemples affichés dans le rapport. Pour les cas où la version la plus
    courte n'est pas la bonne, --garder <id> impose la version portant cet
    identifiant (option répétable).

Usage (depuis le dossier des scripts) :
    # 1) Simulation : affiche le rapport sans rien écrire
    python dedoublonner_vignettes.py --dry-run

    # 2) Écriture
    python dedoublonner_vignettes.py
"""

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

LONGUEUR_PREFIXE = 200

CHEMIN_ENTREE_DEFAUT = "../../data/anonymized/mediqal_mcqm_urgences_complet_anonymise.jsonl"
CHEMIN_SORTIE_DEFAUT = "../../data/anonymized/mediqal_mcqm_urgences_complet_dedup.jsonl"
CHEMIN_RETENUES_DEFAUT = "../../data/normalized/mediqal_mcqm_urgences.jsonl"
CHEMIN_EXCLUES_DEFAUT = "../../data/normalized/mediqal_mcqm_exclues.jsonl"


def calculer_cas_id(texte: str) -> str:
    """Empreinte stable des 200 premiers caractères (espaces normalisés)."""
    normalise = re.sub(r"\s+", " ", texte).strip()[:LONGUEUR_PREFIXE]
    return "cas_" + hashlib.sha1(normalise.encode("utf-8")).hexdigest()[:12]


def lire_jsonl(chemin: Path) -> list:
    lignes = []
    with open(chemin, encoding="utf-8") as f:
        for numero, ligne in enumerate(f, 1):
            ligne = ligne.strip()
            if not ligne:
                continue
            try:
                objet = json.loads(ligne)
            except json.JSONDecodeError as e:
                raise SystemExit(f"Ligne {numero} de {chemin} : JSON invalide ({e})")
            if "id" not in objet:
                raise SystemExit(f"Ligne {numero} de {chemin} : champ 'id' manquant")
            lignes.append(objet)
    return lignes


def compter_decisions_filtre(chemin_retenues: Path, chemin_exclues: Path):
    """Compte, par cas, combien de lignes le filtre a retenues et exclues.

    Les fichiers du filtre (normalized/) contiennent le texte ORIGINAL, alors que
    le fichier à dédoublonner contient le texte anonymisé : on relie les deux par
    l'identifiant. Retourne (id_vers_cle, compte) où compte[cle] = [retenues, exclues].
    """
    id_vers_cle = {}
    compte = defaultdict(lambda: [0, 0])
    for indice, chemin in enumerate((chemin_retenues, chemin_exclues)):
        for ligne in lire_jsonl(chemin):
            texte = ligne.get("contexte_clinique")
            if not texte:
                continue
            cle = calculer_cas_id(texte)
            compte[cle][indice] += 1
            if indice == 0:
                id_vers_cle[ligne["id"]] = cle
    return id_vers_cle, compte


def premiere_difference(a: str, b: str) -> int:
    limite = min(len(a), len(b))
    return next((k for k in range(limite) if a[k] != b[k]), limite)


def main():
    parser = argparse.ArgumentParser(description="Dédoublonnage des vignettes par cas clinique")
    parser.add_argument("--input", default=CHEMIN_ENTREE_DEFAUT, help="JSONL de vignettes (défaut : %(default)s)")
    parser.add_argument("--output", default=CHEMIN_SORTIE_DEFAUT, help="JSONL de sortie (défaut : %(default)s)")
    parser.add_argument("--variantes", choices=["garder", "plus-courte"], default="garder",
                        help="Que faire des versions différentes d'un même cas (même début de 200 caractères)")
    parser.add_argument("--garder", action="append", default=[], metavar="ID",
                        help="Avec --variantes plus-courte : impose, pour son cas, la version portant cet "
                             "identifiant plutôt que la plus courte (option répétable)")
    parser.add_argument("--filtre-retenues", default=CHEMIN_RETENUES_DEFAUT,
                        help="Sortie du filtre : lignes retenues, texte original (défaut : %(default)s)")
    parser.add_argument("--filtre-exclues", default=CHEMIN_EXCLUES_DEFAUT,
                        help="Sortie du filtre : lignes exclues, texte original (défaut : %(default)s)")
    parser.add_argument("--dry-run", action="store_true", help="Affiche le rapport sans écrire aucun fichier")
    parser.add_argument("--force", action="store_true", help="Autorise l'écrasement des fichiers de sortie existants")
    args = parser.parse_args()

    chemin_entree = Path(args.input)
    chemin_sortie = Path(args.output)
    chemin_correspondance = chemin_sortie.with_name(chemin_sortie.stem + ".correspondance.jsonl")

    if args.garder and args.variantes != "plus-courte":
        raise SystemExit("--garder s'utilise avec --variantes plus-courte.")
    if not chemin_entree.exists():
        raise SystemExit(f"Fichier d'entrée introuvable : {chemin_entree}")
    if chemin_entree.resolve() == chemin_sortie.resolve():
        raise SystemExit("Le fichier de sortie doit être différent du fichier d'entrée.")
    if not args.dry_run and not args.force:
        for chemin in (chemin_sortie, chemin_correspondance):
            if chemin.exists():
                raise SystemExit(f"{chemin} existe déjà. Relancez avec --force pour l'écraser.")

    lignes = lire_jsonl(chemin_entree)

    id_vers_cle_filtre, compte_filtre = {}, {}
    stats_filtre_dispo = Path(args.filtre_retenues).exists() and Path(args.filtre_exclues).exists()
    if stats_filtre_dispo:
        id_vers_cle_filtre, compte_filtre = compter_decisions_filtre(
            Path(args.filtre_retenues), Path(args.filtre_exclues))
    else:
        print("Avertissement : fichiers du filtre introuvables, les cas instables ne seront pas marqués "
              f"({args.filtre_retenues} / {args.filtre_exclues}).", file=sys.stderr)

    # --- 1. Regroupement par texte EXACT (ordre de première apparition conservé) ---
    ordre = []            # entrées dans l'ordre du fichier : ("groupe", texte) ou ("seule", ligne)
    groupes = {}          # texte exact -> lignes
    for ligne in lignes:
        texte = ligne.get("contexte_clinique")
        if not texte:
            ordre.append(("seule", ligne))
            continue
        if texte not in groupes:
            groupes[texte] = []
            ordre.append(("groupe", texte))
        groupes[texte].append(ligne)

    # --- 2. Textes différents partageant le même début (= même cas enrichi) ---
    textes_par_cas = {}
    for texte in groupes:
        textes_par_cas.setdefault(calculer_cas_id(texte), []).append(texte)
    cas_a_variantes = {c: t for c, t in textes_par_cas.items() if len(t) > 1}

    # Versions imposées par --garder : cas_id -> texte à conserver
    id_vers_texte = {m["id"]: texte for texte, membres in groupes.items() for m in membres}
    imposes = {}
    for identifiant in args.garder:
        if identifiant not in id_vers_texte:
            raise SystemExit(f"--garder : identifiant inconnu dans le fichier d'entrée : {identifiant}")
        texte = id_vers_texte[identifiant]
        cas_id = calculer_cas_id(texte)
        if cas_id not in cas_a_variantes:
            print(f"Avertissement : {identifiant} n'appartient à aucun cas à plusieurs versions ; --garder ignoré.",
                  file=sys.stderr)
            continue
        imposes[cas_id] = texte

    textes_ecartes = {}   # texte écarté -> texte conservé (uniquement avec --variantes plus-courte)
    if args.variantes == "plus-courte":
        for cas_id, textes in cas_a_variantes.items():
            garde = imposes.get(cas_id) or min(textes, key=len)   # à longueur égale : le premier rencontré
            for texte in textes:
                if texte != garde:
                    textes_ecartes[texte] = garde

    # --- 3. Construction des sorties et de la correspondance ---
    sorties, correspondance = [], []
    for genre, valeur in ordre:
        if genre == "seule":
            sortie = dict(valeur)
            sortie["cas_id"] = "sans_contexte_" + str(valeur["id"])
            sortie["ids_questions"] = [valeur["id"]]
            sortie["n_questions"] = 1
            sorties.append(sortie)
            continue

        texte = valeur
        membres = groupes[texte]
        cas_id = calculer_cas_id(texte)
        if texte in textes_ecartes:
            conserve = groupes[textes_ecartes[texte]][0]["id"]
            for m in membres:
                correspondance.append({"id": m["id"], "conserve": conserve, "cas_id": cas_id,
                                       "raison": "autre_version"})
            continue

        representant = membres[0]
        sortie = dict(representant)
        sortie["cas_id"] = cas_id
        sortie["ids_questions"] = [m["id"] for m in membres]
        sortie["n_questions"] = len(membres)
        ecartes = [m["id"] for t, g in textes_ecartes.items() if g == texte for m in groupes[t]]
        if ecartes:
            sortie["ids_variantes_ecartees"] = ecartes
        if stats_filtre_dispo:
            cles = {id_vers_cle_filtre[i] for i in sortie["ids_questions"] + ecartes if i in id_vers_cle_filtre}
            if cles:
                retenues = sum(compte_filtre[c][0] for c in cles)
                exclues = sum(compte_filtre[c][1] for c in cles)
                sortie["filtre_retenues"] = retenues
                sortie["filtre_exclues"] = exclues
                sortie["filtre_instable"] = retenues > 0 and exclues > 0
        sorties.append(sortie)
        for m in membres[1:]:
            correspondance.append({"id": m["id"], "conserve": representant["id"], "cas_id": cas_id,
                                   "raison": "meme_texte"})

    # --- 4. Rapport ---
    n_sans_contexte = sum(1 for g, _ in ordre if g == "seule")
    n_avec_contexte = len(lignes) - n_sans_contexte
    plus_gros = max((len(v) for v in groupes.values()), default=0)
    print(f"Fichier lu : {chemin_entree}")
    print(f"  {len(lignes)} lignes ({n_avec_contexte} avec contexte clinique, {n_sans_contexte} sans)")
    print(f"  {len(groupes)} textes de cas distincts ; plus gros groupe : {plus_gros} lignes pour le même texte")
    print(f"  {n_avec_contexte - len(groupes)} lignes supprimées car texte identique à une ligne déjà vue")
    print(f"  {len(cas_a_variantes)} cas avec plusieurs versions différentes (même début de {LONGUEUR_PREFIXE} caractères)")
    if args.variantes == "plus-courte":
        n_variantes = sum(len(groupes[t]) for t in textes_ecartes)
        print(f"  --variantes plus-courte : {n_variantes} lignes supplémentaires supprimées (autres versions du même cas)")
    print(f"  => {len(sorties)} lignes conservées")

    if stats_filtre_dispo:
        avec_stats = [s for s in sorties if "filtre_instable" in s]
        instables = [s for s in avec_stats if s["filtre_instable"]]
        egalites = sum(1 for s in instables if s["filtre_retenues"] == s["filtre_exclues"])
        maj_retenue = sum(1 for s in instables if s["filtre_retenues"] > s["filtre_exclues"])
        maj_exclue = len(instables) - egalites - maj_retenue
        print(f"\n  Filtre de pertinence : {len(instables)} cas marqués filtre_instable "
              f"(sur {len(avec_stats)} cas conservés avec statistiques)")
        print(f"     dont {egalites} égalités, {maj_retenue} majorité retenue, {maj_exclue} majorité exclue")
        sans_stats = len(sorties) - len(avec_stats)
        if sans_stats:
            print(f"     {sans_stats} cas conservés sans statistiques du filtre (identifiant introuvable)")

    if cas_a_variantes:
        print("\nExemples de cas à plusieurs versions (à lire avant de choisir --variantes plus-courte) :")
        for cas_id, textes in list(cas_a_variantes.items())[:4]:
            a, b = sorted(textes, key=len)[:2]
            i = premiere_difference(a, b)
            print(f"\n  {cas_id} : {groupes[a][0]['id']} ({len(a)} car.) vs {groupes[b][0]['id']} ({len(b)} car.)"
                  f" — diffèrent à partir du caractère {i}")
            print("     courte :", repr(a[max(0, i - 40): i + 70]))
            print("     longue :", repr(b[max(0, i - 40): i + 70]))

    if args.dry_run:
        print("\n[dry-run] aucun fichier écrit.")
        return

    chemin_sortie.parent.mkdir(parents=True, exist_ok=True)
    with open(chemin_sortie, "w", encoding="utf-8") as f:
        for s in sorties:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    with open(chemin_correspondance, "w", encoding="utf-8") as f:
        for c in correspondance:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"\nÉcrit : {chemin_sortie}")
    print(f"Écrit : {chemin_correspondance}  ({len(correspondance)} lignes supprimées, avec la ligne conservée à leur place)")


if __name__ == "__main__":
    main()