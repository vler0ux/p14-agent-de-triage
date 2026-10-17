"""
filtrer_rejected.py

Forme les paires du DPO v6 à partir de la sortie de preparer_dpo_rejected.py
(plusieurs réponses générées par contexte, une ligne par réponse).

Règles (décidées avant de voir les résultats) :
  1. Le "chosen" (réplique de référence) ne doit avoir AUCUN des défauts retenus ;
     sinon le contexte est écarté (la paire enseignerait une contradiction).
  2. Le "rejected" est une réponse générée qui a au moins un défaut retenu.
     S'il y en a plusieurs pour un même contexte, on prend la plus grave
     (ORDRE_GRAVITE) ; à gravité égale, tirage reproductible (graine + contexte).
  3. Au plus UNE paire par contexte (cas_id + tour).

Sorties :
  --sortie           paires au format attendu par fusionner_dpo.py (--rejected-sft),
                     avec la raison du rejet pour la traçabilité ;
  --relecture        échantillon aléatoire au format de relire_paires_dpo.py, pour vérifier
                     le filtre à l'aveugle (aucun effet sur l'entraînement) ;
  bilan affiché      contextes écartés et pourquoi, paires par défaut, conclusions / milieu.

Usage (depuis la racine du projet, sans GPU) :
    python3 src/dpo/filtrer_rejected.py \\
        --entree data/dpo/rejected_v6.jsonl \\
        --sortie data/dpo/paires_v6.jsonl \\
        --relecture data/dpo/paires_v6_a_relire.jsonl --n-relecture 50
    python3 src/dpo/relire_paires_dpo.py data/dpo/paires_v6_a_relire.jsonl
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from defauts import conclut, evaluer, nombre_mots  # noqa: E402

# Défauts qui font un "rejected", du plus grave au moins grave.
ORDRE_GRAVITE = [
    "vide",
    "artefact",
    "non_francais",
    "rassurance_ou_diagnostic",
    "conclusion_manquee",       # la référence conclut, le modèle continue à questionner
    "conclusion_prematuree",    # la référence questionne encore, le modèle conclut
]
MARQUEURS_FIN = ("<|im_end|>", "atrigesimal", "<|im_start|>", "<|endoftext|>")
SANS_ANGLAIS = "(paire générée en français : pas de version anglaise)"


def lire_jsonl(chemin):
    with open(chemin, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def seuil_trop_longue(lignes):
    """p95 de la longueur des répliques de référence hors conclusion (une par contexte)."""
    vues, longueurs = set(), []
    for l in lignes:
        cle = (l["cas_id"], l["tour"])
        if cle in vues or conclut(l["chosen"]):
            continue
        vues.add(cle)
        longueurs.append(nombre_mots(l["chosen"]))
    longueurs.sort()
    return longueurs[int(0.95 * (len(longueurs) - 1))] if longueurs else 30


def defauts_retenus(d):
    return [nom for nom in ORDRE_GRAVITE if d[nom]]


def choisir_paire(groupe, seuil, graine):
    """Retourne (paire, None) ou (None, raison de l'écart) pour un contexte."""
    ref = groupe[0]
    chosen = ref["chosen"]
    ref_conclut = conclut(chosen)
    if defauts_retenus(evaluer(chosen, reference_conclut=ref_conclut, seuil_mots=seuil)):
        return None, "chosen_avec_defaut"

    candidats = []
    for l in groupe:
        brute = l.get("rejected_brut", l["rejected"])
        tronquee = not any(m in brute for m in MARQUEURS_FIN)   # arrêté par max_new_tokens
        d = evaluer(l["rejected"], reference_conclut=ref_conclut, seuil_mots=seuil,
                    reponse_brute=brute, tronquee=tronquee)
        trouves = defauts_retenus(d)
        if trouves and l["rejected"].strip() != chosen.strip():
            candidats.append((ORDRE_GRAVITE.index(trouves[0]), l, trouves, d))
    if not candidats:
        return None, "aucune_reponse_fautive"

    rang_min = min(c[0] for c in candidats)
    plus_graves = [c for c in candidats if c[0] == rang_min]
    _, l, trouves, d = random.Random(f"{graine}-{ref['cas_id']}-{ref['tour']}").choice(plus_graves)
    return {
        "cas_id": ref["cas_id"],
        "tour": ref["tour"],
        "reponse_n": l.get("reponse_n"),
        "messages_contexte": ref["messages_contexte"],
        "chosen": chosen,
        "rejected": l["rejected"],
        "raison_rejet": trouves[0],
        "defauts_rejected": trouves,
        "motifs": d["_motifs_rassurance"] + d["_artefacts"],
        "chosen_est_conclusion": ref_conclut,
        "source": "derive_sft_locale_v6",
    }, None


def format_relecture(paire, k):
    """Une paire au format de relire_paires_dpo.py (relecture à l'aveugle)."""
    derniers = paire["messages_contexte"][-4:]
    noms = {"user": "PATIENT", "assistant": "AGENT", "system": "SYSTÈME"}
    contexte = "\n".join(f"{noms.get(m['role'], m['role'])} : {m['content']}" for m in derniers)
    return {
        "id": f"v6_{k:03d}_{paire['cas_id']}_t{paire['tour']}",
        "prompt_fr": "Quelle est la meilleure réplique de l'agent à ce moment de l'appel ?\n\n" + contexte,
        "chosen_fr": paire["chosen"], "rejected_fr": paire["rejected"],
        "prompt_en": SANS_ANGLAIS, "chosen_en": SANS_ANGLAIS, "rejected_en": SANS_ANGLAIS,
        "mot_cle_detecte": paire["raison_rejet"],   # affiché par l'outil en en-tête
        "cas_id": paire["cas_id"], "tour": paire["tour"], "raison_rejet": paire["raison_rejet"],
        "a_valider": None,
    }


def main():
    p = argparse.ArgumentParser(description="Forme les paires DPO v6 à partir des réponses générées")
    p.add_argument("--entree", required=True, help="sortie de preparer_dpo_rejected.py")
    p.add_argument("--sortie", required=True, help="paires pour fusionner_dpo.py (--rejected-sft)")
    p.add_argument("--relecture", default=None, help="échantillon à relire avec relire_paires_dpo.py")
    p.add_argument("--n-relecture", type=int, default=50)
    p.add_argument("--graine", type=int, default=42)
    args = p.parse_args()

    for chemin in (args.sortie, args.relecture):
        if chemin and Path(chemin).exists():
            raise SystemExit(f"{chemin} existe déjà : choisir un autre nom (pas d'écrasement silencieux).")

    lignes = lire_jsonl(args.entree)
    groupes = defaultdict(list)
    for l in lignes:
        groupes[(l["cas_id"], l["tour"])].append(l)
    seuil = seuil_trop_longue(lignes)

    paires, ecarts = [], Counter()
    for cle in sorted(groupes):
        paire, raison = choisir_paire(groupes[cle], seuil, args.graine)
        if paire:
            paires.append(paire)
        else:
            ecarts[raison] += 1

    Path(args.sortie).parent.mkdir(parents=True, exist_ok=True)
    with open(args.sortie, "w", encoding="utf-8") as f:
        for paire in paires:
            f.write(json.dumps(paire, ensure_ascii=False) + "\n")

    print(f"{len(lignes)} réponses lues, {len(groupes)} contextes, seuil trop_longue = {seuil} mots")
    print(f"=> {len(paires)} paires ({100 * len(paires) / max(len(groupes), 1):.0f} % des contextes) -> {args.sortie}")
    print("Contextes écartés :", dict(ecarts))
    print("Paires par raison du rejet :", dict(Counter(p["raison_rejet"] for p in paires).most_common()))
    print("Paires sur une conclusion :", sum(p["chosen_est_conclusion"] for p in paires),
          "| au milieu de l'entretien :", sum(not p["chosen_est_conclusion"] for p in paires))
    print("Cas distincts :", len({p["cas_id"] for p in paires}))

    if args.relecture:
        tirage = random.Random(args.graine).sample(paires, min(args.n_relecture, len(paires)))
        with open(args.relecture, "w", encoding="utf-8") as f:
            for k, paire in enumerate(tirage, 1):
                f.write(json.dumps(format_relecture(paire, k), ensure_ascii=False) + "\n")
        print(f"{len(tirage)} paires à relire -> {args.relecture}")


if __name__ == "__main__":
    main()
