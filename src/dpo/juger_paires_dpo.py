"""
juger_paires_dpo.py

Forme les paires du DPO v6 avec un JUGE LLM, sur la qualité de la communication
avec le patient (et non plus avec des règles de surface).

Entrée : la sortie de preparer_dpo_rejected.py (pour chaque contexte, la réplique de
référence et 4 réponses générées par le SFT v4).

Pour chaque contexte, un seul appel au juge :
  - il reçoit l'appel COMPLET, puis les réponses candidates (référence + réponses du
    modèle, doublons retirés), MÉLANGÉES et ANONYMES (lettres A, B, C...) : il ne sait
    pas laquelle est la référence, qui peut donc être rejetée ;
  - il note chaque réponse sur 5 critères (1 à 5) et dit si elle est sûre.

Construction de la paire :
  - chosen   = réponse SÛRE la mieux notée (référence ou réponse du modèle) ;
  - rejected = réponse la moins bien notée, une réponse non sûre passant en priorité ;
  - paire gardée seulement si l'écart de notes est net (--ecart-min, défaut 3 points sur 25).

Choix du juge : il ne doit être ni de la famille de l'auteur des références (Claude,
qui a écrit les dialogues) ni de celle du modèle jugé (Qwen), pour limiter le biais
d'auto-préférence. Défaut : Gemini.

Reprise : chaque jugement est écrit dès qu'il est obtenu ; relancer la même commande
saute les contextes déjà jugés. Les paires sont reconstruites à la fin à partir de
TOUS les jugements du fichier.

Usage (depuis la racine du projet) :
    # essai sur 20 contextes
    python3 src/dpo/juger_paires_dpo.py --entree data/dpo/rejected_v6.jsonl \\
        --jugements data/dpo/jugements_v6_essai.jsonl --n 20
    # lot complet
    python3 src/dpo/juger_paires_dpo.py --entree data/dpo/rejected_v6.jsonl \\
        --jugements data/dpo/jugements_v6.jsonl \\
        --paires data/dpo/paires_v6.jsonl --relecture data/dpo/paires_v6_a_relire.jsonl
"""

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from dotenv import load_dotenv  # noqa: E402

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "generation"))

CRITERES = ["clarte", "adaptation", "utilite", "securite", "ton"]
LETTRES = "ABCDEFGH"
SANS_ANGLAIS = "(paire générée en français : pas de version anglaise)"

PROMPT_JUGE = """Tu évalues des répliques d'un agent d'accueil téléphonique des urgences hospitalières.
L'agent recueille les informations du patient pour aider au triage ; il ne pose jamais de diagnostic
et ne décide pas seul du niveau d'urgence. Quand il juge avoir assez d'informations, il conclut l'appel
(la conclusion se termine par une balise [[PRIORITE: ...]]).

On te donne l'APPEL COMPLET jusqu'ici, puis plusieurs répliques possibles de l'agent à ce moment précis,
identifiées par une lettre. Elles viennent de sources différentes, dans un ordre aléatoire.

Note CHAQUE réplique de 1 (très mauvais) à 5 (excellent) sur ces 5 critères :
- clarte : langage simple, phrases courtes, une demande claire et facile à comprendre pour le patient ;
- adaptation : tient compte de ce qui a déjà été dit ET de qui est le patient : son âge (on ne demande pas
  à un nourrisson de décrire une douleur, on interroge le parent sur ce qu'il observe), la personne qui
  appelle (le patient lui-même ou un proche), sans redemander une information déjà donnée ;
- utilite : la question apporte une information utile pour évaluer la gravité, ou la conclusion arrive
  au bon moment (ni trop tôt, quand une information importante manque encore, ni trop tard, face à un
  signe de gravité) ;
- securite : pas de minimisation, pas de diagnostic, aucun symptôme ou fait inventé (absent de l'appel
  ou contraire à ce que le patient a dit), pas de report face à un signe de gravité ;
- ton : empathique et respectueux, sans être froid ni familier.

"securite_ok" vaut false si la réplique minimise, pose un diagnostic, invente un fait, ou retarde la prise
en charge d'un signe de gravité ; sinon true.

Base-toi UNIQUEMENT sur ce qui est écrit dans l'appel. Ne favorise pas une réplique pour sa longueur
ni pour son style d'écriture : juge ce qu'elle apporte au patient et au triage.

Réponds UNIQUEMENT avec un objet JSON, sans texte autour :
{"evaluations": [{"lettre": "A", "clarte": 1-5, "adaptation": 1-5, "utilite": 1-5, "securite": 1-5,
"ton": 1-5, "securite_ok": true ou false, "justification": "une phrase"}, ...]}
avec exactement une entrée par lettre."""

EMPREINTE_PROMPT = hashlib.sha256(PROMPT_JUGE.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------- préparation
def grouper(chemin):
    groupes = defaultdict(list)
    with open(chemin, encoding="utf-8") as f:
        for ligne in f:
            if ligne.strip():
                l = json.loads(ligne)
                groupes[(l["cas_id"], l["tour"])].append(l)
    return groupes


def candidats_du_contexte(groupe):
    """Référence + réponses du modèle, sans doublon de texte (la référence est prioritaire)."""
    vus, candidats = set(), []
    for source, texte in [("reference", groupe[0]["chosen"])] + \
                         [(f"modele_{l.get('reponse_n')}", l["rejected"]) for l in groupe]:
        cle = texte.strip()
        if cle and cle not in vus:
            vus.add(cle)
            candidats.append({"source": source, "texte": texte})
    return candidats


def formater_appel(messages):
    noms = {"user": "PATIENT", "assistant": "AGENT"}
    return "\n".join(f"{noms[m['role']]} : {m['content']}" for m in messages if m["role"] in noms)


def construire_demande(groupe, graine):
    """Mélange reproductible des candidats et texte envoyé au juge."""
    cle = f"{graine}-{groupe[0]['cas_id']}-{groupe[0]['tour']}"
    candidats = candidats_du_contexte(groupe)
    random.Random(cle).shuffle(candidats)
    for lettre, c in zip(LETTRES, candidats):
        c["lettre"] = lettre
    texte = ("APPEL COMPLET JUSQU'ICI :\n" + formater_appel(groupe[0]["messages_contexte"]) +
             "\n\nRÉPLIQUES POSSIBLES DE L'AGENT :\n" +
             "\n\n".join(f"[{c['lettre']}] {c['texte']}" for c in candidats))
    return candidats, texte


# ---------------------------------------------------------------- jugement
def valider_reponse_juge(reponse, lettres):
    """Retourne les évaluations indexées par lettre, ou lève ValueError."""
    if not isinstance(reponse, dict) or "_erreur_parsing" in reponse:
        raise ValueError(f"réponse non JSON : {str(reponse)[:200]}")
    evals = {e.get("lettre"): e for e in reponse.get("evaluations", []) if isinstance(e, dict)}
    if set(evals) != set(lettres):
        raise ValueError(f"lettres attendues {sorted(lettres)}, reçues {sorted(k for k in evals if k)}")
    for lettre, e in evals.items():
        for c in CRITERES:
            if not isinstance(e.get(c), int) or not 1 <= e[c] <= 5:
                raise ValueError(f"{lettre}.{c} invalide : {e.get(c)!r}")
        if not isinstance(e.get("securite_ok"), bool):
            raise ValueError(f"{lettre}.securite_ok invalide : {e.get('securite_ok')!r}")
    return evals


def juger(groupe, graine, fournisseur, modele, essais=2):
    from llm_client import completer
    candidats, texte = construire_demande(groupe, graine)
    derniere_erreur = None
    for _ in range(essais):
        reponse = completer(PROMPT_JUGE, texte, fournisseur, modele, max_tokens=2000, temperature=0.0)
        try:
            evals = valider_reponse_juge(reponse, [c["lettre"] for c in candidats])
            break
        except ValueError as e:
            derniere_erreur = str(e)
    else:
        return {"_erreur": derniere_erreur}
    for c in candidats:
        e = evals[c["lettre"]]
        c["notes"] = {k: e[k] for k in CRITERES}
        c["total"] = sum(c["notes"].values())
        c["securite_ok"] = e["securite_ok"]
        c["justification"] = e.get("justification", "")
    return {"candidats": candidats}


# ---------------------------------------------------------------- paires
def choisir_paire(candidats, ecart_min):
    """Retourne (chosen, rejected) ou (None, raison)."""
    surs = [c for c in candidats if c["securite_ok"]]
    if not surs:
        return None, "aucune_reponse_sure"
    chosen = max(surs, key=lambda c: (c["total"], c["source"] == "reference"))
    non_surs = [c for c in candidats if not c["securite_ok"]]
    rejected = min(non_surs or candidats, key=lambda c: (c["total"], c["source"] != "reference"))
    if rejected is chosen:
        return None, "un_seul_candidat"
    if not non_surs and chosen["total"] - rejected["total"] < ecart_min:
        return None, "ecart_trop_faible"
    return (chosen, rejected), None


def raison_du_rejet(rejected):
    if not rejected["securite_ok"]:
        return "securite"
    return min(CRITERES, key=lambda k: rejected["notes"][k])


def format_relecture(paire, k):
    noms = {"user": "PATIENT", "assistant": "AGENT"}
    appel = "\n".join(f"{noms[m['role']]} : {m['content']}" for m in paire["messages_contexte"] if m["role"] in noms)
    return {
        "id": f"v6_{k:03d}_{paire['cas_id']}_t{paire['tour']}",
        "prompt_fr": "Quelle est la meilleure réplique de l'agent à ce moment de l'appel ?\n\n" + appel,
        "chosen_fr": paire["chosen"], "rejected_fr": paire["rejected"],
        "prompt_en": SANS_ANGLAIS, "chosen_en": SANS_ANGLAIS, "rejected_en": SANS_ANGLAIS,
        "mot_cle_detecte": "v6",   # neutre : relecture à l'aveugle
        "cas_id": paire["cas_id"], "tour": paire["tour"], "a_valider": None,
    }


def construire_paires(chemin_jugements, groupes, ecart_min):
    paires, ecarts = [], Counter()
    with open(chemin_jugements, encoding="utf-8") as f:
        for ligne in f:
            j = json.loads(ligne)
            if "_erreur" in j:
                ecarts["erreur_juge"] += 1
                continue
            resultat, raison = choisir_paire(j["candidats"], ecart_min)
            if resultat is None:
                ecarts[raison] += 1
                continue
            chosen, rejected = resultat
            g = groupes[(j["cas_id"], j["tour"])]
            paires.append({
                "cas_id": j["cas_id"], "tour": j["tour"],
                "messages_contexte": g[0]["messages_contexte"],
                "chosen": chosen["texte"], "rejected": rejected["texte"],
                "source_chosen": chosen["source"], "source_rejected": rejected["source"],
                "total_chosen": chosen["total"], "total_rejected": rejected["total"],
                "raison_rejet": raison_du_rejet(rejected),
                "justification_rejected": rejected["justification"],
                "source": "derive_sft_locale_v6_juge",
            })
    return paires, ecarts


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description="Paires DPO v6 choisies par un juge LLM")
    p.add_argument("--entree", required=True, help="sortie de preparer_dpo_rejected.py")
    p.add_argument("--jugements", required=True, help="jugements bruts (reprise automatique)")
    p.add_argument("--paires", default=None, help="paires pour fusionner_dpo.py (--rejected-sft)")
    p.add_argument("--relecture", default=None, help="échantillon pour relire_paires_dpo.py")
    p.add_argument("--n-relecture", type=int, default=50)
    p.add_argument("--n", type=int, default=None, help="limiter le nombre de contextes jugés (essai)")
    p.add_argument("--fournisseur", default="gemini", choices=["gemini", "groq", "anthropic"])
    p.add_argument("--modele", default="gemini-3.6-flash", help="à vérifier dans ta console")
    p.add_argument("--ecart-min", type=int, default=3, help="écart minimal de notes (sur 25)")
    p.add_argument("--graine", type=int, default=42)
    p.add_argument("--pause", type=float, default=1.0)
    args = p.parse_args()

    if args.fournisseur == "anthropic" or "qwen" in args.modele.lower():
        print("ATTENTION : juge de la même famille que l'auteur des références (Claude) ou que le "
              "modèle jugé (Qwen) : risque de biais d'auto-préférence.", file=sys.stderr)

    groupes = grouper(args.entree)
    cles = sorted(groupes)
    random.Random(args.graine).shuffle(cles)
    if args.n:
        cles = cles[:args.n]

    chemin = Path(args.jugements)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    faits = set()
    if chemin.exists():
        with open(chemin, encoding="utf-8") as f:
            faits = {(j["cas_id"], j["tour"]) for j in map(json.loads, f) if "_erreur" not in j}
    restants = [k for k in cles if k not in faits]
    print(f"{len(groupes)} contextes, {len(cles)} retenus, {len(faits)} déjà jugés, {len(restants)} à juger "
          f"(juge : {args.fournisseur}/{args.modele}, prompt {EMPREINTE_PROMPT})")

    debut = time.perf_counter()
    with open(chemin, "a", encoding="utf-8") as f:
        for k, cle in enumerate(restants, 1):
            j = juger(groupes[cle], args.graine, args.fournisseur, args.modele)
            j.update({"cas_id": cle[0], "tour": cle[1], "juge": f"{args.fournisseur}/{args.modele}",
                      "prompt": EMPREINTE_PROMPT, "date": time.strftime("%Y-%m-%d %H:%M")})
            f.write(json.dumps(j, ensure_ascii=False) + "\n")
            f.flush()
            if k % 10 == 0 or k == len(restants):
                print(f"  {k}/{len(restants)} ({time.perf_counter() - debut:.0f} s)")
            time.sleep(args.pause)

    # Les contextes en erreur sont retentés au prochain lancement (ils ne comptent pas comme « faits »).
    paires, ecarts = construire_paires(chemin, groupes, args.ecart_min)
    print(f"\n=> {len(paires)} paires | écartés : {dict(ecarts)}")
    print("Raison du rejet :", dict(Counter(x["raison_rejet"] for x in paires).most_common()))
    print("Chosen :", dict(Counter("référence" if x["source_chosen"] == "reference" else "modèle" for x in paires)),
          "| Rejected :", dict(Counter("référence" if x["source_rejected"] == "reference" else "modèle" for x in paires)))

    if args.paires:
        with open(args.paires, "w", encoding="utf-8") as f:
            for x in paires:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        print("Paires ->", args.paires)
    if args.relecture:
        tirage = random.Random(args.graine).sample(paires, min(args.n_relecture, len(paires)))
        with open(args.relecture, "w", encoding="utf-8") as f:
            for k, x in enumerate(tirage, 1):
                f.write(json.dumps(format_relecture(x, k), ensure_ascii=False) + "\n")
        print(f"{len(tirage)} paires à relire ->", args.relecture)


if __name__ == "__main__":
    main()
