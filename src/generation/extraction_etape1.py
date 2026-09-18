"""
extraction_etape1.py

Étape 1 du pipeline : extraction structurée du cas clinique à partir d'une
vignette source en texte libre, au format CasClinique (cf.
src/referentiel/red_flag_detector.py) — c'est cette structure qui alimente
ensuite l'Étape 2 (détection déterministe des red flags).

Le prompt intègre DYNAMIQUEMENT le vocabulaire du référentiel FRENCH (la
liste des motifs et de leurs critères exacts) pour que le LLM ne produise
que des motif_id et criteres_presents réellement exploitables par l'Étape 2
— plutôt que des libellés inventés qui ne correspondraient à rien.

⚠️ Ce prompt est volumineux (~19 500 caractères / ~4 900 tokens, à cause des
119 motifs du référentiel) — il consomme une bonne part du quota par minute
sur Groq. Par défaut, ce script utilise Gemini plutôt que Groq (cf.
llm_client.py), justement pour ne pas entrer en compétition avec
filtre_pertinence.py qui tourne sur Groq. Passez --fournisseur groq si
besoin.

Règle impérative du prompt : ne jamais inventer une donnée absente du texte
source (pas de constante vitale devinée, pas de motif forcé si aucun ne
correspond vraiment).

Usage (mode test, contre le jeu de cas de référence) :
    python extraction_etape1.py --golden ../../tests/golden_etape1_extraction.json

Usage (sur de vraies vignettes filtrées, ex. depuis backups/) :
    python extraction_etape1.py --input /chemin/vers/vignettes_urgences.jsonl \
        --output ../../data/normalized/mediqal_mcqm_extraction.jsonl --limit 20
"""

import argparse
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "referentiel"))
from llm_client import completer  # noqa: E402
from red_flag_detector import charger_referentiel  # noqa: E402


def construire_vocabulaire_motifs(referentiel: dict) -> str:
    """Construit, à partir du référentiel, la liste des motifs et critères
    valides à injecter dans le prompt — reste automatiquement synchronisé
    si le référentiel évolue, pas besoin de dupliquer la liste à la main."""
    lignes = []
    for m in referentiel["motifs"]:
        criteres = [mod["critere"] for mod in m["modulateurs"]]
        criteres_txt = "; ".join(criteres) if criteres else "(aucun modulateur, utiliser tel quel si le motif correspond)"
        lignes.append(f"- {m['id']} : {m['libelle']}\n  Critères possibles : {criteres_txt}")
    return "\n".join(lignes)


def construire_prompt_systeme(referentiel: dict) -> str:
    vocabulaire = construire_vocabulaire_motifs(referentiel)
    return f"""Tu es un assistant d'extraction de données cliniques structurées, pour un projet de triage médical.

À partir du texte d'une vignette clinique, extrait UNIQUEMENT les informations explicitement présentes dans le texte, au format JSON suivant :

{{
  "motif_id": "un identifiant EXACT parmi la liste ci-dessous, ou null si aucun ne correspond vraiment",
  "age_annees": nombre ou null,
  "pas_mmhg": nombre ou null,
  "fc_min": nombre ou null,
  "spo2_pct": nombre ou null,
  "fr_min": nombre ou null,
  "glycemie_mmol_l": nombre ou null,
  "gcs": nombre ou null,
  "fievre": true ou false,
  "criteres_presents": ["critères recopiés EXACTEMENT depuis la liste ci-dessous, uniquement ceux explicitement vérifiés par le texte"]
}}

RÈGLE IMPÉRATIVE : n'invente JAMAIS une donnée absente du texte. Si une information n'est pas mentionnée, mets null (ou [] pour criteres_presents, ou false pour fievre). Ne devine JAMAIS une constante vitale non donnée dans le texte, même si elle te semble plausible cliniquement.

Motifs et critères disponibles (recopie les libellés EXACTEMENT, ne les reformule pas) :
{vocabulaire}

Réponds UNIQUEMENT avec le JSON, sans texte avant ni après, sans balises markdown.
"""


def construire_vocabulaire_motifs_leger(referentiel: dict) -> str:
    """Version allégée du vocabulaire : juste id + libellé, sans les critères
    détaillés. Utilisée pour l'Appel A (choix du motif) — les critères ne
    sont envoyés qu'à l'Appel B, uniquement pour le motif déjà choisi."""
    return "\n".join(f"- {m['id']} : {m['libelle']}" for m in referentiel["motifs"])


def construire_prompt_etape1a(referentiel: dict) -> str:
    vocabulaire = construire_vocabulaire_motifs_leger(referentiel)
    return f"""Tu es un assistant d'extraction de données cliniques structurées, pour un projet de triage médical.

À partir du texte d'une vignette clinique, extrait UNIQUEMENT les informations explicitement présentes dans le texte, au format JSON suivant :

{{
  "motif_id": "un identifiant EXACT parmi la liste ci-dessous, ou null si aucun ne correspond vraiment",
  "age_annees": nombre ou null,
  "pas_mmhg": nombre ou null,
  "fc_min": nombre ou null,
  "spo2_pct": nombre ou null,
  "fr_min": nombre ou null,
  "glycemie_mmol_l": nombre ou null,
  "gcs": nombre ou null,
  "fievre": true ou false
}}

RÈGLE IMPÉRATIVE : n'invente JAMAIS une donnée absente du texte. Si une information n'est pas mentionnée, mets null (ou false pour fievre). Ne devine JAMAIS une constante vitale non donnée dans le texte, même si elle te semble plausible cliniquement.

Motifs disponibles (choisis l'id EXACT le plus pertinent, ou null) :
{vocabulaire}

Réponds UNIQUEMENT avec le JSON, sans texte avant ni après, sans balises markdown.
"""


def construire_prompt_etape1b(motif: dict) -> str:
    criteres = [mod["critere"] for mod in motif["modulateurs"]]
    criteres_txt = "\n".join(f"- {c}" for c in criteres)
    return f"""Tu es un assistant d'extraction clinique. Pour le motif "{motif['libelle']}", détermine lesquels des critères suivants sont EXPLICITEMENT vérifiés dans le texte de la vignette fournie.

Critères possibles :
{criteres_txt}

RÈGLE IMPÉRATIVE : ne coche un critère que s'il est explicitement confirmé par le texte — ne devine jamais.

Réponds UNIQUEMENT avec ce JSON, sans texte avant ni après :
{{"criteres_presents": ["critères recopiés EXACTEMENT depuis la liste ci-dessus, uniquement ceux vérifiés"]}}
"""


def extraire_deux_etapes(referentiel: dict, texte_source: str, fournisseur: str, modele: str) -> dict:
    """Extraction en 2 appels légers plutôt qu'un seul gros appel :
    Appel A (vocabulaire léger, id+libellé seulement) choisit le motif et
    extrait les champs simples ; Appel B (minuscule, seulement les critères
    du motif choisi) vérifie les critères — seulement si le motif en a.
    Résultat final au même format que l'ancien extraire() en un seul appel,
    donc compatible avec le reste du pipeline sans autre changement."""
    prompt_a = construire_prompt_etape1a(referentiel)
    resultat_a = completer(prompt_a, f"Vignette :\n{texte_source}", fournisseur, modele, max_tokens=500)

    if "_erreur_parsing" in resultat_a:
        return resultat_a

    resultat = {
        "motif_id": resultat_a.get("motif_id"),
        "age_annees": resultat_a.get("age_annees"),
        "pas_mmhg": resultat_a.get("pas_mmhg"),
        "fc_min": resultat_a.get("fc_min"),
        "spo2_pct": resultat_a.get("spo2_pct"),
        "fr_min": resultat_a.get("fr_min"),
        "glycemie_mmol_l": resultat_a.get("glycemie_mmol_l"),
        "gcs": resultat_a.get("gcs"),
        "fievre": resultat_a.get("fievre", False),
        "criteres_presents": [],
    }

    motif_id = resultat["motif_id"]
    if motif_id:
        motif = next((m for m in referentiel["motifs"] if m["id"] == motif_id), None)
        if motif and motif["modulateurs"]:
            prompt_b = construire_prompt_etape1b(motif)
            resultat_b = completer(prompt_b, f"Vignette :\n{texte_source}", fournisseur, modele, max_tokens=200)
            if "_erreur_parsing" not in resultat_b:
                resultat["criteres_presents"] = resultat_b.get("criteres_presents", [])
            # si l'Appel B échoue, on garde le reste du résultat plutôt que
            # de tout perdre — criteres_presents reste juste vide, ce qui
            # retombe sur le tri_base du motif à l'Étape 2 (comportement sûr)
        # motif sans aucun modulateur : rien à vérifier, Appel B inutile

    return resultat


def comparer(attendu: dict, obtenu: dict) -> dict:
    champs = ["motif_id", "age_annees", "fievre"]
    resultat = {}
    for champ in champs:
        resultat[champ] = {
            "attendu": attendu.get(champ),
            "obtenu": obtenu.get(champ),
            "ok": attendu.get(champ) == obtenu.get(champ),
        }
    resultat["criteres_ok"] = set(attendu.get("criteres_presents", [])) == set(obtenu.get("criteres_presents", []))
    return resultat


def executer_mode_test(golden_path: str, prompt_systeme: str, fournisseur: str, modele: str, pause: float):
    with open(golden_path, encoding="utf-8") as f:
        golden = json.load(f)

    n_ok, n_total = 0, 0
    for cas in golden["cas"]:
        n_total += 1
        obtenu = completer(prompt_systeme, f"Vignette :\n{cas['vignette_source']}", fournisseur, modele, max_tokens=400)
        attendu = cas["extraction_attendue"]

        print(f"\n=== {cas['id']} ===")
        if "_erreur_parsing" in obtenu:
            print(f"❌ ERREUR DE PARSING JSON : {obtenu['_erreur_parsing']}")
            time.sleep(pause)
            continue

        diff = comparer(attendu, obtenu)
        tout_ok = all(v["ok"] for v in diff.values() if isinstance(v, dict)) and diff["criteres_ok"]

        for champ, detail in diff.items():
            if champ == "criteres_ok":
                continue
            marqueur = "✅" if detail["ok"] else "❌"
            print(f"  {marqueur} {champ}: attendu={detail['attendu']!r}, obtenu={detail['obtenu']!r}")
        marqueur_criteres = "✅" if diff["criteres_ok"] else "❌"
        print(f"  {marqueur_criteres} criteres_presents: attendu={attendu.get('criteres_presents')!r}, "
              f"obtenu={obtenu.get('criteres_presents')!r}")

        if tout_ok:
            n_ok += 1
            print("  => CAS ENTIÈREMENT CORRECT")

        time.sleep(pause)

    print(f"\n{'=' * 50}\n{n_ok}/{n_total} cas entièrement corrects (motif + âge + fièvre + critères)")


def executer_mode_reel(input_path: str, output_path: str, prompt_systeme: str, fournisseur: str,
                        modele: str, pause: float, limit: int):
    vignettes = []
    with open(input_path, encoding="utf-8") as f:
        for ligne in f:
            ligne = ligne.strip()
            if ligne:
                vignettes.append(json.loads(ligne))
    if limit:
        vignettes = vignettes[:limit]

    print(f"{len(vignettes)} vignettes à traiter...", file=sys.stderr)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    n_ok, n_erreurs = 0, 0
    with open(output_file, "a", encoding="utf-8") as f_out:
        for i, vignette in enumerate(vignettes, 1):
            texte_source = vignette.get("contexte_clinique") or vignette.get("question", "")
            extraction = completer(prompt_systeme, f"Vignette :\n{texte_source}", fournisseur, modele, max_tokens=400)

            if "_erreur_parsing" in extraction:
                n_erreurs += 1
            else:
                n_ok += 1

            resultat = {"id": vignette["id"], "vignette_source": texte_source, "extraction": extraction}
            f_out.write(json.dumps(resultat, ensure_ascii=False) + "\n")
            f_out.flush()

            if i % 10 == 0 or i == len(vignettes):
                print(f"  {i}/{len(vignettes)} traitées — {n_ok} ok, {n_erreurs} erreurs de parsing", file=sys.stderr)

            time.sleep(pause)

    print(f"\nTerminé : {n_ok} extractions réussies, {n_erreurs} erreurs -> {output_file}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Étape 1 : extraction structurée du cas clinique")
    parser.add_argument("--golden", default=None,
                         help="Mode test : chemin du jeu de cas de référence")
    parser.add_argument("--input", default=None,
                         help="Mode réel : fichier JSONL de vignettes (ex. sortie de filtre_pertinence.py)")
    parser.add_argument("--output", default=None, help="Mode réel : fichier de sortie JSONL")
    parser.add_argument("--limit", type=int, default=None, help="Mode réel : limiter le nombre de vignettes")
    parser.add_argument("--fournisseur", choices=["groq", "gemini"], default="gemini",
                         help="Défaut : gemini, pour ne pas entrer en compétition avec le quota Groq du filtre")
    parser.add_argument("--modele", default=None,
                         help="Défaut selon le fournisseur : gemini-2.5-flash ou openai/gpt-oss-20b")
    parser.add_argument("--pause", type=float, default=3.0)
    args = parser.parse_args()

    modele = args.modele or ("gemini-2.5-flash" if args.fournisseur == "gemini" else "openai/gpt-oss-20b")

    ref_path = Path(__file__).parent.parent / "referentiel" / "french_referentiel.json"
    referentiel = charger_referentiel(str(ref_path))
    prompt_systeme = construire_prompt_systeme(referentiel)

    if args.input:
        if not args.output:
            print("--output est requis avec --input", file=sys.stderr)
            sys.exit(1)
        executer_mode_reel(args.input, args.output, prompt_systeme, args.fournisseur, modele, args.pause, args.limit)
    else:
        golden_path = args.golden or str(Path(__file__).parent.parent.parent / "tests" / "golden_etape1_extraction.json")
        executer_mode_test(golden_path, prompt_systeme, args.fournisseur, modele, args.pause)


if __name__ == "__main__":
    main()