"""
Étape 1 du pipeline : extraction structurée du cas clinique à partir d'une
vignette source en texte libre, au format CasClinique (cf.
src/referentiel/red_flag_detector.py) — c'est cette structure qui alimente
ensuite l'Étape 2 (détection déterministe des red flags).

Le prompt intègre DYNAMIQUEMENT le vocabulaire du référentiel FRENCH (la
liste des motifs et de leurs critères exacts) pour que le LLM ne produise
que des motif_id et criteres_presents réellement exploitables par l'Étape 2
— plutôt que des libellés inventés qui ne correspondraient à rien.

Règle impérative du prompt : ne jamais inventer une donnée absente du texte
source (pas de constante vitale devinée, pas de motif forcé si aucun ne
correspond vraiment).

Usage (mode test, contre le jeu de cas de référence) :
    python extraction_etape1.py --golden ../../tests/golden_etape1_extraction.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent / "referentiel"))
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


def extraire(client, texte_vignette: str, prompt_systeme: str, modele: str) -> dict:
    reponse = client.chat.completions.create(
        model=modele,
        max_tokens=400,
        messages=[
            {"role": "system", "content": prompt_systeme},
            {"role": "user", "content": f"Vignette :\n{texte_vignette}"},
        ],
    )
    texte = reponse.choices[0].message.content.strip()
    try:
        return json.loads(texte)
    except json.JSONDecodeError:
        return {"_erreur_parsing": texte[:300]}


def comparer(attendu: dict, obtenu: dict) -> dict:
    """Compare extraction attendue vs obtenue, champ par champ."""
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


def main():
    parser = argparse.ArgumentParser(description="Étape 1 : extraction structurée — mode test sur le jeu de référence")
    parser.add_argument("--golden", default=str(Path(__file__).parent.parent.parent / "tests" / "golden_etape1_extraction.json"))
    parser.add_argument("--modele", default="openai/gpt-oss-20b")
    args = parser.parse_args()

    from groq import Groq

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("GROQ_API_KEY manquant dans .env", file=sys.stderr)
        sys.exit(1)
    client = Groq(api_key=api_key)

    ref_path = Path(__file__).parent.parent / "referentiel" / "french_referentiel.json"
    referentiel = charger_referentiel(str(ref_path))
    prompt_systeme = construire_prompt_systeme(referentiel)

    with open(args.golden, encoding="utf-8") as f:
        golden = json.load(f)

    n_ok, n_total = 0, 0
    for cas in golden["cas"]:
        n_total += 1
        obtenu = extraire(client, cas["vignette_source"], prompt_systeme, args.modele)
        attendu = cas["extraction_attendue"]

        print(f"\n=== {cas['id']} ===")
        if "_erreur_parsing" in obtenu:
            print(f"❌ ERREUR DE PARSING JSON : {obtenu['_erreur_parsing']}")
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

    print(f"\n{'=' * 50}\n{n_ok}/{n_total} cas entièrement corrects (motif + âge + fièvre + critères)")


if __name__ == "__main__":
    main()