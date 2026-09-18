"""
generation_dialogue_etape5.py

Étape 5 du pipeline — la dernière : à partir d'un cas clinique structuré et
du niveau de priorité DÉJÀ VALIDÉ par les Étapes 1-2-3-4 (croisement
déterministe + LLM + contre-validation), génère un dialogue multi-tours
réaliste entre un patient et un agent de triage.

Point de conception important : cette étape NE DOIT PAS re-décider la
priorité — elle met en scène, de façon réaliste, un cas dont la conclusion
est déjà figée. Le risque à éviter ici n'est pas le sous-triage (déjà géré
en amont) mais un dialogue trop "scolaire" — un patient qui répond comme un
manuel médical plutôt que comme une vraie personne (hésitations, ordre
désordonné, formulations imprécises).

Usage (mode réel, sur la sortie de orchestrer_pipeline.py) :
    python generation_dialogue_etape5.py \
        --input ../../data/normalized/mediqal_mcqm_pipeline_complet.jsonl \
        --output ../../data/normalized/mediqal_mcqm_dialogues.jsonl
"""

import argparse
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))
from llm_client import completer  # noqa: E402

PROMPT_SYSTEME = """Tu es un générateur de dialogues de triage médical, pour un projet d'agent IA au Centre Hospitalier Saint-Aurélien.

À partir d'un cas clinique structuré et d'un NIVEAU DE PRIORITÉ DÉJÀ VALIDÉ (que tu dois respecter sans le remettre en question), génère un dialogue multi-tours réaliste entre un patient et un agent de triage qui collecte les symptômes avant de conclure.

RÈGLES IMPÉRATIVES :
1. Le dialogue doit aboutir à EXACTEMENT le niveau de priorité fourni — ni plus prudent, ni moins prudent. Ce n'est pas à toi de juger la gravité, elle est déjà tranchée.
2. Le patient s'exprime de façon NATURELLE ET IMPARFAITE : hésitations, informations données dans le désordre, réponses parfois vagues, reformulations — jamais comme un manuel médical. Exemple de registre attendu : "Ben en fait ça a commencé hier soir je crois, ou peut-être avant-hier, j'sais plus trop... et puis ce matin c'était pire" plutôt que "Les symptômes ont débuté il y a 18 heures avec une aggravation progressive."
3. L'agent pose des questions progressives et pertinentes, cohérentes avec le cas structuré fourni — jamais une question dont la réponse est déjà connue.
4. Le dialogue se termine par une phrase de conclusion de l'agent énonçant le niveau de priorité dans un langage NATUREL ET HUMAIN — jamais de jargon administratif ou clinique du type "je vous classe en priorité modérée", "niveau de priorité X", "catégorie Y". L'agent parle comme une vraie personne à l'accueil, pas comme un système qui annonce une classification. Exemples de bon registre : "on va vous prendre en charge tout de suite" (urgence), "on va vous voir rapidement, mais ce n'est pas alarmant" (modérée), "vous pouvez patienter, ce n'est pas urgent mais on va s'occuper de vous" (différée).
5. N'invente AUCUN élément clinique absent du cas structuré fourni, même s'il te semble plausible ou cohérent avec le tableau clinique — pas de nouveau symptôme, pas de nouvelle constante, pas de nouvel antécédent inventé pour "enrichir" le tableau. Tu peux enrichir la mise en scène narrative (contexte de vie, formulations, hésitations), jamais la substance médicale, MÊME de façon plausible. Si le cas structuré est sobre, le dialogue doit rester sobre sur ce point plutôt que d'ajouter un détail non vérifié.
6. Entre 4 et 8 tours d'échange (une question agent + une réponse patient = 1 tour), plus la conclusion finale.

Réponds UNIQUEMENT avec un objet JSON, sans texte avant ni après :
{"dialogue": [{"role": "agent", "content": "..."}, {"role": "patient", "content": "..."}, ...], "conclusion": "phrase finale de l'agent, en langage naturel"}
"""


def generer_dialogue(cas_structure: dict, categorie_finale: str, vignette_source: str,
                      fournisseur: str, modele: str) -> dict:
    contenu = {
        "cas_structure": cas_structure,
        "niveau_de_priorite_valide": categorie_finale,
        "contexte_clinique_original": vignette_source,
    }
    contenu_json = json.dumps(contenu, ensure_ascii=False, indent=2)
    return completer(PROMPT_SYSTEME, contenu_json, fournisseur, modele, max_tokens=2048)


def valider_dialogue(resultat: dict) -> list:
    """Vérifications structurelles minimales (pas une garantie de qualité
    clinique, juste un filet contre les réponses mal formées)."""
    problemes = []
    if "dialogue" not in resultat or not isinstance(resultat["dialogue"], list):
        problemes.append("champ 'dialogue' manquant ou n'est pas une liste")
    elif len(resultat["dialogue"]) < 2:
        problemes.append("dialogue trop court (< 2 tours)")
    if "conclusion" not in resultat or not resultat.get("conclusion"):
        problemes.append("champ 'conclusion' manquant ou vide")
    return problemes


def charger_ids_deja_traites(output_file: Path) -> set:
    ids = set()
    if output_file.exists():
        with open(output_file, encoding="utf-8") as f:
            for ligne in f:
                ligne = ligne.strip()
                if ligne:
                    ids.add(json.loads(ligne)["id"])
    return ids


def main():
    parser = argparse.ArgumentParser(description="Étape 5 : génération du dialogue multi-tours")
    parser.add_argument("--input", required=True, help="Sortie JSONL de orchestrer_pipeline.py")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fournisseur", choices=["groq", "gemini"], default="groq")
    parser.add_argument("--modele", default="openai/gpt-oss-120b")
    parser.add_argument("--pause", type=float, default=3.0)
    args = parser.parse_args()

    lignes = []
    with open(args.input, encoding="utf-8") as f:
        for ligne in f:
            ligne = ligne.strip()
            if ligne:
                lignes.append(json.loads(ligne))

    # On ne garde que les vignettes qui ont abouti à un label final exploitable
    lignes = [l for l in lignes if l.get("categorie_finale")]
    if args.limit:
        lignes = lignes[: args.limit]

    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    ids_deja_traites = charger_ids_deja_traites(output_file)
    if ids_deja_traites:
        print(f"Reprise détectée : {len(ids_deja_traites)} déjà traitées, sautées.", file=sys.stderr)
    lignes = [l for l in lignes if l["id"] not in ids_deja_traites]

    print(f"{len(lignes)} cas à mettre en dialogue...", file=sys.stderr)

    n_ok, n_erreurs = 0, 0
    with open(output_file, "a", encoding="utf-8") as f_out:
        for i, entree in enumerate(lignes, 1):
            cas_structure = entree.get("etape1_extraction", {})
            categorie_finale = entree["categorie_finale"]
            vignette_source = entree.get("vignette_source", "")

            resultat = generer_dialogue(cas_structure, categorie_finale, vignette_source,
                                          args.fournisseur, args.modele)

            if "_erreur_parsing" in resultat:
                n_erreurs += 1
                sortie = {"id": entree["id"], "categorie_finale": categorie_finale, "_erreur": resultat}
            else:
                problemes = valider_dialogue(resultat)
                if problemes:
                    n_erreurs += 1
                    sortie = {"id": entree["id"], "categorie_finale": categorie_finale,
                              "_erreur": {"problemes_structurels": problemes, "brut": resultat}}
                else:
                    n_ok += 1
                    sortie = {"id": entree["id"], "categorie_finale": categorie_finale,
                              "cas_structure": cas_structure, "dialogue": resultat["dialogue"],
                              "conclusion": resultat["conclusion"]}

            f_out.write(json.dumps(sortie, ensure_ascii=False) + "\n")
            f_out.flush()

            print(f"  [{i}/{len(lignes)}] {entree['id']} -> {'OK' if 'dialogue' in sortie else 'ÉCHEC'}",
                  file=sys.stderr)
            time.sleep(args.pause)

    print(f"\nTerminé : {n_ok} dialogues générés, {n_erreurs} échecs -> {output_file}", file=sys.stderr)


if __name__ == "__main__":
    main()