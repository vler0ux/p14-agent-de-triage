"""
generation_dialogue_etape5.py

Étape 5 du pipeline : à partir d'un cas déjà labellisé (sortie de orchestrer_pipeline.py) et de son niveau de
priorité VALIDÉ, génère un dialogue multi-tours réaliste entre un patient et un agent de triage.

Point de conception : cette étape NE re-décide PAS la priorité ; elle met en scène un cas dont la conclusion est figée.
Le risque à éviter est un dialogue « scolaire » (patient qui parle comme un manuel) ou qui fuit des informations
que l'agent ne peut pas connaître.

--- Ce qui a changé au lot 2 (sept. 2026) ---
- Fournisseur par défaut : Anthropic (Haiku), température 0,8 (créativité) ; --fournisseur groq|gemini|anthropic.
- VARIANTES : plusieurs dialogues par cas (par défaut 1 « urgence maximale », 2 « modérée », 3 « différée »),
  chacun avec un style de patient différent. Sert à compenser le déséquilibre des classes SANS toucher aux labels.
  Toutes les variantes d'un même cas partagent `cas_id` : le découpage train / validation / test se fait PAR CAS.
- L'agent NE CONNAÎT PAS les constantes vitales mesurées (tension, pouls, saturation, glycémie...) ni les résultats
  d'examens, ni le diagnostic : le prompt ne reçoit ni les constantes ni le motif codé, et le dialogue est refusé
  s'il contient des valeurs chiffrées de constantes.
- Contrôles automatiques avec nouvelle tentative (--essais) : structure et alternance des tours, 4 à 8 échanges,
  aucun marqueur [PERSON] ni initiale de nom (« Monsieur L. »), aucune valeur de constante vitale. Un dialogue qui
  échoue reste dans le fichier avec `_erreur` (jamais dans le dataset).
- CONCLUSION IMPOSÉE PAR LE CODE : tirée dans CONCLUSIONS selon (id du cas, variante, niveau), donc reproductible et
  cohérente avec le niveau par construction. Elle remplace toujours celle du modèle (`controles.conclusion_corrigee`
  indique si le modèle s'en était écarté). Les anciens contrôles de la conclusion (jargon, mots-clés de cohérence,
  actes promis) n'ont donc plus lieu d'être et ont été retirés ; `controles.conclusion_coherente` vaut toujours True
  (champ gardé pour la compatibilité avec les fichiers existants et recalibrer_conclusions.py).
- La BALISE de catégorie (`[[PRIORITE: moderee]]`) est calculée ici à partir du label validé (jamais par le LLM) et
  enregistrée dans le champ `balise` ; preparer_dataset_sft.py l'ajoute en dernière ligne du message final.
- Provenance : fournisseur, modèle, température, empreinte du prompt, date.
- PREMIÈRE PHRASE IMPOSÉE PAR LE CODE (retour de Véro après l'essai : « Qu'est-ce qui vous amène aujourd'hui ? » sonne
  comme une formule de commerce) : l'agent ouvre avec une formule de service d'urgence, tirée dans une liste selon
  (id du cas, variante), donc reproductible. Le prompt la donne au modèle ; si le modèle la modifie, le CODE la rétablit
  (`controles.ouverture_corrigee`). --contexte appel (défaut) : appel téléphonique au service des urgences
  (« Quel est l'objet de votre appel ? ») ; --contexte accueil : accueil sur place (« Que puis-je faire pour vous ? »).
- FIDÉLITÉ (correctif après l'essai sur 3 cas) : sur 2 dialogues lus, l'un faisait CONFIRMER au patient un signe absent
  de la description (« elle a mal au cou », qui suggère une raideur de nuque) après une question de l'agent. Trois parades :
    * prompt : le patient ne confirme jamais un symptôme absent de la description (il répond « non » ou « je sais pas »),
      la chronologie reste vague quand la description ne la donne pas, la première phrase de l'agent est neutre ;
    * JUGE DE FIDÉLITÉ : un second appel compare les affirmations du patient à la description et liste les éléments
      cliniques inventés ; un élément d'importance « medicale » fait refuser le dialogue (nouvelle tentative) ;
    * la conclusion est imposée par le code (voir plus haut) : elle ne peut donc promettre aucun acte précis.
  Limite : le juge est un LLM ; il réduit les inventions, il ne les élimine pas. `controles.fidelite` le dit
  (« verifiee » ou « non_verifiee » si le juge n'a pas répondu). --sans-juge désactive ce contrôle.

Usage :
    python generation_dialogue_etape5.py \
        --input ../../data/normalized/pipeline_473.jsonl \
        --output ../../data/normalized/dialogues_473.jsonl
    # essai sur 3 cas : --limit 3      ; moins de variantes : --variantes urgence_maximale=1,moderee=1,differee=2
"""

import argparse
import hashlib
import zlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))
from llm_client import completer  # noqa: E402

CATEGORIES = ("urgence_maximale", "moderee", "differee")
FOURNISSEUR_DEFAUT = "anthropic"
MODELE_DEFAUT = "claude-haiku-4-5-20251001"
VARIANTES_DEFAUT = {"urgence_maximale": 1, "moderee": 2, "differee": 3}

OUVERTURES = {
    "appel": [
        "Urgences du Centre Hospitalier Saint-Aurélien, bonjour. Quel est l'objet de votre appel ?",
        "Service des urgences, bonjour. Que puis-je faire pour vous ?",
        "Centre Hospitalier Saint-Aurélien, service des urgences, bonjour. Je vous écoute : que se passe-t-il ?",
        "Urgences, bonjour. Quel est le motif de votre appel ?",
    ],
    "accueil": [
        "Service des urgences, bonjour. Que puis-je faire pour vous ?",
        "Bonjour, accueil des urgences. Je vous écoute.",
        "Bonjour, service des urgences. Que puis-je faire pour vous ?",
    ],
}

CONCLUSIONS = {
    "urgence_maximale": [
        "Vous devez venir aux urgences tout de suite",
        "On s'occupe de vous immédiatement.",
        "On vous prend en charge tout de suite.",
    ],
    "moderee": [
        "Vous devez venir nous voir, on va vous examiner rapidement dans la journée.",
        "On va s'occuper de vous rapidement dans la journée.",
        "Venez nous voir d'ici peu dans la journée.",
    ],
    "differee": [
        "Vous pouvez patienter, mais rappelez-nous si ça s'aggrave.",
        "Il est mieux que vous restiez à votre domicile, mais rappelez-nous si ça empire.",
        "Reposez-vous chez vous, contactez-nous si vos symptômes s'aggravent.",
    ],
}


def choisir_conclusion(id_cas: str, variante: int, categorie: str) -> str:
    """Même principe que choisir_ouverture : choix reproductible (id du cas + variante)."""
    liste = CONCLUSIONS[categorie]
    return liste[zlib.crc32(f"{id_cas}-{variante}-conclusion".encode("utf-8")) % len(liste)]

CONTEXTES = {
    "appel": "il s'agit d'un APPEL TÉLÉPHONIQUE au service des urgences. L'agent ne voit pas la personne : il ne décrit jamais son apparence (« je vous vois pâle »). La personne qui appelle est le patient ou un proche.",
    "accueil": "il s'agit d'un ACCUEIL SUR PLACE, au guichet du service des urgences. Le patient (ou un proche) est devant l'agent.",
}

STYLES = [
    "patient un peu stressé, phrases courtes et hésitantes",
    "patient bavard qui part parfois dans des digressions avant de revenir aux symptômes",
    "patient qui minimise ses symptômes et reste vague",
    "proche ou accompagnant qui répond à la place du patient, en racontant ce qu'il a observé",
    "patient calme et précis, mais avec des mots du quotidien",
    "patient agacé par l'attente, réponses sèches",
]

PROMPT_SYSTEME = """Tu es un générateur de dialogues de triage médical, pour un projet d'agent IA au Centre Hospitalier Saint-Aurélien.

On te donne la description clinique d'un cas (fiche d'un cas d'école) et le NIVEAU DE PRIORITÉ DÉJÀ VALIDÉ, que tu dois respecter sans le remettre en question. Génère un dialogue multi-tours réaliste entre un patient (ou un proche) et un agent de triage du service des urgences, qui pose des questions puis conclut. CONTEXTE : {contexte}

RÈGLES IMPÉRATIVES :
1. Le dialogue doit aboutir EXACTEMENT au niveau de priorité fourni, ni plus prudent ni moins prudent. Ce n'est pas à toi de juger la gravité.
2. Le patient s'exprime de façon NATURELLE ET IMPARFAITE : hésitations, informations dans le désordre, réponses parfois vagues, reformulations, mots du quotidien. Jamais comme un manuel médical. Exemple de registre : « Ben en fait ça a commencé hier soir je crois, ou peut-être avant-hier, j'sais plus trop... et puis ce matin c'était pire ».
3. L'agent pose des questions progressives et pertinentes, une à la fois, jamais une question dont la réponse est déjà connue. Il ne pose JAMAIS de diagnostic et ne cite aucune maladie.
4. Ce que le patient sait : uniquement ce qu'il ressent, ce qu'il a remarqué, ses antécédents et ce qui s'est passé avant son arrivée. Il ne connaît NI les résultats d'examens (radio, analyses, ECG, scanner...), NI le diagnostic, NI l'évolution à l'hôpital, même si la description clinique les mentionne. Ne les mets pas dans sa bouche.
5. L'agent n'a PAS de constantes mesurées : aucune valeur chiffrée de tension, de pouls, de saturation, de fréquence respiratoire ou de glycémie ne doit apparaître dans le dialogue. Une température peut apparaître seulement si le patient dit l'avoir prise lui-même.
6. N'invente AUCUN élément clinique absent de la description : pas de nouveau symptôme, pas de nouvel antécédent, pas de nouveau traitement, pas de nouvelle mesure, même plausible. Tu peux enrichir la mise en scène (contexte de vie, formulations, hésitations), jamais la substance médicale. Si la description est sobre, le dialogue reste sobre sur ce point.
   - Le patient ne CONFIRME JAMAIS un symptôme, un signe ou un antécédent absent de la description : si l'agent l'interroge sur un élément qui n'y figure pas, le patient répond « non », « pas que je sache » ou « je sais pas trop ». L'agent pose donc de préférence des questions sur des éléments présents dans la description.
   - Si la description ne donne pas la chronologie précise (heure, jour), reste vague (« depuis quelque temps », « ces derniers jours ») : n'écris ni « ce matin » ni « hier soir » de ta propre initiative.
7. Ne donne aucun nom propre : ni prénom, ni nom, ni initiale, ni marqueur du type [PERSON]. L'agent dit « madame », « monsieur » ou « vous ». Le dialogue COMMENCE par cette phrase de l'agent, à recopier MOT POUR MOT : « {ouverture} ». Elle n'affirme rien sur le cas ; la suite ne reprend jamais la formule « qu'est-ce qui vous amène ».
8. Si le patient ne peut pas s'exprimer (enfant, confusion, obnubilation, malaise), c'est un proche ou un accompagnant qui répond à sa place.
9. Le dialogue commence par l'agent et finit par une réponse du patient : entre 4 et 8 échanges (un échange = une question de l'agent + une réponse du patient). La CONCLUSION est ensuite donnée à part.
10. La CONCLUSION doit être recopiée MOT POUR MOT, sans aucune modification : « {conclusion_imposee} »
STYLE DU PATIENT pour ce dialogue : {style}.

Réponds UNIQUEMENT avec un objet JSON, sans texte avant ni après :
{{"dialogue": [{{"role": "agent", "content": "..."}}, {{"role": "patient", "content": "..."}}, ...], "conclusion": "phrase finale de l'agent, en langage naturel"}}
"""

# Contrôles automatiques
REGEX_CONSTANTES = re.compile(
    r"\b\d{2,3}\s*/\s*\d{2,3}\b|\bmm\s?hg\b|\bspo2\b|\bsao2\b|\bsaturation\b[^.]{0,20}\b\d{2,3}\b|"
    r"\bpouls\b[^.]{0,15}\b\d{2,3}\b|\b\d{2,3}\s?bpm\b|\bglyc[ée]mie\b[^.]{0,20}\d|"
    r"\bfr[ée]quence (cardiaque|respiratoire)\b[^.]{0,20}\d|"
    r"\btension\b[^.]{0,20}\b\d{1,2}([,.]\d)?\s*/\s*\d{1,2}\b",
    re.IGNORECASE)
REGEX_INITIALE_NOM = re.compile(r"\b(Monsieur|Madame|Mme|M\.)\s+[A-Z]\b\.?")


def balise_categorie(categorie: str) -> str:
    return f"[[PRIORITE: {categorie}]]"


def empreinte(texte: str) -> str:
    return hashlib.sha256(texte.encode("utf-8")).hexdigest()[:8]


def choisir_ouverture(id_cas: str, variante: int, contexte: str = "appel") -> str:
    """Première phrase de l'agent : tirée dans la liste du contexte, de façon reproductible (id du cas + variante)."""
    liste = OUVERTURES[contexte]
    return liste[zlib.crc32(f"{id_cas}-{variante}".encode("utf-8")) % len(liste)]


def construire_prompt(style: str, contexte: str = "appel", ouverture: str = None, conclusion_imposee: str = "") -> str:
    return PROMPT_SYSTEME.format(style=style, contexte=CONTEXTES[contexte],
                                  ouverture=ouverture or OUVERTURES[contexte][0],
                                  conclusion_imposee=conclusion_imposee)


def construire_contenu(trace: dict) -> str:
    """Ce que voit le LLM : la description clinique, l'âge et le niveau validé. PAS de constantes, PAS de motif codé."""
    ext = trace.get("etape1_extraction") or {}
    contenu = {
        "description_clinique": trace.get("vignette_source", ""),
        "age_annees": ext.get("age_annees"),
        "niveau_de_priorite_valide": trace["categorie_finale"],
    }
    return json.dumps(contenu, ensure_ascii=False, indent=2)


PROMPT_JUGE = """Tu es un juge de fidélité pour un jeu de dialogues de triage médical généré à partir d'une description clinique.

On te donne : la DESCRIPTION CLINIQUE (la seule source de vérité) et un DIALOGUE entre un agent de triage et un patient (ou un proche).

Ta tâche : repérer les éléments CLINIQUES que les répliques du PATIENT affirment et qui ne figurent PAS dans la description. Compte comme élément clinique : un symptôme, un signe, un antécédent, un traitement, une mesure chiffrée, une chronologie précise inventée.

Règles :
1. Ne signale PAS : les émotions (peur, pleurs, agacement), le vocabulaire courant, le contexte de vie, les reformulations, ni les développements naturels d'un symptôme cité (par exemple « elle ne garde rien » pour des vomissements incoercibles).
2. Une simple réponse « non » ou « je ne sais pas » à une question de l'agent n'est pas un élément inventé.
3. Une confirmation par le patient d'un symptôme, d'un signe ou d'un antécédent absent de la description EST un élément inventé, même formulée avec prudence (« il me semble », « peut-être », « plutôt », « elle a l'air ») : par exemple, dire qu'un enfant « semble chaud » alors que la description dit que la fièvre a disparu, ou que la fièvre est absente, est un signe inventé.
   Compare aussi l'ÉTAT ACTUEL décrit par le patient à celui de la description : un signe présent dans la description mais présenté comme disparu, ou l'inverse, est une invention.
4. Donne à chaque élément une importance : « medicale » (symptôme, signe, antécédent, traitement, mesure, ou chronologie qui change le tableau clinique) ou « mineure » (détail de chronologie ou de circonstances sans effet sur la gravité). Dater un début (« ce matin », « cette nuit », « hier ») alors que la description ne donne aucune date est un élément « mineure » : signale-le quand même.

Réponds UNIQUEMENT avec un objet JSON, sans texte avant ni après :
{"elements_inventes": [{"element": "description courte de l'élément inventé", "importance": "medicale" ou "mineure"}]}
Si rien n'est inventé : {"elements_inventes": []}
"""


def juger_fidelite(description: str, dialogue: list, fournisseur: str, modele: str):
    """Retourne (graves, mineurs) ou None si le juge n'a pas répondu de façon exploitable."""
    contenu = json.dumps({"description_clinique": description, "dialogue": dialogue}, ensure_ascii=False, indent=2)
    res = completer(PROMPT_JUGE, contenu, fournisseur, modele, max_tokens=800, temperature=0.0)
    if "_erreur_parsing" in res or not isinstance(res.get("elements_inventes"), list):
        return None
    elements = [e for e in res["elements_inventes"] if isinstance(e, dict) and e.get("element")]
    return ([e for e in elements if e.get("importance") == "medicale"], [e for e in elements if e.get("importance") != "medicale"])


def valider_dialogue(resultat: dict) -> list:
    """Retourne la liste des problèmes (vide si le dialogue est accepté). Un dialogue avec problème est refusé.
    La conclusion du modèle n'est PAS validée : elle est toujours remplacée par la conclusion imposée
    juste après (cf. generer_variante), son contenu n'a donc aucune importance ici."""
    problemes = []
    dialogue = resultat.get("dialogue")
    if not isinstance(dialogue, list) or not dialogue:
        return ["champ 'dialogue' absent ou vide"]

    roles_attendus = ["agent", "patient"]
    for i, tour in enumerate(dialogue):
        if not isinstance(tour, dict) or tour.get("role") != roles_attendus[i % 2] or not str(tour.get("content", "")).strip():
            problemes.append(f"tour {i} invalide (rôle attendu : {roles_attendus[i % 2]})")
            break
    if len(dialogue) % 2 != 0:
        problemes.append("le dialogue doit finir par une réponse du patient")
    n_echanges = len(dialogue) // 2
    if not 4 <= n_echanges <= 8:
        problemes.append(f"{n_echanges} échanges (attendu : 4 à 8)")

    texte = " ".join(str(t.get("content", "")) for t in dialogue if isinstance(t, dict))
    if "[PERSON]" in texte or re.search(r"\[[A-Z_]{3,}\]", texte):
        problemes.append("marqueur d'anonymisation dans le texte")
    if REGEX_INITIALE_NOM.search(texte):
        problemes.append("initiale de nom propre dans le texte")
    if REGEX_CONSTANTES.search(texte):
        problemes.append("valeur de constante vitale dans le dialogue")

    return problemes


def generer_variante(trace: dict, variante: int, fournisseur: str, modele: str, temperature: float, essais: int,
                     juge: tuple = None, contexte: str = "appel") -> dict:
    """`juge` = (fournisseur, modele) du juge de fidélité, ou None pour ne pas juger."""
    categorie = trace["categorie_finale"]
    style = STYLES[variante % len(STYLES)]
    ouverture = choisir_ouverture(trace["id"], variante, contexte)
    conclusion_imposee = choisir_conclusion(trace["id"], variante, categorie)
    prompt = construire_prompt(style, contexte, ouverture, conclusion_imposee)
    dernier_probleme = None
    refus = []          # raisons des tentatives refusées (pour comprendre pourquoi il faut plusieurs essais)
    for tentative in range(1, essais + 1):
        resultat = completer(prompt, construire_contenu(trace), fournisseur, modele, max_tokens=2500, temperature=temperature)
        if "_erreur_parsing" in resultat:
            dernier_probleme = {"_erreur_parsing": str(resultat["_erreur_parsing"])[:300]}
            refus.append({"tentative": tentative, "problemes": ["réponse illisible du modèle"]})
            continue
        problemes = valider_dialogue(resultat)
        if problemes:
            dernier_probleme = {"problemes": problemes}
            refus.append({"tentative": tentative, "problemes": problemes})
            continue
        # La première phrase de l'agent est imposée : si le modèle l'a modifiée, le code la rétablit.
        ouverture_corrigee = resultat["dialogue"][0]["content"].strip() != ouverture
        resultat["dialogue"][0] = {"role": "agent", "content": ouverture}
        # Le modèle peut omettre le champ ou le laisser vide : sans importance, la conclusion est imposée.
        conclusion_corrigee = str(resultat.get("conclusion") or "").strip() != conclusion_imposee
        resultat["conclusion"] = conclusion_imposee
        fidelite, mineurs = "non_jugee", []
        if juge:
            verdict = juger_fidelite(trace.get("vignette_source", ""), resultat["dialogue"] + [{"role": "agent", "content": resultat["conclusion"]}],
                                     juge[0], juge[1])
            if verdict is None:
                fidelite = "non_verifiee"
            else:
                graves, mineurs = verdict
                if graves:
                    dernier_probleme = {"problemes": ["élément(s) clinique(s) inventé(s) : " + " ; ".join(e["element"] for e in graves)],
                                        "elements_inventes": graves}
                    refus.append({"tentative": tentative, "problemes": dernier_probleme["problemes"]})
                    continue
                fidelite = "verifiee"
        return {
            "id": trace["id"], "cas_id": trace.get("cas_id"), "variante": variante, "style": style,
            "categorie_finale": categorie, "dialogue": resultat["dialogue"], "conclusion": resultat["conclusion"].strip(),
            "balise": balise_categorie(categorie),
            "a_relire": bool(trace.get("a_relire")), "priorite_relecture": trace.get("priorite_relecture"),
            "controles": {"conclusion_coherente": True, "tentatives": tentative, "fidelite": fidelite,
                          "elements_mineurs": [e["element"] for e in mineurs], "refus_precedents": refus,
                          "ouverture_corrigee": ouverture_corrigee, "conclusion_corrigee": conclusion_corrigee},
            "contexte": contexte, "ouverture": ouverture,
            "provenance": {"fournisseur": fournisseur, "modele": modele, "temperature": temperature,
                           "juge": {"fournisseur": juge[0], "modele": juge[1], "prompt": empreinte(PROMPT_JUGE)} if juge else None,
                           "prompt": empreinte(PROMPT_SYSTEME), "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")},
        }
    return {"id": trace["id"], "cas_id": trace.get("cas_id"), "variante": variante, "categorie_finale": categorie,
            "_erreur": {**(dernier_probleme or {"problemes": ["échec sans détail"]}), "refus": refus}}


def lire_variantes(texte: str) -> dict:
    """« urgence_maximale=1,moderee=2,differee=3 » -> dict."""
    resultat = {}
    for morceau in texte.split(","):
        cat, _, n = morceau.partition("=")
        cat = cat.strip()
        if cat not in CATEGORIES or not n.strip().isdigit() or int(n) < 0:
            raise SystemExit(f"--variantes : « {morceau} » invalide (attendu : catégorie=nombre, catégories : {', '.join(CATEGORIES)})")
        resultat[cat] = int(n)
    for cat in CATEGORIES:
        resultat.setdefault(cat, VARIANTES_DEFAUT[cat])
    return resultat


def deja_traites(output_file: Path) -> set:
    """Couples (id, variante) déjà réussis : mécanisme de reprise. Les échecs (_erreur) seront retentés."""
    faits = set()
    if output_file.exists():
        with open(output_file, encoding="utf-8") as f:
            for ligne in f:
                ligne = ligne.strip()
                if ligne:
                    e = json.loads(ligne)
                    if "_erreur" not in e:
                        faits.add((e["id"], e.get("variante", 0)))
    return faits


def main():
    parser = argparse.ArgumentParser(description="Étape 5 : génération des dialogues (plusieurs variantes par cas)")
    parser.add_argument("--input", required=True, help="Sortie JSONL de orchestrer_pipeline.py")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None, help="Nombre de cas à traiter (essai)")
    parser.add_argument("--fournisseur", choices=["groq", "gemini", "anthropic"], default=FOURNISSEUR_DEFAUT)
    parser.add_argument("--modele", default=None, help=f"Défaut : {MODELE_DEFAUT}")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--variantes", default="urgence_maximale=1,moderee=2,differee=3",
                        help="Dialogues par cas selon le label (défaut : 1 / 2 / 3)")
    parser.add_argument("--essais", type=int, default=3, help="Tentatives par dialogue si un contrôle échoue (défaut : 3)")
    parser.add_argument("--contexte", choices=list(OUVERTURES), default="appel",
                        help="appel : appel téléphonique aux urgences (défaut) ; accueil : accueil sur place")
    parser.add_argument("--sans-juge", action="store_true", help="Désactive le juge de fidélité (déconseillé)")
    parser.add_argument("--fournisseur-juge", choices=["groq", "gemini", "anthropic"], default=None,
                        help="Fournisseur du juge (défaut : celui des dialogues)")
    parser.add_argument("--modele-juge", default=None, help="Modèle du juge (défaut : celui des dialogues)")
    parser.add_argument("--pause", type=float, default=1.0)
    args = parser.parse_args()

    if args.fournisseur != FOURNISSEUR_DEFAUT and not args.modele:
        raise SystemExit(f"--fournisseur {args.fournisseur} demande aussi --modele (nom de modèle à prendre dans ta console).")
    modele = args.modele or MODELE_DEFAUT
    variantes = lire_variantes(args.variantes)
    juge = None
    if not args.sans_juge:
        fj = args.fournisseur_juge or args.fournisseur
        if args.modele_juge:
            mj = args.modele_juge
        elif fj == args.fournisseur:
            mj = modele
        elif fj == FOURNISSEUR_DEFAUT:
            mj = MODELE_DEFAUT
        else:
            raise SystemExit(f"--fournisseur-juge {fj} demande aussi --modele-juge (nom de modèle à prendre dans ta console).")
        juge = (fj, mj)

    traces = []
    with open(args.input, encoding="utf-8") as f:
        for ligne in f:
            ligne = ligne.strip()
            if ligne:
                t = json.loads(ligne)
                if t.get("categorie_finale") in CATEGORIES and not t.get("a_relire"):
                    traces.append(t)
    if args.limit:
        traces = traces[: args.limit]

    sortie = Path(args.output)
    sortie.parent.mkdir(parents=True, exist_ok=True)
    faits = deja_traites(sortie)
    travail = [(t, k) for t in traces for k in range(variantes[t["categorie_finale"]]) if (t["id"], k) not in faits]
    print(f"{len(traces)} cas, {len(travail)} dialogues à générer ({len(faits)} déjà faits) avec "
          f"{args.fournisseur}/{modele}, température {args.temperature}, contexte : {args.contexte}, juge de fidélité : "
          f"{'aucun' if not juge else juge[0] + '/' + juge[1]}.", file=sys.stderr)

    n_ok = n_ko = 0
    with open(sortie, "a", encoding="utf-8") as f_out:
        for i, (trace, k) in enumerate(travail, 1):
            try:
                ligne = generer_variante(trace, k, args.fournisseur, modele, args.temperature, args.essais, juge, args.contexte)
            except RuntimeError as e:
                print(f"\n⚠️  Arrêt propre : {e}\n  {i - 1}/{len(travail)} faits ; relancez la même commande pour reprendre.", file=sys.stderr)
                sys.exit(0)
            f_out.write(json.dumps(ligne, ensure_ascii=False) + "\n")
            f_out.flush()
            ok = "_erreur" not in ligne
            n_ok += ok
            n_ko += (not ok)
            print(f"  [{i}/{len(travail)}] {trace['id']} v{k} ({trace['categorie_finale']}) -> {'OK' if ok else 'ÉCHEC'}", file=sys.stderr)
            time.sleep(args.pause)

    print(f"\nTerminé : {n_ok} dialogues, {n_ko} échecs -> {sortie}", file=sys.stderr)


if __name__ == "__main__":
    main()
