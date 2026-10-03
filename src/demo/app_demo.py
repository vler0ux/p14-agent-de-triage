"""
app_demo.py

Petite interface Gradio pour tester l'agent de triage entraîné par SFT
(Qwen3-1.7B-Base + adaptateur LoRA), en conversation multi-tours.

⚠️ Le prompt système est importé de src/entrainement/prompt_agent.py, la même
source que le dataset d'entraînement : le modèle a appris à répondre dans CE
contexte précis, un prompt différent peut dégrader ses réponses.

Usage :
    python app_demo.py --adapter-dir ../../qwen3-1.7b-triage-sft
    (ou le chemin où tu as dézippé l'archive téléchargée depuis Kaggle)

Ouvre ensuite le lien local affiché dans le terminal (http://127.0.0.1:7860).
"""

import argparse
import sys
import json
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "entrainement"))
from prompt_agent import PROMPT_SYSTEME_AGENT as PROMPT_SYSTEME  # noqa: E402


MODELE_BASE_PAR_DEFAUT = "Qwen/Qwen3-1.7B-Base"

CHAT_TEMPLATE_SECOURS = (
    # Copie de secours uniquement (voir main() : le vrai fichier template_triage.jinja
    # est chargé en priorité). Gardée synchronisée à la main avec ce fichier — en cas de
    # divergence, c'est TOUJOURS template_triage.jinja qui doit faire foi.
    "{%- for message in messages -%}"
    "{%- if message.role == 'assistant' -%}"
    "{{- '<|im_start|>assistant\\n' -}}"
    "{{- (message.content | trim) + '<|im_end|>' -}}"
    "{{- '\\n' -}}"
    "{%- else -%}"
    "{{- '<|im_start|>' + message.role + '\\n' + (message.content | trim) + '<|im_end|>\\n' -}}"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}"
    "{{- '<|im_start|>assistant\\n' -}}"
    "{%- endif -%}"
)


def main():
    parser = argparse.ArgumentParser(description="Démo Gradio de l'agent de triage (Qwen3-1.7B-Base + LoRA)")
    parser.add_argument("--adapter-dir", required=True,
                         help="Dossier de l'adaptateur LoRA téléchargé depuis Kaggle (contient adapter_config.json)")
    parser.add_argument("--modele-base", default=MODELE_BASE_PAR_DEFAUT)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--n-exemples-entrainement", type=int, default=None,
                         help="Nombre d'exemples du dataset d'entraînement utilisé (affiché dans la description ; "
                              "évite d'afficher un chiffre en dur qui devient faux dès que le dataset change)")
    parser.add_argument("--share", action="store_true", help="Génère un lien public temporaire Gradio")
    parser.add_argument("--decideur-dir", default=None,
                        help="Dossier du décideur encodeur (ex. decideur-patient). Sans lui, la décision reste celle de l'hôtesse.")
    # Défaut ancré sur le dossier du script (src/demo/logs/, ignoré par git) : un chemin relatif
    # au dossier de lancement écrirait ailleurs selon l'endroit d'où la démo est lancée.
    parser.add_argument("--journal", default=str(Path(__file__).parent / "logs" / "decisions.jsonl"),
                        help="Journal d'audit des décisions (contient les conversations : ne pas versionner)")
    parser.add_argument("--adapter-sft-dir", default=None,
                        help="Pour un adaptateur DPO : dossier de l'adaptateur SFT à fusionner d'abord (v4)")
    args = parser.parse_args()
    adapter_dir = Path(args.adapter_dir)
    assert (adapter_dir / "adapter_config.json").exists(), (
        f"adapter_config.json introuvable dans {adapter_dir} — vérifie le chemin "
        "(c'est le dossier obtenu en dézippant l'archive téléchargée depuis Kaggle)."
    )

    import torch
    import gradio as gr
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    cuda = torch.cuda.is_available()
    if cuda:
        dtype = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
    else:
        dtype = torch.float32
    print(f"Device : {'cuda' if cuda else 'cpu'} | dtype : {dtype}")

    print(f"Chargement du tokenizer et du modèle de base : {args.modele_base}")
    tokenizer = AutoTokenizer.from_pretrained(args.modele_base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    modele_base = AutoModelForCausalLM.from_pretrained(
        args.modele_base,
        dtype=dtype,
        device_map="auto" if cuda else None,
    )
    
    if args.adapter_sft_dir:
        # L'adaptateur DPO a été entraîné par-dessus le SFT v4 FUSIONNÉ : on reproduit ce point de départ.
        print(f"Fusion préalable de l'adaptateur SFT : {args.adapter_sft_dir}")
        modele_base = PeftModel.from_pretrained(modele_base, args.adapter_sft_dir).merge_and_unload()
        
    print(f"Chargement de l'adaptateur LoRA : {adapter_dir}")
    modele = PeftModel.from_pretrained(modele_base, str(adapter_dir))
    modele.eval()
    
    decideur = None
    if args.decideur_dir:
        from decideur import Decideur, plancher
        decideur = Decideur(args.decideur_dir)
        print(f"Décideur chargé : {args.decideur_dir}")

    chemin_journal = Path(args.journal)
    chemin_journal.parent.mkdir(parents=True, exist_ok=True)

    def journaliser(messages, niveau_hotesse, decision):
        entree = {
            "horodatage": datetime.now(timezone.utc).isoformat(),
            "conversation": [m for m in messages if m["role"] != "system"],
            "niveau_hotesse": niveau_hotesse,
            **decision,
            "modele_hotesse": str(adapter_dir),
        }
        with open(chemin_journal, "a", encoding="utf-8") as f:
            f.write(json.dumps(entree, ensure_ascii=False) + "\n")

    # Template de chat : on charge le vrai fichier template_triage.jinja (source unique)
    # plutôt que de le dupliquer à la main ici, pour ne jamais diverger silencieusement
    # de ce qui a servi à l'entraînement. Chemin relatif : src/demo/ -> src/entrainement/.
    chemin_template = Path(__file__).parent.parent / "entrainement" / "template_triage.jinja"
    if chemin_template.exists():
        tokenizer.chat_template = chemin_template.read_text(encoding="utf-8")
        print(f"Template de chat chargé depuis {chemin_template}")
    else:
        print(f"[avertissement] {chemin_template} introuvable — utilisation d'une copie de secours "
              "intégrée à ce script, qui peut diverger de la vraie source. Corrige le chemin ci-dessus si besoin.")
        tokenizer.chat_template = CHAT_TEMPLATE_SECOURS

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    substitut_fin_id = tokenizer.convert_tokens_to_ids("atrigesimal")
    assert substitut_fin_id is not None and tokenizer.convert_ids_to_tokens(substitut_fin_id) == "atrigesimal", "substitut de fin introuvable dans le vocabulaire"
    im_end_multi_tokens = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    print(f"[diagnostic] <|im_end|> -> id unique {im_end_id} ; encodage complet {im_end_multi_tokens} "
          f"({'OK, un seul token' if len(im_end_multi_tokens) == 1 else 'PROBLÈME : plusieurs tokens, eos_token_id ne peut pas fonctionner seul'})")

    ACCUEIL = (
        "Centre Hospitalier Saint-Aurélien, service des urgences, bonjour. "
        "Quel est le motif de votre appel?"
    )

    def repondre(message, historique):
        # historique est déjà une liste de {"role": ..., "content": ...} (format "messages" de Gradio) :
        # on la reprend telle quelle, pas besoin de la retraduire depuis des paires (user, bot).
        messages = [{"role": "system", "content": PROMPT_SYSTEME}]
        for m in historique:
            contenu = m["content"]
            if isinstance(contenu, list):  # nouveau format Gradio : [{'type': 'text', 'text': ...}]
                contenu = " ".join(c.get("text", "") for c in contenu if isinstance(c, dict))
            messages.append({"role": m["role"], "content": contenu})
        messages.append({"role": "user", "content": message})
        # Plancher évalué à CHAQUE message du patient, avant de laisser parler l'hôtesse :
        # un signe d'alerte suffit à conclure, inutile de poursuivre l'entretien.
        if decideur is not None and plancher(messages):
            decision = decideur.decider(messages)
            decision["source"] = "plancher_immediat"
            print(f"[triage] plancher déclenché : {decision['signes_alerte']} -> {decision['niveau']}")
            journaliser(messages, None, decision)
            return decision["conclusion"]

        texte_entree = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        entrees = tokenizer(texte_entree, return_tensors="pt").to(modele.device)
        

        with torch.no_grad():
            # Réglages d'INFÉRENCE uniquement, sans équivalent côté entraînement (SFT classique,
            # pas de sampling) : do_sample=False (glouton) et repetition_penalty=1.15 sont des
            # mitigations empiriques contre les dérives multilingues observées en génération libre
            # (cf. tête de sortie non adaptée par le LoRA actuel, ciblé q/k/v/o_proj uniquement).
            # À revoir une fois le modèle réentraîné (nouveau découpage + balise).
            sortie = modele.generate(
                **entrees,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                repetition_penalty=1.1,
                eos_token_id=[im_end_id,substitut_fin_id,
                              tokenizer.convert_tokens_to_ids("<|im_start|>"),
                              tokenizer.convert_tokens_to_ids("<|endoftext|>")],
                pad_token_id=tokenizer.pad_token_id,
            )

        nouveaux_tokens = sortie[0][entrees["input_ids"].shape[1]:]
        # skip_special_tokens=False : on garde <|im_end|> dans le texte décodé pour pouvoir
        # couper dessus nous-mêmes. Si le critère d'arrêt eos_token_id ne s'est pas déclenché
        # (ex. <|im_end|> n'est pas un token unique pour ce tokenizer), le modèle peut continuer
        # à générer du bruit au-delà de la vraie réponse — on l'ignore en coupant ici.
        reponse_brute = tokenizer.decode(nouveaux_tokens, skip_special_tokens=False)
        print("[diag] derniers tokens :", tokenizer.convert_ids_to_tokens(nouveaux_tokens.tolist()[:60]))
        reponse = reponse_brute.split("<|im_end|>")[0].split("<|im_start|>")[0].split("atrigesimal")[0].strip()

        # L'hôtesse termine sa conclusion par "\n[[PRIORITE: <categorie>]]". On retire la balise
        # de l'affichage. Si un décideur est chargé, sa décision (plancher, puis encodeur avec
        # seuil) remplace celle de l'hôtesse, et la conclusion affichée est choisie par le code.
        if "[[PRIORITE:" in reponse:
            reponse, _, balise = reponse.partition("[[PRIORITE:")
            reponse = reponse.strip()
            niveau_hotesse = balise.rstrip(']').strip()
            print(f"[triage] catégorie proposée par l'hôtesse : {niveau_hotesse}")
            if decideur is not None:
                decision = decideur.decider(messages)
                reponse = decision["conclusion"]
                print(f"[triage] décision retenue : {decision['niveau']} ({decision['source']}) | {decision['probas']}")
                journaliser(messages, niveau_hotesse, decision)

        if not reponse:
            reponse = "(réponse vide — voir le terminal pour le texte brut généré)"
            print("[avertissement] réponse vide après troncature. Texte brut :", repr(reponse_brute[:300]))
        return reponse

    description_volume = (
        f", entraîné sur {args.n_exemples_entrainement} exemples" if args.n_exemples_entrainement else ""
    )
    demo = gr.ChatInterface(
        fn=repondre,
        chatbot=gr.Chatbot(value=[{"role": "assistant", "content": ACCUEIL}], label="Agent de triage CHSA"),
        title="Agent de triage CHSA — démo (POC)",
        description=(
            "⚠️ POC à but de démonstration uniquement — ne reflète pas une décision médicale validée. "
            f"Qwen3-1.7B-Base + LoRA{description_volume}. "
            "Répondez au message d'accueil ci-dessous comme si vous étiez le patient."
        ),
        examples=[
            "Bonjour, j'ai très mal à la poitrine depuis ce matin et j'ai du mal à respirer.",
            "Mon fils de 4 ans a de la fièvre depuis hier soir, 39 degrés.",
            "J'ai une petite coupure au doigt en cuisinant, ça saigne un peu.",
        ],
    )
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()