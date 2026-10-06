"""
app.py — démo de l'agent de triage CHSA sur Hugging Face Spaces (ZeroGPU).

Même fonctionnement que src/demo/app_demo.py (démo locale) :
  1. l'hôtesse (Qwen3-1.7B-Base + adaptateur LoRA SFT) mène l'entretien ;
  2. le plancher de sécurité est évalué à CHAQUE message du patient : un signe d'alerte conclut tout de suite ;
  3. quand l'hôtesse veut conclure, sa proposition est ignorée : le décideur (encodeur) tranche, avec le
     plancher puis le seuil de prudence ; la phrase de conclusion est choisie par le code.

Différences avec la démo locale :
  - les modèles sont téléchargés depuis le Hub (pas de dossiers locaux) ;
  - ZeroGPU : un GPU n'est attribué que pendant la génération de l'hôtesse (fonction @spaces.GPU) ;
    le décideur reste sur CPU ;
  - le disque d'un Space n'est pas permanent : le journal d'audit est écrit dans les logs du Space
    (onglet « Logs »), une ligne JSON par décision. En production : stockage persistant certifié HDS.

Ce fichier est assemblé avec le code du dépôt par deploiement/assembler_space.sh (source unique).
"""
try:
    import spaces  # ZeroGPU : doit être importé avant torch
    GPU = spaces.GPU(duration=20)  # latence mesurée : 1 à 12 s par réponse
except ImportError:  # GPU dédié (T4) : pas de ZeroGPU, la fonction s'exécute directement
    GPU = lambda fonction: fonction

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

RACINE = Path(__file__).parent
sys.path.insert(0, str(RACINE / "src" / "entrainement"))
sys.path.insert(0, str(RACINE / "src" / "demo"))
from prompt_agent import PROMPT_SYSTEME_AGENT as PROMPT_SYSTEME  # noqa: E402
from decideur import Decideur, plancher  # noqa: E402

MODELE_BASE = "Qwen/Qwen3-1.7B-Base"
DEPOT_HOTESSE = "vler0ux/qwen3-1.7b-triage-chsa-hotesse"
DEPOT_DECIDEUR = "vler0ux/moderncamembert-bio-triage-chsa-decideur"
MAX_NEW_TOKENS = 200

# ------------------------------------------------------------------ chargement (au démarrage du Space)
tokenizer = AutoTokenizer.from_pretrained(MODELE_BASE)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# Même chat template qu'à l'entraînement (source unique : src/entrainement/template_triage.jinja)
tokenizer.chat_template = (RACINE / "src" / "entrainement" / "template_triage.jinja").read_text(encoding="utf-8")

# T4 (architecture Turing) : float16 ; bfloat16 n'y est pas pris en charge nativement.
DTYPE = torch.float16 if os.environ.get("ACCELERATOR", "").startswith("t4") else torch.bfloat16
modele_base = AutoModelForCausalLM.from_pretrained(MODELE_BASE, dtype=DTYPE)
modele = PeftModel.from_pretrained(modele_base, DEPOT_HOTESSE, torch_device="cpu").to("cuda").eval()

decideur = Decideur(snapshot_download(DEPOT_DECIDEUR))  # CPU
decideur.dossier = DEPOT_DECIDEUR  # trace lisible dans le journal (au lieu du chemin de cache)

FIN_IDS = [tokenizer.convert_tokens_to_ids(t) for t in ("<|im_end|>", "atrigesimal", "<|im_start|>", "<|endoftext|>")]

ACCUEIL = ("Centre Hospitalier Saint-Aurélien, service des urgences, bonjour. "
           "Quel est le motif de votre appel ?")


def journaliser(messages, niveau_hotesse, decision, latence_s=None):
    """Journal d'audit : une ligne JSON par décision, dans les logs du Space."""
    entree = {
        "horodatage": datetime.now(timezone.utc).isoformat(),
        "conversation": [m for m in messages if m["role"] != "system"],
        "niveau_hotesse": niveau_hotesse,
        **decision,
        "modele_hotesse": DEPOT_HOTESSE,
        "latence_generation_s": latence_s,
    }
    print("[audit] " + json.dumps(entree, ensure_ascii=False), flush=True)


@GPU
def generer(texte_entree):
    """Réponse de l'hôtesse (seule étape qui utilise le GPU)."""
    entrees = tokenizer(texte_entree, return_tensors="pt").to("cuda")
    with torch.no_grad():
        sortie = modele.generate(
            **entrees,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.1,
            eos_token_id=FIN_IDS,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(sortie[0][entrees["input_ids"].shape[1]:], skip_special_tokens=False)


def repondre(message, historique):
    messages = [{"role": "system", "content": PROMPT_SYSTEME}]
    for m in historique:
        contenu = m["content"]
        if isinstance(contenu, list):  # format Gradio : [{'type': 'text', 'text': ...}]
            contenu = " ".join(c.get("text", "") for c in contenu if isinstance(c, dict))
        messages.append({"role": m["role"], "content": contenu})
    messages.append({"role": "user", "content": message})

    # Plancher évalué à chaque message du patient : un signe d'alerte suffit à conclure.
    if plancher(messages):
        decision = decideur.decider(messages)
        decision["source"] = "plancher_immediat"
        journaliser(messages, None, decision)
        return decision["conclusion"]

    texte_entree = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    debut = time.perf_counter()
    reponse_brute = generer(texte_entree)
    latence = round(time.perf_counter() - debut, 2)
    print(f"[latence] génération de l'hôtesse : {latence} s", flush=True)
    reponse = reponse_brute.split("<|im_end|>")[0].split("<|im_start|>")[0].split("atrigesimal")[0].strip()

    # L'hôtesse conclut par "[[PRIORITE: ...]]" : sa proposition est remplacée par celle du décideur.
    if "[[PRIORITE:" in reponse:
        _, _, balise = reponse.partition("[[PRIORITE:")
        niveau_hotesse = balise.rstrip("]").strip()
        decision = decideur.decider(messages)
        journaliser(messages, niveau_hotesse, decision, latence)
        reponse = decision["conclusion"]

    return reponse or "Pouvez-vous reformuler, s'il vous plaît ?"


demo = gr.ChatInterface(
    fn=repondre,
    chatbot=gr.Chatbot(value=[{"role": "assistant", "content": ACCUEIL}], label="Agent de triage CHSA"),
    title="Agent de triage CHSA — démo (POC)",
    description=(
        "⚠️ Preuve de concept à but de démonstration, sur données synthétiques : ne constitue pas une décision "
        "médicale. En cas d'urgence réelle, appelez le 15. "
        "L'hôtesse (Qwen3-1.7B + LoRA) mène l'entretien ; la priorité est décidée par un plancher de sécurité "
        "puis un classifieur (ModernCamemBERT-bio). Une décision « différée » devrait toujours être validée par "
        "un soignant. Répondez au message d'accueil comme si vous étiez le patient. "
        "GPU partagé (ZeroGPU) : quota de quelques minutes par jour et par visiteur."
    ),
    examples=[
        "Bonjour, j'ai très mal à la poitrine depuis ce matin et j'ai du mal à respirer.",
        "Mon fils de 4 ans a de la fièvre depuis hier soir, 39 degrés.",
        "J'ai une petite coupure au doigt en cuisinant, ça saigne un peu.",
    ],
)

if __name__ == "__main__":
    demo.launch()
