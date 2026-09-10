"""
schema.py

Définit le format commun ("vignette normalisée") vers lequel tous les
scripts d'ingestion (load_mediqal.py, load_frenchmedmcqa.py, load_medquad.py)
doivent convertir leurs corpus respectifs.

L'objectif : que la suite du pipeline (Étapes 1 à 5) puisse traiter n'importe
quelle vignette de la même façon, quelle que soit sa source d'origine.
"""

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class VignetteNormalisee:
    id: str                                # identifiant unique, préfixé par la source (ex. "mediqal_mcqm_11399")
    source: str                            # "MediQAl" | "FrenchMedMCQA" | "MedQuAD"
    subset: Optional[str] = None           # ex. "mcqm", "mcqu", "oeq" (spécifique à MediQAl)
    split: Optional[str] = None            # "train" | "validation" | "test" (si fourni par la source)
    langue: str = "fr"                     # "fr" | "en"
    contexte_clinique: Optional[str] = None  # description du cas patient, si disponible
    question: Optional[str] = None
    choix: Optional[dict] = None           # {"a": "...", "b": "...", ...} pour les QCM, None sinon
    reponse_correcte: Optional[str] = None # lettre(s) ou texte libre selon la source
    specialite: Optional[str] = None       # discipline médicale, si disponible
    type_question: Optional[str] = None    # ex. "Understanding" / "Reasoning" (MediQAl)
    task: Optional[str] = None             # type de tâche source (QCM, QROC, QA...)

    def to_dict(self) -> dict:
        return asdict(self)


CHAMPS_OBLIGATOIRES = ["id", "source", "langue", "question"]


def valider_vignette(vignette: dict) -> list:
    """Retourne la liste des champs obligatoires manquants ou vides."""
    manquants = []
    for champ in CHAMPS_OBLIGATOIRES:
        if not vignette.get(champ):
            manquants.append(champ)
    return manquants
