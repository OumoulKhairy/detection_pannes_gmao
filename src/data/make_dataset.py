"""Chargement des données brutes."""
from pathlib import Path

import pandas as pd

from settings.params import FICHIER_OT
from src.utils.logger import logger


def load_data(chemin=FICHIER_OT):
    """Lit l'export GMAO (un OT par ligne). Le fichier n'est jamais modifié :
    toutes les transformations sont faites ensuite, sur une copie."""
    chemin = Path(chemin)
    if not chemin.exists():
        raise FileNotFoundError(f"{chemin} introuvable : placer FaitOT.xlsx dans data/raw/")
    df = pd.read_excel(chemin)
    logger.info(f"Chargé : {chemin.name} | {df.shape[0]:,} OT x {df.shape[1]} colonnes")
    return df
