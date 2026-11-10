"""Nettoyage des OT.

Chaque étape est une petite fonction, utilisée à la fois par le notebook d'exploration et par `prepare`
(pour le modèle) : le code n'est écrit qu'une fois.
"""
import numpy as np
import pandas as pd

from settings.params import DATE_COL, SAISIE_COL, DEBUT, FIN, FILTRE_SAISIE, EXCLURE_FAMILLES
from src.utils.logger import logger


# ----------------------------------------------------------------------------
# Étapes de nettoyage
# ----------------------------------------------------------------------------
def nettoyer_texte(s):
    """Retire les espaces en trop (y compris l'espace insécable) ; une chaîne vide devient manquante."""
    propre = s.astype(str).str.replace("\xa0", " ").str.strip()
    return propre.where(s.notna() & (propre != ""), np.nan)


def uniformiser(df):
    """Précautions de format (sans effet sur l'export actuel, utiles si un export est moins propre) :
    type d'intervention en majuscules, codes équipement en texte, textes sans espaces parasites."""
    df = df.copy()
    df["typeIntervention"] = df["typeIntervention"].astype(str).str.strip().str.upper()
    df["codeEquipement"] = df["codeEquipement"].astype(str).str.strip()
    for c in ["zone", "fonction", "familleEquipement", "Equipement_Parent"]:
        df[c] = nettoyer_texte(df[c])
    return df


def supprimer_doublons(df):
    """Un même OT peut apparaître deux fois (une ligne avec l'unité, une avec 'DD') :
    on garde la ligne avec l'unité."""
    n0 = len(df)
    df = df.assign(_dd=df["entiteIntervention"].eq("DD"))
    df = df.sort_values(["codeOT", "_dd"]).drop_duplicates("codeOT").drop(columns="_dd")
    logger.info(f"Doublons de codeOT : {n0 - len(df)} lignes supprimées")
    return df


def convertir_dates(df):
    """Convertit les deux dates utiles : réalisation (= date de la panne) et saisie (= OT connu)."""
    df = df.copy()
    for c in [DATE_COL, SAISIE_COL]:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def prepare(df):
    """Fichier brut -> OT prêts pour le feature engineering.
    Ajoute :
      - date_realisation : jour de l'intervention (date de la panne) ;
      - date_connue : jour où l'OT devient visible dans la GMAO = le plus tardif entre la réalisation
        et la saisie (avec FILTRE_SAISIE = False : date_connue = date_realisation).
    Garde les OT réalisés entre DEBUT et FIN."""
    df = convertir_dates(supprimer_doublons(uniformiser(df)))
    df["date_realisation"] = df[DATE_COL].dt.normalize()
    df = df[(df["date_realisation"] >= DEBUT) & (df["date_realisation"] <= FIN)].copy()
    if FILTRE_SAISIE:
        saisie = df[SAISIE_COL].dt.normalize().fillna(df["date_realisation"])
        df["date_connue"] = np.maximum(df["date_realisation"], saisie)
    else:
        df["date_connue"] = df["date_realisation"]
    df = df.sort_values(["codeEquipement", "date_realisation"]).reset_index(drop=True)
    logger.info(f"OT préparés : {len(df):,} OT réalisés du {DEBUT:%d/%m/%Y} au {FIN:%d/%m/%Y}, "
                f"{df['codeEquipement'].nunique():,} équipements")
    return df


# ----------------------------------------------------------------------------
# Périmètre et groupes
# ----------------------------------------------------------------------------
def est_exclu(familles):
    """True pour les codes qui ne sont pas des équipements du réseau (localités, unités, groupes, véhicules)."""
    return pd.Series(familles).astype(str).isin(EXCLURE_FAMILLES).values


def groupe_zone(z):
    """Zone -> 'Dakar', 'Régions' (zones commençant par DR) ou 'Autres'."""
    z = pd.Series(z).astype(object).fillna("INCONNU").astype(str)
    return np.where(z == "DAKAR", "Dakar", np.where(z.str.startswith("DR"), "Régions", "Autres"))
