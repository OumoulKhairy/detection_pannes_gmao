"""
Paramètres du projet "Détection de pannes avec la GMAO".

C'est le SEUL endroit où l'on règle le projet : les notebooks et les fonctions de src/ lisent leurs
valeurs ici. Pour tester un autre choix, on change la valeur ici et on relance.

Question du projet (validée avec le maître de stage) : pour chaque équipement, à une date donnée,
  - est-il à risque de panne dans les h prochains jours (h choisi librement : 10, 34, 90 jours...) ?
  - dans combien de jours la prochaine panne est-elle attendue ?
On y répond avec une approche de SURVIE : la cible est le délai jusqu'à la prochaine panne.
"""
from pathlib import Path

import pandas as pd

# ----------------------------------------------------------------------------
# Chemins
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
FICHIER_OT = DATA_RAW / "FaitOT.xlsx"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# ----------------------------------------------------------------------------
# Définition de la panne et des dates (validées avec l'encadrant)
# ----------------------------------------------------------------------------
TYPE_PANNE = "CORR"              # une panne = une intervention corrective
TYPE_PREVENTIF = "PREV"          # maintenance préventive
DATE_COL = "dateDebutPrevue"     # date réelle de l'intervention : date de la panne
SAISIE_COL = "dateCreation"      # date d'enregistrement dans la GMAO : date où l'OT devient connu
FILTRE_SAISIE = True             # True : à une date donnée, on ne voit que les OT déjà saisis

# ----------------------------------------------------------------------------
# Période étudiée (au jour)
# ----------------------------------------------------------------------------
DEBUT = pd.Timestamp("2019-01-01")        # premier jour étudié
FIN = pd.Timestamp("2025-12-31")          # dernier jour observé
TROU_DEBUT = pd.Timestamp("2025-06-01")   # extraction incomplète : juin et juillet 2025
TROU_FIN = pd.Timestamp("2025-07-31")     # (aucune date de coupure, aucun suivi dans ce trou)

# ----------------------------------------------------------------------------
# Dates de coupure et découpage temporel
# ----------------------------------------------------------------------------
# Une "date de coupure" est le jour où l'on se place pour prédire : on regarde le passé de chaque
# équipement jusqu'à ce jour, et on mesure le délai jusqu'à sa prochaine panne. On prend le dernier jour
# de chaque mois. Chaque ensemble a une "fin d'observation" : au-delà, on ne regarde pas les pannes
# (sinon l'entraînement verrait la période de validation ou de test).
COUPURES_TRAIN = ("2019-01-31", "2024-10-31")
FIN_OBS_TRAIN = "2024-11-30"
COUPURES_VAL = ("2024-11-30", "2025-04-30")
FIN_OBS_VAL = "2025-05-31"
COUPURES_TEST = ("2025-08-31", "2025-11-30")
FIN_OBS_TEST = "2025-12-31"
DATE_FUTUR = "2025-12-31"        # dernière date connue : sert à la prévision réelle (pas de cible)

H_MAX = 365                      # on suit chaque date de coupure au plus 365 jours
HORIZONS = [7, 30, 90]           # horizons (en jours) utilisés pour l'analyse et l'évaluation
HORIZON_REFERENCE = 30           # horizon des graphiques "panne dans les 30 jours"

# ----------------------------------------------------------------------------
# Périmètre : codes qui ne sont pas des équipements du réseau
# (exclus de l'analyse des variables, de l'évaluation et de la liste de risque)
# ----------------------------------------------------------------------------
EXCLURE_FAMILLES = [
    "LOCALITE",                  # zones géographiques
    "FAMDEF",                    # "famille par défaut" : unités
    "GRPMT",                     # groupes d'équipements
    "V-EXPLO", "V-CADRE",        # véhicules (zone GARAGE)
    "VEH_LEGER", "VEH_LOURD",    # véhicules
]
# À confirmer avec l'encadrant : codes C4, 401MD, GE401 (zone 1000)

# ----------------------------------------------------------------------------
# Colonnes du fichier FaitOT
# ----------------------------------------------------------------------------
COLONNES_GARDEES = [
    "codeOT", "codeEquipement", "typeIntervention", "dateDebutPrevue", "dateCreation",
    "familleEquipement", "fonction", "zone", "coutMaintenance", "Equipement_Parent",
    "entiteIntervention",
]
COLONNES_ECARTEES = {
    "SOURCE": "vide à 100 %",
    "Alimente_par": "décision de l'encadrant, vide à 91 %",
    "etatOT": "décision de l'encadrant : état administratif de l'OT",
    "numeroPlan": "vide à 78 %, ne concerne que les préventifs",
    "entiteEquipement": "vide à 72 %",
    "CoutMainDoeuvre": "détail de coutMaintenance",
    "CoutMateriel": "détail de coutMaintenance",
    "CoutService": "détail de coutMaintenance",
    "CoutMoyen": "détail de coutMaintenance",
    "intervention": "répète typeIntervention",
    "entiteAction": "répète en partie fonction et zone",
    "classeIntervention": "décrit l'intervention",
    "dateFinPrevue": "décrit l'intervention",
    "DatePlanifiee": "décrit l'intervention",
}

# ----------------------------------------------------------------------------
# Variables du modèle (créées au feature engineering, au jour près)
# ----------------------------------------------------------------------------
FENETRES = [7, 30, 90, 365]      # fenêtres (en jours) pour compter les pannes passées
JOURS_JAMAIS = 3000              # "jours depuis" quand l'équipement n'a jamais eu l'intervention
                                 # (plus long que toute la période étudiée : 2 556 jours)
FEATURES_NUM = ["n_corr_7j", "n_corr_30j", "n_corr_90j", "n_corr_365j", "n_corr_total",
                "n_prev_365j", "n_prev_total", "jours_depuis_corr", "jours_depuis_prev",
                "ecart_moyen_corr", "log_cout_365j", "parent_corr_90j", "anciennete_jours", "mois"]
FEATURES_CAT = ["familleEquipement", "zone", "fonction"]
DUREE, EVENEMENT = "duree", "evenement"   # cible de survie : délai (jours) et panne observée (1) ou non (0)

# Listes décidées dans 01_01_exploration_donnees (phase 7, règle de redondance)
FEATURES_ALL = FEATURES_NUM + FEATURES_CAT          # modèles à arbres : supportent les variables redondantes
FEATURES_LINEAR = [                                 # modèles linéaires : sans les variables redondantes
    "n_corr_7j", "n_corr_30j", "n_corr_90j", "n_corr_365j", "n_corr_total", "jours_depuis_prev",
    "parent_corr_90j", "anciennete_jours", "mois", "familleEquipement", "zone", "fonction",
]   # retirées : log_cout_365j, jours_depuis_corr, ecart_moyen_corr, n_prev_365j, n_prev_total

# ----------------------------------------------------------------------------
# Analyse et sélection
# ----------------------------------------------------------------------------
SEUIL_CORRELATION = 0.8          # deux variables plus corrélées que ça (Spearman) : on n'en garde qu'une
TAILLE_ECHANTILLON_GRAPHIQUES = 200_000   # lignes utilisées pour les calculs lourds de l'exploration
TOP_FRAC = 0.10                  # part des équipements alertés
RANDOM_STATE = 42                # graine : mêmes résultats à chaque exécution
