# Détection de pannes avec la GMAO

**Question :** pour chaque équipement du réseau, à une date donnée :
- est-il **à risque de panne** (intervention corrective `CORR`) dans les **h prochains jours**, h étant choisi librement (10, 34, 90 jours…) ?
- **dans combien de jours** la prochaine panne est-elle attendue ?

On y répond avec une **approche de survie** : la cible est le délai jusqu'à la prochaine panne.

## Organisation

```
detection_pannes_gmao/
├── data/
│   ├── raw/FaitOT.xlsx                  # export GMAO brut (non versionné)
│   └── processed/                       # train / val / test en Parquet (non versionnés)
├── notebooks/
│   ├── 01_01_exploration_donnees.ipynb  # exploration et feature engineering
│   ├── 01_02_splitting.ipynb            # variables, cible, découpage temporel, sauvegarde
│   ├── 02_modeling.ipynb                # (à venir) modèles de survie
│   └── 03_interpretability.ipynb        # (à venir)
├── settings/params.py                   # tous les paramètres
├── src/
│   ├── data/                            # chargement (make_dataset.py), nettoyage (preprocess.py)
│   ├── features/build.py                # variables au jour, cible de survie, découpage, Kaplan-Meier
│   ├── models/                          # (à venir) entraînement, évaluation
│   └── utils/                           # logger, graphiques
├── models/                              # modèles sauvegardés
├── reports/                             # rapports HTML des notebooks exécutés (non versionnés)
└── requirements.txt
```

Les fonctions sont écrites une seule fois dans `src/` ; les notebooks les importent.

## Installation

```
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1          (Windows)
python -m pip install -r requirements.txt
```

Placer `FaitOT.xlsx` dans `data/raw/`, puis exécuter les notebooks dans l'ordre (`01_01`, puis `01_02`).

## Règles du projet (validées avec l'encadrant)

- Toutes les interventions `CORR` sont des pannes.
- Une panne est datée par `dateDebutPrevue` (réalisation) ; `dateCreation` est la date de saisie.
- À une date donnée, le modèle ne voit que les OT réalisés **et déjà saisis**.
- Les codes qui ne sont pas des équipements du réseau (localités, unités, groupes d'équipements, véhicules) sont exclus de l'évaluation et de la liste de risque.
- Dates de coupure : dernier jour de chaque mois. Découpage temporel : entraînement (coupures de janvier 2019 à octobre 2024, observé jusqu'au 30/11/2024), validation (novembre 2024 à avril 2025, jusqu'au 31/05/2025), test (août à novembre 2025, jusqu'au 31/12/2025).
- Suivi maximal : 365 jours ; aucun suivi pendant le trou d'extraction de juin-juillet 2025.
