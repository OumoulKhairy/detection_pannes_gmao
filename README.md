# Détection de pannes avec la GMAO

**Question :** un équipement du réseau aura-t-il au moins une intervention corrective (`CORR`) le mois suivant ?

## Organisation

```
detection_pannes_gmao/
├── data/
│   ├── raw/FaitOT.xlsx          # export GMAO brut (non versionné)
│   └── processed/               # train / val / test générés
├── notebooks/
│   ├── 01_01_eda.ipynb          # exploration et feature engineering
│   ├── 01_02_splitting.ipynb    # panel, découpage temporel, sauvegarde
│   ├── 02_modeling.ipynb        # baselines, LightGBM, réglage, comparaison
│   └── 03_interpretability.ipynb
├── settings/params.py           # tous les paramètres
├── src/
│   ├── data/                    # chargement, nettoyage
│   ├── features/                # création des variables
│   ├── models/                  # entraînement, réglage, évaluation
│   └── utils/                   # outils communs (logger)
├── models/                      # modèles sauvegardés
├── reports/figures/             # graphiques
└── requirements.txt
```

Les fonctions sont écrites une seule fois dans `src/` ; les notebooks les importent.

## Installation

```
python -m venv .venv
.venv\Scripts\activate          (Windows)
python -m pip install -r requirements.txt
```

Placer `FaitOT.xlsx` dans `data/raw/`.

## Règles du projet (validées avec l'encadrant)

- Toutes les interventions `CORR` sont des pannes.
- Une panne est datée par `dateDebutPrevue` ; `dateCreation` est la date de saisie.
- À un mois donné, le modèle ne voit que les OT déjà saisis.
- Les codes qui ne sont pas des équipements du réseau (localités, unités, groupes d'équipements, véhicules) sont exclus de l'évaluation et de la liste de risque.
- Découpage temporel : entraînement jusqu'à novembre 2024, validation de décembre 2024 à mai 2025, test de septembre à décembre 2025.
