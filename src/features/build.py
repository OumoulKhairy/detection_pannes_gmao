"""Feature engineering au jour près, cible de survie, découpage temporel,
puis mesures de lien entre variables et règle de redondance.

Principe
  - On se place à une "date de coupure" d (le dernier jour de chaque mois).
  - Pour chaque équipement déjà connu à d, les variables résument son passé, en n'utilisant que les OT
    RÉALISÉS au plus tard le jour d ET DÉJÀ SAISIS dans la GMAO le jour d (aucune information du futur).
  - La cible est le délai (en jours) entre d et la prochaine panne réalisée :
      evenement = 1 : une panne est observée, duree = nombre de jours jusqu'à elle ;
      evenement = 0 : aucune panne observée avant la fin du suivi (censure), duree = durée du suivi.
    Le suivi s'arrête au plus tard après H_MAX jours, à la fin d'observation de l'ensemble, ou au début
    du trou d'extraction (juin-juillet 2025).
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from settings.params import (TYPE_PANNE, TYPE_PREVENTIF, FENETRES, JOURS_JAMAIS, FEATURES_CAT, H_MAX,
                             TROU_DEBUT, TROU_FIN, COUPURES_TRAIN, FIN_OBS_TRAIN, COUPURES_VAL, FIN_OBS_VAL,
                             COUPURES_TEST, FIN_OBS_TEST, DATE_FUTUR, DUREE, EVENEMENT)
from src.data.preprocess import est_exclu
from src.utils.logger import logger

EQ = "codeEquipement"
R = "date_realisation"


# ----------------------------------------------------------------------------
# 1. Variables à une date de coupure
# ----------------------------------------------------------------------------
def ot_visibles(ot, d):
    """OT réalisés au plus tard le jour d et déjà saisis le jour d."""
    return ot[(ot[R] <= d) & (ot["date_connue"] <= d)]


def variables_a_date(ot, d):
    """Une ligne par équipement connu le jour d, avec ses variables calculées sur les OT visibles."""
    d = pd.Timestamp(d)
    v = ot_visibles(ot, d)
    corr = v[v["typeIntervention"] == TYPE_PANNE]
    prev = v[v["typeIntervention"] == TYPE_PREVENTIF]
    jours = lambda n: pd.Timedelta(days=n)

    res = pd.DataFrame(index=pd.Index(v[EQ].unique(), name=EQ))
    res["anciennete_jours"] = (d - v.groupby(EQ)["date_connue"].min()).dt.days

    # Historique des pannes : nombre de correctifs sur 7, 30, 90, 365 jours, et depuis le début
    for w in FENETRES:
        res[f"n_corr_{w}j"] = corr[corr[R] > d - jours(w)].groupby(EQ).size()
    res["n_corr_total"] = corr.groupby(EQ).size()

    # Maintenance préventive
    res["n_prev_365j"] = prev[prev[R] > d - jours(365)].groupby(EQ).size()
    res["n_prev_total"] = prev.groupby(EQ).size()

    # Récence : jours depuis la dernière panne et le dernier préventif
    res["jours_depuis_corr"] = (d - corr.groupby(EQ)[R].max()).dt.days
    res["jours_depuis_prev"] = (d - prev.groupby(EQ)[R].max()).dt.days

    # Rythme des pannes : écart moyen (en jours) entre deux pannes connues
    gc = corr.groupby(EQ)[R]
    n = gc.count()
    res["ecart_moyen_corr"] = ((gc.max() - gc.min()).dt.days / (n - 1)).where(n >= 2)

    # Gravité : coût total des pannes sur 365 jours, en log(1 + coût)
    cout = corr[corr[R] > d - jours(365)].groupby(EQ)["coutMaintenance"].sum().clip(lower=0)
    res["log_cout_365j"] = np.log1p(cout)

    # Descriptions de l'équipement : dernière valeur connue le jour d
    for col in FEATURES_CAT + ["Equipement_Parent"]:
        res[col] = v.dropna(subset=[col]).groupby(EQ)[col].last()

    # Pannes de l'équipement parent sur 90 jours (0 si pas de parent, ou parent absent du fichier)
    comptes = [f"n_corr_{w}j" for w in FENETRES] + ["n_corr_total", "n_prev_365j", "n_prev_total"]
    res[comptes] = res[comptes].fillna(0).astype(int)
    res["parent_corr_90j"] = res["Equipement_Parent"].map(res["n_corr_90j"]).fillna(0).astype(int)

    # Valeurs quand l'intervention n'a jamais eu lieu
    res[["jours_depuis_corr", "jours_depuis_prev", "ecart_moyen_corr"]] = \
        res[["jours_depuis_corr", "jours_depuis_prev", "ecart_moyen_corr"]].fillna(JOURS_JAMAIS)
    res["log_cout_365j"] = res["log_cout_365j"].fillna(0)
    res["mois"] = d.month

    res = res.drop(columns="Equipement_Parent").reset_index()
    res.insert(1, "date_coupure", d)
    return res


def dates_coupure(debut, fin):
    """Derniers jours de chaque mois entre debut et fin, sans le trou d'extraction."""
    dates = pd.date_range(debut, fin, freq="ME")
    return dates[(dates < TROU_DEBUT) | (dates > TROU_FIN)]


def build_snapshots(ot, debut, fin):
    """Variables à chaque date de coupure entre debut et fin (une ligne par équipement et par date)."""
    snap = pd.concat([variables_a_date(ot, d) for d in dates_coupure(debut, fin)], ignore_index=True)
    for col in FEATURES_CAT:                                  # mêmes catégories pour tous les ensembles
        snap[col] = pd.Categorical(snap[col], categories=sorted(ot[col].dropna().unique()))
    snap["hors_perimetre"] = est_exclu(snap["familleEquipement"])
    return snap


# ----------------------------------------------------------------------------
# 2. Cible de survie
# ----------------------------------------------------------------------------
def ajouter_cible(snap, ot, fin_obs, h_max=H_MAX):
    """Ajoute duree (jours jusqu'à la prochaine panne ou fin du suivi) et evenement (1 = panne observée).
    Le suivi d'une date d s'arrête au plus tôt de : d + h_max, fin_obs, veille du trou d'extraction."""
    fin_obs = pd.Timestamp(fin_obs)
    pannes = ot.loc[ot["typeIntervention"] == TYPE_PANNE, [EQ, R]]
    morceaux = []
    for d, g in snap.groupby("date_coupure"):
        limite = min(d + pd.Timedelta(days=h_max), fin_obs)
        if d < TROU_DEBUT:
            limite = min(limite, TROU_DEBUT - pd.Timedelta(days=1))
        p = pannes[(pannes[R] > d) & (pannes[R] <= limite)]
        prochaine = g[EQ].map(p.groupby(EQ)[R].min())
        g = g.copy()
        g[EVENEMENT] = prochaine.notna().astype(int)
        g[DUREE] = np.where(prochaine.notna(), (prochaine - d).dt.days, (limite - d).days).astype(int)
        morceaux.append(g)
    out = pd.concat(morceaux, ignore_index=True)
    return out[out[DUREE] > 0].reset_index(drop=True)


def etiquette_horizon(df, h):
    """Panne dans les h jours ? 1 = oui ; 0 = non (suivi d'au moins h jours sans panne) ;
    NaN = on ne sait pas (suivi arrêté avant h jours sans panne)."""
    oui = (df[EVENEMENT] == 1) & (df[DUREE] <= h)
    non = ~oui & (df[DUREE] >= h)
    return pd.Series(np.where(oui, 1.0, np.where(non, 0.0, np.nan)), index=df.index)


# ----------------------------------------------------------------------------
# 3. Découpage temporel
# ----------------------------------------------------------------------------
def construire_ensembles(ot):
    """Entraînement, validation, test (avec cible) et futur (dernière date, sans cible)."""
    ensembles = {}
    for nom, (debut, fin), fin_obs in [("train", COUPURES_TRAIN, FIN_OBS_TRAIN),
                                       ("val", COUPURES_VAL, FIN_OBS_VAL),
                                       ("test", COUPURES_TEST, FIN_OBS_TEST)]:
        ensembles[nom] = ajouter_cible(build_snapshots(ot, debut, fin), ot, fin_obs)
        logger.info(f"{nom} : {len(ensembles[nom]):,} lignes, {ensembles[nom]['date_coupure'].nunique()} dates de coupure")
    ensembles["futur"] = build_snapshots(ot, DATE_FUTUR, DATE_FUTUR)
    logger.info(f"futur : {len(ensembles['futur']):,} équipements au {DATE_FUTUR}")
    return ensembles


# ----------------------------------------------------------------------------
# 4. Courbe de survie de Kaplan-Meier
# ----------------------------------------------------------------------------
def kaplan_meier(duree, evenement):
    """Probabilité d'être encore sans panne après t jours, en tenant compte de la censure.
    À chaque jour t : survie(t) = survie(t-1) x (1 - pannes ce jour / équipements encore suivis)."""
    duree, evenement = np.asarray(duree), np.asarray(evenement)
    jours, idx = np.unique(duree, return_inverse=True)
    pannes = np.bincount(idx, weights=evenement)
    a_risque = np.bincount(idx)[::-1].cumsum()[::-1]
    survie = np.cumprod(1 - pannes / a_risque)
    return pd.DataFrame({"jour": jours, "survie": survie, "a_risque": a_risque, "pannes": pannes.astype(int)})


def proba_panne(km, h):
    """Probabilité de panne dans les h jours = 1 - survie(h)."""
    s = km.loc[km["jour"] <= h, "survie"]
    return 1 - (s.iloc[-1] if len(s) else 1.0)


def delai_median(km):
    """Jour où la moitié des équipements a eu une panne (NaN si jamais atteint dans le suivi)."""
    j = km.loc[km["survie"] <= 0.5, "jour"]
    return j.iloc[0] if len(j) else np.nan


# ----------------------------------------------------------------------------
# 5. Lien entre variables et règle de redondance
# ----------------------------------------------------------------------------
def cramers_v(x, y):
    """V de Cramér : force du lien entre deux variables catégorielles, de 0 (aucun) à 1 (total)."""
    t = pd.crosstab(x, y)
    chi2 = chi2_contingency(t, correction=False)[0]
    return np.sqrt(chi2 / (t.values.sum() * (min(t.shape) - 1)))


def correlation_cible(df, variables, cible):
    """Corrélation de Spearman de chaque variable avec la cible, triée par valeur absolue."""
    rangs = df[variables + [cible]].rank()
    c = rangs.corr()[cible].drop(cible)
    return c.reindex(c.abs().sort_values(ascending=False).index)


def paires_redondantes(matrice, seuil):
    """Paires de variables dont la corrélation (en valeur absolue) dépasse le seuil."""
    m = matrice.abs()
    haut = m.where(np.triu(np.ones(m.shape, dtype=bool), 1)).stack()
    p = haut[haut > seuil].sort_values(ascending=False).reset_index()
    p.columns = ["variable_1", "variable_2", "correlation"]
    p["correlation"] = [matrice.loc[a, b] for a, b in zip(p["variable_1"], p["variable_2"])]
    return p


def regle_redondance(paires, corr_cible):
    """Pour chaque paire redondante (de la plus corrélée à la moins), on garde la variable la plus liée
    à la cible et on retire l'autre. Une paire dont une variable est déjà retirée n'est plus traitée.
    Renvoie (tableau des décisions, liste des variables retirées)."""
    retirees, lignes = set(), []
    for _, r in paires.iterrows():
        a, b = r["variable_1"], r["variable_2"]
        if a in retirees or b in retirees:
            lignes.append({**r, "gardée": "-", "retirée": "-", "remarque": "déjà traitée"})
            continue
        garde, retire = (a, b) if abs(corr_cible[a]) >= abs(corr_cible[b]) else (b, a)
        retirees.add(retire)
        lignes.append({**r, "gardée": garde, "retirée": retire,
                       "remarque": f"|corr. avec la cible| {abs(corr_cible[garde]):.3f} contre {abs(corr_cible[retire]):.3f}"})
    return pd.DataFrame(lignes), sorted(retirees)
