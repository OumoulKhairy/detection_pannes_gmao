"""Graphiques réutilisables (Plotly).

Les boxplots sont construits à partir de statistiques déjà calculées (quartiles, moustaches) :
on peut ainsi les tracer sur plus d'un million de lignes sans alourdir le notebook.
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def stats_box(x):
    """Quartiles et moustaches (1,5 x écart interquartile) d'une série."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    iqr = q3 - q1
    bas = x[x >= q1 - 1.5 * iqr].min()
    haut = x[x <= q3 + 1.5 * iqr].max()
    return dict(q1=[q1], median=[med], q3=[q3], lowerfence=[bas], upperfence=[haut], mean=[x.mean()])


def boxplots_par_classe(df, variables, classe, noms_classes, couleurs, titre, ncols=3, log=False):
    """Un boxplot par variable, une boîte par classe (par exemple panne / pas de panne).
    log=True : on trace log(1 + valeur), utile pour les comptages et les coûts très asymétriques."""
    nrows = int(np.ceil(len(variables) / ncols))
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=variables)
    for i, v in enumerate(variables):
        r, c = i // ncols + 1, i % ncols + 1
        for k, nom in noms_classes.items():
            x = df.loc[df[classe] == k, v].to_numpy(float)
            if log:
                x = np.log1p(np.clip(x, 0, None))
            fig.add_trace(go.Box(name=nom, x=[nom], marker_color=couleurs[k], showlegend=(i == 0),
                                 legendgroup=nom, **stats_box(x)), row=r, col=c)
    fig.update_layout(title=titre + (" (échelle log(1 + valeur))" if log else ""),
                      height=320 * nrows, template="plotly_white")
    return fig


def courbes_survie(df, groupes, titre, ordre=None, horizons=(7, 30, 90), duree="duree", evenement="evenement"):
    """Une courbe "probabilité de panne dans les h jours" (Kaplan-Meier) par groupe d'équipements,
    et un tableau résumé : probabilité de panne à quelques horizons et délai médian.
    Renvoie (figure, tableau)."""
    import pandas as pd
    import plotly.express as px
    from src.features.build import kaplan_meier, proba_panne, delai_median
    ordre = list(ordre) if ordre is not None else sorted(pd.unique(pd.Series(groupes).dropna()))
    couleurs = px.colors.sequential.Reds[2:] if len(ordre) <= 7 else px.colors.qualitative.Plotly
    fig, lignes = go.Figure(), []
    for i, g in enumerate(ordre):
        m = (np.asarray(groupes) == g)
        if m.sum() == 0:
            continue
        k = kaplan_meier(df.loc[m, duree], df.loc[m, evenement])
        fig.add_trace(go.Scatter(x=k["jour"], y=1 - k["survie"], name=str(g), line_color=couleurs[i % len(couleurs)]))
        lignes.append({"groupe": g, "lignes": int(m.sum()),
                       **{f"P(panne ≤ {h} j)": round(proba_panne(k, h), 3) for h in horizons},
                       "délai médian (jours)": delai_median(k)})
    fig.update_layout(title=titre, template="plotly_white", height=420, yaxis_tickformat=".0%",
                      xaxis_title="horizon h (jours)", yaxis_title="probabilité de panne dans les h jours")
    return fig, pd.DataFrame(lignes)
