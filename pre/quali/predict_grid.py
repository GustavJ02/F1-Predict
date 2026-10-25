"""Predict a qualifying session before it starts: pole probabilities and expected grid.

Model: ridge regression of each driver's quali gap to pole (%) on season form, practice pace and (sprint
weekends) sprint-quali / sprint pace, trained on every earlier weekend we have data for. Then a Monte Carlo:
each driver's gap = prediction + a shared team shock + own noise (both sized from the training residuals), and
the order of the simulated gaps gives pole probability, P(top 3) and the expected position.

Only columns that exist for the target weekend are used (a normal weekend has no sq_gap, the first race of a year
no form), and the model is trained only on earlier weekends that have the same columns.

Usage (from the repo root):
    python -m pre.quali.predict_grid 2026 Austin
    python -m pre.quali.predict_grid 2026 Austin --columns fp_gap fp_soft_gap      # restrict the inputs
"""
import argparse
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from f1live.paths import DATA
from pre.history import available
from pre.quali.features import FEATURES, Tables, fill, weekend_features

RIDGE = 1.0
MIN_TRAIN = 5            # earlier weekends needed to fit; optional columns are dropped until there are enough
DROP_ORDER = ["sprint_gap", "sq_gap", "fp_soft_gap", "form_team", "form_driver", "fp_gap"]
N_SIMS = 20_000


@dataclass
class Fit:
    columns: list
    coef: pd.Series
    intercept: float
    sigma_team: float
    sigma_driver: float
    n_train: int


def _present(df, columns):
    return [c for c in columns if df[c].notna().any()]


def training_set(tables, target, columns):
    """Earlier weekends (same year or before) with the target's columns, as filled frames."""
    av = tables.av
    quali = av[(av["session"] == "quali") & (av["start"] < target["start"].iloc[0])]
    frames = []
    for r in quali.itertuples():
        df = weekend_features(tables, r.year, r.race)
        if "y" in df and len(_present(df, columns)) == len(columns):
            frames.append(fill(df, columns))
    return frames


def fit_model(tables, target, columns=None):
    columns = _present(target, FEATURES if columns is None else columns)
    while True:
        frames = training_set(tables, target, columns)
        if len(frames) >= MIN_TRAIN or not columns:
            break
        drop = next((c for c in DROP_ORDER if c in columns), None)
        if drop is None:
            break
        columns = [c for c in columns if c != drop]
    if len(frames) < MIN_TRAIN:
        return None
    X = np.vstack([np.c_[np.ones(len(f)), f[columns].to_numpy()] for f in frames]) if columns else None
    y = np.concatenate([f["y"].to_numpy() for f in frames])
    if columns:
        pen = RIDGE * np.eye(X.shape[1]); pen[0, 0] = 0
        beta = np.linalg.solve(X.T @ X + pen, X.T @ y)
        intercept, coef = beta[0], pd.Series(beta[1:], index=columns)
        resid = [f["y"].to_numpy() - np.c_[np.ones(len(f)), f[columns].to_numpy()] @ beta for f in frames]
    else:
        intercept, coef = y.mean(), pd.Series(dtype=float)
        resid = [f["y"].to_numpy() - intercept for f in frames]
    # team shock vs own noise from teammates' residual covariance, on the faster half (the pole contenders)
    cov, var = [], []
    for f, r in zip(frames, resid):
        r = pd.Series(r, index=f.index)
        fast = f["y"] <= f["y"].median()
        var += list(r[fast] ** 2)
        for _, g in r[fast].groupby(f["team"][fast]):
            if len(g) == 2:
                cov.append(g.iloc[0] * g.iloc[1])
    s_team = np.sqrt(max(np.mean(cov), 0)) if cov else 0.0
    s_driver = np.sqrt(max(np.mean(var) - s_team ** 2, 1e-4))
    return Fit(columns, coef, intercept, s_team, s_driver, len(frames))


def predict(fit, target):
    f = fill(target, fit.columns)
    pred = fit.intercept + (f[fit.columns] @ fit.coef if fit.columns else 0.0)
    return pd.Series(pred, index=target.index)


def simulate(pred, teams, fit, n_sims=N_SIMS, seed=0):
    """-> DataFrame per driver: gap (predicted %), pole, top3, position (expected)."""
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(teams)
    shock = fit.sigma_team * rng.standard_normal((n_sims, len(uniq)))[:, codes]
    draw = pred.to_numpy() + shock + fit.sigma_driver * rng.standard_normal((n_sims, len(pred)))
    pos = draw.argsort(axis=1).argsort(axis=1) + 1
    return pd.DataFrame({"gap": pred, "pole": (pos == 1).mean(axis=0), "top3": (pos <= 3).mean(axis=0),
                         "position": pos.mean(axis=0)}, index=pred.index)


def predict_weekend(tables, year, race, columns=None, with_target=False):
    target = weekend_features(tables, year, race, with_target=with_target)
    fit = fit_model(tables, target, columns)
    if fit is None:
        return None, None, target
    return fit, simulate(predict(fit, target), target["team"], fit), target


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("year", type=int)
    ap.add_argument("race", help="race folder, e.g. Austin or 'United States'")
    ap.add_argument("--columns", nargs="+", choices=FEATURES, help="restrict the inputs (default: all available)")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--data-dir", default=DATA, type=type(DATA))
    a = ap.parse_args()
    tables = Tables(available(a.data_dir))
    fit, res, target = predict_weekend(tables, a.year, a.race, a.columns)
    if fit is None:
        raise SystemExit("not enough earlier weekends to fit a model")
    print(f"{a.year} {a.race}: trained on {fit.n_train} weekends, inputs: {', '.join(fit.columns) or '-'}")
    print("coefficients: " + "  ".join(f"{c} {v:+.2f}" for c, v in fit.coef.items()))
    print(f"noise: team {fit.sigma_team:.2f}%  driver {fit.sigma_driver:.2f}%\n")
    out = res.assign(team=target["team"]).sort_values("position").head(a.top)
    out["pole"], out["top3"] = out["pole"].map("{:.1%}".format), out["top3"].map("{:.1%}".format)
    print(out[["team", "gap", "position", "top3", "pole"]].round(2).to_string())
