"""Backtest pre.quali.predict_grid: predict every qualifying we have from only what was known before it.

For each quali (chronological, once MIN_TRAIN earlier weekends exist) the model is fit on earlier weekends only
and scored against the real result. Variants show which inputs help.

Usage (from the repo root):
    python -m pre.quali.backtest
    python -m pre.quali.backtest --csv backtest_quali.csv
"""
import argparse
import warnings

import numpy as np
import pandas as pd
from f1live.paths import DATA
from pre.history import available
from pre.quali.features import Tables
from pre.quali.predict_grid import predict_weekend

VARIANTS = {
    "none (random)": [],
    "form": ["form_driver", "form_team"],
    "practice": ["fp_gap", "fp_soft_gap", "sq_gap", "sprint_gap"],
    "form+practice": None,          # all available
}


def score(res, y):
    res = res.loc[res.index.intersection(y.index)]
    y = y.loc[res.index]
    actual = y.rank(method="first")
    winner = y.idxmin()
    onehot = (res.index == winner).astype(float)
    pred_top3 = set(res["position"].nsmallest(3).index)
    return dict(pole_hit=res["pole"].idxmax() == winner, p_top1=res["pole"].max(), p_winner=res.loc[winner, "pole"],
                brier=float(((res["pole"].to_numpy() - onehot) ** 2).sum()),
                pos_mae=float((res["position"] - actual).abs().mean()),
                spearman=float(res["position"].corr(actual, method="spearman")),
                top3=len(pred_top3 & set(y.nsmallest(3).index)) / 3)


def run(tables, variants=VARIANTS):
    quali = tables.av[tables.av["session"] == "quali"].sort_values("start")
    rows = []
    for r in quali.itertuples():
        for name, cols in variants.items():
            fit, res, target = predict_weekend(tables, r.year, r.race, cols, with_target=True)
            if fit is None:
                continue
            rows.append(dict(variant=name, year=r.year, race=r.race, n_train=fit.n_train,
                             inputs=",".join(fit.columns), **score(res, target["y"].dropna())))
    return pd.DataFrame(rows)


def summarize(df):
    g = df.groupby("variant", sort=False)
    print(f"{df.groupby('variant').size().max()} weekends scored\n")
    print(g.agg(pole_hit=("pole_hit", "mean"), p_top1=("p_top1", "mean"), p_winner=("p_winner", "mean"), brier=("brier", "mean"),
                pos_mae=("pos_mae", "mean"), spearman=("spearman", "mean"), top3=("top3", "mean")).round(3).to_string())
    print("\nby year:")
    print(df.groupby(["variant", "year"], sort=False).agg(n=("brier", "size"), pole_hit=("pole_hit", "mean"),
          brier=("brier", "mean"), spearman=("spearman", "mean")).round(3).to_string())


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--data-dir", default=DATA, type=type(DATA))
    a = ap.parse_args()
    df = run(Tables(available(a.data_dir)))
    if a.csv:
        df.to_csv(a.csv, index=False)
    summarize(df)
