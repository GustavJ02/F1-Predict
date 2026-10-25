"""Backtest pre.race.predict_result: predict every recorded race from only what was known before it.

Each race gets priors fitted without it (leave one race out; other years of the circuit are kept) and a pace model fitted on
the races before it, then its finishing distribution is scored against the real result. Baselines: finishing in grid order.

Usage (from the repo root):
    python -m pre.race.backtest
    python -m pre.race.backtest --sims 1000 --csv backtest_race.csv
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from f1live.events import load
from f1live.paths import DATA
from live.race import priors as P
from live.race.pace import lap_rows
from pre.history import available
from pre.race.pace import RaceArchive, actual_order
from pre.race.predict_result import predict_weekend, quali_table

MIN_EARLIER = 4         # races before this one needed to score it


def score(out, actual, grid):
    """Metrics of one race: ranks among the drivers in both the prediction and the result."""
    d = [t for t in out["tla"] if t in actual.index]
    out = out[out["tla"].isin(d)].copy()
    act = actual.loc[d].rank(method="first")
    pred_rank = out.set_index("tla")["mean_pos"].rank(method="first")
    grid_rank = pd.Series({t: grid[t] for t in d}).rank(method="first")
    n = len(d)
    res = {}
    for name, rank in (("simulation", pred_rank.loc[d]), ("grid order", grid_rank.loc[d])):
        res[name] = dict(spearman=float(rank.corr(act, method="spearman")), mae=float((rank - act).abs().mean()),
                         winner=bool(rank.idxmin() == act.idxmin()),
                         podium=len(set(rank.nsmallest(3).index) & set(act.nsmallest(3).index)) / 3,
                         top10=len(set(rank.nsmallest(10).index) & set(act.nsmallest(10).index)) / 10)
    o = out.set_index("tla")
    res["simulation"]["p_winner"] = float(o.loc[act.idxmin(), "p_win"])
    res["simulation"]["podium_brier"] = float(((o["p_podium"] - o.index.isin(act.nsmallest(3).index)) ** 2).mean())
    res["simulation"]["points_brier"] = float(((o["p_points"] - o.index.isin(act.nsmallest(10).index)) ** 2).mean())
    cols = [f"pos_{k}" for k in range(1, 23) if f"pos_{k}" in o.columns]
    probs = np.array([o.loc[t, cols[int(act[t]) - 1]] if int(act[t]) <= len(cols) else 0.0 for t in d])
    res["simulation"]["log_pos"] = float(np.mean(np.log(np.clip(probs, 1e-3, 1))))
    res["samples"] = pd.DataFrame(dict(p_win=o["p_win"], win=(act.rank(method="first") == 1).astype(float).reindex(o.index),
                                       p_podium=o["p_podium"], podium=o.index.isin(act.nsmallest(3).index).astype(float),
                                       p_points=o["p_points"], points=o.index.isin(act.nsmallest(10).index).astype(float)))
    return res


def run(collected, data_dir=DATA, sims=600, use_rating=True):
    av = available(data_dir)
    races = av[av["session"] == "race"].sort_values("start").reset_index(drop=True)
    rows, samples = [], []
    priors_all = P.load_priors()
    archive = RaceArchive(races, priors_all)
    for i, r in enumerate(races.itertuples()):
        if i < MIN_EARLIER or quali_table(av, r.year, r.race) is None:
            continue
        key = next((k for k in collected if k[0] == str(r.year) and P.circuit_of(*k) == P.circuit_of(str(r.year), r.race)
                    and race_match(k, r)), None)
        pri = P.fit(collected, exclude=[key] if key else [])
        archive.priors = priors_all                      # race-pace tables don't depend on the held-out race
        out, info = predict_weekend(archive, av, pri, r.year, r.race, sims=sims, use_rating=use_rating, before=r.start)
        if out is None:
            continue
        actual = archive.result(r.path)
        q = quali_table(av, r.year, r.race)
        sc = score(out, actual, q["rank"].to_dict())
        samples.append(sc.pop("samples").assign(race=f"{r.year}/{r.race}"))
        for name, m in sc.items():
            rows.append(dict(race=f"{r.year}/{r.race}", model=name, **m))
    return pd.DataFrame(rows), pd.concat(samples)


def race_match(k, r):
    return k[1] == r.race or P.circuit_of(*k) == P.circuit_of(str(r.year), r.race) and k[0] == str(r.year)


def summarize(df, samples):
    print(f"{df['race'].nunique()} races\n")
    t = df.groupby("model", sort=False).agg(spearman=("spearman", "mean"), mae=("mae", "mean"), winner=("winner", "mean"),
                                           podium=("podium", "mean"), top10=("top10", "mean")).round(3)
    print(t.to_string())
    sim = df[df["model"] == "simulation"]
    print(f"\nsimulation: P(actual winner) {sim['p_winner'].mean():.3f}, podium Brier {sim['podium_brier'].mean():.3f}, "
          f"points Brier {sim['points_brier'].mean():.3f}, mean log P(actual position) {sim['log_pos'].mean():.2f}")
    for col, y, bins in (("p_win", "win", [-.01, .02, .1, .3, .5, 1.0]), ("p_podium", "podium", [-.01, .05, .2, .4, .6, .8, 1.0]),
                         ("p_points", "points", [-.01, .2, .5, .8, .95, 1.0])):
        b = pd.cut(samples[col], bins)
        print(f"\n{col} calibration")
        print(samples.groupby(b, observed=True).agg(n=(col, "size"), predicted=(col, "mean"), actual=(y, "mean")).round(3).to_string())


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=600)
    ap.add_argument("--no-rating", action="store_true", help="pace from qualifying only (ablation)")
    ap.add_argument("--csv")
    ap.add_argument("--data-dir", default=DATA, type=Path)
    a = ap.parse_args()
    df, samples = run(P.collect(a.data_dir), a.data_dir, a.sims, not a.no_rating)
    if a.csv:
        df.to_csv(a.csv, index=False)
    summarize(df, samples)
