"""Backtest the projected finishing order on recorded races.

At 25%, 50% and 75% of each race the model sees only the rows up to that lap, with priors fitted WITHOUT that race
(leave-one-race-out; other years of the same circuit are still used), and is scored against the real result.
Baselines: 'current order' and 'pace only' (current order plus the pace difference over the remaining laps, no stops).

Usage (from the repo root):
    python -m live.race.backtest_projection
    python -m live.race.backtest_projection --sims 800 --csv backtest_projection.csv
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from f1live.events import load
from f1live.paths import DATA
from live.race import priors as P
from live.race.battles import latest
from live.race.pace import driver_pace
from live.race.projection import car_states, project

FRACTIONS = (0.25, 0.5, 0.75)


def final_order(h):
    """{driver: final classified position} from the last row of each driver."""
    last = h[h["driver"].notna()].groupby("driver").tail(1).set_index("driver")["position"]
    return pd.to_numeric(last, errors="coerce").dropna().astype(int)


def score(rank, actual, drivers):
    """rank: {driver: predicted rank}. Compares with the real finishing positions of the same cars."""
    d = [x for x in drivers if x in actual.index and x in rank]
    pred = pd.Series({x: rank[x] for x in d})
    act = actual.loc[d].rank(method="first")
    top3 = set(pred.nsmallest(3).index)
    return dict(spearman=float(pred.corr(act, method="spearman")), mae=float((pred - act).abs().mean()),
                winner=bool(pred.idxmin() == act.idxmin()), podium=len(top3 & set(act.nsmallest(3).index)) / 3,
                top10=len(set(pred.nsmallest(10).index) & set(act.nsmallest(10).index)) / 10)


def run(collected, data_dir=DATA, sims=400):
    rows = []
    for p in sorted(Path(data_dir).glob("*/*/race/events.csv")):
        year, race = p.parent.parent.parent.name, p.parent.parent.name
        key, circuit = (year, race), P.circuit_of(year, race)
        pri = P.fit(collected, exclude=[key])
        h = load(p)
        total = int(h["total_laps"].max())
        actual = final_order(h)
        for frac in FRACTIONS:
            lap = int(round(frac * total))
            up = h[h["race_lap"] <= lap]
            out, info = project(up, pri, circuit, n_sims=sims)
            if out is None:
                continue
            drivers = list(out["driver"])
            cur, pac = baselines_with(up, pri)
            sim_rank = dict(zip(out["driver"], out["proj_rank"]))
            win = actual.idxmin()
            for name, rank in (("simulation", sim_rank), ("current order", cur), ("pace only", pac)):
                s = score(rank, actual, drivers)
                if name == "simulation":
                    s["p_winner"] = float(out.set_index("driver")["p_win"].get(win, 0.0))
                    s["p_podium_brier"] = float(((out.set_index("driver")["p_podium"]
                                                  - out["driver"].isin(actual.nsmallest(3).index).astype(float).values) ** 2).mean())
                rows.append(dict(race=f"{year}/{race}", frac=frac, lap=lap, model=name, **s))
    return pd.DataFrame(rows)


def baselines_with(h, pri):
    cars, meta = car_states(h, pri)
    t_lead, n0, total = meta
    pace = driver_pace(h)["pace_s"]
    lead = cars["driver"].iloc[0]
    rel = (pace - pace.get(lead, np.nan)).reindex(cars["driver"]).fillna(0).to_numpy()
    pace_time = cars["gap"].to_numpy() + (total - n0) * rel
    cur = dict(zip(cars["driver"], cars["position"].rank(method="first")))
    pac = dict(zip(cars["driver"], pd.Series(pace_time, index=cars.index).rank(method="first")))
    return cur, pac


def summarize(df):
    print(f"{df['race'].nunique()} races\n")
    g = df.groupby(["frac", "model"], sort=False)
    t = g.agg(spearman=("spearman", "mean"), mae=("mae", "mean"), winner=("winner", "mean"),
              podium=("podium", "mean"), top10=("top10", "mean")).round(3)
    print(t.to_string())
    sim = df[df["model"] == "simulation"]
    print(f"\nsimulation: mean P(actual winner) {sim['p_winner'].mean():.3f}, podium Brier {sim['p_podium_brier'].mean():.3f}")
    print("\nall laps pooled:")
    print(df.groupby("model", sort=False).agg(spearman=("spearman", "mean"), mae=("mae", "mean"), winner=("winner", "mean"),
                                              podium=("podium", "mean"), top10=("top10", "mean")).round(3).to_string())


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--csv")
    ap.add_argument("--data-dir", default=DATA, type=Path)
    a = ap.parse_args()
    df = run(P.collect(a.data_dir), a.data_dir, a.sims)
    if a.csv:
        df.to_csv(a.csv, index=False)
    summarize(df)
