"""Backtest the battle prediction on recorded races: when it says "in N laps", does the contact really happen?

At every lap L of every race, the model sees only the rows up to that lap. For each adjacent pair it predicts the laps
until the car behind is within THRESH of the car ahead. The outcome comes from the later rows: the first lap on which
the rear car's interval is <= THRESH (or it has passed the front car). A pair is censored, and not scored, if either car
pits, the track is not green, or the interval becomes unknown before either happens.

Usage (from the repo root):
    python -m live.race.backtest_battles                       # every data/*/*/race
    python -m live.race.backtest_battles 2025 --slope
    python -m live.race.backtest_battles --window 8 --min-laps 5
"""
import argparse
import warnings

import numpy as np
import pandas as pd
from f1live.events import has_events, load
from f1live.paths import DATA
from live.race.battles import HORIZON, THRESH, pair_table

LOOKAHEAD = 15
FIRST_L, LAST_MARGIN = 8, 3


def outcome(laps, pair, lap0):
    """-> (laps until contact, True) if it happened, (LOOKAHEAD, False) if it didn't, None if censored."""
    f, r = laps.get(pair["front"]), laps.get(pair["rear"])
    if f is None or r is None:
        return None
    r0 = r[r["race_lap"] <= lap0].tail(1)
    f0 = f[f["race_lap"] <= lap0].tail(1)
    if r0.empty or f0.empty:
        return None
    pos0, stops_r, stops_f = r0["position"].iloc[0], r0["n_pit_stops"].iloc[0], f0["n_pit_stops"].iloc[0]
    n0 = r0["lap_number"].iloc[0]
    for k in range(1, LOOKAHEAD + 1):
        rk = r[r["lap_number"] == n0 + k]
        fk = f[f["lap_number"] == f0["lap_number"].iloc[0] + k]
        if rk.empty or fk.empty:
            return None
        rk, fk = rk.iloc[0], fk.iloc[0]
        if rk["track_status"] != "AllClear" or rk["n_pit_stops"] != stops_r or fk["n_pit_stops"] != stops_f:
            return None
        if rk["position"] < pos0 and rk["laps_down"] == r0["laps_down"].iloc[0]:
            return float(k), True                         # passed someone: contact certainly happened
        if pd.isna(rk["interval_s"]):
            return None
        if rk["interval_s"] <= THRESH and rk["position"] == pos0:
            return float(k), True
    return float(LOOKAHEAD), False


def backtest_race(name, hist, **kw):
    laps = {d: g.sort_values("ts") for d, g in hist[hist["event"] == "lap"].groupby("driver")}
    total = hist["total_laps"].dropna().max()
    rows = []
    for lap0 in range(FIRST_L, int(total) - LAST_MARGIN):
        upto = hist[hist["race_lap"] <= lap0]
        _, pairs = pair_table(upto, **kw)
        if pairs.empty or hist[hist["race_lap"] == lap0]["track_status"].iloc[-1] != "AllClear":
            continue
        for _, p in pairs.iterrows():
            if p["laps"] == 0:
                continue                                  # already in contact: nothing to predict
            res = outcome(laps, p, lap0)
            if res is None:
                continue
            rows.append(dict(race=name, lap=lap0, front=p["front_tla"], rear=p["rear_tla"], gap=p["gap_s"],
                             delta=p["delta_s"], pred=p["laps"], actual=res[0], contact=res[1]))
    return rows


def summarize(df):
    print(f"{df['race'].nunique()} races, {len(df)} resolved pair-laps\n")
    bins = [(0, 2), (2, 4), (4, 7), (7, HORIZON), (HORIZON, np.inf)]
    out = []
    for lo, hi in bins:
        s = df[(df["pred"] > lo) & (df["pred"] <= hi)] if lo else df[(df["pred"] > 0) & (df["pred"] <= hi)]
        if s.empty:
            continue
        within = s["actual"] <= s["pred"] + 2
        hit = s["contact"] & within
        done = s[s["contact"]]
        out.append(dict(predicted="inf (not closing / > horizon)" if hi == np.inf else f"{lo}-{hi} laps", n=len(s),
                        contact_by_pred_plus2=round(hit.mean(), 3), contact_within_15=round(s["contact"].mean(), 3),
                        median_actual=done["actual"].median() if len(done) else np.nan,
                        median_err=round((done["actual"] - done["pred"]).median(), 2) if len(done) else np.nan))
    print(pd.DataFrame(out).to_string(index=False))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("year", nargs="?")
    ap.add_argument("--slope", action="store_true", help="extrapolate each driver's lap time trend")
    ap.add_argument("--window", type=int)
    ap.add_argument("--min-laps", type=int)
    ap.add_argument("--csv")
    ap.add_argument("--data-dir", default=DATA, type=type(DATA))
    a = ap.parse_args()
    kw = {"use_slope": a.slope, **({"window": a.window} if a.window else {}), **({"min_laps": a.min_laps} if a.min_laps else {})}
    rows = []
    for p in sorted(q for q in a.data_dir.glob(f"{a.year or '*'}/*/race/events.csv") if has_events(q)):
        rows += backtest_race(f"{p.parent.parent.parent.name}/{p.parent.parent.name}", load(p), **kw)
    df = pd.DataFrame(rows)
    if a.csv:
        df.to_csv(a.csv, index=False)
    summarize(df)
