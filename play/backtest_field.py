"""Does projecting the race with every driver on his optimal strategy predict it better than the usual stop plans?

At 25%, 50% and 75% of each race the projection is made twice from the same state, with the same random draws:
  usual    every car follows the usual stochastic stop plans (what live.race.projection does)
  optimal  every car follows the plan found by play.field.optimal_field (best response given the others, tyre sets applied)
and both are scored against the real finishing order. Priors are fitted without the race being tested.

Usage (from the repo root; slow: one field search per race and distance):
    python -m play.backtest_field --year 2026
    python -m play.backtest_field --year 2026 --fractions 0.5 --passes 2 --csv backtest_field.csv
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from f1live.events import has_events, load
from f1live.paths import DATA, Session
from live.race import priors as P
from live.race.backtest_projection import final_order, score
from play.field import optimal_field
from play.scenario import Scenario


def finished(path):
    """A race whose recording reached the last lap (one still being recorded has no result to score against)."""
    h = load(path)
    return bool(h["race_lap"].max() >= h["total_laps"].max() - 1)


def evaluate(year, race, collected, fraction, passes, search_sims, sims, data_dir, jobs=6):
    path = Path(data_dir) / year / race / "race" / "events.csv"
    h = load(path)
    total = int(h["total_laps"].max())
    lap = int(round(fraction * total))
    up = h[h["race_lap"] <= lap]
    pri = P.fit(collected, exclude=[(year, race)])
    sc = Scenario.live(Session(year, race, "race", Path(data_dir)), history=up, sims=search_sims, priors=pri)
    plans, log = optimal_field(sc, passes=passes, sims=search_sims, verbose=False, jobs=jobs)
    sc_final = Scenario(sc.cars, sc.meta, sc.track, sc.priors, sc.circuit, None, sc.opt, sims, sc.seed, sc.label, tyres=sc.tyres)
    actual = final_order(h)
    drivers = list(sc.cars["driver"])
    tla_of = dict(zip(sc.cars["tla"], sc.cars["driver"]))
    rows = []
    for name, run in (("usual", sc_final.run()), ("optimal", sc_final.run(plans=plans))):
        tab = run.table()
        rank = {tla_of[t]: r for t, r in zip(tab.index, range(1, len(tab) + 1))}
        s = score(rank, actual, drivers)
        win = actual.idxmin()
        by_driver = tab.rename(index=tla_of)
        s["p_winner"] = float(by_driver["p_win"].get(win, 0.0))
        podium = set(actual.loc[[d for d in drivers if d in actual.index]].nsmallest(3).index)
        s["podium_brier"] = float(np.mean([(by_driver.loc[d, "p_podium"] - (d in podium)) ** 2 for d in drivers]))
        top10 = set(actual.loc[[d for d in drivers if d in actual.index]].nsmallest(10).index)
        s["points_brier"] = float(np.mean([(by_driver.loc[d, "p_points"] - (d in top10)) ** 2 for d in drivers]))
        rows.append(dict(race=f"{year}/{race}", frac=fraction, model=name, changed_last_pass=len(log[-1]) if log else 0, **s))
    return rows


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", default="2026")
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--search-sims", type=int, default=250)
    ap.add_argument("--sims", type=int, default=1500)
    ap.add_argument("--limit", type=int, help="only the first N races")
    ap.add_argument("--jobs", type=int, default=6, help="processes per search (keep it low if something else needs the CPU)")
    ap.add_argument("--csv")
    ap.add_argument("--data-dir", default=DATA, type=Path)
    a = ap.parse_args()
    collected = P.collect(a.data_dir)
    races = sorted(p.parent.parent.name for p in (a.data_dir / a.year).glob("*/race/events.csv") if has_events(p) and finished(p))
    rows = []
    for race in races[: a.limit]:
        for frac in a.fractions:
            try:
                rows += evaluate(a.year, race, collected, frac, a.passes, a.search_sims, a.sims, a.data_dir, a.jobs)
            except Exception as e:                                   # a race with too little data at that point
                print(f"  skipped {race} at {frac:.0%}: {type(e).__name__}: {e}", flush=True)
                continue
            u, o = rows[-2], rows[-1]
            print(f"{race:12s} {frac:.0%}  rank error usual {u['mae']:.2f} optimal {o['mae']:.2f} | P(winner) {u['p_winner']:.2f} / {o['p_winner']:.2f}", flush=True)
    df = pd.DataFrame(rows)
    if a.csv:
        df.to_csv(a.csv, index=False)
    g = df.groupby(["frac", "model"]).agg(n=("mae", "size"), spearman=("spearman", "mean"), mae=("mae", "mean"), winner=("winner", "mean"),
                                           podium=("podium", "mean"), top10=("top10", "mean"), p_winner=("p_winner", "mean"),
                                           podium_brier=("podium_brier", "mean"), points_brier=("points_brier", "mean")).round(3)
    print("\n" + g.to_string())
    print("\nall distances:\n" + df.groupby("model").agg(n=("mae", "size"), spearman=("spearman", "mean"), mae=("mae", "mean"),
          winner=("winner", "mean"), podium=("podium", "mean"), top10=("top10", "mean"), p_winner=("p_winner", "mean"),
          podium_brier=("podium_brier", "mean"), points_brier=("points_brier", "mean")).round(3).to_string())


if __name__ == "__main__":
    main()
