"""Score the fantasy simulation on finished races: expected driver points (with the real grid known) against the points the drivers
really scored, from the recorded data (finish, lap-counted overtakes, quickest lap, not-classified). Driver of the Day is left out on
both sides, nothing says how fans voted.

    python -m fantasy.backtest 6            # the last 6 races of the newest season
    python -m fantasy.backtest 10 --year 2025
"""
import argparse
import warnings

import numpy as np
import pandas as pd

from f1live.events import load
from f1live.paths import DATA
from fantasy import overtakes as OT
from fantasy import scenarios as SC
from fantasy import scoring as S
from pre.history import available, pace_table


def realised(path, grid):
    """Series tla -> points a driver really scored in a race (qualifying + race, without Driver of the Day), given the grid."""
    ev = load(path)
    laps = ev[(ev["event"] == "lap") & ev["tla"].notna()]
    done = laps.groupby("tla")["lap_number"].max()
    final = pd.to_numeric(ev[ev["driver"].notna()].groupby("tla").tail(1).set_index("tla")["position"], errors="coerce")
    made = OT.count_by_lap(ev, grid)
    fastest = laps[laps["lap_kind"] == "racing"].groupby("tla")["lap_time_s"].min().idxmin()
    out = {}
    for d in grid.index:
        if d in final.index and pd.notna(final[d]):
            classified = done.get(d, 0) >= OT.NOT_CLASSIFIED * done.max()
            out[d] = S.quali_points(int(grid[d])) + S.race_points(int(grid[d]), int(final[d]), classified, int(made.get(d, 0)), d == fastest, False)
    return pd.Series(out)


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("races", type=int, nargs="?", default=6)
    ap.add_argument("--year", type=int)
    ap.add_argument("--runs", type=int, default=300)
    ap.add_argument("--data-dir", default=DATA, type=type(DATA))
    a = ap.parse_args()
    av = available(a.data_dir)
    races = av[av["session"] == "race"]
    races = races[races["year"] == (a.year or races["year"].max())].sort_values("start").tail(a.races)
    rows = []
    for r in races.itertuples():
        quali = av[(av["year"] == r.year) & (av["race"] == r.race) & (av["session"] == "quali")]
        if quali.empty:
            continue
        grid = pace_table(quali["path"].iloc[0]).reset_index(drop=True).set_index("tla")["rank"]
        sc = SC.simulate_weekend(r.year, r.race, grids=1, runs=a.runs, seed=1, data_dir=a.data_dir)
        expected = pd.Series(sc.driver_points.mean(0) - S.DRIVER_OF_THE_DAY * sc.parts["dotd"].mean(0), index=sc.drivers)
        df = pd.DataFrame(dict(exp=expected, real=realised(r.path, grid))).dropna().assign(race=r.race)
        rows.append(df)
        print(f"{r.race:16s} drivers {len(df):2d}  corr {np.corrcoef(df['exp'], df['real'])[0, 1]:.2f}  "
              f"mean expected {df['exp'].mean():5.1f}  mean real {df['real'].mean():5.1f}  MAE {(df['exp'] - df['real']).abs().mean():4.1f}")
    d = pd.concat(rows)
    print(f"\nall {len(d)} driver-weekends: corr {np.corrcoef(d['exp'], d['real'])[0, 1]:.2f}, bias {(d['exp'] - d['real']).mean():+.2f}, "
          f"MAE {(d['exp'] - d['real']).abs().mean():.1f} (guessing the average: {(d['real'] - d['real'].mean()).abs().mean():.1f})")
    d["bucket"] = pd.qcut(d["exp"], 5, duplicates="drop")
    print(d.groupby("bucket", observed=True).agg(expected=("exp", "mean"), real=("real", "mean"), n=("exp", "size")).round(1).to_string())


if __name__ == "__main__":
    main()
