"""Pace of each qualifying part (Q1, Q2, Q3), predicted before the part starts.

For part k the inputs are everything known when it starts: season form, this weekend's practice / sprint pace
(pre.quali.features) and the pace in the parts before it (Q1 for Q2, Q1 and Q2 for Q3), each for the driver and for the
team's better car. The target is the driver's best valid lap in the part, as a gap to the part's fastest lap (%).
One ridge per part, trained only on earlier weekends (pre.quali.predict_grid.fit_model).

Columns (on top of pre.quali.features.FEATURES):
  fp_team_gap     the team's better car in practice
  q1_gap, q2_gap  gap in an earlier part (%); q1_team_gap / q2_team_gap: the team's better car in it

Validate against simple baselines on every earlier weekend we have:
    python -m pre.quali.parts
"""
import argparse
import warnings

import numpy as np
import pandas as pd
from f1live.paths import DATA
from pre.history import available, race_key, valid_laps
from pre.quali import predict_grid
from pre.quali.features import FEATURES, Tables, weekend_features

PARTS = (1, 2, 3)
GAP_CAP = 10.0     # %: a wet part or a spoiled lap gives gaps of 30%+ that a squared-loss fit would chase
DROP_ORDER = ["sprint_gap", "sq_gap", "fp_soft_gap", "fp_team_gap", "form_team", "form_driver", "fp_gap",
              "q1_team_gap", "q2_team_gap", "q1_gap", "q2_gap"]


def part_features_for(part):
    cols = FEATURES + ["fp_team_gap"]
    for j in range(1, part):
        cols += [f"q{j}_gap", f"q{j}_team_gap"]
    return cols


def tables_from_laps(laps):
    """{part: DataFrame indexed by tla (team, best_s, gap_pct)} from valid_laps() output of a qualifying session."""
    out = {}
    laps = laps[laps["session_part"].notna()]
    for part, g in laps.groupby(laps["session_part"].astype(int)):
        t = g.groupby("tla").agg(team=("team", "last"), best_s=("lap_time_s", "min"))
        t["gap_pct"] = (t["best_s"] / t["best_s"].min() - 1) * 100
        out[part] = t
    return out


def part_tables(tables, path):
    """Cached tables_from_laps for one events.csv (a finished session)."""
    cache = tables.__dict__.setdefault("_parts", {})
    if path not in cache:
        cache[path] = tables_from_laps(valid_laps(path))
    return cache[path]


def part_features(tables, year, race, part, drivers=None, earlier=None):
    """One row per driver of `part` (index = tla). Without `drivers`/`earlier` this is a finished weekend: the rows are
    the drivers with a time in the part, earlier parts come from the quali's events.csv and `y` is the part's gap.
    Live, pass `drivers` (DataFrame indexed by tla with a `team` column) and `earlier` ({part: table} so far)."""
    live = drivers is not None
    av = tables.av
    q = av[(av["year"] == year) & (av["race"] == race_key(race)) & (av["session"] == "quali")]
    parts = earlier if live else (part_tables(tables, q["path"].iloc[0]) if len(q) else {})
    if not live:
        if part not in parts:
            return pd.DataFrame()
        drivers = parts[part][["team"]]
    df = weekend_features(tables, year, race, with_target=False, drivers=drivers)
    df["fp_team_gap"] = df.groupby("team")["fp_gap"].transform("min")
    for j in range(1, part):
        t = parts.get(j)
        df[f"q{j}_gap"] = t["gap_pct"].reindex(df.index) if t is not None else np.nan
        df[f"q{j}_team_gap"] = df["team"].map(t.groupby("team")["gap_pct"].min()) if t is not None else np.nan
    if not live:
        df["y"] = parts[part]["gap_pct"]
    gaps = [c for c in df.columns if c.endswith("_gap") or c == "y"]
    df[gaps] = df[gaps].clip(upper=GAP_CAP)
    return df


def fit_part(tables, target, part, columns=None):
    featurize = lambda t, y, r: part_features(t, y, r, part)
    return predict_grid.fit_model(tables, target, columns, featurize, part_features_for(part), DROP_ORDER)


# ---------------------------------------------------------------- validation
def validate(tables):
    quali = tables.av[tables.av["session"] == "quali"].sort_values("start")
    rows = []
    for r in quali.itertuples():
        for part in PARTS:
            target = part_features(tables, r.year, r.race, part)
            if len(target) < 8:
                continue
            y = target["y"]
            variants = {"all inputs": None, "no earlier parts": [c for c in part_features_for(part) if not c.startswith("q")]}
            preds = {}
            for name, cols in variants.items():
                fit = fit_part(tables, target, part, cols)
                if fit is not None:
                    preds[name] = predict_grid.predict(fit, target)
            if "all inputs" not in preds:
                continue
            preds["field mean"] = pd.Series(y.mean(), index=y.index)       # (optimistic: uses the target's own mean)
            if part > 1:
                preds["previous part"] = fillna(target[f"q{part - 1}_gap"])
            for name, p in preds.items():
                p = p.reindex(y.index)
                rows.append(dict(year=r.year, race=r.race, part=part, model=name, mae=(p - y).abs().mean(),
                                 spearman=p.corr(y, method="spearman"), n=len(y)))
    return pd.DataFrame(rows)


def fillna(s):
    return s.fillna(s.quantile(0.8))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA, type=type(DATA))
    a = ap.parse_args()
    df = validate(Tables(available(a.data_dir)))
    g = df.groupby(["part", "model"], sort=False)
    print(g.agg(weekends=("mae", "size"), mae_pct=("mae", "mean"), spearman=("spearman", "mean")).round(3).to_string())
