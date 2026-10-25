"""Per-driver features for predicting a qualifying session from what is known before it.

One row per driver (index = tla) for a weekend, built only from sessions that started before the target quali:
  form_driver  recency-weighted average gap to pole (%) over this year's earlier qualifying sessions
  form_team    same for the team's better car
  fp_gap       best practice gap (%) over the weekend's FP sessions (each normalised to its own fastest lap)
  fp_soft_gap  same using soft-tyre laps only
  sq_gap       sprint-qualifying gap (%), sprint weekends only
  sprint_gap   best-lap gap in the sprint (%), sprint weekends only
  y            (if the quali exists) the target: gap to pole in %, from the best valid lap of the whole session
Missing values (a driver without a time, a column that doesn't exist for the weekend) are NaN; `fill` replaces
them per weekend with the 80th percentile of the drivers that have one.
"""
import numpy as np
import pandas as pd
from pre.history import available, pace_table, race_key

FEATURES = ["form_driver", "form_team", "fp_gap", "fp_soft_gap", "sq_gap", "sprint_gap"]
FORM_HALFLIFE = 3.0       # sessions


class Tables:
    """Caches pace tables per events.csv, since backtests read the same sessions many times."""
    def __init__(self, av):
        self.av, self._t = av, {}

    def get(self, path):
        if path not in self._t:
            self._t[path] = pace_table(path)
        return self._t[path]

    def session(self, year, race, session):
        row = self.av[(self.av["year"] == year) & (self.av["race"] == race) & (self.av["session"] == session)]
        t = self.get(row["path"].iloc[0]) if len(row) else None
        return t if t is not None and not t.empty else None


def ewma(series_list, halflife=FORM_HALFLIFE):
    """Recency-weighted mean of per-driver values, oldest session first; drivers may miss sessions."""
    n = len(series_list)
    num, den = pd.Series(dtype=float), pd.Series(dtype=float)
    for j, s in enumerate(series_list):
        w = 0.5 ** ((n - 1 - j) / halflife)
        num = num.add(s * w, fill_value=0)
        den = den.add(s.notna() * w, fill_value=0)
    return num / den.where(den > 0)


def weekend_features(tables, year, race, with_target=True, drivers=None):
    """`drivers` (DataFrame indexed by tla with a `team` column) fixes the rows and keeps the quali's own events.csv
    unread: for a quali that is still being recorded, whose pace table would be incomplete (and cached)."""
    race = race_key(race)
    av = tables.av
    wk = av[(av["year"] == year) & (av["race"] == race)]
    q = wk[wk["session"] == "quali"]
    cutoff = q["start"].iloc[0] if len(q) else wk["start"].max()
    quali_t = tables.get(q["path"].iloc[0]) if len(q) and drivers is None else None

    # form: this year's earlier qualifying sessions
    prior = av[(av["year"] == year) & (av["session"] == "quali") & (av["start"] < cutoff)].sort_values("start")
    pts = [tables.get(p) for p in prior["path"]]
    pts = [t for t in pts if not t.empty]
    form_driver = ewma([t.set_index("tla")["gap_pct"] for t in pts]) if pts else pd.Series(dtype=float)
    form_team = ewma([t.groupby("team")["gap_pct"].min() for t in pts]) if pts else pd.Series(dtype=float)

    # this weekend's other sessions
    fp, fp_soft = [], []
    for s in ("fp1", "fp2", "fp3"):
        t = tables.session(year, race, s)
        if t is not None:
            fp.append(t.set_index("tla")["gap_pct"])
            soft = t.set_index("tla")["best_soft_s"]
            fp_soft.append((soft / soft.min() - 1) * 100)
    sq = tables.session(year, race, "sq")
    sprint = tables.session(year, race, "sprint")

    # drivers: those in the quali, else everyone seen in this weekend's sessions
    if drivers is not None:
        base = drivers[["team"]]
    elif quali_t is not None and not quali_t.empty:
        base = quali_t.set_index("tla")[["team"]]
    else:
        seen = [tables.session(year, race, s) for s in ("fp1", "fp2", "fp3", "sq", "sprint")]
        seen = pd.concat([t for t in seen if t is not None and not t.empty])
        base = seen.groupby("tla")[["team"]].last()
    df = base.copy()
    df["form_driver"] = form_driver.reindex(df.index)
    df["form_team"] = df["team"].map(form_team)
    df["fp_gap"] = pd.concat(fp, axis=1).min(axis=1).reindex(df.index) if fp else np.nan
    df["fp_soft_gap"] = pd.concat(fp_soft, axis=1).min(axis=1).reindex(df.index) if fp_soft else np.nan
    df["sq_gap"] = sq.set_index("tla")["gap_pct"].reindex(df.index) if sq is not None else np.nan
    df["sprint_gap"] = sprint.set_index("tla")["gap_pct"].reindex(df.index) if sprint is not None else np.nan
    if with_target and quali_t is not None:
        df["y"] = quali_t.set_index("tla")["gap_pct"]
    df["year"], df["race"], df["start"] = year, race, cutoff
    return df


def fill(df, columns):
    """Replace NaNs in `columns` with the weekend's 80th percentile (a driver without a time is probably slow)."""
    df = df.copy()
    for c in columns:
        df[c] = df[c].fillna(df[c].quantile(0.8))
    return df
