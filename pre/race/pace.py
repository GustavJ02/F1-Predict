"""Race pace of each driver, and the rating that predicts it before a race.

race_pace: from one finished race, every driver's pace relative to the field (seconds per lap, negative = faster),
from all their clean laps after taking out tyre wear, compound and fuel with the lap model of live.race.priors.

ratings: recency-weighted average of race_pace over the races before a given time, shrunk towards the team average
(a driver with few races, a new car). Only races of the same season are used: cars change between seasons.
"""
import numpy as np
import pandas as pd
from f1live.events import load
from live.race.pace import clean_laps, lap_rows, lap_temps
from live.race.priors import COMPOUNDS, MIN_TEMP
from live.race.projection import lap_time

MIN_LAPS = 8
HALFLIFE = 5.0          # races
TEAM_SHRINK = 2.0       # races' worth of weight pulling a driver towards the team average


def race_pace(history, priors):
    """DataFrame indexed by tla: team, rel (s per lap vs the field median, negative = faster), n (laps used)."""
    laps = lap_rows(history)
    clean = clean_laps(laps)
    clean = clean[clean["compound"].isin(COMPOUNDS) & clean["tyre_laps"].notna()].copy()
    if clean.empty:
        return pd.DataFrame(columns=["team", "rel", "n"])
    ci = clean["compound"].map({c: i for i, c in enumerate(COMPOUNDS)}).to_numpy()
    model = priors["lap_model"]
    temp = lap_temps(clean, model.get("temp_ref", 0.0), MIN_TEMP)
    clean["resid"] = clean["lap_time_s"].to_numpy() - lap_time(model, 0.0, ci, clean["tyre_laps"].to_numpy(),
                                                              clean["lap_number"].to_numpy(), temp=temp)
    g = clean.groupby("driver")
    keep = g["resid"].transform(lambda r: r <= r.median() + 1.5)
    clean = clean[keep]
    g = clean.groupby("driver")
    t = pd.DataFrame({"tla": laps.groupby("driver")["tla"].last(), "team": laps.groupby("driver")["team"].last(),
                      "base": g["resid"].median(), "n": g.size()}).dropna(subset=["base"])
    t = t[t["n"] >= MIN_LAPS]
    t["rel"] = t["base"] - t["base"].median()
    return t.set_index("tla")[["team", "rel", "n"]]


def actual_order(history):
    """{tla: final classified position} of a finished race."""
    laps = lap_rows(history)
    tla = laps.groupby("driver")["tla"].last()
    last = pd.to_numeric(history[history["driver"].notna()].groupby("driver").tail(1).set_index("driver")["position"],
                         errors="coerce").dropna()
    return pd.Series({tla[d]: int(p) for d, p in last.items() if d in tla.index})


class RaceArchive:
    """Race pace and results of every recorded race, computed once."""
    def __init__(self, av_races, priors):
        self.races = av_races            # DataFrame year, race, start, path (race sessions only), sorted by start
        self.priors = priors
        self._pace, self._result = {}, {}

    def result(self, path):
        if path not in self._result:
            self._result[path] = actual_order(load(path))
        return self._result[path]

    def pace(self, path):
        if path not in self._pace:
            self._pace[path] = race_pace(load(path), self.priors)
        return self._pace[path]


def ratings(archive, year, before, halflife=HALFLIFE, shrink=TEAM_SHRINK):
    """DataFrame indexed by tla: team, rating (s per lap vs the field, negative = faster), weight (races' worth of data)."""
    races = archive.races[(archive.races["year"] == year) & (archive.races["start"] < before)].sort_values("start")
    rows = []
    for age, r in enumerate(reversed(list(races.itertuples()))):
        w = 0.5 ** (age / halflife)
        t = archive.pace(r.path)
        rows += [dict(tla=tla, team=x.team, rel=x.rel, w=w * min(x.n / 40.0, 1.0)) for tla, x in t.iterrows()]
    if not rows:
        return pd.DataFrame(columns=["team", "rating", "weight"])
    d = pd.DataFrame(rows)
    drv = d.groupby("tla").apply(lambda g: pd.Series(dict(team=g["team"].iloc[0], rating=np.average(g["rel"], weights=g["w"]),
                                                          weight=g["w"].sum())))
    team = drv.groupby("team").apply(lambda g: np.average(g["rating"], weights=g["weight"]))
    drv["rating"] = (drv["weight"] * drv["rating"] + shrink * drv["team"].map(team)) / (drv["weight"] + shrink)
    return drv
