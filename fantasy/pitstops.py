"""Pit stop points of a constructor (fantasy.scoring.constructor_pit_points) from each team's distribution of stop times.

Where the distributions come from, in order of preference:
  1. data/fantasy/pit_stops.json: per team the fastest stop, quartiles and median of the season's stop times, copied from a published
     source (formula1dashboard.com "pit stop consistency by team", read off the chart to about 0.03 s). Use medians and quartiles, not
     averages: a few 10 s stops (a penalty, a stuck wheel) drag an average up by seconds and say nothing about the next stop.
  2. the recorded pit-lane times (pit_lane_s on pit_out rows), for a team the file does not have. The drive through the speed-limited
     lane takes about the same for everybody, so a stop compared with the other stops of its race (relative time r = pit_lane_s minus
     the race median, green flag only) shows how long the car stood still. One stop's lane time has about 0.5 s of noise, but a team's
     average is stable (split-half correlation 0.7) and agrees with the published medians (correlation 0.89 over the 2026 teams, a second
     in the lane = 0.78 s of median stop, see MEDIAN_AT_AVERAGE / MEDIAN_PER_LANE_S).

Only the quick end of a distribution is worth anything (under 2.0 s 20, 2.0-2.19 10, 2.2-2.49 5, 2.5-2.99 2, from 3.0 s 0, the quickest stop of
the race +5, under 1.8 s +15), so a stop is drawn through the team's quantile function: linear between its fastest stop (taken as the 3rd
percentile: about 33 stops a season), Q1, median and Q3, then a tail for slow stops that all score 0. Time penalties (5 or 10 s served in the
pit lane) land in that tail; in the lane data they are lane times 4+ s above the race median and are cut from the team effects.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from fantasy import scoring as S

STATS_FILE = Path(__file__).resolve().parents[1] / "data" / "fantasy" / "pit_stops.json"
P_FASTEST = 0.03            # the season's fastest stop is about the 3rd percentile of a team's ~33 stops
TAIL_P = (0.90, 1.0)        # beyond Q3: the 90th percentile is Q3 + 1 IQR, the slowest stop Q3 + 4 IQR (all of it scores 0 anyway)
MEDIAN_AT_AVERAGE = 2.88    # s, median stop of a team whose lane times equal the field's average (fit over the 2026 teams)
MEDIAN_PER_LANE_S = 0.78    # s of median stop per second of average lane time relative to the race median
TRIM = (-3.0, 4.0)          # relative lane times outside this range (s from the race median) are not ordinary stops
HALFLIFE = 12               # races: how fast an old stop's information on a team fades
SHRINK = 10.0               # a team's effect counts this many stops' worth of "average team" (effect 0)
FIRST_YEAR = 2023           # the pit lane timing of earlier seasons is a different measurement
MAX_STOPS = 8               # per team and race, enough for any simulated scenario


def load_stats(path=STATS_FILE):
    """{team: dict(low, q1, med, q3)} from the JSON file, {} if there is none."""
    try:
        return {t: dict(v) for t, v in json.loads(Path(path).read_text())["teams"].items()}
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return {}


def stats_from_effect(effect):
    """Stop time summary of a team known only by its lane-time effect (s relative to the average team)."""
    med = MEDIAN_AT_AVERAGE + MEDIAN_PER_LANE_S * effect
    q1, q3 = med - 0.3, med + 0.35 + 0.5 * max(med - 2.5, 0.0)
    return dict(low=max(2.15, q1 - 0.2), q1=q1, med=med, q3=q3)


def quantile_knots(st):
    """(probabilities, stop times) of the piecewise-linear quantile function of a team's stop time."""
    iqr = st["q3"] - st["q1"]
    p = [0.0, P_FASTEST, 0.25, 0.5, 0.75, *TAIL_P]
    t = [st["low"] - 0.05, st["low"], st["q1"], st["med"], st["q3"], st["q3"] + iqr, st["q3"] + 4 * iqr]
    return p, t


def pit_lane_stops(av):
    """Ordinary green-flag stops of the recorded races: DataFrame year, race, start, team, tla, r (s relative to the race median)."""
    from f1live.events import load
    rows = []
    for x in av[(av["session"] == "race") & (av["year"] >= FIRST_YEAR)].itertuples():
        ev = load(x.path)
        p = ev[(ev["event"] == "pit_out") & ev["pit_lane_s"].notna() & (ev["race_lap"] >= 2) & ev["team"].notna()
               & (ev["track_status"] == "AllClear")]
        if len(p) < 6:
            continue
        r = p["pit_lane_s"] - p["pit_lane_s"].median()
        keep = r.between(*TRIM)
        rows.append(pd.DataFrame(dict(year=x.year, race=x.race, start=x.start, team=p["team"][keep], tla=p["tla"][keep], r=r[keep])))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["year", "race", "start", "team", "tla", "r"])


def team_effects(stops, before):
    """Series team -> seconds slower (+) / quicker (-) than the average team, from stops of races started before `before`;
    recency weighted (HALFLIFE races) and shrunk toward 0. A team with no stops yet has no entry (the model uses 0)."""
    s = stops[stops["start"] < before]
    if s.empty:
        return pd.Series(dtype=float)
    order = s["start"].drop_duplicates().sort_values().reset_index(drop=True)
    age = pd.Series(np.arange(len(order))[::-1], index=order)
    w = 0.5 ** (s["start"].map(age).to_numpy() / HALFLIFE)
    num = (s["r"].to_numpy() * w)
    out = pd.DataFrame(dict(team=s["team"].to_numpy(), num=num, w=w)).groupby("team").sum()
    return out["num"] / (out["w"] + SHRINK)


@dataclass
class PitModel:
    stats: dict = field(default_factory=dict)       # team -> dict(low, q1, med, q3), from the file
    effect: dict = field(default_factory=dict)      # team -> lane-time effect, for teams the file does not have

    def team_stats(self, team):
        if team in self.stats:
            return self.stats[team]
        return stats_from_effect(self.effect.get(team, 0.0))

    def stop_times(self, rng, teams, n_stops):
        """Simulated stop times, shape (scenarios, teams, MAX_STOPS), NaN where the team makes no such stop.
        n_stops: int array (scenarios, teams), the number of stops of both the team's cars in that scenario."""
        n_stops = np.minimum(np.asarray(n_stops), MAX_STOPS)
        n_scen, n_teams = n_stops.shape
        u = rng.random((n_scen, n_teams, MAX_STOPS))
        t = np.empty_like(u)
        for k, team in enumerate(teams):
            p, q = quantile_knots(self.team_stats(team))
            t[:, k, :] = np.interp(u[:, k, :], p, q)
        return np.where(np.arange(MAX_STOPS)[None, None, :] < n_stops[:, :, None], t, np.nan)

    def points(self, rng, teams, n_stops):
        """Constructor pit stop points per scenario and team, (scenarios, teams): every stop by its time, 5 for the team that
        made the quickest stop of the race, 15 for each stop under the world record time."""
        times = self.stop_times(rng, teams, n_stops)
        filled = np.nan_to_num(times, nan=np.inf)
        pts = np.zeros(times.shape)
        for limit, p in reversed(S.PIT):                 # from the slowest tier to the quickest: later assignments win
            pts = np.where(filled < limit, p, pts)
        pts = np.where(filled < S.RECORD_TIME, pts + S.PIT_RECORD, pts)
        total = pts.sum(axis=2)
        flat = filled.reshape(filled.shape[0], -1)
        quickest = flat.argmin(axis=1) // MAX_STOPS      # team index of each scenario's quickest stop
        has_stop = np.isfinite(flat.min(axis=1))
        bonus = np.zeros(total.shape)
        bonus[np.arange(len(bonus))[has_stop], quickest[has_stop]] = S.PIT_FASTEST
        return total + bonus


def fit_model(av, before=None, stats_file=STATS_FILE):
    """PitModel: the file's team statistics plus lane-time effects (measured from the recorded races before `before`)."""
    before = before if before is not None else "9999"
    return PitModel(stats=load_stats(stats_file), effect=team_effects(pit_lane_stops(av), before).to_dict())
