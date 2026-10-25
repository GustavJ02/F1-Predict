"""Predict battles: which cars will be within THRESH seconds of the car ahead, and in how many laps.

For each pair of adjacent cars (A ahead, B behind) with a trustworthy pace for both, B closes on A by
delta = pace_A - pace_B seconds per lap, so the gap shrinks to THRESH in (gap - THRESH) / delta laps. Pairs that are
already within THRESH are "in battle now". Neighbouring pairs that are in or will be in contact within HORIZON laps
are grouped into one battle ("Battle for 4th: ANT, PIA, NOR").

A pair is only predicted when both cars have a pace estimate (live.race.pace: enough clean laps on the current
stint), neither is in the pit lane or out of the race, the interval to the car ahead is known and the track is green.
"""
import numpy as np
import pandas as pd
from live.race.hazards import chequered
from live.race.pace import driver_pace

THRESH = 1.0            # s, DRS range
HORIZON = 12            # laps
MIN_CLOSING = 0.05      # s per lap; slower than this is not closing
STALE_S = 240           # no lap for this long while the race goes on: the car is out
JUST_PITTED = 2         # laps on the stint before a car's pace and gap are meaningful again


def latest(history):
    """One row per driver: the most recent row (any event), plus last_lap_ts and out (retired or stale)."""
    rows = history[history["driver"].notna()]
    if rows.empty:
        return pd.DataFrame()
    last = rows.groupby("driver").tail(1).set_index("driver")
    now = history["ts"].max()
    lap_ts = rows[rows["event"] == "lap"].groupby("driver")["ts"].max()
    retired = set(rows.loc[rows["event"] == "retired", "driver"])
    last["last_lap_ts"] = lap_ts.reindex(last.index)
    red = str(history["status_code"].dropna().iloc[-1]) == "5" if history["status_code"].notna().any() else False
    stale = ((now - last["last_lap_ts"]).dt.total_seconds() > STALE_S) & (not red)      # nobody laps under a red flag
    last["out"] = last.index.isin(retired) | (stale & (last["last_lap_ts"].notna()))
    last["position"] = pd.to_numeric(last["position"], errors="coerce")
    return last.sort_values("position")


def _true(v):
    return pd.notna(v) and bool(v)


def laps_to_contact(gap, delta, slope_delta=0.0, thresh=THRESH, horizon=HORIZON):
    """Whole laps until the gap is <= thresh when closing by delta s/lap (+ slope_delta s/lap each lap); inf if never."""
    if gap <= thresh:
        return 0.0
    if delta < MIN_CLOSING:
        return np.inf
    closed = 0.0
    for k in range(1, horizon + 1):
        closed += delta + slope_delta * (k - 1)
        if gap - closed <= thresh:
            prev = closed - (delta + slope_delta * (k - 1))
            return round(k - 1 + (gap - thresh - prev) / max(closed - prev, 1e-9), 1)
    return np.inf


def pair_table(history, use_slope=False, **pace_kw):
    """Eligible adjacent pairs: front/rear driver, gap, delta (s/lap, + = rear closing), laps (to contact), status."""
    lat = latest(history)
    if lat.empty:
        return lat, pd.DataFrame()
    pace = driver_pace(history, use_slope=use_slope, **pace_kw)
    rows = []
    cars = lat[~lat["out"] & lat["position"].notna()]
    for (_, a), (_, b) in zip(cars.iloc[:-1].iterrows(), cars.iloc[1:].iterrows()):
        da, db = a.name, b.name
        gap = b["interval_s"]
        if pd.isna(gap) or _true(a["in_pit"]) or _true(b["in_pit"]):
            continue
        pa, pb = pace["pace_s"].get(da, np.nan), pace["pace_s"].get(db, np.nan)
        if np.isnan(pa) or np.isnan(pb):
            continue
        delta = pa - pb
        slope_d = (pace["slope_s"].get(da, 0.0) - pace["slope_s"].get(db, 0.0)) if use_slope else 0.0
        rows.append(dict(front=da, rear=db, front_tla=a["tla"], rear_tla=b["tla"], position=int(a["position"]),
                         gap_s=float(gap), delta_s=float(delta), laps=laps_to_contact(float(gap), delta, slope_d)))
    return lat, pd.DataFrame(rows)


def find_battles(history, use_slope=False, horizon=HORIZON, **pace_kw):
    """-> (status, battles). status is a message when predictions are paused (None otherwise); battles are dicts:
    position (of the front car), drivers (tlas front to back), numbers, gaps (s, per pair), laps (per pair, 0 = in
    contact now), in_laps (soonest contact of the group), now (bool)."""
    lat, pairs = pair_table(history, use_slope, **pace_kw)
    if lat.empty:
        return "waiting for race data", []
    lap = history[history["race_lap"].notna() & history["total_laps"].notna()].tail(1)
    if len(lap):                      # no contact can happen after the flag
        remaining = int(lap["total_laps"].iloc[0] - lap["race_lap"].iloc[0])
        if remaining <= 0 and chequered(history):      # the counter shows total/total for the whole final lap
            return "race finished", []
        horizon = min(horizon, max(remaining, 1))
    track = history["track_status"].dropna().iloc[-1] if history["track_status"].notna().any() else None
    if track not in (None, "AllClear"):
        return f"{track}: predictions paused", []
    if pairs.empty:
        return "collecting pace data", []
    linked = pairs[pairs["laps"] <= horizon]
    battles, group = [], []
    for _, p in linked.sort_values("position").iterrows():
        if group and group[-1]["position"] + 1 == p["position"]:
            group.append(p)
        else:
            if group:
                battles.append(group)
            group = [p]
    if group:
        battles.append(group)
    out = []
    for g in battles:
        out.append(dict(position=int(g[0]["position"]),
                        drivers=[g[0]["front_tla"]] + [p["rear_tla"] for p in g],
                        numbers=[g[0]["front"]] + [p["rear"] for p in g],
                        gaps=[round(p["gap_s"], 2) for p in g], laps=[float(p["laps"]) for p in g],
                        in_laps=float(min(p["laps"] for p in g)), now=any(p["laps"] == 0 for p in g)))
    return None, sorted(out, key=lambda b: (b["in_laps"], b["position"]))
