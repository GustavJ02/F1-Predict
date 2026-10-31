"""What a practice dashboard needs from events.csv, no predictions involved: the session clock, flags, what each car is doing,
and the timesheet with long-run pace.

Car mode (car_modes), from timing and the car's progress along the lap:
    PIT     in the pit lane or garage
    OUT     out lap: left the pits and has not yet crossed the line
    PUSH    on a lap that, projected from the time since the line and the distance covered, is within PUSH_RATIO of the
            driver's best (the session's best x FIRST_RATIO if he has no time yet)
    SLOW    projected lap more than SLOW_RATIO of that: cool-down, in-lap or a crawl
    RUN     anything in between (long run, a lap with traffic); also when there is too little of the lap to tell
    STOPPED stationary on track after running (hazards.classify)
"""
import re

import numpy as np
import pandas as pd

from live.race.battles import latest
from live.race.hazards import chequered

# Calibrated on the Austin FP1/FP2 telemetry: at 35-80 % of the lap, laps run within 5 % of the driver's best project to about
# 0.97-1.02, cool-down and in-laps to 1.4+. Very early in the lap the projection reads fast, hence MIN_PROGRESS. A quick long-run
# lap and a push lap overlap, so PUSH means 'at flying-lap pace', not 'a timed attempt'.
PUSH_RATIO, SLOW_RATIO, FIRST_RATIO = 1.02, 1.2, 1.04
MIN_PROGRESS = 0.3              # of the lap before its pace can be projected
MIN_RUN = 5                     # clean laps in one stint before it counts as a long run
MAX_SLOWER = 1.07               # a lap slower than this x the driver's best is a cool-down or traffic, not run pace
OUTLIER = 1.5                   # s above the run's median: dropped (mistake, traffic)

YELLOW = re.compile(r"(DOUBLE )?YELLOW IN TRACK SECTOR (\d+)")
CLEAR = re.compile(r"CLEAR IN TRACK SECTOR (\d+)")


def clock_left(history):
    """Seconds left in the session as of the latest row, or None."""
    left = history["clock_left_s"].dropna()
    return float(left.iloc[-1]) if len(left) else None


def messages(history):
    """Race control text in order."""
    rc = history[history["event"] == "race_control"]
    return list(zip(rc["ts"], rc["message"].astype(str)))


def yellow_sectors(history):
    """{marshal sector: 'YELLOW' | 'DOUBLE YELLOW'} still showing, from the race control messages."""
    out = {}
    for _, text in messages(history):
        if hit := YELLOW.search(text):
            out[int(hit.group(2))] = "DOUBLE YELLOW" if hit.group(1) else "YELLOW"
        elif hit := CLEAR.search(text):
            out.pop(int(hit.group(1)), None)
        elif "ALL CLEAR" in text or "CLEAR" == text.strip():
            out.clear()
    return out


def pit_exit(history):
    """'open', 'closed' or None (not announced) from the latest GREEN LIGHT / PIT EXIT message."""
    state = None
    for _, text in messages(history):
        if "PIT EXIT OPEN" in text:
            state = "open"
        elif "PIT EXIT CLOSED" in text:
            state = "closed"
    return state


def recent_messages(history, n=4):
    """The last n race control messages that are not lap deletions: [(ts, text)]."""
    return [m for m in messages(history) if "LAP DELETED" not in m[1]][-n:]


def car_modes(history, progress, flags=None, measured=True):
    """{driver: 'PIT' | 'OUT' | 'PUSH' | 'SLOW' | 'RUN' | 'STOPPED'} for every driver in the timing data.
    progress: {driver: lap fraction}; flags: {driver: 'stopped' | 'slow'} from the hazards (only 'stopped' is used here:
    a cool-down lap is slow all the time in practice). measured=False (progress estimated from lap timing, not telemetry)
    skips PUSH/SLOW: that estimate is made from the car's own pace, so it says nothing about it."""
    lat = latest(history)
    if lat.empty:
        return {}
    now = history["ts"].max()
    laps = history[history["event"] == "lap"]
    last_lap = laps.groupby("driver")["ts"].max()
    pit_out = history[history["event"] == "pit_out"].groupby("driver")["ts"].max()
    session_best = lat["best_lap_s"].min()
    modes = {}
    for drv, r in lat.iterrows():
        if flags and flags.get(drv) == "stopped":
            modes[drv] = "STOPPED"
        elif r["in_pit"] is True or (pd.notna(r["in_pit"]) and bool(r["in_pit"])):
            modes[drv] = "PIT"
        elif drv in pit_out.index and (drv not in last_lap.index or pit_out[drv] > last_lap[drv]):
            modes[drv] = "OUT"
        else:
            modes[drv] = "RUN"
            frac, since = progress.get(drv), last_lap.get(drv)
            ref = r["best_lap_s"] if pd.notna(r["best_lap_s"]) else session_best * FIRST_RATIO
            if measured and frac is not None and frac >= MIN_PROGRESS and pd.notna(since) and pd.notna(ref):
                ratio = (now - since).total_seconds() / frac / ref
                if ratio <= PUSH_RATIO:
                    modes[drv] = "PUSH"
                elif ratio > SLOW_RATIO:
                    modes[drv] = "SLOW"
    return modes


def clean_laps(history):
    """Timed laps that say something about pace: not in/out, under green, not deleted, within MAX_SLOWER of the driver's best."""
    laps = history[(history["event"] == "lap") & history["driver"].notna()]
    laps = laps[laps["lap_kind"].isin(["push", "other"]) & laps["lap_time_s"].notna()]
    deleted = history[history["event"] == "lap_deleted"]
    if len(deleted):
        gone = set(zip(deleted["driver"], deleted["lap_time_s"].round(3)))
        laps = laps[[(d, round(t, 3)) not in gone for d, t in zip(laps["driver"], laps["lap_time_s"])]]
    laps = laps[laps["track_status"].eq("AllClear")]
    return laps[laps["lap_time_s"] <= MAX_SLOWER * laps.groupby("driver")["lap_time_s"].transform("min")]


def long_runs(history):
    """DataFrame indexed by driver, their latest stint with at least MIN_RUN clean laps: run_compound, run_laps, run_pace_s (median),
    run_deg_s (s per lap of tyre age, NaN if it can't be fitted), run_gap_s (to the quickest run on the same compound)."""
    out = {}
    for drv, g in clean_laps(history).groupby("driver"):
        for _, s in g.groupby("stint", sort=True):
            if len(s) < MIN_RUN:
                continue
            t = s["lap_time_s"].to_numpy()
            keep = t <= np.median(t) + OUTLIER
            if keep.sum() < MIN_RUN:
                continue
            x, t = s["tyre_laps"].to_numpy(float)[keep], t[keep]
            deg = float(np.clip(np.polyfit(x, t, 1)[0], -0.5, 0.5)) if len(np.unique(x)) > 2 else np.nan
            out[drv] = dict(run_compound=s["compound"].iloc[-1], run_laps=int(keep.sum()),
                            run_pace_s=float(np.median(t)), run_deg_s=deg)
    runs = pd.DataFrame.from_dict(out, orient="index", columns=["run_compound", "run_laps", "run_pace_s", "run_deg_s"])
    runs["run_gap_s"] = runs["run_pace_s"] - runs.groupby("run_compound")["run_pace_s"].transform("min")
    return runs


def best_compound(history):
    """{driver: compound of the quickest timed lap}."""
    laps = history[(history["event"] == "lap") & history["lap_kind"].isin(["push", "other"]) & history["lap_time_s"].notna()]
    return laps.sort_values("lap_time_s").groupby("driver")["compound"].first()
