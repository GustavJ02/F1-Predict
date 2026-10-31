"""A car that has stopped or is crawling on the track, from its telemetry speed.

STOPPED  speed at or below STOP_KMH for the last STOP_S seconds, after the car was running (above RAN_KMH at some point in the
         last RAN_S seconds): a car stationary on the grid, in the pit lane or garage doesn't count.
SLOW     the median speed over the last SLOW_S seconds is below SLOW_RATIO of what cars normally do at that part of the lap
         (learned from this session: SpeedProfile), where that normal speed is at least MIN_TYPICAL, so slow corners, braking
         zones and pit lanes don't trigger it. Only judged under green flags: under a safety car or VSC everybody is slow.

Both are only raised while the race is running (lap 2 on, not red-flagged, not after the flag) and the car is not in the pit lane.
BOT in the 2026 US GP is the pattern this was built from: about a minute in first gear at 30-55 km/h, then stationary, and
the VSC ten seconds later.
"""
import bisect
from collections import deque

import numpy as np
import pandas as pd

STOP_KMH, STOP_S = 5.0, 3.0
RAN_KMH, RAN_S = 100.0, 240.0
SLOW_RATIO, SLOW_S, MIN_TYPICAL = 0.45, 6.0, 80.0
BINS, KEEP, MIN_BIN_SAMPLES = 60, 150, 25
MIN_SPEED_LEARNED = 30.0          # km/h: slower samples don't teach the profile what a normal speed is


class SpeedProfile:
    """What speed cars normally have at each part of the lap: recent speeds per slice of the lap."""
    def __init__(self, bins=BINS, keep=KEEP):
        self.bins = bins
        self.speeds = [deque(maxlen=keep) for _ in range(bins)]

    def _bin(self, progress):
        return min(int(progress * self.bins), self.bins - 1)

    def add(self, progress, speed):
        if speed is not None and speed >= MIN_SPEED_LEARNED:
            self.speeds[self._bin(progress)].append(float(speed))

    def typical(self, progress):
        """Median speed there, or None before enough cars have been through."""
        s = self.speeds[self._bin(progress)]
        return float(np.median(s)) if len(s) >= MIN_BIN_SAMPLES else None


def classify(samples, at, progress_of=None, profile=None, green=True, last_ran=None):
    """None, 'stopped' or 'slow' for one car at event time `at`.

    samples: [(ts, x, y, speed), ...] in time order; progress_of(x, y) -> lap fraction (or None) and the profile are only needed
    for 'slow'. last_ran: when the car was last seen above RAN_KMH, if known beyond `samples` (so a car that has been stopped
    for longer than the trail stays stopped)."""
    recent = [s for s in samples if s[0] <= at]
    if not recent:
        return None
    stop = [s for s in recent if (at - s[0]).total_seconds() <= STOP_S]
    if len(stop) >= 2 and all(_speed(s) <= STOP_KMH for s in stop):
        earlier = [_speed(s) for s in recent if STOP_S < (at - s[0]).total_seconds() <= RAN_S]
        if (earlier and max(earlier) > RAN_KMH) or (last_ran is not None and last_ran <= at):
            return "stopped"
    if green and progress_of is not None and profile is not None:
        window = [s for s in recent if (at - s[0]).total_seconds() <= SLOW_S]
        if len(window) >= 3:
            typical = [profile.typical(p) for p in (progress_of(s[1], s[2]) for s in window) if p is not None]
            typical = [t for t in typical if t is not None]
            if len(typical) >= 3 and np.median(typical) >= MIN_TYPICAL \
                    and np.median([_speed(s) for s in window]) < SLOW_RATIO * np.median(typical):
                return "slow"
    return None


def _speed(sample):
    v = sample[3] if len(sample) > 3 else np.nan
    return float(v) if v == v else np.inf             # an unknown speed is never "stopped" or "slow"


def track_state(history):
    """The latest track status ('AllClear', 'Yellow', 'SCDeployed', 'VSCDeployed', 'VSCEnding', 'Red', ...), or None."""
    if history is None or history.empty:
        return None
    ts = history.loc[history["event"] == "track_status", "track_status"].dropna()
    return ts.iloc[-1] if len(ts) else None


NEUTRAL = ("SCDeployed", "SCEnding", "VSCDeployed", "VSCEnding", "Red")
BANNERS = {"SCDeployed": "SAFETY CAR", "SCEnding": "SAFETY CAR ENDING", "VSCDeployed": "VIRTUAL SAFETY CAR",
           "VSCEnding": "VSC ENDING", "Red": "RED FLAG"}


def chequered(history):
    """Has the chequered flag been shown? (The lap counter reads total/total for the whole final lap, so it can't tell.)"""
    if history is None or history.empty or "message" not in history:
        return False
    msg = history.loc[history["event"] == "race_control", "message"].dropna()
    return bool(msg.astype(str).str.contains("CHEQUERED FLAG", case=False).any())


def race_is_live(history):
    """Is it a moment at which a stopped car means something: past lap 1, not red-flagged, not finished."""
    if history is None or history.empty:
        return False
    row = history[history["race_lap"].notna()].tail(1)
    if row.empty:
        return False
    if row["race_lap"].iloc[0] < 2 or chequered(history):
        return False
    return track_state(history) != "Red"


PIT_BEFORE, PIT_AFTER = 15.0, 25.0     # s around a pit_in/pit_out pair in which the pit-lane limiter explains a slow car


def pit_windows(history):
    """{driver: [(start, end), ...]} event-time windows in which a car is in or just entering/leaving the pit lane.
    A pit_in without its pit_out yet stays open."""
    out = {}
    if history is None or history.empty:
        return out
    ev = history[history["event"].isin(["pit_in", "pit_out"])].sort_values("ts")
    before, after = pd.Timedelta(PIT_BEFORE, unit="s"), pd.Timedelta(PIT_AFTER, unit="s")
    for drv, g in ev.groupby("driver"):
        windows, opened = [], None
        for r in g.itertuples():
            if r.event == "pit_in":
                opened = r.ts - before
            elif opened is not None:
                windows.append((opened, r.ts + after))
                opened = None
            else:                                      # a pit_out whose pit_in we never saw (session start in the pit lane)
                windows.append((r.ts - after, r.ts + after))
        if opened is not None:
            windows.append((opened, pd.Timestamp.max.tz_localize(opened.tzinfo)))
        out[str(drv)] = windows
    return out


def in_pit(windows, driver, at):
    return any(a <= at <= b for a, b in windows.get(str(driver), ()))


def retired_at(history):
    """{driver: when the retirement was reported}."""
    if history is None or history.empty:
        return {}
    r = history[history["event"] == "retired"]
    return {str(d): ts for d, ts in zip(r["driver"], r["ts"])}


def retired(history):
    """Drivers who have retired (their 'slow' is no news; a stopped car still is)."""
    if history is None or history.empty:
        return set()
    return set(history.loc[history["event"] == "retired", "driver"].astype(str))


def hazards(trail, history, at, progress_of=None, profile=None, ran=None, out=(), race=True):
    """{driver: 'stopped' | 'slow'} as of event time `at`. trail: {driver: [(ts, x, y, speed), ...]}; ran: {driver: [ts, ...]}
    sorted moments the car was seen racing (Feed keeps these for the whole session). A car that has retired or is `out` (marked OUT
    in the standings) is no longer a hazard to flag: whatever happened, it is over for the race. race=False (practice) skips the
    check that a race is running; the lap counter doesn't exist there."""
    if race and not race_is_live(history):
        return {}
    windows, gone = pit_windows(history), retired(history) | set(map(str, out))
    green = track_state(history) not in NEUTRAL
    out = {}
    for drv, samples in trail.items():
        if in_pit(windows, drv, at):
            continue
        if drv in gone:
            continue
        times = (ran or {}).get(drv)
        i = bisect.bisect_right(times, at) if times else 0
        kind = classify(samples, at, progress_of, profile, green, times[i - 1] if i else None)
        if kind:
            out[drv] = kind
    return out


def neutral_at(history, ts):
    """Boolean array: was the race neutralised (SC, VSC, red) at each timestamp of `ts`?"""
    ev = history.loc[history["event"] == "track_status", ["ts", "track_status"]].sort_values("ts") \
        if history is not None and not history.empty else None
    ts = pd.DatetimeIndex(ts)
    if ev is None or ev.empty:
        return np.zeros(len(ts), bool)
    state = ev["track_status"].isin(NEUTRAL).to_numpy()
    i = np.searchsorted(pd.DatetimeIndex(ev["ts"]).asi8, ts.asi8, side="right") - 1
    return np.where(i >= 0, state[np.maximum(i, 0)], False)
