"""Race pace per driver from the lap rows of events.csv.

A driver's pace is the median of their last WINDOW clean racing laps on the current stint. A lap is clean if it is
a racing lap (no in/out lap), run under green on the lap before and on this one, and not lap 1-2. Laps more than
MAX_SLOWER seconds off the stint's median (traffic, a mistake) are dropped. With fewer than MIN_LAPS clean laps the
driver has no pace estimate: after a stop, or early in the race, the model says nothing rather than guesses.
"""
import numpy as np
import pandas as pd

WINDOW = 4              # backtested on 25 races: 4/3 beat 6/4, 8/5 and 10/6 slightly on precision, recall and ranking
MIN_LAPS = 3
MAX_SLOWER = 1.0        # s above the stint median
FIRST_CLEAN_LAP = 3


def lap_rows(history):
    laps = history[(history["event"] == "lap") & history["driver"].notna()].sort_values(["driver", "ts"])
    return laps


def clean_laps(laps):
    """Racing laps under green (this lap and the previous one of that driver), from lap 3 on."""
    green = laps["track_status"].eq("AllClear")
    prev = green.groupby(laps["driver"]).shift(1)
    prev_green = prev.where(prev.notna(), True).astype(bool)          # a driver's first lap has no previous lap
    keep = (laps["lap_kind"] == "racing") & laps["lap_time_s"].notna() & green & prev_green \
        & (laps["lap_number"] >= FIRST_CLEAN_LAP)
    return laps[keep]


def driver_pace(history, window=WINDOW, min_laps=MIN_LAPS, use_slope=False):
    """DataFrame indexed by driver: pace_s, slope_s (per lap, 0 unless use_slope), noise_s, n (laps used), last_lap_ts.

    Drivers without enough clean laps in their current stint have NaN pace but still a last_lap_ts."""
    laps = lap_rows(history)
    if laps.empty:
        return pd.DataFrame(columns=["pace_s", "slope_s", "noise_s", "n", "last_lap_ts"])
    stint = laps.groupby("driver")["stint"].last()
    last_ts = laps.groupby("driver")["ts"].last()
    clean = clean_laps(laps)
    rows = {}
    for drv, g in clean.groupby("driver"):
        g = g[g["stint"] == stint[drv]].tail(window)
        if len(g) >= 2:
            g = g[g["lap_time_s"] <= g["lap_time_s"].median() + MAX_SLOWER]
        if len(g) < min_laps:
            continue
        t = g["lap_time_s"].to_numpy()
        pace = float(np.median(t))
        noise = max(1.4826 * float(np.median(np.abs(t - pace))), 0.05)
        slope = float(np.clip(np.polyfit(g["lap_number"].to_numpy(), t, 1)[0], -0.3, 0.3)) if use_slope and len(g) >= 5 else 0.0
        rows[drv] = dict(pace_s=pace, slope_s=slope, noise_s=noise, n=len(g))
    out = pd.DataFrame.from_dict(rows, orient="index", columns=["pace_s", "slope_s", "noise_s", "n"])
    out = out.reindex(last_ts.index)
    out["last_lap_ts"] = last_ts
    return out
