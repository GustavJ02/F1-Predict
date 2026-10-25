"""Rain, from the weather feed and the tyres the cars are on.

The race model (priors, lap model, stint lengths) is a dry-weather model: on wet tyres lap times, stop plans and the safety
car rate are all different. So a race that is wet now can't be projected with it (live.race.model pauses the projection),
and a race that was wet is a different kind of race in the backtests: in the 6 wet races of 2025-26 there were twice the
safety cars and virtual safety cars of the dry ones (1.3 against 0.7 per race each), and the grid was a worse guide to the
finish (rank correlation 0.67 against 0.80; the pole-sitter won half of them, not three quarters).
"""
import numpy as np
import pandas as pd

WET_COMPOUNDS = ("INTERMEDIATE", "WET")
NOW_WINDOW_S = 300           # how far back "raining now" looks, in event time


def rain_rows(history):
    """Share of the weather rows that say it is raining (0 if the recording has no weather)."""
    if "rainfall" not in history or not history["rainfall"].notna().any():
        return 0.0
    return float((history["rainfall"].astype(float).dropna() > 0).mean())


def wet_lap_share(history):
    """Share of the lap rows run on intermediate or wet tyres."""
    laps = history[history["event"] == "lap"]
    return float(laps["compound"].isin(WET_COMPOUNDS).mean()) if len(laps) else 0.0


def is_wet_race(history, share=0.05):
    """A race that had rain on the weather feed or wet tyres on the cars for more than `share` of the time."""
    return rain_rows(history) > share or wet_lap_share(history) > share


def wet_now(history, window_s=NOW_WINDOW_S):
    """True if it is raining at the end of `history` (event time): the weather feed says so in the last few minutes, or at
    least two cars finished their last lap on intermediates or wets."""
    if history is None or history.empty:
        return False
    end = history["ts"].max()
    recent = history[history["ts"] >= end - pd.Timedelta(window_s, unit="s")]
    if "rainfall" in recent and (recent["rainfall"].astype(float).fillna(0) > 0).any():
        return True
    laps = recent[recent["event"] == "lap"]
    return bool(laps.groupby("driver")["compound"].last().isin(WET_COMPOUNDS).sum() >= 2) if len(laps) else False
