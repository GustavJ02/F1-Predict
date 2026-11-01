"""Fit the calibration of the Q1/Q2 cutoff band (live.quali.predict_elimination.BAND_SCALE) on backtest results.

The simulated 90 % band is too narrow. For every prediction the real cutoff is compared with the band's two sides, in units of the
band's own half widths: (truth - pred) / (hi - pred) above the prediction and (pred - truth) / (pred - lo) below it. The factor of a
side is the 95th percentile of those numbers in a bucket of "time left in the part", so 5 % of the real cutoffs fall beyond each side
and the band covers 90 %. The factors are put at the median time left of their bucket and interpolated between them.

Usage (from the repo root; the CSVs are written by live.quali.backtest_cutoff --prior --csv FILE, with the bands as simulated):
    python -m live.quali.calibrate_band train_2023.csv train_2024.csv --test 2026.csv
"""
import argparse

import numpy as np
import pandas as pd

EDGES = [-1, 0, 60, 180, 420, 600, 800, 10_000]
MIN_FACTOR, MAX_FACTOR = 0.5, 8.0
MIN_HALF_WIDTH = 1e-3            # s, so that a band that has collapsed onto the prediction does not give infinite ratios


def bands(df):
    """Lower and upper side of the band as simulated (the _raw columns if the backtest wrote them)."""
    lo = df["lo_raw"] if "lo_raw" in df else df["lo"]
    hi = df["hi_raw"] if "hi_raw" in df else df["hi"]
    return lo, hi


def ratios(df):
    """(truth - pred) / upper half width and (pred - truth) / lower half width, per prediction."""
    lo, hi = bands(df)
    up = np.maximum(hi - df["pred"], MIN_HALF_WIDTH)
    down = np.maximum(df["pred"] - lo, MIN_HALF_WIDTH)
    return (df["truth"] - df["pred"]) / up, (df["pred"] - df["truth"]) / down


def fit(df, level=0.90, edges=EDGES):
    """-> list of (median seconds left, factor below, factor above), one per bucket of the time left."""
    r_up, r_down = ratios(df)
    q = 1 - (1 - level) / 2
    table = []
    for lo_e, hi_e in zip(edges[:-1], edges[1:]):
        m = ((df["clock"] > lo_e) & (df["clock"] <= hi_e)).to_numpy()
        if m.sum() < 30:
            continue
        table.append((round(float(df["clock"][m].median()), 1),
                      round(float(np.clip(np.quantile(r_down[m], q), MIN_FACTOR, MAX_FACTOR)), 2),
                      round(float(np.clip(np.quantile(r_up[m], q), MIN_FACTOR, MAX_FACTOR)), 2)))
    return table


def apply(df, table):
    """DataFrame with the calibrated band: columns lo_cal, hi_cal, covered_cal."""
    lo, hi = bands(df)
    t, down, up = (np.array(c, float) for c in zip(*table))
    out = df.copy()
    out["lo_cal"] = df["pred"] - np.interp(df["clock"], t, down) * (df["pred"] - lo)
    out["hi_cal"] = df["pred"] + np.interp(df["clock"], t, up) * (hi - df["pred"])
    out["covered_cal"] = (out["lo_cal"] <= df["truth"]) & (df["truth"] <= out["hi_cal"])
    return out


def report(df, table, title):
    """Coverage and mean width of the band before and after the calibration, by time left."""
    d = apply(df, table)
    lo, hi = bands(d)
    d["raw_covered"] = (lo <= d["truth"]) & (d["truth"] <= hi)
    d["raw_width"], d["cal_width"] = hi - lo, d["hi_cal"] - d["lo_cal"]
    d["bucket"] = pd.cut(d["clock"], EDGES, labels=["0s", "0-60s", "60-180s", "180-420s", "420-600s", "600-800s", ">800s"])
    t = d.groupby("bucket", observed=True).agg(n=("clock", "size"), cover_before=("raw_covered", "mean"), cover_after=("covered_cal", "mean"),
                                               width_before=("raw_width", "mean"), width_after=("cal_width", "mean"))
    print(f"\n{title}: {len(d)} predictions from {d['session'].nunique()} sessions")
    print(t.round(2).to_string())
    print(f"  all: coverage {d['raw_covered'].mean():.0%} -> {d['covered_cal'].mean():.0%}, mean width {d['raw_width'].mean():.2f} -> {d['cal_width'].mean():.2f} s")
    for part, g in d.groupby("part"):
        print(f"  Q{part}: coverage {g['raw_covered'].mean():.0%} -> {g['covered_cal'].mean():.0%}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("train", nargs="+", help="backtest CSVs to fit on")
    ap.add_argument("--test", nargs="*", default=[], help="backtest CSVs to check the fit on (not used for fitting)")
    ap.add_argument("--level", type=float, default=0.90)
    a = ap.parse_args()
    train = pd.concat([pd.read_csv(f) for f in a.train], ignore_index=True).dropna(subset=["pred", "lo", "hi", "truth", "clock"])
    table = fit(train, a.level)
    print("BAND_SCALE =", table)
    report(train, table, "fit data")
    if a.test:
        test = pd.concat([pd.read_csv(f) for f in a.test], ignore_index=True).dropna(subset=["pred", "lo", "hi", "truth", "clock"])
        report(test, table, "held-out data")


if __name__ == "__main__":
    main()
