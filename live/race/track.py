"""How far along the lap each car is: from car telemetry projected onto a reference lap, else from lap timing.

Reference lap: the x/y samples of one clean flying lap (lap crossing to lap crossing, same driver, green, no stop),
resampled to N points of equal distance, so progress 0 is the start/finish line (to within the crossing's timing
accuracy) and 1 is one full lap. It is built from the session's own telemetry once a clean lap exists, or loaded from
data/<year>/<race>/track.csv, which `python -m live.race.track 2026 Austin quali` writes from any session's telemetry
(handy before a race: the first laps then don't need to be waited for).

Progress of a car = the nearest reference point to its latest position, searched near where it was last time so
parallel stretches of track can't be confused. Without a reference (or telemetry) `timing_progress` estimates it as
the time since the car's last lap crossing divided by its pace.

Usage (from the repo root):
    python -m live.race.track 2026 Austin quali           # write data/2026/Austin/track.csv
"""
import argparse
import numpy as np
import pandas as pd
from f1live.events import prepare
from f1live.paths import add_session_args, session_from_args

FLYING = ("racing", "push", "other")     # lap kinds of a lap run at speed (race laps, qualifying/practice flying laps)
N_POINTS = 600
MIN_POINTS = 40         # distinct positions needed on the reference lap (the feed gives about 70 per lap)
MAX_CLOSING_GAP = 0.05  # start and end of the reference lap must be within this fraction of its length
OFF_TRACK = 150         # units (1/10 m); beyond this a car is not near where it was, e.g. in the pit lane: search everywhere
AHEAD, BEHIND = 0.25, 0.03     # search window around the last progress, in laps


def resample(xy, n=N_POINTS):
    """Closed polyline through the points, resampled to n points of equal distance. -> (n, 2) array."""
    xy = np.asarray(xy, float)
    keep = np.r_[True, (np.diff(xy, axis=0) != 0).any(axis=1)]          # drop repeated samples
    xy = xy[keep]
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    grid = np.linspace(0, d[-1], n, endpoint=False)
    return np.c_[np.interp(grid, d, xy[:, 0]), np.interp(grid, d, xy[:, 1])]


def build_reference(telemetry, laps):
    """Reference polyline from one clean lap, or None if telemetry/laps don't contain one yet.

    telemetry: ts, driver, x, y (prepared); laps: lap rows of events.csv (event == 'lap')."""
    tel = telemetry.dropna(subset=["x", "y"])
    if tel.empty:
        return None
    laps = laps[laps["event"] == "lap"].sort_values(["driver", "ts"])
    for drv, g in laps.groupby("driver"):
        best = g["lap_time_s"].min()
        g = g.reset_index(drop=True)
        for i in range(1, len(g)):
            a, b = g.iloc[i - 1], g.iloc[i]
            same_stops = (pd.isna(a["n_pit_stops"]) and pd.isna(b["n_pit_stops"])) or a["n_pit_stops"] == b["n_pit_stops"]
            if not (b["lap_kind"] in FLYING and a["lap_kind"] in FLYING + ("out",) and b["track_status"] == "AllClear"
                    and a["track_status"] == "AllClear" and same_stops and b["lap_time_s"] < 1.05 * best
                    and b["lap_number"] >= 3):
                continue
            s = tel[(tel["driver"] == drv) & (tel["ts"] >= a["ts"]) & (tel["ts"] <= b["ts"])]
            xy = s[["x", "y"]].to_numpy(float)
            if len(xy) < MIN_POINTS or len(np.unique(xy, axis=0)) < MIN_POINTS:
                continue
            length = np.hypot(*np.diff(xy, axis=0).T).sum()
            if np.hypot(*(xy[-1] - xy[0])) > MAX_CLOSING_GAP * length:    # didn't cover the whole lap
                continue
            return resample(xy)
    return None


class TrackProgress:
    """Fractional lap progress (0-1) of each car from its position on the reference lap."""
    def __init__(self, ref):
        self.ref, self.last = np.asarray(ref, float), {}

    def progress(self, driver, x, y):
        n = len(self.ref)
        dist = np.hypot(self.ref[:, 0] - x, self.ref[:, 1] - y)
        prev = self.last.get(driver)
        if prev is not None:
            rel = (np.arange(n) - prev) % n
            near = np.where((rel <= AHEAD * n) | (rel >= n - BEHIND * n), dist, np.inf)
            if near.min() <= OFF_TRACK:
                dist = near
        i = int(dist.argmin())
        self.last[driver] = i
        return i / n


def timing_progress(lat, now, pace):
    """Fallback: {driver: progress} = time since the last lap crossing / pace (or last lap time), clipped to [0, 1)."""
    out = {}
    for drv, r in lat.iterrows():
        ts = r.get("last_lap_ts")
        t = pace.get(drv) if pace is not None else None
        if pd.isna(ts) or t is None or pd.isna(t):
            continue
        out[drv] = float(np.clip((now - ts).total_seconds() / t, 0, 0.999))
    return out


def save_reference(ref, path):
    pd.DataFrame(ref, columns=["x", "y"]).to_csv(path, index=False)


def load_reference(path):
    try:
        df = pd.read_csv(path)
        return df[["x", "y"]].to_numpy(float) if len(df) >= MIN_POINTS else None
    except (FileNotFoundError, KeyError, pd.errors.EmptyDataError):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_session_args(ap)
    a = ap.parse_args()
    sess = session_from_args(a)
    tel = pd.read_csv(sess.telemetry)
    tel["ts"] = pd.to_datetime(tel["ts"], format="ISO8601")
    tel["driver"] = tel["driver"].astype(int).astype(str)
    ref = build_reference(tel, prepare(pd.read_csv(sess.events)))
    if ref is None:
        raise SystemExit(f"no clean lap with telemetry found in {sess}")
    out = sess.dir.parent / "track.csv"
    save_reference(ref, out)
    length = np.hypot(*np.diff(np.vstack([ref, ref[:1]]), axis=0).T).sum() / 10
    print(f"wrote {out}  ({len(ref)} points, lap length about {length:.0f} m)")


if __name__ == "__main__":
    main()
