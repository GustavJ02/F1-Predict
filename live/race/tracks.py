"""Track layouts, corners and timing sectors for every circuit of a season's calendar, written next to the sessions:

    data/<year>/<race>/track.csv     reference lap polyline (x, y in 1/10 m), start/finish at the first point (live.race.track)
    data/<year>/<race>/corners.csv   corner number -> lap fraction (FastF1 circuit info; needs network)
    data/<year>/<race>/sectors.csv   the three timing sectors: where they start and end (lap fraction and metres along the lap)

The layout is built from the car telemetry of a session at that circuit: this year's if the race has been run, otherwise last
year's (so the whole calendar is covered before the season ends). Existing track.csv files are kept (the dashboard already uses
them); pass --force to rebuild. Sector boundaries are found where the cars actually were when they completed S1 and S2: for
each clean flying lap the position at (lap end - S2 - S3) and (lap end - S3) is projected onto the reference lap and the median
over the laps is taken. They are as accurate as the telemetry (about 1 Hz, so tens of metres) and the timing clock, not exact.
Distances along the lap are measured on the resampled polyline, so lap lengths are a little short of the official figure.

Usage (from the repo root):
    python -m live.race.tracks                      # 2026 calendar
    python -m live.race.tracks --year 2026 --only Austin Monaco --force
    python -m live.race.tracks --dry-run            # which session each circuit would be built from
"""
import argparse
import numpy as np
import pandas as pd
from f1live.events import load
from f1live.paths import DATA, Session, folder_name
from live.race import track as T
from live.race.priors import circuit_of

PREFER = ["quali", "race", "sq", "sprint", "fp3", "fp2", "fp1"]          # sessions to read telemetry from, best first
CLEAN_SLACK = 1.05             # laps within this factor of the session's best count as clean flying laps
SECTOR_TOL = 0.1               # s: s1 + s2 + s3 must match the lap time
SEARCH = 0.12                  # a boundary is looked for within this fraction of the lap around where the sector times put it
MAX_GAP_S = 3.0                # telemetry samples further apart than this around a boundary are not interpolated over
MIN_LAPS = 5
COLUMNS = ["sector", "start_frac", "end_frac", "start_m", "end_m", "length_m", "n_laps", "spread_m", "source"]


def event_folder(event_name, location, country, existing):
    """The race folder for a calendar entry: an existing folder that matches, else the event name without 'Grand Prix'."""
    stripped = folder_name(event_name.replace("Grand Prix", "").strip())
    for cand in (stripped, folder_name(location), folder_name(country)):
        if cand in existing:
            return cand
    return stripped


def calendar(year, data_dir=DATA):
    """[(round, race folder)] for the season's calendar (needs network or FastF1's cache)."""
    import fastf1
    existing = {p.name for p in (data_dir / str(year)).glob("*") if p.is_dir()}
    sched = fastf1.get_event_schedule(int(year), include_testing=False)
    return [(int(r["RoundNumber"]), event_folder(str(r["EventName"]), str(r["Location"]), str(r["Country"]), existing))
            for _, r in sched.iterrows()]


def usable(sess):
    """Does the session have events and telemetry we can build from?"""
    try:
        return sess.events.stat().st_size > 0 and sess.telemetry.stat().st_size > 0
    except FileNotFoundError:
        return False


def source_session(year, folder, data_dir=DATA):
    """The session to build the circuit from: this year's folder if it has one with telemetry, else the same circuit's folder
    in earlier years (newest first). None if there is nothing."""
    for y in range(int(year), int(year) - 3, -1):
        names = [folder] if y == int(year) else sorted(
            p.name for p in (data_dir / str(y)).glob("*") if p.is_dir() and circuit_of(y, p.name) == circuit_of(year, folder))
        for name in names:
            for s in PREFER:
                sess = Session(str(y), name, s, data_dir)
                if usable(sess):
                    return sess
    return None


def read_telemetry(sess):
    tel = pd.read_csv(sess.telemetry, usecols=["ts", "driver", "x", "y"])
    tel["ts"] = pd.to_datetime(tel["ts"], format="ISO8601")
    tel["driver"] = tel["driver"].astype(int).astype(str)
    return tel.dropna(subset=["x", "y"])


def lap_length_m(ref):
    ref = np.asarray(ref, float)
    return float(np.hypot(*np.diff(np.vstack([ref, ref[:1]]), axis=0).T).sum() / 10)


def clean_sector_laps(events):
    """Lap rows with all three sector times that add up to the lap time, run at speed under green flags and close to the best."""
    laps = events[(events["event"] == "lap") & events["lap_kind"].isin(T.FLYING) & (events["track_status"] == "AllClear")]
    laps = laps.dropna(subset=["s1_s", "s2_s", "s3_s", "lap_time_s"])
    laps = laps[(laps[["s1_s", "s2_s", "s3_s"]].sum(axis=1) - laps["lap_time_s"]).abs() <= SECTOR_TOL]
    return laps[laps["lap_time_s"] <= CLEAN_SLACK * laps["lap_time_s"].min()]


def sector_boundaries(ref, tel, events):
    """-> (fractions [end of S1, end of S2], per-lap fractions array (n, 2)) from where cars were when they completed each sector,
    or None with fewer than MIN_LAPS usable laps."""
    ref = np.asarray(ref, float)
    n = len(ref)
    laps = clean_sector_laps(events)
    per_driver = {d: (g["ts"].astype("int64").to_numpy() / 1e9, g["x"].to_numpy(float), g["y"].to_numpy(float))
                  for d, g in tel.groupby("driver")}
    found = []
    for r in laps.itertuples():
        if r.driver not in per_driver:
            continue
        t, x, y = per_driver[r.driver]
        end = pd.Timestamp(r.ts).value / 1e9
        total = r.s1_s + r.s2_s + r.s3_s
        marks = []
        for when, share in ((end - r.s2_s - r.s3_s, r.s1_s / total), (end - r.s3_s, (r.s1_s + r.s2_s) / total)):
            i = np.searchsorted(t, when)
            if i == 0 or i == len(t) or t[i] - t[i - 1] > MAX_GAP_S:
                break
            px, py = np.interp(when, t, x), np.interp(when, t, y)
            idx = np.arange(n)
            near = np.abs((idx - share * n + n / 2) % n - n / 2) <= SEARCH * n
            dist = np.where(near, np.hypot(ref[:, 0] - px, ref[:, 1] - py), np.inf)
            j = int(dist.argmin())
            marks.append(((j - share * n + n / 2) % n - n / 2 + share * n) / n)       # unwrapped next to the time share
        if len(marks) == 2:
            found.append(marks)
    if len(found) < MIN_LAPS:
        return None
    found = np.array(found)
    return [float(np.median(found[:, 0]) % 1), float(np.median(found[:, 1]) % 1)], found


def sectors_table(ref, bounds, per_lap, source):
    """The sectors.csv rows from boundary fractions."""
    ref = np.asarray(ref, float)
    n = len(ref)
    seg = np.hypot(*np.diff(np.vstack([ref, ref[:1]]), axis=0).T) / 10
    dist = np.r_[0, np.cumsum(seg)]                                       # metres along the lap at each reference point
    total = float(dist[-1])
    edges = [0.0, *bounds, 1.0]
    rows = []
    for k in range(3):
        spread = 0.0
        if k < 2:
            q = np.percentile(per_lap[:, k], [25, 75])
            spread = float((q[1] - q[0]) * total)
        rows.append(dict(sector=k + 1, start_frac=edges[k], end_frac=edges[k + 1], start_m=edges[k] * total,
                         end_m=edges[k + 1] * total, length_m=(edges[k + 1] - edges[k]) * total, n_laps=len(per_lap),
                         spread_m=spread, source=source))
    return pd.DataFrame(rows, columns=COLUMNS).round(4)


def build_circuit(year, folder, data_dir=DATA, force=False, network=True, dry_run=False):
    """Write track.csv, corners.csv and sectors.csv for one race folder. -> a one-line report."""
    src = source_session(year, folder, data_dir)
    if src is None:
        return f"{folder}: no session with telemetry, skipped"
    race_dir = data_dir / str(year) / folder
    label = f"{src.year}/{src.race}/{src.name}"
    if dry_run:
        return f"{folder}: would build from {label}"
    race_dir.mkdir(parents=True, exist_ok=True)
    events, tel = load(src.events), read_telemetry(src)
    ref = T.load_reference(race_dir / "track.csv") if not force else None
    made = "kept"
    if ref is None:
        ref = T.build_reference(tel, events)
        if ref is None:
            return f"{folder}: no clean lap with telemetry in {label}, skipped"
        T.save_reference(ref, race_dir / "track.csv")
        made = "built"
    out = f"{folder}: track {made} ({lap_length_m(ref):.0f} m, from {label})"
    if network and (force or not (race_dir / "corners.csv").exists()):
        corners = T.fetch_corners(src.year, src.race)
        if corners is not None:
            T.save_corners(T.corner_fractions(ref, corners), race_dir / "corners.csv")
            out += f", {len(corners)} corners"
    found = sector_boundaries(ref, tel, events)
    if found is None:
        return out + ", too few clean laps with sector times for sectors"
    bounds, per_lap = found
    table = sectors_table(ref, bounds, per_lap, label)
    table.to_csv(race_dir / "sectors.csv", index=False)
    lengths = ", ".join(f"{m:.0f}" for m in table["length_m"])
    return out + f", sectors {lengths} m (±{table['spread_m'].max():.0f} m, {len(per_lap)} laps)"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--only", nargs="+", help="race folders to do (default: the whole calendar)")
    ap.add_argument("--force", action="store_true", help="rebuild track.csv and corners.csv that already exist")
    ap.add_argument("--no-network", action="store_true", help="don't fetch corners")
    ap.add_argument("--dry-run", action="store_true", help="only say which session each circuit would be built from")
    ap.add_argument("--data-dir", help=f"data directory (default {DATA})")
    a = ap.parse_args()
    from pathlib import Path
    data_dir = Path(a.data_dir) if a.data_dir else DATA
    races = calendar(a.year, data_dir)
    only = {folder_name(r) for r in a.only} if a.only else None
    for rnd, folder in races:
        if only is None or folder in only:
            print(f"{rnd:2d} " + build_circuit(a.year, folder, data_dir, a.force, not a.no_network, a.dry_run), flush=True)


if __name__ == "__main__":
    main()
