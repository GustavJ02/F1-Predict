"""Past sessions as pace tables, for the pre-session predictions (pre.quali, pre.race).

Everything is read from data/<year>/<race>/<session>/events.csv (fetch with python -m f1live.fetch_season).
Session folders: fp1 fp2 fp3 sq sprint quali race. Race folders are the meeting name without "Grand Prix", spaces as underscores.
"""
import pandas as pd
from f1live.events import load
from f1live.paths import DATA, folder_name

SESSIONS = ["fp1", "fp2", "fp3", "sq", "sprint", "quali", "race"]    # weekend order for a sprint-less weekend: fp1 fp2 fp3 quali race
FLYING = ["push", "other"]                                           # lap_kind of laps that are neither in- nor out-laps

# Folders that changed name between years / recordings (calendar order matters, so everything keys on race folder)
ALIASES = {"Austin": "United_States"}


def race_key(race):
    race = folder_name(race)
    return ALIASES.get(race, race)


def _first_ts(path):
    """Timestamp of the first event of a session, or None if the file has no events (yet)."""
    try:
        ts = pd.read_csv(path, usecols=["ts"], nrows=1)["ts"]
    except (pd.errors.EmptyDataError, ValueError):
        return None
    return ts.iloc[0] if len(ts) else None


def available(data_dir=DATA):
    """DataFrame (year, race, session, path, start) of every session that has events in its events.csv, in calendar order.
    A session that is being recorded right now and has no events yet is left out."""
    rows = []
    for ev in sorted(data_dir.glob("*/*/*/events.csv")):
        sess, race, year = ev.parent.name, ev.parent.parent.name, ev.parent.parent.parent.name
        if year.isdigit() and sess in SESSIONS:
            start = _first_ts(ev)
            if start is not None:
                rows.append(dict(year=int(year), race=race_key(race), session=sess, path=ev, start=start))
    df = pd.DataFrame(rows, columns=["year", "race", "session", "path", "start"])
    return df.sort_values("start").reset_index(drop=True) if len(df) else df


def valid_laps(path):
    """Flying laps of one session (no in/out laps, deleted laps removed): driver, tla, team, lap_time_s, ..."""
    return valid_laps_from(load(path))


def valid_laps_from(ev):
    """valid_laps for events already in memory (e.g. a session that is still being recorded)."""
    laps = ev[(ev["event"] == "lap") & ev["lap_kind"].isin(FLYING) & ev["lap_time_s"].notna()]
    deleted = ev[ev["event"] == "lap_deleted"][["driver", "lap_time_s"]].assign(deleted=True)
    laps = laps.merge(deleted, on=["driver", "lap_time_s"], how="left")
    laps = laps[laps["deleted"].isna()].drop(columns="deleted")
    laps = laps[laps["tla"].notna()]            # a recording without a DriverList (2021 Bahrain FP1) has no driver codes to key on
    keep = ["ts", "session_part", "driver", "tla", "team", "lap_number", "lap_time_s", "lap_kind", "compound",
            "tyre_new", "tyre_laps", "track_temp", "air_temp"]
    return laps[keep].reset_index(drop=True)


def pace_table(path):
    """One row per driver: best valid lap, its gap to the session's fastest in % (gap_pct), laps, best per compound.

    In qualifying all parts count, so a driver's best is their final grid time (gap_pct 0 = pole)."""
    laps = valid_laps(path)
    if laps.empty:
        return pd.DataFrame()
    g = laps.groupby("driver")
    t = pd.DataFrame({"tla": g["tla"].last(), "team": g["team"].last(),
                      "best_s": g["lap_time_s"].min(), "n_laps": g.size()})
    for comp in ("SOFT", "MEDIUM", "HARD"):
        t[f"best_{comp.lower()}_s"] = laps[laps["compound"] == comp].groupby("driver")["lap_time_s"].min()
    fastest = t["best_s"].min()
    t["gap_pct"] = (t["best_s"] / fastest - 1) * 100
    t["rank"] = t["best_s"].rank(method="min").astype(int)
    return t.sort_values("best_s")


def weekend(race, year, data_dir=DATA):
    """{session: pace_table} for every session of that weekend we have data for."""
    out = {}
    for s in available(data_dir).query("year == @year and race == @race").itertuples():
        t = pace_table(s.path)
        if not t.empty:
            out[s.session] = t
    return out
