"""Past sessions as pace tables, for the pre-session predictions (pre.quali, pre.race).

Everything is read from data/<year>/<race>/<session>/events.csv (fetch with python -m f1live.fetch_season).
Session folders: fp1 fp2 fp3 sq sprint quali race. Race folders are the meeting name without "Grand Prix".
"""
import pandas as pd
from f1live.events import load
from f1live.paths import DATA

SESSIONS = ["fp1", "fp2", "fp3", "sq", "sprint", "quali", "race"]    # weekend order for a sprint-less weekend: fp1 fp2 fp3 quali race
FLYING = ["push", "other"]                                           # lap_kind of laps that are neither in- nor out-laps

# Folders that changed name between years / recordings (calendar order matters, so everything keys on race folder)
ALIASES = {"Austin": "United States"}


def race_key(race):
    return ALIASES.get(race, race)


def available(data_dir=DATA):
    """DataFrame (year, race, session, path) of every session that has an events.csv, in calendar order."""
    rows = []
    for ev in sorted(data_dir.glob("*/*/*/events.csv")):
        sess, race, year = ev.parent.name, ev.parent.parent.name, ev.parent.parent.parent.name
        if year.isdigit() and sess in SESSIONS:
            rows.append(dict(year=int(year), race=race_key(race), session=sess, path=ev))
    df = pd.DataFrame(rows, columns=["year", "race", "session", "path"])
    if df.empty:
        return df
    df["start"] = [pd.read_csv(p, usecols=["ts"], nrows=1)["ts"].iloc[0] for p in df["path"]]
    return df.sort_values("start").reset_index(drop=True)


def valid_laps(path):
    """Flying laps of one session (no in/out laps, deleted laps removed): driver, tla, team, lap_time_s, ..."""
    ev = load(path)
    laps = ev[(ev["event"] == "lap") & ev["lap_kind"].isin(FLYING) & ev["lap_time_s"].notna()]
    deleted = ev[ev["event"] == "lap_deleted"][["driver", "lap_time_s"]].assign(deleted=True)
    laps = laps.merge(deleted, on=["driver", "lap_time_s"], how="left")
    laps = laps[laps["deleted"].isna()].drop(columns="deleted")
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
