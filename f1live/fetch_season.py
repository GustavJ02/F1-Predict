"""Download every (or selected) session of a season from the archive into data/<year>/<race>/<session>/ and parse it.

Usage (from the repo root):
    python -m f1live.fetch_season 2025                                  # everything
    python -m f1live.fetch_season 2025 --sessions fp1 fp2 fp3 sq sprint quali race
    python -m f1live.fetch_season 2026 --sessions fp1 fp2 fp3 sq sprint     # the US GP goes to Austin, see default_alias
    python -m f1live.fetch_season 2025 --only Monaco Italian --dry-run
    python -m f1live.fetch_season 2025 --telemetry      # also CarData/Position -> raw_telemetry.txt -> telemetry.csv

Folders: <race> is the meeting name without "Grand Prix", spaces as underscores (Australian, Saudi_Arabian), except the renames in
default_alias (United_States -> Austin, Mexican -> Mexico_City, Brazilian -> Sao_Paulo, Spanish -> Barcelona before 2026);
<session> is fp1 fp2 fp3 sq sprint quali race. Sessions that already have a raw.txt are skipped, and
sessions without events.csv are (re-)parsed, so an interrupted run can simply be started again. Races the archive's index leaves out
are found through FastF1's schedule (unlisted_meetings).
"""
import argparse, subprocess, sys, time
import pandas as pd
from f1live.archive import BASE, download, download_telemetry, get_json
from f1live.paths import DATA, Session, folder_name

FOLDERS = {"Practice 1": "fp1", "Practice 2": "fp2", "Practice 3": "fp3", "Sprint Qualifying": "sq", "Sprint Shootout": "sq",
           "Sprint": "sprint", "Qualifying": "quali", "Race": "race"}


def default_alias(year):
    """Folder names the data uses for races whose meeting name changed or collides: the US GP is Austin, the Mexican and Brazilian GPs are
    Mexico_City and Sao_Paulo, and the Spanish GP was Barcelona until Madrid took the name in 2026."""
    alias = {"United_States": "Austin", "Mexican": "Mexico_City", "Brazilian": "Sao_Paulo"}
    if int(year) < 2026:
        alias["Spanish"] = "Barcelona"
    return alias


def unlisted_meetings(year, listed):
    """[(meeting name, [(session name, archive path)])] for the season's races that the archive's Index.json leaves out (the 2024 index
    starts at the Spanish GP) but whose files exist: the path is <year>/<race date>_<Event_Name>/<session date>_<Session_Name>/, built from
    FastF1's schedule and checked with a HEAD request. Empty if the schedule can't be fetched."""
    import fastf1
    from f1live.archive import HTTP
    try:
        schedule = fastf1.get_event_schedule(int(year), include_testing=False)
    except Exception as e:                       # offline, API change: only what the index lists
        print(f"no schedule for {year}: {e}")
        return []
    out = []
    for _, ev in schedule.iterrows():
        if ev["EventName"] in listed:
            continue
        found = []
        for i in range(1, 6):
            name, day = ev.get(f"Session{i}"), ev.get(f"Session{i}Date")
            if not isinstance(name, str) or pd.isna(day):
                continue
            path = f"{year}/{ev['EventDate']:%Y-%m-%d}_{ev['EventName'].replace(' ', '_')}/{day:%Y-%m-%d}_{name.replace(' ', '_')}/"
            if HTTP.head(f"{BASE}{path}SessionInfo.jsonStream", timeout=30).status_code == 200:
                found.append((name, path))
        if found:
            out.append((ev["EventName"], found))
    return out


def plan(year, sessions, only, alias, data_dir):
    """-> [(Session, archive path)] for sessions that exist in the archive and match the filters. `alias` adds to / overrides default_alias."""
    alias = {**default_alias(year), **alias}
    meetings = [(mt["Name"], [(s.get("Name"), s.get("Path")) for s in mt["Sessions"]])
                for mt in get_json(f"{BASE}{year}/Index.json")["Meetings"]]
    meetings += unlisted_meetings(year, {name for name, _ in meetings})
    out = []
    for name, listed in meetings:
        race = folder_name(name.removesuffix(" Grand Prix"))
        if "Testing" in name or (only and race not in only):
            continue
        race = alias.get(race, race)
        for session, path in listed:
            folder = FOLDERS.get(session)               # some indexes have unnamed entries
            if folder in sessions and path:
                out.append((Session(str(year), race, folder, data_dir), path))
    return out


def fetch_one(sess, path, retries=3):
    for attempt in range(1, retries + 1):
        try:
            download(path, sess.raw, verbose=False)
            return
        except Exception as e:
            print(f"  attempt {attempt} failed: {type(e).__name__}: {e}", flush=True)
            time.sleep(5 * attempt)
    raise RuntimeError(f"giving up on {sess}")


def fetch_telemetry(sess, path, retries=3):
    for attempt in range(1, retries + 1):
        try:
            download_telemetry(path, sess.raw_telemetry, verbose=False)
            return
        except Exception as e:
            print(f"  telemetry attempt {attempt} failed: {type(e).__name__}: {e}", flush=True)
            time.sleep(5 * attempt)
    raise RuntimeError(f"giving up on telemetry of {sess}")


def parse_telemetry(sess, every):
    subprocess.run([sys.executable, "-m", "f1live.telemetry", sess.year, sess.race, sess.name, "--replay",
                    "--every", str(every), "--data-dir", str(sess.data_dir)],
                   check=True)


def parse(sess):
    subprocess.run([sys.executable, "-m", "f1live.parse", sess.year, sess.race, sess.name, "--replay", "--quiet",
                    "--data-dir", str(sess.data_dir)],
                   check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("year", type=int)
    ap.add_argument("--sessions", nargs="+", default=list(FOLDERS.values()), choices=list(FOLDERS.values()))
    ap.add_argument("--only", nargs="+", help="race folder names to include (default: all)")
    ap.add_argument("--alias", action="append", default=[], metavar="RACE=FOLDER",
                    help="use another folder name for a race, e.g. 'Styrian=Austrian' (adds to default_alias)")
    ap.add_argument("--data-dir", default=DATA, type=type(DATA))
    ap.add_argument("--telemetry", action="store_true",
                    help="also fetch CarData/Position (raw_telemetry.txt) and parse it to telemetry.csv; "
                         "sessions that already have raw.txt only get the telemetry")
    ap.add_argument("--tel-every", type=float, default=1.0,
                    help="seconds between telemetry rows per driver (0 = every sample, ~3.7 Hz; ~16 MB per quali)")
    ap.add_argument("--skip", nargs="+", default=[], metavar="RACE/SESSION",
                    help="sessions to leave alone, e.g. Austin/race (a session being recorded live)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    alias = {folder_name(k): v for k, v in (x.split("=", 1) for x in a.alias)}

    todo = plan(a.year, a.sessions, {folder_name(x) for x in a.only or []}, alias, a.data_dir)
    skip = {x.lower() for x in a.skip}
    todo = [(s, p) for s, p in todo if f"{s.race}/{s.name}".lower() not in skip]
    failed = []
    for i, (sess, path) in enumerate(todo, 1):
        have_raw, have_events = sess.raw.exists(), sess.events.exists()
        status = "done" if have_raw and have_events else "parse" if have_raw else "fetch"
        print(f"[{i}/{len(todo)}] {sess}: {status}", flush=True)
        need_tel = a.telemetry and not sess.telemetry.exists()
        if need_tel:
            print("  + telemetry", flush=True)
        if a.dry_run or (status == "done" and not need_tel):
            continue
        try:
            if status == "fetch":
                fetch_one(sess, path)
            if status != "done":
                parse(sess)
            if need_tel:
                if not sess.raw_telemetry.exists():
                    fetch_telemetry(sess, path)
                parse_telemetry(sess, a.tel_every)
        except Exception as e:
            print(f"  FAILED {sess}: {e}", flush=True)
            failed.append(str(sess))
    print(f"\n{len(todo) - len(failed)}/{len(todo)} sessions ok" + (f"; failed: {', '.join(failed)}" if failed else ""))


if __name__ == "__main__":
    main()
