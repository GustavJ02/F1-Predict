"""Download every (or selected) session of a season from the archive into data/<year>/<race>/<session>/ and parse it.

Usage (from the repo root):
    python -m f1live.fetch_season 2025                                  # everything
    python -m f1live.fetch_season 2025 --sessions fp1 fp2 fp3 sq sprint quali race
    python -m f1live.fetch_season 2026 --sessions fp1 fp2 fp3 sq sprint --alias "United States=Austin"
    python -m f1live.fetch_season 2025 --only Monaco Italian --dry-run
    python -m f1live.fetch_season 2025 --telemetry      # also CarData/Position -> raw_telemetry.txt -> telemetry.csv

Folders: <race> is the meeting name without "Grand Prix", spaces as underscores (Australian, Saudi_Arabian, United_States);
<session> is fp1 fp2 fp3 sq sprint quali race. Sessions that already have a raw.txt are skipped, and
sessions without events.csv are (re-)parsed, so an interrupted run can simply be started again.
"""
import argparse, subprocess, sys, time
from f1live.archive import BASE, download, download_telemetry, get_json
from f1live.paths import DATA, Session, folder_name

FOLDERS = {"Practice 1": "fp1", "Practice 2": "fp2", "Practice 3": "fp3", "Sprint Qualifying": "sq",
           "Sprint": "sprint", "Qualifying": "quali", "Race": "race"}


def plan(year, sessions, only, alias, data_dir):
    """-> [(Session, archive path)] for sessions that exist in the archive and match the filters."""
    out = []
    for mt in get_json(f"{BASE}{year}/Index.json")["Meetings"]:
        race = folder_name(mt["Name"].removesuffix(" Grand Prix"))
        if "Testing" in mt["Name"] or (only and race not in only):
            continue
        race = alias.get(race, race)
        for s in mt["Sessions"]:
            folder = FOLDERS.get(s["Name"])
            if folder in sessions and s.get("Path"):
                out.append((Session(str(year), race, folder, data_dir), s["Path"]))
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
                    help="use an existing folder name for a race, e.g. 'United States=Austin'")
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
