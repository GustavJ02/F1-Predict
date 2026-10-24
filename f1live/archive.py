"""Download a finished session from F1's live-timing archive and write it in the same raw format
the FastF1 live client records, so it can be parsed and replayed like a live recording.
No F1TV login is needed for finished sessions.

Usage (from the repo root); writes data/<year>/<race>/<session>/raw.txt:
    python -m f1live.archive --list 2026                          # list meetings and sessions
    python -m f1live.archive 2026 Singapore race                  # meeting search text + our session name
    python -m f1live.archive 2026 Singapore quali --archive-name Qualifying
    python -m f1live.archive 2026 Austin quali --path "2026/2026-10-25_United_States_Grand_Prix/2026-10-24_Qualifying/"

NOTE: written against the archive layout FastF1 uses (Index.json + <Topic>.jsonStream);
not tested against the live server yet. Validate it by fetching a session you also recorded live.
"""
import argparse, json, sys
from datetime import timedelta
import requests
from f1live.feed import iso
from f1live.paths import add_session_args, session_from_args

BASE = "https://livetiming.formula1.com/static/"
TOPICS = ["SessionInfo", "DriverList", "TimingData", "TimingAppData", "TimingStats", "LapCount",
          "TrackStatus", "RaceControlMessages", "WeatherData", "ExtrapolatedClock", "SessionData",
          "SessionStatus", "Heartbeat", "TopThree", "PitLaneTimeCollection"]


def get_json(url):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return json.loads(r.content.decode("utf-8-sig"))


def stream(path, topic):
    """Yield (offset_seconds, payload) from a .jsonStream file. Lines look like 00:01:23.456{...}."""
    r = requests.get(f"{BASE}{path}{topic}.jsonStream", timeout=60)
    if r.status_code == 404:
        return
    r.raise_for_status()
    for line in r.content.decode("utf-8-sig").splitlines():
        if len(line) < 13:
            continue
        h, m, s = line[:12].split(":")
        yield int(h) * 3600 + int(m) * 60 + float(s), json.loads(line[12:])


def find_path(year, meeting, session):
    idx = get_json(f"{BASE}{year}/Index.json")
    for mt in idx["Meetings"]:
        if meeting.lower() in (mt.get("Name", "") + mt.get("Location", "")).lower():
            for ss in mt["Sessions"]:
                if ss.get("Name", "").lower() == session.lower() and ss.get("Path"):
                    return ss["Path"]
    sys.exit(f"no session '{session}' found for '{meeting}' in {year}")


# our folder name -> the archive's session name
ARCHIVE_NAMES = {"quali": "Qualifying", "race": "Race", "sprint": "Sprint",
                 "sq": "Sprint Qualifying", "fp1": "Practice 1", "fp2": "Practice 2", "fp3": "Practice 3"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", metavar="YEAR", help="list meetings and sessions, then exit")
    if "--list" in sys.argv:
        a = ap.parse_args()
    else:
        add_session_args(ap)
        ap.add_argument("--path", help="archive path of the session (skips the search)")
        ap.add_argument("--archive-name", help="session name in the archive (default: derived from session)")
        a = ap.parse_args()

    if a.list:
        for mt in get_json(f"{BASE}{a.list}/Index.json")["Meetings"]:
            print(mt.get("Name"), "->", ", ".join(s.get("Name", "?") for s in mt.get("Sessions", [])))
        return

    sess = session_from_args(a)
    path = a.path or find_path(a.year, a.race, a.archive_name or ARCHIVE_NAMES.get(a.session, a.session))
    out = sess.raw
    if out.exists():
        sys.exit(f"{out} already exists; move it away first")
    out.parent.mkdir(parents=True, exist_ok=True)
    print("fetching", path)

    events = []
    for topic in TOPICS:
        n = 0
        for off, payload in stream(path, topic):
            events.append((off, topic, payload))
            n += 1
        print(f"  {topic}: {n} messages")

    # Offsets are relative to the stream start; Heartbeat carries absolute UTC to anchor them.
    hb = next(((off, p) for off, t, p in events if t == "Heartbeat" and isinstance(p, dict) and "Utc" in p), None)
    if not hb:
        sys.exit("no Heartbeat in archive; cannot anchor timestamps")
    t0 = iso(hb[1]["Utc"]) - timedelta(seconds=hb[0])

    events.sort(key=lambda e: e[0])
    with open(out, "w") as f:
        for off, topic, payload in events:
            ts = (t0 + timedelta(seconds=off)).isoformat(timespec="milliseconds") + "Z"
            f.write(str([topic, payload, ts]) + "\n")
    print(f"wrote {len(events)} messages to {out}")


if __name__ == "__main__":
    main()
