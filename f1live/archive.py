"""Download a finished session from F1's live-timing archive and write it in the same raw format
the FastF1 live client records, so it can be parsed and replayed like a live recording.
The archive requires an F1TV login now (401 otherwise): requests go through fastf1's auth adapter, which
sends the token from `python -m fastf1 auth f1tv` to livetiming.formula1.com only.

Usage (from the repo root); writes data/<year>/<race>/<session>/raw.txt:
    python -m f1live.archive --list 2026                          # list meetings and sessions
    python -m f1live.archive 2026 Singapore race                  # meeting search text + our session name
    python -m f1live.archive 2026 Singapore quali --archive-name Qualifying
    python -m f1live.archive 2026 Austin quali --path "2026/2026-10-25_United_States_Grand_Prix/2026-10-24_Qualifying/"
    python -m f1live.archive 2026 Singapore quali --telemetry     # car telemetry -> raw_telemetry.txt (raw.txt untouched)

Uses the archive layout FastF1 uses (Index.json + <Topic>.jsonStream).
"""
import argparse, json, sys
from datetime import timedelta
import requests
from fastf1.internals.f1account import LIVETIMING_URL, LivetimingAuthAdapter, token_manager
from f1live.feed import iso
from f1live.paths import add_session_args, session_from_args

BASE = "https://livetiming.formula1.com/static/"
HTTP = requests.Session()
HTTP.mount(LIVETIMING_URL, LivetimingAuthAdapter(token_manager))
TOPICS = ["SessionInfo", "DriverList", "TimingData", "TimingAppData", "TimingStats", "LapCount",
          "TrackStatus", "RaceControlMessages", "WeatherData", "ExtrapolatedClock", "SessionData",
          "SessionStatus", "Heartbeat", "TopThree", "PitLaneTimeCollection"]
TELEMETRY_TOPICS = ["CarData.z", "Position.z"]      # big (10s of MB for a race): fetched only on request


def get_json(url):
    r = HTTP.get(url, timeout=30)
    r.raise_for_status()
    return json.loads(r.content.decode("utf-8-sig"))


def stream(path, topic):
    """Yield (offset_seconds, payload) from a .jsonStream file. Lines look like 00:01:23.456{...}."""
    r = HTTP.get(f"{BASE}{path}{topic}.jsonStream", timeout=60)
    if r.status_code in (403, 404):      # the archive answers 403 for topics a session doesn't have
        return
    r.raise_for_status()
    for line in r.content.decode("utf-8-sig").splitlines():
        if len(line) < 13:
            continue
        h, m, s = line[:12].split(":")
        yield int(h) * 3600 + int(m) * 60 + float(s), json.loads(line[12:])


def anchor(events, topic):
    """Median stream start time implied by the payloads' own `Utc` field, or None."""
    t0s = sorted(iso(p["Utc"]) - timedelta(seconds=off)
                 for off, t, p in events if t == topic and isinstance(p, dict) and "Utc" in p)
    return t0s[len(t0s) // 2] if t0s else None


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
        ap.add_argument("--telemetry", action="store_true",
                        help="fetch CarData/Position into raw_telemetry.txt instead (raw.txt may already exist)")
        ap.add_argument("--archive-name", help="session name in the archive (default: derived from session)")
        a = ap.parse_args()

    if a.list:
        for mt in get_json(f"{BASE}{a.list}/Index.json")["Meetings"]:
            print(mt.get("Name"), "->", ", ".join(s.get("Name", "?") for s in mt.get("Sessions", [])))
        return

    sess = session_from_args(a)
    path = a.path or find_path(a.year, a.race, a.archive_name or ARCHIVE_NAMES.get(a.session, a.session))
    target = sess.raw_telemetry if a.telemetry else sess.raw
    if target.exists():
        sys.exit(f"{target} already exists; move it away first")
    try:
        (download_telemetry if a.telemetry else download)(path, target)
    except RuntimeError as e:
        sys.exit(str(e))


def download_telemetry(path, out, verbose=True):
    """Fetch CarData/Position of the archive session at `path` into `out`, in the recorder's line format.
    Lines are in stream order with an empty ts: every sample carries its own UTC stamp, so no anchoring is needed."""
    out.parent.mkdir(parents=True, exist_ok=True)
    events = []
    for topic in TELEMETRY_TOPICS:
        n = 0
        for off, payload in stream(path, topic):
            events.append((off, topic, payload))
            n += 1
        if verbose:
            print(f"  {topic}: {n} messages")
    if not events:
        raise RuntimeError(f"no telemetry in the archive for {path}")
    events.sort(key=lambda e: e[0])
    tmp = out.with_suffix(".tmp")
    with open(tmp, "w") as f:
        for off, topic, payload in events:
            f.write(str([topic, payload, ""]) + "\n")
    tmp.replace(out)
    if verbose:
        print(f"wrote {len(events)} messages to {out}")


def download(path, out, verbose=True):
    """Fetch every topic of the archive session at `path` and write it to `out` in the live recorder's format.
    Written to a temp file first, so an interrupted download never leaves a half-written raw.txt behind."""
    out.parent.mkdir(parents=True, exist_ok=True)
    if verbose:
        print("fetching", path)

    events = []
    for topic in TOPICS:
        n = 0
        for off, payload in stream(path, topic):
            events.append((off, topic, payload))
            n += 1
        if verbose:
            print(f"  {topic}: {n} messages")

    # Offsets are relative to the stream start; anchor them with absolute UTC stamps in the payloads.
    # ExtrapolatedClock anchors agree with the live recording to the ms (constant across the session);
    # a single Heartbeat is ~2 s off and later ones drift, so Heartbeat is only the fallback.
    t0 = anchor(events, "ExtrapolatedClock") or anchor(events, "Heartbeat")
    if t0 is None:
        raise RuntimeError("no ExtrapolatedClock/Heartbeat in archive; cannot anchor timestamps")

    events.sort(key=lambda e: e[0])
    first_seen = set()
    tmp = out.with_suffix(".tmp")
    with open(tmp, "w") as f:
        for off, topic, payload in events:
            ts = (t0 + timedelta(seconds=off)).isoformat(timespec="milliseconds") + "Z"
            # The first message of a topic is its full state: write it as a keyframe (ts '') like the live
            # recorder does, so the parser only builds state from it. RaceControlMessages starts with a real message.
            if topic not in first_seen and topic != "RaceControlMessages":
                ts = ""
            first_seen.add(topic)
            f.write(str([topic, payload, ts]) + "\n")
    tmp.replace(out)
    if verbose:
        print(f"wrote {len(events)} messages to {out}")


if __name__ == "__main__":
    main()
