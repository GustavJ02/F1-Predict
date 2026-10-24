"""Replay a recorded raw file into a new file in (scaled) real time, so the whole pipeline
can be tested as if a session were live.

Usage (from the repo root; src is year/race/session, dst is the session to write):
    python -m f1live.replay 2026/Austin/quali 2026/Austin/replay              # real speed
    python -m f1live.replay 2026/Austin/quali 2026/Austin/replay --speed 20   # 20x faster
    python -m f1live.replay 2026/Austin/race 2026/Austin/replay --start 2026-10-25T19:40:00Z
        (write everything before --start instantly, then replay from there)

Then run the parser / runner on the dst session, e.g.  python -m f1live.parse 2026 Austin replay
"""
import argparse, time
from f1live.feed import parse_line, iso
from f1live.paths import Session


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="year/race/session to read")
    ap.add_argument("dst", help="year/race/session to write (must differ from src)")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--start", help="fast-forward to this UTC time before replaying in real time")
    args = ap.parse_args()
    start = iso(args.start) if args.start else None
    src_path, dst_path = Session.parse(args.src).raw, Session.parse(args.dst).raw
    if src_path.resolve() == dst_path.resolve():
        raise SystemExit("src and dst are the same session; that would erase the recording")
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with open(src_path) as src, open(dst_path, "w") as dst:
        prev = None
        for i, line in enumerate(src):
            parsed = parse_line(line)
            ts = iso(parsed[2]) if parsed and parsed[2] else None
            if ts and (start is None or ts >= start):
                if prev is not None:
                    wait = (ts - prev).total_seconds() / args.speed
                    if wait > 0:
                        time.sleep(min(wait, 60))   # cap long gaps (e.g. between segments)
                prev = ts
            dst.write(line)
            dst.flush()
            if ts and i % 200 == 0:
                print(f"replaying {ts.isoformat()}Z", end="\r")
    print("\nreplay finished")


if __name__ == "__main__":
    main()
