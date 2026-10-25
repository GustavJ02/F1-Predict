"""Parse car telemetry (CarData + Position) from a recording into telemetry.csv.

Usage (from the repo root):
    python -m f1live.telemetry 2026 Austin quali            # tail live: raw.txt -> telemetry.csv
    python -m f1live.telemetry 2026 Austin quali --replay   # parse a finished recording and exit
    python -m f1live.telemetry 2026 Austin quali --every 0  # keep every sample (~3.7 Hz per car)

One row per driver per --every seconds (default 1), stamped with the sample's own UTC time:
    ts, driver, speed (km/h), rpm, gear, throttle (0-100), brake (0 or 100), x, y, z, pos_status
x/y/z are the latest Position sample held for that car. throttle/brake are empty when the feed
reports its 104 "no data" marker (parked cars); x/y/z are empty while the feed has no fix (0,0,0). The 2026 feed has no DRS channel.

Reads raw_telemetry.txt (from `f1live.archive --telemetry`) if it exists, otherwise raw.txt.
Kept apart from parse.py so events.csv stays small and unchanged.
"""
import argparse, csv
from f1live.feed import decode, follow, iso, parse_line
from f1live.paths import add_session_args, session_from_args

FIELDS = ["ts", "driver", "speed", "rpm", "gear", "throttle", "brake", "x", "y", "z", "pos_status"]
CHANNELS = {"0": "rpm", "2": "speed", "3": "gear", "4": "throttle", "5": "brake"}   # channel 45 = DRS (not sent in 2026)
NO_DATA = 100          # throttle/brake above this are the 104 marker, not a pedal position


class TelemetryParser:
    def __init__(self, writer, every=1.0):
        self.w = writer
        self.every = every
        self.pos = {}              # driver -> latest Position entry
        self.last = {}             # driver -> time of the last emitted row

    def feed(self, line):
        parsed = parse_line(line)
        if not parsed or parsed[0] not in ("CarData", "Position"):
            return
        topic, data, _ = parsed        # line ts is arrival time (empty in archive files); samples carry their own
        try:
            data = decode(data)
        except Exception:
            return                     # a corrupt payload costs a few samples, not the run
        if not isinstance(data, dict):
            return
        if topic == "Position":
            for sample in data.get("Position", []):
                for drv, e in sample.get("Entries", {}).items():
                    self.pos[drv] = e
        else:
            for sample in data.get("Entries", []):
                self.car_sample(sample)

    def car_sample(self, sample):
        utc = sample["Utc"]
        t = iso(utc)
        for drv, car in sample.get("Cars", {}).items():
            prev = self.last.get(drv)
            if prev is not None and (t - prev).total_seconds() < self.every:
                continue
            self.last[drv] = t
            ch = car.get("Channels", {})
            r = dict.fromkeys(FIELDS)
            r.update(ts=utc, driver=drv)
            for k, name in CHANNELS.items():
                r[name] = ch.get(k)
            for name in ("throttle", "brake"):
                if r[name] is not None and r[name] > NO_DATA:
                    r[name] = None
            p = self.pos.get(drv)
            if p and (p.get("X"), p.get("Y"), p.get("Z")) != (0, 0, 0):    # 0,0,0 = no fix yet, not a place on track
                r.update(x=p.get("X"), y=p.get("Y"), z=p.get("Z"), pos_status=p.get("Status"))
            self.w(r)


def main():
    ap = argparse.ArgumentParser()
    add_session_args(ap)
    ap.add_argument("--replay", action="store_true", help="parse to end of file and exit")
    ap.add_argument("--every", type=float, default=1.0, help="seconds between rows per driver (0 = every sample)")
    args = ap.parse_args()
    sess = session_from_args(args)
    sess.dir.mkdir(parents=True, exist_ok=True)

    # --replay builds into a temp file, so an interrupted run never leaves a partial telemetry.csv that looks finished
    dest = sess.telemetry.with_suffix(".tmp") if args.replay else sess.telemetry
    out = open(dest, "w", newline="")
    w = csv.DictWriter(out, fieldnames=FIELDS)
    w.writeheader()

    state = {"n": 0}

    def write(r):
        w.writerow(r)
        state["n"] += 1
        if args.replay and state["n"] % 5000:
            return
        out.flush()            # flush so the runner can tail the file live

    p = TelemetryParser(write, args.every)
    if sess.raw_telemetry.exists():
        src, live = sess.raw_telemetry, False       # a finished archive download
    else:
        src, live = sess.raw, not args.replay
    for line in follow(src, live=live):
        p.feed(line)
    out.close()
    if args.replay:
        dest.replace(sess.telemetry)


if __name__ == "__main__":
    main()
