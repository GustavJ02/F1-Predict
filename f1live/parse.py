"""Parse a FastF1 live-timing recording (race, sprint or qualifying) into events.csv.

Usage (from the repo root):
    python -m f1live.parse 2026 Austin quali                # tail live: raw.txt -> events.csv
    python -m f1live.parse 2026 Austin quali --replay       # parse a finished recording and exit
    python -m f1live.parse 2026 Austin quali --snapshot 5   # + one row per driver every 5 s (0 = off)

Every row is one event (see the `event` column):
    lap                 a completed lap
    pit_in / pit_out    car entered / left the pit lane (pit_out carries pit_lane_s and the new tyre)
    lap_deleted         track-limits deletion from race control
    retired             car retired
    track_status        green / yellow / SC / VSC / red changes
    race_control        any other race control message (penalties, investigations, flags)
    session_part_start  Q2 / Q3 started
    snapshot            periodic per-driver state (only with --snapshot)
Both race and qualifying scripts read this file.
"""
import argparse, csv, re
from f1live.feed import decode, merge, parse_line, follow, val, to_sec, parse_gap, iso
from f1live.paths import add_session_args, session_from_args

FIELDS = [
    "ts", "event", "session_type", "session_part", "clock_left_s", "race_lap", "total_laps",
    "driver", "tla", "team", "position",
    "lap_number", "lap_time_s", "lap_kind", "personal_best", "overall_best",
    "s1_s", "s2_s", "s3_s", "speed_i1", "speed_i2", "speed_fl", "speed_st",
    "gap_to_leader_s", "laps_down", "interval_s", "catching",
    "in_pit", "n_pit_stops", "pit_lane_s", "stint", "compound", "tyre_new", "tyre_laps",
    "best_lap_s", "best_in_part_s", "n_entries", "n_advance",
    "track_status", "status_code", "message",
    "track_temp", "air_temp", "rainfall",
]

# Line Status bits as observed in the feed (inferred, not documented):
# 16 = in pit, 32 = pit out, 64 = running, 512 = on out lap, 1024 = taken chequered flag
OUT_LAP_BIT = 512
DELETED = re.compile(r"CAR (\d+) \(\w+\) TIME ([\d:.]+) DELETED")


class Parser:
    def __init__(self, writer):
        self.S = {}                # merged state per topic
        self.w = writer
        self.drv = {}              # per-driver bookkeeping (out-lap flag, pit entry time)
        self.last_part = None
        self.last_track = None
        self.last_snapshot = None

    # ---------- state helpers ----------
    def session_type(self):
        return self.S.get("SessionInfo", {}).get("Type")

    def is_race(self):
        return self.session_type() in ("Race", "Sprint")

    def clock_left(self, now):
        c = self.S.get("ExtrapolatedClock")
        if not c or "Remaining" not in c:
            return None
        h, m, s = map(int, c["Remaining"].split(":"))
        left = h * 3600 + m * 60 + s
        if c.get("Extrapolating") and now:
            left -= (iso(now) - iso(c["Utc"])).total_seconds()
        return round(max(left, 0), 1)

    def tyre(self, drv):
        stints = val(self.S, "TimingAppData", "Lines", drv, "Stints") or []
        if isinstance(stints, dict):
            stints = [stints[k] for k in sorted(stints, key=int)]
        if not stints:
            return None, None, None, None
        st = stints[-1]
        return len(stints), st.get("Compound"), st.get("New") == "true", st.get("TotalLaps")

    def row(self, ts, event, drv=None, **kw):
        td = self.S.get("TimingData", {})
        part = td.get("SessionPart")
        entries = td.get("NoEntries") or []
        weather = self.S.get("WeatherData", {})
        lc = self.S.get("LapCount", {})
        r = dict.fromkeys(FIELDS)
        r.update(ts=ts, event=event, session_type=self.session_type(), session_part=part,
                 clock_left_s=self.clock_left(ts),
                 race_lap=lc.get("CurrentLap"), total_laps=lc.get("TotalLaps"),
                 n_entries=entries[part - 1] if part and len(entries) >= part else None,
                 n_advance=entries[part] if part and len(entries) > part else None,
                 track_status=self.S.get("TrackStatus", {}).get("Message"),
                 status_code=self.S.get("TrackStatus", {}).get("Status"),
                 track_temp=weather.get("TrackTemp"), air_temp=weather.get("AirTemp"),
                 rainfall=weather.get("Rainfall"))
        if drv:
            ln = val(td, "Lines", drv) or {}
            dl = val(self.S, "DriverList", drv) or {}
            stint, comp, new, age = self.tyre(drv)
            # race: GapToLeader / IntervalToPositionAhead; qualifying/practice: TimeDiffToFastest
            gap, laps_down = parse_gap(ln.get("GapToLeader", ln.get("TimeDiffToFastest")))
            interval, _ = parse_gap(val(ln, "IntervalToPositionAhead", "Value")
                                    or ln.get("TimeDiffToPositionAhead"))
            r.update(driver=drv, tla=dl.get("Tla"), team=dl.get("TeamName"),
                     position=ln.get("Position"), lap_number=ln.get("NumberOfLaps"),
                     gap_to_leader_s=gap, laps_down=laps_down, interval_s=interval,
                     catching=val(ln, "IntervalToPositionAhead", "Catching"),
                     in_pit=ln.get("InPit"), n_pit_stops=ln.get("NumberOfPitStops"),
                     stint=stint, compound=comp, tyre_new=new, tyre_laps=age,
                     best_lap_s=to_sec(val(ln, "BestLapTime", "Value")),
                     best_in_part_s=to_sec(val(ln, "BestLapTimes", (part or 1) - 1, "Value")))
        r.update(kw)
        return r

    def emit(self, r):
        self.w(r)

    # ---------- main entry ----------
    def feed(self, line):
        parsed = parse_line(line)
        if not parsed:
            return
        topic, data, ts = parsed
        if topic in ("CarData", "Position"):
            return                                   # not used; skip the decompress cost
        data = decode(data)
        if not isinstance(data, dict):
            return
        snapshot = ts == ""
        merge(self.S.setdefault(topic, {}), data)
        if snapshot:
            if topic == "TimingData":
                for d, ln in self.S["TimingData"].get("Lines", {}).items():
                    self.drv[d] = {"out": bool(ln.get("PitOut") or ln.get("InPit")
                                               or ln.get("Status", 0) & OUT_LAP_BIT),
                                   # in the pits when we connected: pit_out will have no duration
                                   "pit_t": "unknown" if ln.get("InPit") else None}
                self.last_part = self.S["TimingData"].get("SessionPart")
            if topic == "TrackStatus":
                self.last_track = self.S["TrackStatus"].get("Status")
            return

        if topic == "TrackStatus":
            st = self.S["TrackStatus"].get("Status")
            if st != self.last_track:
                self.last_track = st
                self.emit(self.row(ts, "track_status"))
        elif topic == "RaceControlMessages":
            msgs = data.get("Messages", {})
            for m in (msgs.values() if isinstance(msgs, dict) else msgs):
                text = m.get("Message", "")
                hit = DELETED.search(text)
                if hit:
                    self.emit(self.row(ts, "lap_deleted", hit.group(1), lap_time_s=to_sec(hit.group(2)),
                                       message=text))
                else:
                    self.emit(self.row(ts, "race_control", m.get("RacingNumber"), message=text))
        elif topic == "TimingData":
            self.timing(ts, data)

        self.maybe_snapshot(ts)

    def timing(self, ts, data):
        td = self.S["TimingData"]
        if td.get("SessionPart") != self.last_part:
            self.last_part = td.get("SessionPart")
            self.emit(self.row(ts, "session_part_start"))

        for drv, delta in data.get("Lines", {}).items():
            if not isinstance(delta, dict):
                continue
            ln = td["Lines"][drv]
            f = self.drv.setdefault(drv, {"out": False, "pit_t": None})

            if delta.get("InPit") is True and f["pit_t"] is None:
                f["pit_t"] = ts
                self.emit(self.row(ts, "pit_in", drv))
            if delta.get("PitOut") or delta.get("InPit") or delta.get("Status", 0) & OUT_LAP_BIT:
                f["out"] = True       # cleared when the lap completes; Status may flip first
            if delta.get("InPit") is False and f["pit_t"] is not None:
                dur = None if f["pit_t"] == "unknown" else round((iso(ts) - iso(f["pit_t"])).total_seconds(), 2)
                f["pit_t"] = None
                self.emit(self.row(ts, "pit_out", drv, pit_lane_s=dur))
            if delta.get("Retired") is True:
                self.emit(self.row(ts, "retired", drv))

            if not val(delta, "LastLapTime", "Value"):
                continue
            llt = ln["LastLapTime"]
            pb = bool(llt.get("PersonalFastest"))
            if ln.get("InPit"):
                kind = "in"
            elif f["out"]:
                kind = "out"
            elif self.is_race():
                kind = "racing"
            else:
                kind = "push" if pb else "other"
            f["out"] = False
            self.emit(self.row(
                ts, "lap", drv,
                lap_time_s=to_sec(llt["Value"]), lap_kind=kind,
                personal_best=pb, overall_best=bool(llt.get("OverallFastest")),
                s1_s=to_sec(val(ln, "Sectors", 0, "Value")),
                s2_s=to_sec(val(ln, "Sectors", 1, "Value")),
                s3_s=to_sec(val(ln, "Sectors", 2, "Value")),
                speed_i1=val(ln, "Speeds", "I1", "Value"), speed_i2=val(ln, "Speeds", "I2", "Value"),
                speed_fl=val(ln, "Speeds", "FL", "Value"), speed_st=val(ln, "Speeds", "ST", "Value"),
            ))

    def maybe_snapshot(self, ts):
        if not self.snapshot_every or not ts:
            return
        t = iso(ts)
        if self.last_snapshot and (t - self.last_snapshot).total_seconds() < self.snapshot_every:
            return
        self.last_snapshot = t
        for drv in self.S.get("TimingData", {}).get("Lines", {}):
            self.emit(self.row(ts, "snapshot", drv))


def main():
    ap = argparse.ArgumentParser()
    add_session_args(ap)
    ap.add_argument("--replay", action="store_true", help="parse to end of file and exit")
    ap.add_argument("--snapshot", type=float, default=0, help="seconds between per-driver snapshots (0 = off)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    sess = session_from_args(args)
    sess.dir.mkdir(parents=True, exist_ok=True)

    out = open(sess.events, "w", newline="")
    w = csv.DictWriter(out, fieldnames=FIELDS)
    w.writeheader()

    def write(r):
        w.writerow(r)
        out.flush()            # flush so the runner can tail the file live
        if not args.quiet and r["event"] != "snapshot":
            print(r["ts"], r["event"], r["tla"] or "", r["lap_time_s"] or "", r["lap_kind"] or "",
                  r["message"] or "")

    p = Parser(write)
    p.snapshot_every = args.snapshot
    for line in follow(sess.raw, live=not args.replay):
        p.feed(line)


if __name__ == "__main__":
    main()
