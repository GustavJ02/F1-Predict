"""Parse FastF1 live-timing recordings into a CSV of lap events.

Usage:
    python parse_live.py austin_quali_raw.txt                 # tail live, write laps.csv
    python parse_live.py austin_quali_raw.txt --replay        # parse a finished recording
    python parse_live.py raw.txt --replay --out q_usa.csv     # custom output path
"""
import ast, json, base64, zlib, re, time, sys, csv, os, argparse
from datetime import datetime

# ---------- decoding / state ----------

def decode(d):
    if isinstance(d, dict): return d
    try: d = json.loads(d)            # snapshot payloads are JSON text...
    except ValueError: pass
    if isinstance(d, str):            # ...and .z topics are base64 + raw deflate
        d = json.loads(zlib.decompress(base64.b64decode(d), -zlib.MAX_WBITS))
    return d

def merge(dst, src):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), (dict, list)):
            if isinstance(dst[k], list):  # deltas address list items by string index
                for i, sub in v.items():
                    i = int(i)
                    while len(dst[k]) <= i: dst[k].append({})
                    if isinstance(sub, dict) and isinstance(dst[k][i], dict): merge(dst[k][i], sub)
                    else: dst[k][i] = sub
            else:
                merge(dst[k], v)
        else:
            dst[k] = v

def to_sec(t):
    if not t: return None
    m, _, s = t.rpartition(":")
    return round((int(m) * 60 if m else 0) + float(s), 3)

def iso(t): return datetime.fromisoformat(t.rstrip("Z")[:26])

def follow(path, live):
    with open(path) as f:
        while True:
            line = f.readline()
            if line: yield line
            elif live: time.sleep(0.2)
            else: return

# ---------- helpers reading the merged state ----------

S = {}  # merged state per topic

def clock_left(now):
    """Seconds left in the segment; the feed sends occasional anchors to extrapolate from."""
    c = S.get("ExtrapolatedClock")
    if not c: return None
    h, m, sec = map(int, c["Remaining"].split(":"))
    left = h * 3600 + m * 60 + sec
    if c.get("Extrapolating"):
        left -= (iso(now) - iso(c["Utc"])).total_seconds()
    return round(max(left, 0), 1)

def driver_info(drv):
    d = S.get("DriverList", {}).get(drv, {})
    return d.get("Tla"), d.get("TeamName")

def tyre(drv):
    stints = S.get("TimingAppData", {}).get("Lines", {}).get(drv, {}).get("Stints", [])
    if isinstance(stints, dict): stints = [stints[k] for k in sorted(stints, key=int)]
    if not stints: return None, None, None
    st = stints[-1]
    age = (st.get("TotalLaps") or 0)
    return st.get("Compound"), st.get("New") == "true", age

def val(x, *path):
    for p in path:
        if isinstance(x, list): x = x[p] if len(x) > p else {}
        else: x = (x or {}).get(p, {})
    return x if x not in ({}, "") else None

# ---------- CSV ----------

FIELDS = [
    "ts", "event", "session_part", "clock_left_s", "driver", "tla", "team",
    "lap_number", "lap_time_s", "lap_kind", "personal_best", "overall_best",
    "s1_s", "s2_s", "s3_s", "speed_i1", "speed_i2", "speed_fl", "speed_st",
    "compound", "tyre_new", "tyre_laps",
    "best_in_part_s", "position", "n_entries", "n_advance",
    "track_status", "track_temp", "air_temp", "rainfall",
]

def make_row(ts, event, drv=None, **kw):
    td = S.get("TimingData", {})
    part = td.get("SessionPart")
    entries = td.get("NoEntries") or []
    row = dict.fromkeys(FIELDS)
    row.update(ts=ts, event=event, session_part=part, clock_left_s=clock_left(ts),
               n_entries=entries[part - 1] if part and len(entries) >= part else None,
               n_advance=entries[part] if part and len(entries) > part else None,
               track_status=S.get("TrackStatus", {}).get("Message"),
               track_temp=S.get("WeatherData", {}).get("TrackTemp"),
               air_temp=S.get("WeatherData", {}).get("AirTemp"),
               rainfall=S.get("WeatherData", {}).get("Rainfall"))
    if drv:
        ln = td.get("Lines", {}).get(drv, {})
        tla, team = driver_info(drv)
        comp, new, age = tyre(drv)
        row.update(driver=drv, tla=tla, team=team, position=ln.get("Position"),
                   compound=comp, tyre_new=new, tyre_laps=age,
                   best_in_part_s=to_sec(val(ln, "BestLapTimes", (part or 1) - 1, "Value")))
    row.update(kw)
    return row

# ---------- main ----------

# Line Status bits as observed in this feed (inferred, not documented):
# 16 = in pit, 32 = pit out, 64 = running, 512 = on out lap, 1024 = taken chequered flag
OUT_LAP_BIT = 512
DELETED = re.compile(r"CAR (\d+) \(\w+\) TIME ([\d:.]+) DELETED")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", nargs="?", default="austin_quali_raw.txt")
    ap.add_argument("--out", default="laps.csv")
    ap.add_argument("--replay", action="store_true", help="parse to end of file and exit")
    args = ap.parse_args()

    out = open(args.out, "w", newline="")
    w = csv.DictWriter(out, fieldnames=FIELDS)
    w.writeheader()
    def emit(row):
        w.writerow(row); out.flush()          # flush so a reader can tail the CSV live
        print(row["ts"], row["event"], row["tla"] or "", row["lap_time_s"] or "",
              row["lap_kind"] or "", f"Q{row['session_part']}")

    lap_flags, last_part = {}, None
    for line in follow(args.raw, live=not args.replay):
        try: topic, data, ts = ast.literal_eval(line.strip())
        except Exception: continue
        topic = topic.removesuffix(".z")
        if topic in ("CarData", "Position"): continue   # not used yet; skip the decompress cost
        data = decode(data)
        snapshot = ts == ""                  # initial keyframe: build state, emit nothing
        merge(S.setdefault(topic, {}), data)

        if topic == "RaceControlMessages" and not snapshot:
            msgs = data.get("Messages", {})
            for m in (msgs.values() if isinstance(msgs, dict) else msgs):
                if (hit := DELETED.search(m.get("Message", ""))):
                    emit(make_row(ts, "lap_deleted", hit.group(1), lap_time_s=to_sec(hit.group(2))))

        if topic != "TimingData": continue
        td = S["TimingData"]
        if snapshot:  # seed out-lap flags for cars already on an out-lap when we connected
            for drv, ln in td["Lines"].items():
                lap_flags[drv] = {"out": bool(ln.get("PitOut") or ln.get("InPit")
                                              or ln.get("Status", 0) & OUT_LAP_BIT)}
            last_part = td.get("SessionPart")
            continue

        if td.get("SessionPart") != last_part:
            last_part = td.get("SessionPart")
            emit(make_row(ts, "session_part_start"))

        for drv, delta in data.get("Lines", {}).items():
            ln = td["Lines"][drv]
            f = lap_flags.setdefault(drv, {"out": False})
            if delta.get("PitOut") or delta.get("InPit") or delta.get("Status", 0) & OUT_LAP_BIT:
                f["out"] = True   # cleared only when the lap completes; Status may flip first
            if not val(delta, "LastLapTime", "Value"): continue

            llt = ln["LastLapTime"]
            pb = bool(llt.get("PersonalFastest"))
            kind = ("in" if ln.get("InPit") else "out" if f["out"] else "push" if pb else "other")
            f["out"] = False
            emit(make_row(
                ts, "lap", drv,
                lap_number=ln.get("NumberOfLaps"), lap_time_s=to_sec(llt["Value"]), lap_kind=kind,
                personal_best=pb, overall_best=bool(llt.get("OverallFastest")),
                s1_s=to_sec(val(ln, "Sectors", 0, "Value")),
                s2_s=to_sec(val(ln, "Sectors", 1, "Value")),
                s3_s=to_sec(val(ln, "Sectors", 2, "Value")),
                speed_i1=val(ln, "Speeds", "I1", "Value"), speed_i2=val(ln, "Speeds", "I2", "Value"),
                speed_fl=val(ln, "Speeds", "FL", "Value"), speed_st=val(ln, "Speeds", "ST", "Value"),
            ))

if __name__ == "__main__":
    main()