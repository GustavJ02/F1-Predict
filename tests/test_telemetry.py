"""Telemetry decoding: a synthetic fixture (always runs) and the recorded Austin quali (skipped if data/ is absent)."""
import base64, json, zlib

import pandas as pd
import pytest

from f1live.feed import follow
from f1live.paths import Session
from f1live.runner import add_telemetry, prepare_telemetry
from f1live.telemetry import FIELDS, TelemetryParser

RAW = Session("2026", "Austin", "quali").raw


def car(rpm, speed, gear, throttle, brake):
    return {"Channels": {"0": rpm, "2": speed, "3": gear, "4": throttle, "5": brake}}


def line(topic, payload, ts="2026-10-24T21:00:00.000Z"):
    return repr([topic, payload, ts]) + "\n"


def compress(payload):
    c = zlib.compressobj(wbits=-zlib.MAX_WBITS)
    return base64.b64encode(c.compress(json.dumps(payload).encode()) + c.flush()).decode()


def run(lines, every=0):
    rows = []
    p = TelemetryParser(rows.append, every)
    for l in lines:
        p.feed(l)
    return pd.DataFrame(rows, columns=FIELDS)


def test_decodes_plain_and_compressed_payloads_and_joins_position():
    pos = {"Position": [{"Timestamp": "2026-10-24T21:00:00.5Z",
                         "Entries": {"1": {"Status": "OnTrack", "X": 10, "Y": 20, "Z": 30}}}]}
    cd = {"Entries": [{"Utc": "2026-10-24T21:00:01.0Z", "Cars": {"1": car(11000, 250, 6, 100, 0)}}]}
    df = run([line("Position.z", compress(pos)), line("CarData.z", compress(cd), ts="")])
    assert len(df) == 1
    r = df.iloc[0]
    assert (r.driver, r.speed, r.gear, r.throttle, r.brake, r.x, r.y, r.pos_status) == \
        ("1", 250, 6, 100, 0, 10, 20, "OnTrack")
    assert r.ts == "2026-10-24T21:00:01.0Z"      # the sample's own time, not the line's


def test_no_data_marker_is_nulled():
    cd = {"Entries": [{"Utc": "2026-10-24T21:00:01.0Z", "Cars": {"3": car(0, 0, 0, 104, 104)}}]}
    r = run([line("CarData", cd)]).iloc[0]
    assert pd.isna(r.throttle) and pd.isna(r.brake) and r.speed == 0


def test_position_without_fix_is_nulled():
    pos = {"Position": [{"Timestamp": "2026-10-24T21:00:00.5Z",
                         "Entries": {"1": {"Status": "OnTrack", "X": 0, "Y": 0, "Z": 0}}}]}
    cd = {"Entries": [{"Utc": "2026-10-24T21:00:01.0Z", "Cars": {"1": car(0, 0, 0, 0, 0)}}]}
    r = run([line("Position", pos), line("CarData", cd)]).iloc[0]
    assert pd.isna(r.x) and pd.isna(r.y) and pd.isna(r.z)


def test_every_downsamples_per_driver():
    entries = [{"Utc": f"2026-10-24T21:00:0{i}.{f}Z", "Cars": {"1": car(1, 1, 1, 1, 0)}}
               for i in range(3) for f in (0, 5)]
    assert len(run([line("CarData", {"Entries": entries})], every=0)) == 6
    assert len(run([line("CarData", {"Entries": entries})], every=1)) == 3


def test_ignores_other_topics_and_garbage():
    assert run([line("TimingData", {"Lines": {}}), "not a line\n", line("CarData", "###")]).empty


def test_add_telemetry_uses_latest_row_even_if_null():
    tel = prepare_telemetry(pd.DataFrame({
        "ts": ["2026-10-24T21:00:00Z", "2026-10-24T21:00:01Z"], "driver": [1, 1], "speed": [300, 0],
        "rpm": [1, 0], "gear": [8, 0], "throttle": [100, None], "brake": [0, None], "x": [1, 2], "y": [1, 2]}))
    out = add_telemetry(pd.DataFrame({"tla": ["VER"]}, index=["1"]), tel)
    assert out.loc["1", "car_speed"] == 0 and pd.isna(out.loc["1", "car_throttle"])


@pytest.mark.skipif(not RAW.exists(), reason=f"{RAW} not found")
def test_austin_quali():
    rows = []
    p = TelemetryParser(rows.append, 1.0)
    for l in follow(RAW, live=False):
        p.feed(l)
    df = pd.DataFrame(rows, columns=FIELDS)
    assert df["driver"].nunique() == 22 and 50_000 < len(df) < 60_000
    assert df["speed"].between(0, 400).all() and df["gear"].between(0, 8).all()
    assert df["throttle"].dropna().between(0, 100).all() and df["brake"].dropna().isin([0, 100]).all()
    assert (df["pos_status"] == "OnTrack").all()
    assert df["speed"].max() > 300
