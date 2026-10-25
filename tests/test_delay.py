"""Showing the race as it was a few seconds ago, to match a delayed broadcast."""
import io

import numpy as np
import pandas as pd
import pytest
from rich.console import Console

from f1live.paths import Session
from live.race import watch
from live.race.track import TrackProgress, build_reference
from test_battles import T0, race
from test_track import LAP_S, ring_telemetry


def feed_on_a_ring(tmp_path):
    """A feed with a car lapping a circle every 90 s: lap rows up to T0+270 s, positions to T0+360 s."""
    tel = ring_telemetry(laps=3)
    history = race({"AAA": LAP_S}, {"AAA": 0}, laps=3, total=10)
    feed = watch.Feed(Session("2026", "Test", "race", tmp_path))
    feed.history = history
    feed.trail = {"10": list(zip(tel["ts"], tel["x"], tel["y"]))}
    feed.tracker = TrackProgress(build_reference(tel, race({"AAA": LAP_S}, {"AAA": 0}, laps=5)))
    return feed


def expected_progress(at):
    return (((at - (T0 + pd.Timedelta(90, unit="s"))).total_seconds()) % LAP_S) / LAP_S


def test_the_head_is_the_newest_data_and_a_delay_looks_back_from_it(tmp_path):
    feed = feed_on_a_ring(tmp_path)
    head = feed.head()
    assert head == max(feed.trail["10"][-1][0], feed.history["ts"].max())
    for delay in (0, 10, 25, 40):
        v = watch.delayed_view(feed, None, delay)
        assert v["at"] == head - pd.Timedelta(delay, unit="s")
        prog, mode = v["progress"]
        assert mode == "telemetry"
        assert abs(((prog["10"] - expected_progress(v["at"]) + 0.5) % 1) - 0.5) < 0.02      # the car was there, delay seconds ago
        assert v["history"]["ts"].max() <= v["at"]


def test_a_delayed_view_hides_everything_newer_in_every_panel(tmp_path):
    feed = feed_on_a_ring(tmp_path)
    head = feed.head()
    preds = pd.DataFrame(dict(ts=[head - pd.Timedelta(s, unit="s") for s in (50, 20, 5, 0)], kind="status", note="x"))
    v = watch.delayed_view(feed, preds, 10)
    assert list(v["preds"]["ts"]) == list(preds["ts"].iloc[:2])
    assert len(watch.delayed_view(feed, preds, 0)["preds"]) == 4 and watch.delayed_view(feed, preds, 100)["preds"] is None
    assert watch.delayed_view(feed, preds, 100)["history"] is not None            # 100 s back is still inside this short recording


def test_a_delay_longer_than_the_data_has_nothing_to_show_yet(tmp_path):
    feed = feed_on_a_ring(tmp_path)
    v = watch.delayed_view(feed, None, 10_000)
    assert v["history"] is None and v["progress"] == ({}, None)
    empty = watch.Feed(Session("2026", "Test", "race", tmp_path))
    assert empty.head() is None and watch.delayed_view(empty, None, 30)["history"] is None


def test_position_at_is_the_last_sample_before_the_moment(tmp_path):
    feed = feed_on_a_ring(tmp_path)
    t = feed.trail["10"][100][0]
    assert feed.position_at("10", t)[0] == t
    assert feed.position_at("10", t + pd.Timedelta(0.1, unit="s"))[0] == t
    assert feed.position_at("10", T0) is None and feed.position_at("99", t) is None


def test_the_position_trail_keeps_only_the_last_few_minutes(tmp_path):
    sess = Session("2026", "Test", "race", tmp_path)
    sess.dir.mkdir(parents=True)
    t = pd.date_range(T0, periods=601, freq="1s")                            # ten minutes at 1 Hz
    pd.DataFrame(dict(ts=t.strftime("%Y-%m-%dT%H:%M:%S.000Z"), driver=10, speed=200, rpm=1, gear=1, throttle=1, brake=0,
                      x=np.arange(601.0), y=0.0, z=0.0, pos_status="OnTrack")).to_csv(sess.telemetry, index=False)
    feed = watch.Feed(sess)
    feed.poll()
    samples = feed.trail["10"]
    assert samples[-1][0] == t[-1] and (samples[-1][0] - samples[0][0]).total_seconds() <= watch.Feed.TRAIL_S
    assert len(samples) < 320


def test_keys_change_the_delay_within_limits():
    assert [watch.adjust_delay(45, k) for k in "][}{0"] == [46.0, 44.0, 55.0, 35.0, 0.0]
    assert watch.adjust_delay(0.5, "[") == 0.0 and watch.adjust_delay(295, "}") == watch.Feed.TRAIL_S
    assert watch.adjust_delay(45, "x") == 45.0


def test_the_delay_is_remembered_between_runs(tmp_path):
    assert watch.load_delay(tmp_path) == 0.0
    watch.save_delay(tmp_path, 47.5)
    assert watch.load_delay(tmp_path) == 47.5
    watch.delay_file(tmp_path).write_text("not a number")
    assert watch.load_delay(tmp_path) == 0.0
    watch.delay_file(tmp_path).write_text("9999")
    assert watch.load_delay(tmp_path) == watch.Feed.TRAIL_S


def frame(delay, tmp_path):
    feed = feed_on_a_ring(tmp_path)
    v = watch.delayed_view(feed, None, delay)
    console = Console(width=170, height=44, file=io.StringIO(), force_terminal=False, color_system=None)
    console.print(watch.render(v["history"], v["preds"], "2026/Test/race", 44, v["progress"], 170, (), v["sectors"], delay))
    return console.file.getvalue()


def test_the_header_shows_the_delay_only_when_there_is_one(tmp_path):
    assert "DELAY 45 s" in frame(45, tmp_path) and "0 live" in frame(45, tmp_path)
    assert "DELAY" not in frame(0, tmp_path)
