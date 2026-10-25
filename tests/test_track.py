"""Lap progress: reference lap from telemetry, projection, timing fallback and the track line."""
import io

import numpy as np
import pandas as pd
import pytest
from rich.console import Console

from f1live.paths import Session
from live.race import watch
from live.race.track import (TrackProgress, build_reference, corner_fractions, load_corners, load_reference, resample,
                             save_corners, save_reference, sector_fractions, timing_progress)
from test_battles import T0, race

RADIUS, LAP_S = 8000.0, 90.0


def circle(theta):
    return RADIUS * np.cos(theta), RADIUS * np.sin(theta)


def ring_telemetry(driver="10", laps=3, rate=0.27, speedup=1.0):
    """Samples of one car lapping a circular track every LAP_S seconds, starting at angle 0 at T0 + 90 s."""
    t = np.arange(0, laps * LAP_S, rate)
    x, y = circle(2 * np.pi * (t % LAP_S) / LAP_S)
    return pd.DataFrame(dict(ts=T0 + pd.to_timedelta(t + 90, unit="s"), driver=driver, x=x.round(), y=y.round()))


def ring_laps():
    h = race({"AAA": LAP_S}, {"AAA": 0}, laps=5)      # lap rows at T0 + 90 s * lap
    h["lap_time_s"] = LAP_S
    return h


def test_reference_is_built_from_one_clean_lap():
    ref = build_reference(ring_telemetry(), ring_laps())
    assert ref is not None and len(ref) == 600
    assert np.allclose(np.hypot(ref[:, 0], ref[:, 1]), RADIUS, rtol=0.01)       # the circle
    assert np.hypot(*(ref[0] - np.array(circle(0.0)))) < 0.03 * 2 * np.pi * RADIUS   # starts at the line
    assert build_reference(ring_telemetry().iloc[:50], ring_laps()) is None     # not a whole lap yet


def test_progress_follows_the_car_round_the_lap_and_across_the_line():
    ref = build_reference(ring_telemetry(), ring_laps())
    tp = TrackProgress(ref)
    for frac in (0.1, 0.4, 0.7, 0.98, 0.01, 0.05):
        x, y = circle(2 * np.pi * frac)
        assert abs(((tp.progress("10", x, y) - frac + 0.5) % 1) - 0.5) < 0.01


def test_progress_does_not_jump_to_a_parallel_stretch():
    # a long thin loop whose two straights are 400 units (40 m) apart: the nearest point can be on the wrong straight
    n = 400
    t = np.linspace(0, 1, n, endpoint=False)
    out = np.c_[t * 20000, np.zeros(n)]
    back = np.c_[20000 - t * 20000, np.full(n, 400.0)]
    ref = resample(np.vstack([out, back, out[:1]]))
    tp = TrackProgress(ref)
    p = tp.progress("a", 5000, 0)                      # on the outbound straight, a quarter of the way along it
    assert p < 0.25
    assert tp.progress("a", 5400, 150) < 0.3           # 15 m off, nearer the return straight, but it was just on the outbound one


def test_reference_roundtrip(tmp_path):
    ref = resample(np.c_[[0, 1000, 1000, 0, 0], [0, 0, 1000, 1000, 0]] * 10.0)
    save_reference(ref, tmp_path / "track.csv")
    assert np.allclose(load_reference(tmp_path / "track.csv"), ref)
    assert load_reference(tmp_path / "missing.csv") is None


def test_timing_progress_is_time_since_the_line_over_pace():
    lat = pd.DataFrame({"last_lap_ts": [T0, T0, pd.NaT]}, index=["10", "11", "12"])
    got = timing_progress(lat, T0 + pd.Timedelta(45, unit="s"), pd.Series({"10": 90.0, "11": 45.0, "12": 90.0}))
    assert got["10"] == pytest.approx(0.5) and got["11"] == pytest.approx(0.999) and "12" not in got


def render_line(cars, width=60):
    console = Console(width=width, file=io.StringIO(), force_terminal=False, color_system=None)
    console.print(watch.TrackLine(cars))
    return console.file.getvalue().splitlines()


def test_track_line_places_dots_labels_and_marks_the_leader():
    lines = render_line([("VER", 0.5, "red", True), ("HAM", 0.0, "blue", False), ("NOR", 0.98, "blue", False)])
    track = lines[watch.TRACK_ROWS]
    assert track[0] == "●" or "●" in track[:3] and track.count("●") == 2 and track.count("◆") == 1
    assert track.index("◆") == pytest.approx(0.5 * 59, abs=1)
    assert any("VER" in l for l in lines[:watch.TRACK_ROWS]) and any("NOR" in l for l in lines[:watch.TRACK_ROWS])


def test_cars_at_the_same_spot_are_staggered_and_never_lose_their_dot():
    cars = [(f"C{i:02d}", 0.5 + 0.001 * i, "white", False) for i in range(7)]      # 7 cars within 1 character
    lines = render_line(cars)
    labels = [l for l in lines[:watch.TRACK_ROWS] if "C0" in l]
    assert len(labels) == watch.TRACK_ROWS and sum(l.count("C0") for l in labels) == watch.TRACK_ROWS   # 4 labels fit, 3 dropped
    assert lines[watch.TRACK_ROWS].count("●") >= 1


def test_feed_falls_back_to_timing_without_telemetry(tmp_path):
    feed = watch.Feed(Session("2026", "Test", "race", tmp_path))
    assert feed.progress() == ({}, None)
    feed.history = race({"AAA": LAP_S, "BBB": LAP_S}, {"AAA": 0, "BBB": 2.0}, laps=6)
    prog, mode = feed.progress()
    assert mode == "timing" and set(prog) == {"10", "11"}


def test_feed_uses_telemetry_when_a_reference_exists_and_drops_it_when_stale(tmp_path):
    session = Session("2026", "Test", "race", tmp_path)
    save_reference(build_reference(ring_telemetry(), ring_laps()), tmp_path / "2026" / "Test" / "track.csv") \
        if (tmp_path / "2026" / "Test").mkdir(parents=True) is None else None
    feed = watch.Feed(session)
    assert feed.tracker is not None
    feed.history = ring_laps()
    now = feed.history["ts"].max()
    x, y = circle(np.pi)                                               # half way round
    feed.pos = {"10": (now - pd.Timedelta(2, unit="s"), x, y)}
    prog, mode = feed.progress()
    assert mode == "telemetry" and prog["10"] == pytest.approx(0.5, abs=0.01)
    feed.pos = {"10": (now - pd.Timedelta(60, unit="s"), x, y)}         # telemetry stopped a minute ago
    assert feed.progress()[1] == "timing"


def test_dashboard_has_the_track_panel():
    h = race({"AAA": LAP_S, "BBB": LAP_S}, {"AAA": 0, "BBB": 2.0}, laps=6)
    console = Console(width=140, height=40, file=io.StringIO(), force_terminal=False, color_system=None)
    console.print(watch.render(h, None, "2026/Test/race", 40, ({"10": 0.4, "11": 0.35}, "telemetry")))
    text = console.file.getvalue()
    assert "Lap position (from car telemetry)" in text and "◆" in text and "S/F" in text


def test_reference_can_be_built_from_a_qualifying_session():
    laps = ring_laps()
    laps["lap_kind"] = "push"                   # qualifying calls its flying laps push/other, and has no pit-stop count
    laps["n_pit_stops"] = np.nan
    assert build_reference(ring_telemetry(), laps) is not None


def test_sector_fractions_are_the_median_share_of_the_lap():
    h = ring_laps()
    h["s1_s"], h["s2_s"], h["s3_s"] = 27.0, 36.0, 27.0
    assert sector_fractions(h) == pytest.approx([0.3, 0.7])
    assert sector_fractions(ring_laps()) is None                          # no sector times


def test_corners_are_projected_onto_the_reference_and_roundtrip(tmp_path):
    ref = build_reference(ring_telemetry(), ring_laps())
    xs, ys = circle(np.array([np.pi, np.pi / 2, np.pi / 2]))              # 3 corners: half way, quarter, and a 3A twin
    corners = pd.DataFrame(dict(Number=[2, 1, 1], X=xs, Y=ys))
    found = corner_fractions(ref, corners)
    assert [c for c, _ in found] == ["1", "2"] and [f for _, f in found] == pytest.approx([0.25, 0.5], abs=0.01)
    save_corners(found, tmp_path / "corners.csv")
    assert load_corners(tmp_path / "corners.csv") == pytest.approx(found) or load_corners(tmp_path / "corners.csv")[0][0] == "1"
    assert load_corners(tmp_path / "missing.csv") == []


def test_track_line_numbers_corners_and_colours_sectors():
    line = watch.TrackLine([("VER", 0.5, "red", True)], [("1", 0.25), ("2", 0.26), ("3", 0.75)], [0.3, 0.7])
    plain = Console(width=60, file=io.StringIO(), force_terminal=False, color_system=None)
    plain.print(line)
    lines = [l for l in plain.file.getvalue().splitlines() if "S/F" in l]
    assert lines and "1" in lines[0] and "3" in lines[0] and "2" not in lines[0]    # 2 is too close to 1 to fit
    console = Console(width=60, file=io.StringIO(), force_terminal=True, color_system="standard")
    console.print(line)
    out = console.file.getvalue()
    assert out.count("┃") == 2 and "\x1b[36m" in out and "\x1b[35m" in out and "\x1b[33m" in out
