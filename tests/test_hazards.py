"""Stopped and crawling cars, the SC/VSC banner and the pit-lane exemption."""
import io

import numpy as np
import pandas as pd
from rich.console import Console

from live.race import watch
from live.race.hazards import (SpeedProfile, classify, hazards, in_pit, neutral_at, pit_windows, race_is_live, retired,
                               track_state)
from live.race.track import nearest_progress
from test_battles import race

T0 = pd.Timestamp("2026-10-25 20:00:00", tz="UTC")


def at(s):
    return T0 + pd.Timedelta(float(s), unit="s")


def samples(speeds, start=0.0, x=0.0):
    """One sample a second: (ts, x, y, speed)."""
    return [(at(start + i), x, 0.0, v) for i, v in enumerate(speeds)]


def profile(speed=200.0, n=100):
    p = SpeedProfile()
    for q in np.linspace(0, 0.99, 100):
        for _ in range(n // 4):
            p.add(q, speed)
    return p


def events(rows):
    return pd.DataFrame(rows).assign(ts=lambda d: pd.to_datetime(d["ts"]))


def history(status="AllClear", lap=10, extra=()):
    rows = [dict(ts=at(0), event="lap", race_lap=lap, total_laps=56, driver="77", tla="BOT"),
            dict(ts=at(1), event="track_status", track_status=status, status_code=1),
            *extra]
    return pd.DataFrame(rows)


def test_a_car_that_was_racing_and_has_stood_still_is_stopped():
    s = samples([180] * 20 + [60, 20, 4, 0, 0, 0, 0])
    assert classify(s, at(26)) == "stopped"
    assert classify(s, at(21)) is None                      # still rolling


def test_a_car_that_never_ran_is_not_stopped():
    assert classify(samples([0] * 30), at(29)) is None      # on the grid or in the garage


def test_crawling_is_judged_against_the_speed_cars_normally_have_there():
    p = profile(200)
    prog = lambda x, y: 0.5
    crawling = samples([180] * 20 + [45] * 8)
    assert classify(crawling, at(27), prog, p) == "slow"
    assert classify(samples([180] * 28), at(27), prog, p) is None
    assert classify(crawling, at(27), prog, p, green=False) is None      # under SC/VSC everyone is slow


def test_a_slow_part_of_the_lap_is_not_slow():
    p = profile(70)                                          # a hairpin: cars normally do 70
    assert classify(samples([180] * 20 + [45] * 8), at(27), lambda x, y: 0.5, p) is None


def test_the_profile_needs_enough_cars_before_it_judges():
    assert classify(samples([180] * 20 + [45] * 8), at(27), lambda x, y: 0.5, SpeedProfile()) is None


def test_unknown_speed_is_never_a_hazard():
    assert classify(samples([180] * 20 + [np.nan] * 8), at(27), lambda x, y: 0.5, profile()) is None


def test_the_pit_lane_and_its_edges_are_exempt():
    h = history(extra=[dict(ts=at(100), event="pit_in", driver="77"), dict(ts=at(130), event="pit_out", driver="77")])
    w = pit_windows(h)
    assert in_pit(w, "77", at(90)) and in_pit(w, "77", at(150)) and not in_pit(w, "77", at(60)) and not in_pit(w, "77", at(170))
    open_ = pit_windows(history(extra=[dict(ts=at(100), event="pit_in", driver="77")]))
    assert in_pit(open_, "77", at(5000))                     # still in the pits


def test_hazards_skips_cars_in_the_pits_and_names_the_stopped_one():
    trail = {"77": samples([180] * 20 + [0] * 8), "1": samples([180] * 28)}
    assert hazards(trail, history(), at(27)) == {"77": "stopped"}
    h = history(extra=[dict(ts=at(15), event="pit_in", driver="77")])
    assert hazards(trail, h, at(27)) == {}


def test_nothing_is_flagged_on_lap_one_after_the_flag_or_under_red():
    trail = {"77": samples([180] * 20 + [0] * 8)}
    assert hazards(trail, history(lap=1), at(27)) == {}
    flag = dict(ts=at(20), event="race_control", message="CHEQUERED FLAG")
    assert hazards(trail, history(lap=56), at(27)) == {"77": "stopped"}      # the final lap still counts
    assert hazards(trail, history(lap=56, extra=[flag]), at(27)) == {}
    assert hazards(trail, history(status="Red"), at(27)) == {}
    assert not race_is_live(None)


def test_a_retired_car_is_still_stopped_but_never_slow():
    h = history(extra=[dict(ts=at(5), event="retired", driver="77")])
    assert retired(h) == {"77"}
    trail = {"77": samples([180] * 20 + [45] * 8)}
    assert hazards(trail, h, at(27), lambda x, y: 0.5, profile()) == {}
    assert hazards(trail, history(), at(27), lambda x, y: 0.5, profile()) == {"77": "slow"}


def test_slow_is_only_judged_under_green_flags():
    trail = {"77": samples([180] * 20 + [45] * 8)}
    args = (lambda x, y: 0.5, profile())
    assert hazards(trail, history("VSCDeployed"), at(27), *args) == {}
    assert hazards(trail, history("Yellow"), at(27), *args) == {"77": "slow"}


def test_track_state_and_neutral_stretches():
    h = pd.DataFrame([dict(ts=at(10), event="track_status", track_status="VSCDeployed"),
                      dict(ts=at(50), event="track_status", track_status="AllClear")])
    assert track_state(h) == "AllClear" and track_state(None) is None
    assert list(neutral_at(h, [at(5), at(20), at(60)])) == [False, True, False]
    assert not neutral_at(None, [at(1)]).any()


def test_nearest_progress_is_the_fraction_of_the_closest_reference_point():
    ref = np.array([[0, 0], [10, 0], [20, 0], [30, 0]], float)
    assert list(nearest_progress(ref, [1, 19, 31], [0, 3, -2])) == [0.0, 0.5, 0.75]


def race_history(status="AllClear", **kw):
    h = race({"BOT": 90.0}, {"BOT": 0}, laps=3, total=56, **kw)
    ev = pd.DataFrame([dict(ts=h["ts"].max() + pd.Timedelta(1, unit="s"), event="track_status", track_status=status,
                            status_code=1)])
    return pd.concat([h, ev], ignore_index=True)


def frame(flags, status="AllClear", flash=True):
    console = Console(width=120, height=40, file=io.StringIO(), force_terminal=False, color_system=None)
    console.print(watch.render(race_history(status), None, "x", 40, ({"10": 0.3}, "telemetry"), 120, flags=flags, flash=flash))
    return console.file.getvalue()


def test_the_banner_names_a_safety_car_or_vsc_and_stays_readable_in_both_blink_phases():
    assert "VIRTUAL SAFETY CAR" in frame({}, "VSCDeployed")
    assert "SAFETY CAR" in frame({}, "SCDeployed")
    assert "RED FLAG" in frame({}, "Red")
    assert "SAFETY CAR" not in frame({}, "AllClear")
    assert "VIRTUAL SAFETY CAR" in frame({}, "VSCDeployed", flash=False)       # the off phase is dimmer, never blank


def test_a_stopped_car_is_named_and_marked_on_the_line():
    out = frame({"10": "stopped"})
    assert "BOT stopped on track" in out and "✖" in out
    out = frame({"10": "slow"})
    assert "BOT slow on track" in out and "▼" in out
    assert "on track" not in frame({})


def test_a_stopped_car_stays_on_the_line_after_it_is_marked_out():
    h = race({"BOT": 90.0}, {"BOT": 0}, laps=3, total=56, skip_after={"BOT": 1})
    h["laps_down"] = 5
    out = watch.latest(h)
    out_flag = bool(out.loc["10", "out"])
    cars = watch.track_cars(h, {"10": 0.3}, {"10": "stopped"})
    assert [c[0] for c in cars] == ["BOT"] and cars[0][4] == "stopped"
    if out_flag:
        assert watch.track_cars(h, {"10": 0.3}) == []


def test_track_line_marks_flagged_cars():
    cars = [("BOT", 0.3, "green", False, "stopped"), ("VER", 0.31, "blue", True, None)]
    console = Console(width=60, file=io.StringIO(), force_terminal=False, color_system=None)
    console.print(watch.TrackLine(cars, flash=True))
    out = console.file.getvalue()
    assert "✖" in out and "◆" in out and "BOT" in out and "VER" in out


def test_a_car_stopped_for_longer_than_the_trail_stays_stopped():
    trail = {"77": samples([0] * 30, start=400)}                       # nothing but standing still in view
    h = history()
    assert hazards(trail, h, at(429)) == {}                            # no sign it was ever racing
    assert hazards(trail, h, at(429), ran={"77": [at(100), at(250)]}) == {"77": "stopped"}
    assert hazards(trail, h, at(429), ran={"77": [at(500)]}) == {}     # that was later than now (viewer delay)


def test_a_retired_car_is_shown_stopped_for_a_while_then_dropped():
    trail = {"77": samples([0] * 30, start=1000)}
    h = history(extra=[dict(ts=at(900), event="retired", driver="77")])
    ran = {"77": [at(100)]}
    assert hazards(trail, h, at(1029), ran=ran) == {"77": "stopped"}
    assert hazards(trail, h, at(1029 + 400), ran=ran) == {}


def test_the_final_lap_is_not_the_finish_until_the_flag_has_fallen():
    from live.race.battles import find_battles
    from live.race.hazards import chequered
    h = race({"AAA": 90.0, "BBB": 90.0}, {"AAA": 0, "BBB": 1.0}, laps=10, total=10)       # lap counter 10/10
    status, _ = find_battles(h)
    assert status != "race finished" and not chequered(h)
    flag = pd.DataFrame([dict(ts=h["ts"].max() + pd.Timedelta(60, unit="s"), event="race_control", message="CHEQUERED FLAG")])
    h = pd.concat([h, flag], ignore_index=True)
    assert chequered(h) and find_battles(h)[0] == "race finished"
    assert not race_is_live(h)
