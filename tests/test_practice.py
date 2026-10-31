"""Practice dashboard: flags, car modes, long runs, the circuit map and a full frame, on small hand-made data."""
import numpy as np
import pandas as pd
from rich.console import Console

import live.practice.watch as watch
from live.practice import status
from live.practice.trackmap import TrackMap, fit

T0 = pd.Timestamp("2026-10-30T18:30:00Z")


def row(sec, event, driver=None, **kw):
    base = dict(ts=T0 + pd.Timedelta(sec, unit="s"), event=event, driver=driver, track_status="AllClear", clock_left_s=3000.0,
                tla=None, team="Ferrari", in_pit=False, best_lap_s=None, position=1, stint=1, compound="SOFT", tyre_laps=1)
    base.update(kw)
    return base


def frame(rows):
    cols = ["ts", "event", "driver", "tla", "team", "lap_kind", "lap_time_s", "best_lap_s", "in_pit", "position", "stint", "compound",
            "tyre_laps", "track_status", "status_code", "message", "clock_left_s", "track_temp", "air_temp", "rainfall",
            "race_lap", "total_laps", "gap_to_leader_s", "interval_s", "n_pit_stops", "pit_lane_s"]
    return pd.DataFrame(rows).reindex(columns=cols)


def test_yellow_sectors_and_pit_exit():
    h = frame([row(0, "race_control", message="GREEN LIGHT - PIT EXIT OPEN"),
               row(1, "race_control", message="YELLOW IN TRACK SECTOR 7"),
               row(2, "race_control", message="DOUBLE YELLOW IN TRACK SECTOR 8"),
               row(3, "race_control", message="CLEAR IN TRACK SECTOR 7"),
               row(4, "race_control", message="CAR 5 (BOR) LAP DELETED - TRACK LIMITS AT TURN 4")])
    assert status.yellow_sectors(h) == {8: "DOUBLE YELLOW"}
    assert status.pit_exit(h) == "open"
    assert all("LAP DELETED" not in m for _, m in status.recent_messages(h))
    assert status.clock_left(h) == 3000.0


def drivers():
    return frame([row(0, "snapshot", d, tla=t, best_lap_s=b) for d, t, b in (("1", "AAA", 90.0), ("2", "BBB", 91.0), ("3", "CCC", None),
                                                                              ("4", "DDD", 92.0), ("5", "EEE", 90.5))]
                 + [row(10, "lap", "1", tla="AAA", lap_kind="push", lap_time_s=90.0, best_lap_s=90.0),
                    row(10, "lap", "2", tla="BBB", lap_kind="other", lap_time_s=91.0, best_lap_s=91.0),
                    row(10, "lap", "4", tla="DDD", lap_kind="other", lap_time_s=92.0, best_lap_s=92.0),
                    row(10, "lap", "5", tla="EEE", lap_kind="other", lap_time_s=90.5, best_lap_s=90.5),
                    row(20, "pit_out", "3", tla="CCC"),
                    row(60, "snapshot", "4", tla="DDD", in_pit=True, best_lap_s=92.0)])


def test_car_modes():
    h = drivers()                          # latest row is at +60 s; everybody's last line crossing is at +10 s: 50 s into the lap
    modes = status.car_modes(h, {"1": 0.55, "2": 0.4, "3": 0.5}, {})
    assert modes["1"] == "PUSH"            # 50 s / 0.55 = 90.9 s, within 2 % of his 90.0 s best
    assert modes["2"] == "SLOW"            # 50 s / 0.4 = 125 s, vs a 91 s best
    assert modes["3"] == "OUT"             # left the pits, no lap since
    assert modes["4"] == "PIT"
    assert status.car_modes(h, {"1": 0.55}, {"1": "stopped"})["1"] == "STOPPED"
    assert status.car_modes(h, {"1": 0.55}, {}, measured=False)["1"] == "RUN"       # no pace judgement from timing estimates
    assert status.car_modes(h, {"1": 0.1}, {})["1"] == "RUN"                        # too early in the lap to tell


def test_long_run_pace():
    rows = [row(i * 100, "lap", "1", tla="AAA", lap_kind="other", lap_time_s=100.0 + 0.1 * i, stint=2, compound="MEDIUM",
                tyre_laps=i + 1, best_lap_s=95.0) for i in range(6)]
    rows.append(row(700, "lap", "1", tla="AAA", lap_kind="push", lap_time_s=95.0, stint=1, compound="SOFT", best_lap_s=95.0))
    runs = status.long_runs(frame(rows))
    assert runs.loc["1", "run_compound"] == "MEDIUM" and runs.loc["1", "run_laps"] == 6
    assert abs(runs.loc["1", "run_deg_s"] - 0.1) < 1e-6
    assert status.best_compound(frame(rows))["1"] == "SOFT"


def circle(n=200):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.c_[1000 * np.cos(a), 600 * np.sin(a)]


def test_fit_keeps_the_track_inside_the_box():
    to_dots = fit(circle(), 80, 40)
    col, row_ = to_dots(*circle().T)
    assert col.min() >= 0 and col.max() < 80 and row_.min() >= 0 and row_.max() < 40


def test_track_map_draws_cars_but_not_those_in_the_pits():
    cars = [("AAA", 0.25, "red", "PUSH"), ("BBB", 0.5, "blue", "RUN"), ("CCC", 0.75, "green", "PIT"), ("DDD", 0.1, "white", "STOPPED")]
    console = Console(width=60, height=20, force_terminal=False)
    with console.capture() as out:
        console.print(TrackMap(circle(), cars, sectors=[0.3, 0.6], corners=[("1", 0.0)]), height=20)
    text = out.get()
    assert "AAA" in text and "BBB" in text and "DDD" in text and "✖" in text and "CCC" not in text
    assert any(0x2800 <= ord(c) < 0x2900 for c in text)              # the outline


def test_full_frame_without_telemetry_and_with():
    h = drivers()
    console = Console(width=170, height=45, force_terminal=False)
    for progress, ref in (({}, None), (({"1": 0.3, "2": 0.5}, "telemetry"), circle())):
        with console.capture() as out:
            console.print(watch.render(h, None, "2026/Test/fp1", 45, progress if progress else ({}, None), 170, [("1", 0.0)],
                                       [0.3, 0.6], 0.0, None, "usual", {}, False, ref=ref))
        text = out.get()
        assert "AAA" in text and "50:00 LEFT" in text and "Timesheet" in text
    assert "AAA" in text.split("Circuit")[1]                           # drawn on the map in the second frame
