"""The qualifying dashboard: classification order, prediction lines and a frame, on small hand-made data."""
import numpy as np
import pandas as pd
from rich.console import Console

import live.quali.watch as watch
from live.quali.classification import classification, part_bests, part_sizes

T0 = pd.Timestamp("2026-10-30T18:00:00Z")


def row(sec, part, driver, tla, best, entries, advance, event="snapshot", **kw):
    base = dict(ts=T0 + pd.Timedelta(sec, unit="s"), event=event, session_part=part, driver=driver, tla=tla, team="Ferrari",
                best_in_part_s=best, n_entries=entries, n_advance=advance, track_status="AllClear", clock_left_s=600.0,
                in_pit=True, compound="SOFT", tyre_laps=2)
    base.update(kw)
    return base


def frame(rows):
    cols = ["ts", "event", "session_part", "driver", "tla", "team", "best_in_part_s", "n_entries", "n_advance", "track_status",
            "clock_left_s", "in_pit", "compound", "tyre_laps", "lap_time_s", "lap_kind", "message", "race_lap", "total_laps",
            "track_temp", "air_temp", "rainfall", "status_code", "position", "best_lap_s"]
    return pd.DataFrame(rows).reindex(columns=cols)


def six_cars_into_q2():
    """Q1: six cars, the first four go through; in Q2 D (fourth in Q1) is quickest, A has no time yet, E and F are out."""
    cars = [("1", "AAA", 90.0), ("2", "BBB", 90.5), ("3", "CCC", 91.0), ("4", "DDD", 91.5), ("5", "EEE", 92.0), ("6", "FFF", 92.5)]
    rows = [row(10, 1, d, t, b, 6, 4) for d, t, b in cars]
    rows += [row(100, 2, "4", "DDD", 89.5, 4, 2), row(101, 2, "2", "BBB", 89.9, 4, 2), row(102, 2, "3", "CCC", 91.9, 4, 2),
             row(103, 2, "1", "AAA", None, 4, 2)]
    rows.append(row(99, 2, None, None, None, 4, 2, event="session_part_start"))
    return frame(rows)


def test_part_bests_and_sizes():
    h = six_cars_into_q2()
    best = part_bests(h)
    assert best.loc["4", 1] == 91.5 and best.loc["4", 2] == 89.5 and np.isnan(best.loc["1", 2]) and best[3].isna().all()
    assert part_sizes(h) == {1: (6, 4), 2: (4, 2)}


def test_classification_order_cutoff_and_eliminated_cars():
    table, current, advance = classification(six_cars_into_q2())
    assert current == 2 and advance == 2
    # Q2 by time (D, B, C), the car without a Q2 time yet (A) after them, then the two knocked out in Q1 (E, F) in their Q1 order
    assert list(table["tla"]) == ["DDD", "BBB", "CCC", "AAA", "EEE", "FFF"]
    assert list(table["part"]) == [2, 2, 2, 2, 1, 1]
    assert list(table["rank"]) == [1, 2, 3, 4, 5, 6]
    assert table.loc["4", "q1"] == 91.5 and table.loc["4", "q2"] == 89.5


def test_a_car_with_a_time_is_in_the_part_even_if_the_size_says_otherwise():
    h = six_cars_into_q2()
    h["n_entries"] = np.where(h["session_part"] == 2, 2, h["n_entries"])          # a late recording: the size is too small
    table, _, _ = classification(h)
    assert list(table["tla"][:3]) == ["DDD", "BBB", "CCC"]


def test_prediction_lines():
    ts = pd.to_datetime(["2026-10-24T21:30:00Z", "2026-10-24T21:30:05Z"])
    cutoff = pd.DataFrame(dict(ts=ts, part=2, current=95.3, pred=95.04, lo=94.9, hi=95.2, n_pending=6))
    text = "\n".join(str(l) for l in watch.prediction_lines(cutoff))
    assert "Q2 cutoff" in text and "95.04" in text
    pole = pd.DataFrame(dict(ts=ts, part=3, current=94.035, n_pending=3, pole1="VER", p1=0.5, pole2="HAD", p2=0.3, pole3="LEC", p3=0.1))
    text = "\n".join(str(l) for l in watch.prediction_lines(pole))
    assert "Pole probability" in text and "VER" in text and "50.0%" in text
    assert "no predictions" in str(watch.prediction_lines(None)[0])


def test_full_frame_with_and_without_telemetry():
    h = six_cars_into_q2()
    t = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    ref = np.c_[1000 * np.cos(t), 600 * np.sin(t)]
    console = Console(width=170, height=45, force_terminal=False)
    for progress, track in (({}, None), (({"4": 0.3, "2": 0.6}, "telemetry"), ref)):
        with console.capture() as out:
            console.print(watch.render(h, None, "2026/Test/quali", 45, progress if progress else ({}, None), 170, [("1", 0.0)],
                                       [0.3, 0.6], 0.0, None, "usual", {}, False, ref=track))
        text = out.get()
        assert "Q2" in text and "DDD" in text and "Classification" in text and "Predictions" in text
    assert "─" * 4 in text                                   # the cutoff line after P2
