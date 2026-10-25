"""The terminal viewers render from a predictions DataFrame; no data files needed."""
import pandas as pd

import live.quali.watch
import live.race.watch
from f1live.watch import screen


def quali_rows():
    ts = pd.to_datetime(["2026-10-24T22:00:00Z", "2026-10-24T22:00:05Z"])
    return pd.DataFrame(dict(ts=ts, part=3, current=94.035, n_pending=[6, 3],
                             pole1="VER", p1=[0.3, 0.5], pole2="HAD", p2=[0.3, 0.3], pole3="LEC", p3=[0.2, 0.1]))


def test_quali_pole_view():
    text = screen("s", quali_rows(), live.quali.watch.render)
    assert "Pole probability" in text and "VER" in text and "50.0%" in text


def test_quali_cutoff_view():
    ts = pd.to_datetime(["2026-10-24T21:30:00Z"])
    df = pd.DataFrame(dict(ts=ts, part=2, current=95.3, pred=95.04, lo=94.9, hi=95.2, n_pending=6))
    text = screen("s", df, live.quali.watch.render)
    assert "Cutoff" in text and "95.04" in text


def test_race_view_shows_latest_batch_only():
    ts = pd.to_datetime(["2026-10-11T12:00:00Z"] * 2 + ["2026-10-11T12:00:05Z"] * 2)
    df = pd.DataFrame(dict(ts=ts, driver=["3", "16", "16", "3"], tla=["VER", "LEC", "LEC", "VER"],
                           predicted_position=[1, 2, 1, 2]))
    lines = live.race.watch.render(df)
    assert "12:00:05" in lines[0] and len(lines) == 2 + 3     # header, blank, table header + 2 rows


def test_waiting_screen():
    assert "waiting for predictions" in screen("s", None, live.quali.watch.render)
