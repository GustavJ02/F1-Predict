"""The terminal viewers render from a predictions DataFrame; no data files needed."""
import pandas as pd

import live.quali.watch
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


def test_waiting_screen():
    assert "waiting for predictions" in screen("s", None, live.quali.watch.render)


def test_delayed_view_shows_the_predictions_as_they_were():
    from f1live.watch import delayed
    df = quali_rows()                                  # two rows, 5 s apart
    assert delayed(df, 0) is df
    assert list(delayed(df, 5)["n_pending"]) == [6]
    assert delayed(df, 60) is None                     # nothing that old yet
    assert delayed(None, 5) is None
    text = screen("s", delayed(df, 5), live.quali.watch.render, 5)
    assert "DELAY 5 s" in text and "50.0%" not in text
