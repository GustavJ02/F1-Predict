"""The cutoff chart of the qualifying dashboard: which projections it draws, its size, and its place in the predictions panel."""
import numpy as np
import pandas as pd
from rich.console import Console

import live.quali.watch as watch
from live.quali import cutoff_chart as C
from test_quali_dashboard import six_cars_into_q2

T0 = pd.Timestamp("2026-10-30T18:00:00Z")


def predictions(parts=(1,), n=40, current_from=10):
    rows = []
    for part in parts:
        for k in range(n):
            drift = 0.3 * np.exp(-k / 12)
            rows.append(dict(ts=T0 + pd.Timedelta(5 * k + 600 * part, unit="s"), part=part, current=None if k < current_from else 90.4 - drift,
                             pred=90.2 + drift, lo=89.9 + drift, hi=90.7 + drift, n_pending=max(6 - k // 8, 0)))
    return pd.DataFrame(rows)


def lines_of(renderable, width):
    console = Console(width=width, force_terminal=False)
    with console.capture() as out:
        console.print(renderable)
    return out.get().rstrip("\n").split("\n")


def test_part_rows_picks_the_running_part_and_skips_q3_and_other_files():
    assert C.part_rows(None) == (None, None)
    assert C.part_rows(pd.DataFrame([dict(ts=T0, kind="status", note="x")])) == (None, None)             # another model's file
    part, d = C.part_rows(predictions((1, 2)))
    assert part == 2 and (d["part"] == 2).all() and d["ts"].is_monotonic_increasing
    q3 = pd.DataFrame(dict(ts=[T0], part=[3], current=[90.0], n_pending=[3], pole1=["VER"], p1=[0.5]))      # pole probabilities, no cutoff
    assert C.part_rows(q3) == (None, None)


def test_the_chart_is_none_without_room_or_projections():
    p = predictions()
    assert C.cutoff_chart(p, C.MIN_ROWS - 1) is None
    assert C.cutoff_chart(p, 8, width=C.MIN_WIDTH) is None                       # narrower than the labels and a few columns
    assert C.cutoff_chart(None, 8) is None
    assert C.cutoff_chart(p, 8, width=60) is not None


def test_the_chart_has_the_asked_rows_labels_axis_and_legend():
    out = lines_of(C.cutoff_chart(predictions(), 8), 60)
    assert len(out) == 8 + 2 and all(len(l) <= 60 for l in out)
    assert out[0].startswith(" 1:31.") and out[-3].startswith(" 1:29.")            # lap time labels at the top and bottom of the axis
    assert "18:10:00" in out[-2] and "now 18:13:15" in out[-2] and "projected cutoff" in out[-1] and "cutoff now" in out[-1]
    body = "".join(out[:8])
    assert any(0x2800 < ord(ch) <= 0x28FF for ch in body)                        # braille dots were drawn


def test_a_chart_with_a_single_projection_still_draws():
    out = lines_of(C.cutoff_chart(predictions(n=1, current_from=5), 6), 50)
    assert len(out) == 8


def test_the_panel_trades_the_history_lines_for_the_chart_only_when_there_is_room():
    h = six_cars_into_q2()
    p = predictions((2,))
    console = Console(width=170, height=60, force_terminal=False)
    text = {}
    for height in (60, 30, 26):
        with console.capture() as out:
            console.print(watch.render(h, p, "2026/Test/quali", height, ({}, None), 170, [("1", 0.0)], [0.3, 0.6], 0.0, None, "usual", {}, False))
        text[height] = out.get()
    assert "projected cutoff" in text[60] and "Q2 cutoff" in text[60] and text[60].count("pred ") == 0   # the chart replaces the list
    assert "projected cutoff" in text[30]                                       # a short terminal gets a shorter chart
    assert "projected cutoff" not in text[26]                                   # no room at all: the list of the latest projections
    assert text[26].count("pred ") >= 3 and "Q2 cutoff" in text[26]
