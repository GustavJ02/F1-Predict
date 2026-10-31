"""The cutoff plot: the real cutoff of a finished part, reading predictions.csv and drawing, on small hand-made data."""
import matplotlib.pyplot as plt
import pandas as pd

from f1live.paths import Session
from live.quali import plot_cutoff as plot
from test_quali_dashboard import six_cars_into_q2

T0 = pd.Timestamp("2026-10-30T18:00:00Z")


def predictions(parts):
    rows = []
    for part in parts:
        for k in range(6):
            rows.append(dict(ts=T0 + pd.Timedelta(seconds=5 * k + 100 * part), part=part, current=None if k < 2 else 90.0 + part,
                             pred=90.2 + part, lo=89.9 + part, hi=90.6 + part, n_pending=6 - k))
    return pd.DataFrame(rows)


def test_final_cutoff_is_the_slowest_time_that_went_through_and_only_for_finished_parts():
    assert plot.final_cutoffs(six_cars_into_q2()) == {1: 91.5}           # Q1: four of six advance, the fourth is 91.5; Q2 is still running
    assert plot.final_cutoffs(None) == {}


def test_lap_label():
    assert plot.lap_label(78.493) == "1:18.493"
    assert plot.lap_label(60.0) == "1:00.000"


def test_read_predictions_ignores_another_models_file(tmp_path):
    other = tmp_path / "race.csv"
    pd.DataFrame([dict(ts=T0, kind="status", note="collecting pace data")]).to_csv(other, index=False)
    assert plot.read_predictions(other).empty
    assert plot.read_predictions(tmp_path / "missing.csv").empty
    mine = tmp_path / "quali.csv"
    predictions([1]).assign(ts=lambda d: d["ts"].astype(str)).to_csv(mine, index=False)
    got = plot.read_predictions(mine)
    assert len(got) == 6 and (got["part"] == 1).all() and pd.api.types.is_datetime64_any_dtype(got["ts"])


def test_draw_has_a_panel_per_part_and_nothing_without_projections():
    assert plot.draw(predictions([]).reindex(columns=["ts", "part", "current", "pred", "lo", "hi"]), {}) is None
    one = plot.draw(predictions([1]), {1: 91.5})
    two = plot.draw(predictions([1, 2]), {1: 91.5})
    assert len(one.axes) == 1 and len(two.axes) == 2
    assert one.axes[0].get_ylim()[0] < 89.9 + 1 and one.axes[0].get_ylim()[1] > 91.5          # the final cutoff is inside the view
    plt.close("all")


def test_render_writes_a_png_for_a_session_with_predictions_only(tmp_path):
    sess = Session("2026", "Mexico_City", "quali", tmp_path)
    sess.dir.mkdir(parents=True)
    predictions([1, 2]).to_csv(sess.predictions, index=False)
    out = tmp_path / "cutoff.png"
    assert plot.render(sess, out) and out.stat().st_size > 1000
    assert not plot.render(Session("2026", "Nowhere", "quali", tmp_path), tmp_path / "none.png")
