"""Regression test on the recorded 2026 Austin qualifying session (data/ is gitignored, so skipped if absent)."""
import pandas as pd
import pytest

from f1live.events import prepare
from f1live.feed import follow
from f1live.parse import FIELDS, Parser
from f1live.paths import Session
from live.quali.predict_elimination import predict
from live.quali.predict_pole import predict as predict_pole

RAW = Session("2026", "Austin", "quali").raw
AS_OF = "2026-10-24T21:40:28.476Z"
PART = 2
EXPECTED_CUTOFF = 95.0466      # median predicted cutoff, recorded after the '<NA>' fix

pytestmark = pytest.mark.skipif(not RAW.exists(), reason=f"{RAW} not found")


@pytest.fixture(scope="module")
def events():
    rows = []
    parser = Parser(rows.append)
    for line in follow(RAW, live=False):
        parser.feed(line)
    return pd.DataFrame(rows, columns=FIELDS)


def test_event_counts(events):
    counts = events["event"].value_counts()
    assert counts["lap"] == 178
    assert counts["lap_deleted"] == 5
    assert counts["session_part_start"] == 2


def test_cutoff_prediction(events):
    res = predict(prepare(events.copy()), PART, AS_OF, verbose=False)
    assert res["pred"] == pytest.approx(EXPECTED_CUTOFF, abs=0.01)
    assert res["n_pending"] == 6


def test_no_phantom_driver(events):
    assert prepare(events.copy())["driver"].isin(["<NA>", "nan"]).sum() == 0


def test_pole_prediction(events):
    rows = prepare(events.copy())
    late = predict_pole(rows, 3, "2026-10-24T22:00:30Z", verbose=False)    # six drivers still on a flying lap
    probs = [t["prob"] for t in late["top3"]]
    assert len(probs) == 3 and probs == sorted(probs, reverse=True) and sum(probs) <= 1
    assert late["current_driver"] == "12"
    done = predict_pole(rows, 3, verbose=False)                               # session over: pole is settled
    assert done["top3"][0]["tla"] == "ANT" and done["top3"][0]["prob"] == 1.0


AZ = Session("2025", "Azerbaijan", "quali")


@pytest.mark.skipif(not AZ.events.exists(), reason=f"{AZ.events} not found")
def test_pole_not_overconfident_after_red_flag():
    # Q3 red-flagged: three spoiled laps used to set the shift for the whole field, giving 99% to SAI (VER took pole)
    rows = prepare(pd.read_csv(AZ.events))
    res = predict_pole(rows, 3, "2025-09-20T13:41:00Z", verbose=False)
    assert res["top3"][0]["prob"] < 0.8
