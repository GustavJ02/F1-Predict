"""The qualifying pace prior: synthetic checks (always run) and the recorded Austin quali (skipped if data/ is absent)."""
import numpy as np
import pandas as pd
import pytest

from f1live.events import load
from f1live.paths import Session
from live.quali.predict_elimination import predict, simulate_cutoff
from live.quali.prior import Prior, expected_times, make_predictor
from pre.quali.parts import tables_from_laps


def prior(gaps, t_ref=90.0, sigma=0.3):
    return Prior(2, pd.Series(gaps), sigma, t_ref, 10)


def state(best, pending, tlas):
    return pd.DataFrame({"best": best, "pending": pending, "tla": tlas})


def test_expected_time_for_driver_without_a_time_is_t_ref_plus_gap():
    st = state([np.nan], [True], ["AAA"])
    center, extra = expected_times(prior({"AAA": 1.0}), st, mu=0.3, clock_left=900)
    assert center[0] == pytest.approx(90 * 1.01) and extra[0] == pytest.approx(0.3 / 100 * 90)


def test_t_ref_never_above_the_fastest_lap_already_set():
    st = state([88.0], [False], ["AAA"])
    center, _ = expected_times(prior({"AAA": 0.0}, t_ref=90.0), st, mu=0.3, clock_left=900)
    assert center[0] == pytest.approx(88.0 - 0.3)     # the 90.0 prior would put the expected lap above his own best


def test_unknown_driver_keeps_the_old_rule():
    st = state([np.nan, 90.0], [True, True], ["ZZZ", "AAA"])
    center, extra = expected_times(prior({"AAA": 0.0}), st, mu=0.3, clock_left=900)
    assert np.isnan(center[0]) and extra[0] > 0 and not np.isnan(center[1])


def test_pull_is_a_share_of_the_improvement_the_prior_sees():
    st = state([91.0, 91.0], [True, True], ["AAA", "BBB"])
    p = prior({"AAA": 0.0, "BBB": 5.0}, t_ref=89.0)
    c, _ = expected_times(p, st, mu=0.3, clock_left=900)
    assert c[1] == pytest.approx(91.0 - 0.3)          # prior expects no improvement beyond mu: unchanged
    assert 89.0 < c[0] < 91.0 - 0.3                   # prior expects a lot more: part of the way there


def test_simulation_without_prior_is_unchanged_and_with_prior_can_start_with_no_times():
    best = np.array([90.0, 91.0, 92.0, np.nan])
    kw = dict(pending=np.ones(4, bool), p_run=np.full(4, 0.9), mu=0.3, sigma=0.2, n_advance=2)
    assert simulate_cutoff(best, **kw)[0] == simulate_cutoff(best, **kw, center=None)[0]
    assert np.isnan(simulate_cutoff(np.full(4, np.nan), **kw)[0])
    median, _ = simulate_cutoff(np.full(4, np.nan), **kw, center=np.array([90.0, 90.5, 91.0, 91.5]), extra=np.full(4, .2))
    assert 90.0 < median < 91.5


def test_tables_from_laps_gap_is_relative_to_the_parts_fastest():
    laps = pd.DataFrame({"session_part": [1, 1, 1, 2], "tla": ["A", "B", "A", "A"], "team": ["x", "y", "x", "x"],
                         "lap_time_s": [90.0, 91.8, 90.5, 89.0]})
    t = tables_from_laps(laps)
    assert t[1].loc["B", "gap_pct"] == pytest.approx(2.0) and t[1].loc["A", "best_s"] == 90.0
    assert t[2].loc["A", "gap_pct"] == 0.0


AUSTIN = Session("2026", "Austin", "quali")
EXPECTED_WITH_PRIOR = 94.9932      # the same moment without the prior gives 95.0466 (test_regression)


@pytest.mark.skipif(not AUSTIN.events.exists(), reason=f"{AUSTIN.events} not found")
def test_austin_cutoff_with_prior():
    rows = load(AUSTIN.events)
    as_of = "2026-10-24T21:40:28.476Z"
    plain = predict(rows, 2, as_of, verbose=False)
    res = make_predictor(predict, "2026", "Austin")(rows, 2, as_of, verbose=False)
    assert res["pred"] == pytest.approx(EXPECTED_WITH_PRIOR, abs=0.01)
    assert res["n_pending"] == plain["n_pending"] == 6
    assert abs(res["pred"] - 95.0) < 1.0


def test_prior_is_built_once_per_part_even_when_it_cannot_be_built(monkeypatch):
    import live.quali.prior as pr
    calls = []
    monkeypatch.setattr(pr, "build_prior", lambda *a: calls.append(a[3]) or None)     # e.g. too few earlier weekends
    monkeypatch.setattr(pr, "available", lambda *a: None)
    monkeypatch.setattr(pr, "Tables", lambda av: None)
    seen = []
    f = make_predictor(lambda rows, part, as_of, verbose, prior: seen.append(prior) or {}, "2025", "Australian")
    rows = pd.DataFrame({"tla": ["AAA", "BBB"]})
    for _ in range(3):
        f(rows, 1, None, verbose=False)
    f(rows, 2, None, verbose=False)
    f(pd.DataFrame({"tla": ["AAA", "BBB", "CCC"]}), 1, None, verbose=False)         # a new driver: one rebuild
    assert calls == [1, 2, 1] and seen == [None] * 5
