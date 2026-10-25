"""Priors and the race projection, on synthetic races where the answer is known."""
import numpy as np
import pandas as pd
import pytest
from rich.console import Console
import io

from live.race import model, priors as P, watch
from live.race.projection import project, sample_stint
from test_battles import T0, race, with_flag

NEVER = {"age": [500], "surv": [0.5]}               # a stint curve that never stops within a race (half the draws are inf)


def priors(passing=(2.0, 3.0), sc=0.0, effect=0.0):
    """Minimal priors: no tyre wear differences, nobody stops, the given passing logit and safety-car rate."""
    return dict(
        lap_model=dict(fuel=0.0, offset={"SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.0}, wear={"SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.0},
                       lap_noise=0.2),
        stint=dict(pooled={c: NEVER for c in P.COMPOUNDS}, circuits={}),
        transitions={a: {b: 1 for b in P.COMPOUNDS} for a in P.COMPOUNDS},
        pit_loss=dict(pooled=22.0, circuits={}), passing=dict(a=passing[0], b=passing[1], circuits={"Test": effect}),
        safety=dict(sc_rate=sc, vsc_rate=0.0, sc_laps=4.0, vsc_laps=2.0, circuits={}),
        start={"SOFT": 0.2, "MEDIUM": 0.6, "HARD": 0.2})


def test_kaplan_meier_treats_censored_stints_as_at_least_that_long():
    s = pd.DataFrame({"length": [10, 20, 30, 30], "pit": [True, True, True, False]})
    km = P.kaplan_meier(s)
    assert km["age"] == [10, 20, 30]
    assert km["surv"] == pytest.approx([0.75, 0.5, 0.25])      # 4 at risk at 10, 3 at 20, 2 at 30 (one of them censored)


def test_sample_stint_is_conditional_and_can_be_never():
    rng = np.random.default_rng(0)
    km = {c: {"age": [10, 20, 30], "surv": [0.8, 0.5, 0.2]} for c in P.COMPOUNDS}
    out = sample_stint(km, np.zeros(4000, int), np.full(4000, 15.0), rng.random(4000), 4000)
    finite = out[np.isfinite(out)]
    assert finite.min() >= 16 and set(np.unique(finite)) <= {20.0, 30.0}      # lasted 15 laps already: next stop is 20 or 30
    assert np.isinf(out).mean() == pytest.approx(0.25, abs=0.03)              # S(30) / S(15) = 0.2 / 0.8 never stop
    assert np.isinf(sample_stint(km, np.zeros(10, int), np.full(10, 40.0), rng.random(10), 10)).all()   # past the longest known stint
    same = sample_stint(km, np.zeros(5, int), np.full(5, 0.0), np.full(5, 0.3), 5)
    assert len(set(same)) == 1                                                  # the same draw gives the same stint


def test_fit_logistic_recovers_the_passing_curve():
    rng = np.random.default_rng(1)
    n = 6000
    delta = rng.uniform(-1, 2, n)
    circuit = np.where(rng.random(n) < 0.5, "easy", "hard")
    p = 1 / (1 + np.exp(-(-3.0 + 2.0 * delta + np.where(circuit == "easy", 0.8, -0.8))))
    a, b, eff = P.fit_logistic(delta, rng.random(n) < p, circuit.tolist())
    assert b == pytest.approx(2.0, abs=0.3) and eff["easy"] - eff["hard"] == pytest.approx(1.6, abs=0.4)


def test_circuit_names_are_stable_across_years():
    assert P.circuit_of("2025", "Spanish") == "Barcelona" and P.circuit_of("2026", "Spanish") == "Spanish"
    assert P.circuit_of("2026", "Austin") == P.circuit_of("2025", "United_States") == "United_States"


def test_shipped_priors_are_sane():
    pri = P.load_priors()
    assert 0.02 < pri["lap_model"]["fuel"] < 0.08 and pri["lap_model"]["offset"]["SOFT"] < pri["lap_model"]["offset"]["HARD"]
    assert pri["passing"]["b"] > 0 and pri["passing"]["circuits"]["Monaco"] < pri["passing"]["circuits"]["Abu_Dhabi"]
    assert 10 < pri["pit_loss"]["pooled"] < 35 and 0 < pri["safety"]["sc_rate"] < 0.05
    assert sum(pri["start"].values()) == pytest.approx(1.0) and pri["start"]["MEDIUM"] > 0.3
    km = pri["stint"]["pooled"]["MEDIUM"]
    assert all(a < b for a, b in zip(km["age"], km["age"][1:])) and all(a >= b for a, b in zip(km["surv"], km["surv"][1:]))


def synth(gap_b=6.0, laps=10, total=40, pace_b=89.5):
    return race({"AAA": 90.0, pace_b and "BBB": pace_b, "CCC": 92.0}, {"AAA": 0, "BBB": gap_b, "CCC": gap_b + 8.0},
                laps=laps, total=total)


def test_faster_car_behind_wins_when_it_can_pass():
    out, info = project(synth(), priors(), "Test", n_sims=300)
    by = out.set_index("tla")
    assert info["laps_left"] == 30 and by.loc["BBB", "p_win"] > by.loc["AAA", "p_win"] > by.loc["CCC", "p_win"]
    assert out["p_win"].sum() == pytest.approx(1.0, abs=0.01) and list(out["proj_rank"]) == [1, 2, 3]
    assert out.iloc[0]["tla"] == "BBB"


def test_a_circuit_where_passing_is_impossible_keeps_the_order():
    out, _ = project(synth(), priors(effect=-40.0), "Test", n_sims=300)
    by = out.set_index("tla")
    assert by.loc["AAA", "p_win"] > 0.95 and list(out["tla"]) == ["AAA", "BBB", "CCC"]


def test_projection_is_repeatable_and_ranks_by_mean_position():
    a, _ = project(synth(), priors(), "Test", n_sims=200, seed=3)
    b, _ = project(synth(), priors(), "Test", n_sims=200, seed=3)
    pd.testing.assert_frame_equal(a, b)
    assert a["mean_pos"].is_monotonic_increasing


def test_safety_car_chance_and_nothing_to_project_at_the_flag():
    _, info = project(synth(), priors(sc=1.0), "Test", n_sims=50)
    assert info["p_sc"] > 0.99
    _, info0 = project(synth(), priors(sc=0.0), "Test", n_sims=50)
    assert info0["p_sc"] == 0.0
    out, why = project(synth(laps=10, total=10), priors(), "Test")
    assert out is None and why == "last lap"                    # final lap: the flag has not fallen
    out, why = project(with_flag(synth(laps=10, total=10)), priors(), "Test")
    assert out is None and why == "race finished"
    assert project(synth(laps=9, total=10), priors(), "Test")[1] == "last lap"


def test_hook_writes_projection_once_per_lap():
    h = synth()
    info = dict(race_lap=10, total_laps=40, session=None)
    model._last["key"] = None
    first = model.predict(pd.DataFrame({"x": [1]}), h, info)
    assert {"battle", "projection"} <= set(first["kind"]) or {"status", "projection"} <= set(first["kind"])
    proj = first[first["kind"] == "projection"]
    assert len(proj) == 3 and {"proj_rank", "p_win", "p_podium", "p_points", "laps_left", "p_sc"} <= set(proj.columns)
    second = model.predict(pd.DataFrame({"x": [1]}), h, info)             # same lap, same track status: no new projection
    assert "projection" not in set(second["kind"])
    third = model.predict(pd.DataFrame({"x": [1]}), synth(laps=11), dict(info, race_lap=11))
    assert "projection" in set(third["kind"])


def frame(history, preds, width=170):
    console = Console(width=width, height=44, file=io.StringIO(), force_terminal=False, color_system=None)
    console.print(watch.render(history, preds, "2026/Test/race", console.height, None, width))
    return console.file.getvalue()


def test_dashboard_shows_the_projection_panel():
    h = synth()
    model._last["key"] = None
    preds = model.predict(pd.DataFrame({"x": [1]}), h, dict(race_lap=10, total_laps=40, session=None))
    preds.insert(0, "ts", T0)
    for width in (170, 120):                                   # three columns, then the stacked layout
        text = frame(h, preds, width)
        assert "Projected finish" in text and "WIN%" in text and "PODIUM%" in text and "laps left" in text
        assert "safety car / VSC chance" in text
    assert "waiting for the first projection" in frame(h, pd.DataFrame([dict(ts=T0, kind="status", note="collecting pace data")]))
    done = pd.concat([preds, pd.DataFrame([dict(ts=T0 + pd.Timedelta(60, unit="s"), kind="status", note="race finished")])])
    assert "race finished: the standings are the result" in frame(h, done)
