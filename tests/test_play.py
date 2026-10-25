"""The strategy playground and optimizer, on a small synthetic race where the answer is known."""
import numpy as np
import pandas as pd
import pytest

from f1live.paths import Session
from play import optimize as O
from play.playground import describe, parse_plan
from play.scenario import Scenario, compare, parse_tweak
from play.strategy import Plan
from test_projection import priors, synth


def scenario(sims=400, wear=None, offset=None, **kw):
    pri = priors(**kw)
    pri["lap_model"]["wear"] = wear or {"SOFT": 0.30, "MEDIUM": 0.05, "HARD": 0.02}       # soft tyres die quickly
    pri["lap_model"]["offset"] = offset or {"SOFT": -0.4, "MEDIUM": 0.0, "HARD": 0.1}
    return Scenario.live(Session("2026", "Test", "race"), history=synth(), sims=sims, priors=pri)


def test_plan_text_roundtrip_and_rules():
    p = Plan.parse("m-18-h-40-s")
    assert str(p) == "M-18-H-40-S" and p.compounds == ("MEDIUM", "HARD", "SOFT") and p.laps == (18, 40)
    assert p.stops == [(18, 1 + 1), (40, 0)] and p.legal() and not Plan.parse("M").legal() and Plan.parse("H").laps == ()
    with pytest.raises(ValueError, match="compound"):
        Plan.parse("X-10-H")
    with pytest.raises(ValueError, match="alternate"):
        Plan.parse("M-10")
    with pytest.raises(ValueError, match="after lap"):
        Plan.parse("M-20-H-15-S").validate(56)
    with pytest.raises(ValueError, match="before the last lap"):
        Plan.parse("M-56-H").validate(56)
    with pytest.raises(ValueError, match="after lap 10"):
        Plan.parse("M-8-H").validate(40, done=10)
    Plan.parse("M-25-H").validate(56)


def test_command_line_pieces():
    assert parse_tweak("ham.pace=-0.2") == ("HAM", "pace", -0.2) and parse_tweak("sc=2") == (None, "sc", 2.0)
    with pytest.raises(ValueError):
        parse_tweak("HAM.pace")
    assert parse_plan("ham=S-10-H")[0] == "HAM"
    assert "illegal" in describe("AAA", Plan.parse("M"), 40) and "illegal" not in describe("AAA", Plan.parse("M-20-H"), 40)


def test_runs_are_repeatable_and_identical_runs_compare_as_zero():
    sc = scenario()
    a, b = sc.run(), sc.run()
    assert np.array_equal(a.finish, b.finish)
    c = compare(a, b, ["AAA", "BBB"])
    assert (c[["d_pos", "d_win", "d_points", "se_pos"]] == 0).all().all()


def test_a_change_to_one_car_leaves_cars_it_never_meets_exactly_unchanged():
    sc = scenario()
    base = sc.run()
    alt = sc.run(plans={"CCC": Plan.parse("M-15-H")}, tweaks={"CCC": {"pace": -0.2}})       # CCC runs 38 s behind the others
    assert np.array_equal(base.column("AAA"), alt.column("AAA")) and np.array_equal(base.column("BBB"), alt.column("BBB"))
    assert (alt.out.set_index("tla").loc["CCC", "stops_left"]) == pytest.approx(1.0)


def test_tweaks_do_what_they_say():
    sc = scenario(passing=(2.0, 3.0))
    base = sc.run()
    faster = sc.run(tweaks={"AAA": {"pace": -1.0}})
    assert compare(base, faster, ["AAA"]).loc["AAA", "d_pos"] <= 0 and compare(base, faster, ["AAA"]).loc["AAA", "d_win"] > 0
    slow_stops = sc.run(plans={"AAA": Plan.parse("M-20-H")}, tweaks={"AAA": {"pit": 0.0}})
    slower_stops = sc.run(plans={"AAA": Plan.parse("M-20-H")}, tweaks={"AAA": {"pit": 15.0}})
    assert compare(slow_stops, slower_stops, ["AAA"]).loc["AAA", "d_pos"] >= 0
    better_tyres = sc.run(plans={"AAA": Plan.parse("S-20-H")}, tweaks={"AAA": {"wear": 0.2}})
    plain = sc.run(plans={"AAA": Plan.parse("S-20-H")})
    assert compare(plain, better_tyres, ["AAA"]).loc["AAA", "d_pos"] <= 0


def test_global_settings_and_bad_input():
    sc = scenario(sc=0.01)
    assert sc.run().info["p_sc"] < sc.run(sc=50).info["p_sc"]
    with pytest.raises(ValueError, match="not in the race"):
        sc.run(plans={"ZZZ": Plan.parse("M-20-H")})
    with pytest.raises(ValueError, match="after lap 10"):
        sc.run(plans={"AAA": Plan.parse("M-5-H")})
    with pytest.raises(ValueError, match="grid"):
        sc.run(grid={"AAA": 2})                                  # a race under way has no grid


def test_a_pit_stop_costs_time_when_nothing_else_changes():
    sc = scenario(passing=(2.0, 3.0))
    sc.cars["needs_stop"] = False                     # nobody is forced to stop in the baseline, so a stop is pure cost
    base, alt = sc.run(), sc.run(plans={"AAA": Plan.parse("M-20-H")})
    assert compare(base, alt, ["AAA"]).loc["AAA", "base_win"] > compare(base, alt, ["AAA"]).loc["AAA", "alt_win"]


def test_objectives_and_tyre_survival():
    f = np.array([1, 1, 3, 11])
    assert O.objective_value(f, "win") == 0.5 and O.objective_value(f, "podium") == 0.75 and O.objective_value(f, "position") == -4.0
    assert O.objective_value(f, "points") == pytest.approx((25 + 25 + 15 + 0) / 4)
    with pytest.raises(ValueError):
        O.objective_value(f, "nonsense")
    pri = priors()
    pri["stint"]["pooled"]["SOFT"] = {"age": [10, 20], "surv": [0.8, 0.3]}
    assert O.tyre_survival(pri, "Test", "SOFT", 5) == 1.0 and O.tyre_survival(pri, "Test", "SOFT", 15) == 0.8 \
        and O.tyre_survival(pri, "Test", "SOFT", 25) == 0.3
    assert len(O.sequences(1)) == 6 and all(len(set(s)) == 2 for s in O.sequences(1)) and len(O.sequences(0, must_change=False)) == 3
    assert O.sequences(1, first="HARD") == [("HARD", "SOFT"), ("HARD", "MEDIUM")]
    assert O.lap_grids(2, 5, 30, 5, 10) == [(5, 15), (5, 20), (5, 25), (5, 30), (10, 20), (10, 25), (10, 30), (15, 25), (15, 30), (20, 30)]


def test_optimizer_prefers_not_burning_the_soft_tyres_and_returns_windows():
    sc = scenario()
    table, info = O.optimize(sc, "BBB", "position", stops=(1,), step=4, top=5, search_sims=150, final_sims=300, jobs=1,
                             verbose=False)
    assert list(table["value"]) == sorted(table["value"], reverse=True) and len(table) == 5
    best = table.loc[0, "plan"]
    assert best.legal() and not best.compounds[0] == "SOFT" or best.laps[0] < 20     # wear 0.30 s/lap: never a long soft stint
    w = info["windows"][0]
    assert w["first"] <= w["lap"] <= w["last"] and info["baseline"]["mean_pos"] > 0
    assert (table["life"] > 0).all() and "d_points" in table and info["scenario"].startswith("2026/Test/race")


def test_a_driver_who_is_already_legal_may_stay_out_and_that_wins_when_stops_are_expensive():
    flat = {"SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.0}
    sc = scenario(wear={c: 0.003 for c in flat}, offset=flat)               # tyres hardly wear and are equally fast: a 22 s stop never pays
    sc.cars["needs_stop"] = False                               # has already used two compounds
    table, info = O.optimize(sc, "BBB", "position", stops=(1,), step=6, top=3, search_sims=100, final_sims=300, jobs=1,
                             verbose=False)
    assert table.loc[0, "stops"] == 0
    assert info["windows"] == [] and list(info["by_stops"]["stops"])[0] == 0
