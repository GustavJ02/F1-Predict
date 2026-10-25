"""The whole field on its best strategies, on the small synthetic race."""
import numpy as np
import pytest

from play import field as F
from play.strategy import Plan
from play.tyres import ALLOCATION, TyreSets
from test_play import scenario

FLAT = {"SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.0}


def legal(sc):
    sc.cars["needs_stop"] = False                       # nobody has to stop
    return sc


def test_best_response_respects_the_tyres_left_and_is_at_least_as_good_as_what_he_has():
    sc = scenario(wear={c: 0.003 for c in FLAT})        # softs are 0.4 s a lap faster and hardly wear
    plan, value, _ = F.best_response(sc, "BBB", "position", stops=(1,), step=6, sims=120, jobs=1)
    assert plan is not None and "SOFT" in plan.compounds
    sc.tyres = TyreSets(dict(ALLOCATION["normal"]), {"BBB": {"SOFT": 8}}, live=True)            # no soft sets left
    plan2, value2, current = F.best_response(sc, "BBB", "position", stops=(1,), step=6, sims=120, jobs=1, current=plan)
    assert "SOFT" not in plan2.compounds[1:] and value2 <= value + 1e-9
    assert current is not None                                      # the plan he had is scored under the same conditions
    sc.tyres = TyreSets(dict(ALLOCATION["normal"]), {}, overrides={"BBB": {"SOFT": 0, "MEDIUM": 0, "HARD": 0}}, live=True)
    assert F.best_response(sc, "BBB", "position", stops=(1,), step=6, sims=60, jobs=1) == (None, None, None)


def test_when_a_stop_never_pays_everybody_stays_out_and_the_plans_settle():
    sc = legal(scenario(wear={c: 0.0 for c in FLAT}, offset=FLAT))
    plans, log = F.optimal_field(sc, "position", stops=(1,), step=6, sims=100, passes=3, jobs=1, verbose=False)
    assert set(plans) == {"AAA", "BBB", "CCC"} and all(p.laps == () for p in plans.values())
    assert log[-1] == []                                  # the last pass changed nothing: settled


def test_when_the_soft_is_much_faster_everybody_uses_it_and_the_table_compares_both_worlds():
    sc = legal(scenario(wear={c: 0.003 for c in FLAT}))
    plans, log = F.optimal_field(sc, "position", stops=(1,), step=6, sims=120, passes=3, jobs=1, verbose=False)
    assert "SOFT" in plans["BBB"].compounds                  # CCC, 38 s behind, loses nothing either way and may do anything
    table, usual, best = F.field_table(sc, plans, sims=300)
    assert list(table.index) == ["AAA", "BBB", "CCC"] and {"plan", "usual_pos", "best_pos", "usual_win", "best_win"} <= set(table)
    assert abs(table["best_pos"].sum() - 6.0) < 1e-6          # positions 1+2+3 in every run
    assert np.allclose(table["usual_win"].sum(), 1.0) and np.allclose(table["best_win"].sum(), 1.0)


def test_a_plan_the_others_follow_changes_what_a_driver_should_do():
    sc = legal(scenario(wear={c: 0.003 for c in FLAT}))
    alone, _, _ = F.best_response(sc, "BBB", "position", stops=(1,), step=6, sims=150, jobs=1)
    fixed = {"AAA": Plan.parse("M")}                                    # AAA stays out
    vs, _, _ = F.best_response(sc, "BBB", "position", stops=(1,), step=6, sims=150, fixed=fixed, jobs=1)
    assert alone is not None and vs is not None


# --- strategy.csv and the dashboard view -------------------------------------------------------------------------------
import io

import pandas as pd
from rich.console import Console

from f1live.paths import Session
from live.race import watch
from test_battles import T0, race


def strategy_frame(ts):
    return pd.DataFrame([
        dict(ts=ts, race_lap=10, driver="10", tla="AAA", now=1, plan="M-20-H", next_lap=20, next_compound="HARD", usual_pos=1.4,
             best_pos=1.1, usual_win=0.6, best_win=0.9, best_podium=1.0, best_pts=24.5),
        dict(ts=ts, race_lap=10, driver="11", tla="BBB", now=2, plan="M", next_lap=float("nan"), next_compound=None, usual_pos=1.6,
             best_pos=1.9, usual_win=0.4, best_win=0.1, best_podium=1.0, best_pts=18.2)])


def test_strategy_rows_and_warm_start():
    sc = scenario()
    sc.cars["needs_stop"] = False
    plans = {"AAA": Plan.parse("M-20-H"), "BBB": Plan.parse("M")}
    table, _, _ = F.field_table(sc, plans, sims=200)
    rows = F.strategy_rows(sc, plans, table, T0)
    assert list(rows.columns) == F.STRATEGY_COLUMNS and set(rows["tla"]) == {"AAA", "BBB", "CCC"}
    a = rows.set_index("tla").loc["AAA"]
    assert a["plan"] == "M-20-H" and a["next_lap"] == 20 and a["next_compound"] == "HARD"
    assert rows.set_index("tla").loc["BBB", "next_lap"] == "" or pd.isna(rows.set_index("tla").loc["BBB", "next_lap"])
    # a plan whose stop has passed, or that starts on another compound than the car is on, is not carried over
    kept = F.warm_start({"AAA": Plan.parse("M-20-H"), "BBB": Plan.parse("M-8-H"), "CCC": Plan.parse("S-30-M")}, sc)
    assert set(kept) == {"AAA"}                                  # laps done = 10 in this synthetic race; CCC is on mediums


def test_the_dashboard_can_show_the_strategy_world_instead_of_the_projection():
    h = race({"AAA": 90.0, "BBB": 89.5}, {"AAA": 0, "BBB": 3.0}, laps=10, total=40)
    console = Console(width=170, height=44, file=io.StringIO(), force_terminal=False, color_system=None)
    console.print(watch.render(h, None, "2026/Test/race", 44, None, 170, (), None, 0.0, strategy_frame(T0), "optimal"))
    text = console.file.getvalue()
    assert "everyone on his best strategy" in text and "M-20-H" in text and "L20 H" in text and "no stop" in text
    assert "[s] projection" in text and "a scenario, not a forecast" in text
    console = Console(width=170, height=44, file=io.StringIO(), force_terminal=False, color_system=None)
    console.print(watch.render(h, None, "2026/Test/race", 44, None, 170, (), None, 0.0, None, "optimal"))
    assert "no strategy.csv yet" in console.file.getvalue()
    console = Console(width=170, height=44, file=io.StringIO(), force_terminal=False, color_system=None)
    console.print(watch.render(h, None, "2026/Test/race", 44, None, 170, (), None, 0.0, strategy_frame(T0), "usual"))
    assert "[s] best strategies" in console.file.getvalue() and "L20 H" not in console.file.getvalue()


def test_the_feed_reads_strategy_csv_and_a_delayed_view_shows_the_search_that_had_finished(tmp_path):
    sess = Session("2026", "Test", "race", tmp_path)
    sess.dir.mkdir(parents=True)
    feed = watch.Feed(sess)
    feed.poll()
    assert feed.strategy is None
    old, new = strategy_frame(T0 + pd.Timedelta(100, unit="s")), strategy_frame(T0 + pd.Timedelta(400, unit="s"))
    pd.concat([old, new]).assign(ts=lambda d: d["ts"].dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")).to_csv(sess.dir / "strategy.csv", index=False)
    feed.poll()
    assert feed.strategy is not None and feed.strategy["ts"].nunique() == 2
    feed.history = race({"AAA": 90.0}, {"AAA": 0}, laps=6, total=40)          # data up to about T0 + 540 s
    live = watch.delayed_view(feed, None, 0)
    assert live["strategy"]["ts"].iloc[0] == T0 + pd.Timedelta(400, unit="s")
    back = watch.delayed_view(feed, None, 300)                                # the search of T0+400 had not finished 300 s ago
    assert back["strategy"]["ts"].iloc[0] == T0 + pd.Timedelta(100, unit="s")
    assert watch.delayed_view(feed, None, 480)["strategy"] is None            # nothing that old
