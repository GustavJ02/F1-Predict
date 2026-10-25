"""Tyre sets: what each driver has left, and the optimizer keeping to it."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from play import optimize as O
from play.strategy import Plan
from play.tyres import ALLOCATION, TyreSets, add_tyre_args, configure_tyres, fresh_sets, parse_counts
from test_play import scenario

T0 = pd.Timestamp("2026-10-24T12:00:00Z")


def session_events(stints):
    """Lap rows of a session. stints: [(tla, stint number, compound, starts on a new set, laps)]."""
    rows = []
    for i, (tla, stint, compound, new, laps) in enumerate(stints):
        for lap in range(laps):
            rows.append(dict(ts=T0 + pd.Timedelta(float(i * 1000 + lap), unit="s"), event="lap", driver=str(10 + hash(tla) % 50),
                             tla=tla, stint=stint, compound=compound, tyre_new=bool(new) if lap == 0 else False, tyre_laps=lap + 1))
    df = pd.DataFrame(rows)
    df["driver"] = df["tla"].map({t: str(i) for i, t in enumerate(sorted(set(df["tla"])))})
    return df


def test_counts_parse():
    assert parse_counts("H2,m3, S8") == {"HARD": 2, "MEDIUM": 3, "SOFT": 8} and parse_counts("S0") == {"SOFT": 0}
    with pytest.raises(ValueError):
        parse_counts("X2")
    with pytest.raises(ValueError):
        parse_counts("S")


def test_only_stints_that_start_on_a_new_set_use_one_up():
    h = session_events([("HAM", 1, "SOFT", True, 3), ("HAM", 2, "SOFT", False, 4), ("HAM", 3, "MEDIUM", True, 2),
                        ("VER", 1, "HARD", True, 5), ("VER", 2, "SOFT", True, 2)])
    c = fresh_sets(h)
    assert c[("HAM", "SOFT")] == 1 and c[("HAM", "MEDIUM")] == 1 and c[("VER", "HARD")] == 1 and c[("VER", "SOFT")] == 1
    assert fresh_sets(h.drop(columns=["tyre_new"])) == {}                        # a recording without tyre data


def test_available_sets_and_plan_checks():
    t = TyreSets(dict(ALLOCATION["normal"]), {"ANT": {"SOFT": 8, "MEDIUM": 1, "HARD": 1}, "VER": {"SOFT": 5, "MEDIUM": 2, "HARD": 1}})
    assert t.available("ANT") == {"SOFT": 0, "MEDIUM": 2, "HARD": 1} and t.available("VER")["SOFT"] == 3
    assert t.available("NEW") == ALLOCATION["normal"]                              # no usage recorded: the full allocation
    assert t.check("ANT", Plan.parse("M-24-H")) == [] and t.check("ANT", Plan.parse("S-24-H")) == ["needs 1 SOFT set, 0 available"]
    assert t.check("ANT", Plan.parse("M-10-H-30-H")) == ["needs 2 HARD sets, 1 available"]
    t.overrides["ANT"] = {"SOFT": 2}
    assert t.available("ANT")["SOFT"] == 2 and t.check("ANT", Plan.parse("S-24-H")) == []
    t.allow_used = {"HARD": 1}
    assert t.available("VER")["HARD"] == 1 + 1 and "set by hand" in t.describe("ANT") and "used so far: 5 soft" in t.describe("VER")
    over = TyreSets(dict(ALLOCATION["normal"]), {"X": {"HARD": 5}})
    assert over.available("X")["HARD"] == 0                                         # never negative


def test_in_a_race_under_way_the_set_on_the_car_is_not_one_to_choose():
    t = TyreSets(dict(ALLOCATION["normal"]), {"HAM": {"MEDIUM": 3}}, live=True)     # all three mediums used: he is on one now
    assert t.available("HAM")["MEDIUM"] == 0
    assert t.check("HAM", Plan.parse("M-30-H")) == []                              # the M is the set he is on; the H is the choice
    assert TyreSets(dict(ALLOCATION["normal"]), {"HAM": {"MEDIUM": 3}}).check("HAM", Plan.parse("M-30-H")) != []   # before the race it counts


def test_usage_is_read_from_the_weekend_sessions(tmp_path):
    for name, stints in {"fp1": [("HAM", 1, "SOFT", True, 5), ("HAM", 2, "SOFT", True, 5)], "quali": [("HAM", 1, "SOFT", True, 3)],
                         "fp2": [("HAM", 1, "HARD", True, 6)]}.items():
        d = tmp_path / "2026" / "Test" / name
        d.mkdir(parents=True)
        session_events(stints).to_csv(d / "events.csv", index=False)
    t = TyreSets.from_weekend(tmp_path, "2026", "Test")
    assert t.used["HAM"] == {"SOFT": 3, "HARD": 1} and t.allocation == ALLOCATION["normal"]
    sprint = tmp_path / "2026" / "Test" / "sprint"
    sprint.mkdir()
    session_events([("HAM", 1, "MEDIUM", True, 4)]).to_csv(sprint / "events.csv", index=False)
    s = TyreSets.from_weekend(tmp_path, "2026", "Test")
    assert s.allocation == ALLOCATION["sprint"] and s.used["HAM"]["MEDIUM"] == 1                   # a sprint weekend has other sets
    assert TyreSets.from_weekend(tmp_path, "2026", "Test", allocation={"HARD": 1, "MEDIUM": 1, "SOFT": 1}).allocation["SOFT"] == 1


def parse(*argv):
    ap = argparse.ArgumentParser()
    add_tyre_args(ap)
    return ap.parse_args(list(argv))


def test_command_line_options_change_what_a_driver_has():
    sc = scenario()
    sc.tyres = TyreSets(dict(ALLOCATION["normal"]), {}, live=True)
    configure_tyres(sc, parse("--sets", "AAA=S1,M0", "--ban", "BBB=SH", "--allocation", "H1,M1,S1", "--allow-used", "M2"))
    assert sc.tyres.allocation == {"HARD": 1, "MEDIUM": 1, "SOFT": 1}
    assert sc.tyres.available("AAA") == {"SOFT": 1, "MEDIUM": 0, "HARD": 1}                       # M0 set by hand beats allow-used
    assert sc.tyres.available("BBB") == {"SOFT": 0, "MEDIUM": 3, "HARD": 0}
    configure_tyres(sc, parse("--tyres", "none"))
    assert sc.tyres is None
    with pytest.raises(ValueError):
        sc2 = scenario()
        sc2.tyres = TyreSets(dict(ALLOCATION["normal"]), {})
        configure_tyres(sc2, parse("--ban", "AAA=X"))


def test_the_optimizer_only_suggests_plans_the_driver_has_tyres_for_and_prices_the_limit():
    sc = scenario(wear={c: 0.003 for c in ("SOFT", "MEDIUM", "HARD")})     # soft is 0.4 s a lap faster and hardly wears: attractive
    free_table, free_info = O.optimize(sc, "BBB", "position", stops=(1,), step=5, top=3, search_sims=150, final_sims=300, jobs=1,
                                       verbose=False, tyres=None)
    assert "SOFT" in str(free_table.loc[0, "plan"].compounds) and free_info["unrestricted"] is None and free_info["tyres"] is None
    sc.tyres = TyreSets(dict(ALLOCATION["normal"]), {"BBB": {"SOFT": 8}}, live=True)            # no soft sets left
    table, info = O.optimize(sc, "BBB", "position", stops=(1,), step=5, top=5, search_sims=150, final_sims=300, jobs=1, verbose=False)
    assert all("SOFT" not in p.compounds[1:] for p in table["plan"])                             # live: the car's own set doesn't count
    assert info["unrestricted"] and info["unrestricted"]["missing"] and info["unrestricted"]["cost"] >= 0
    assert "soft 0" in info["tyres"]
    only_one = TyreSets(dict(ALLOCATION["normal"]), {"BBB": {"SOFT": 7, "MEDIUM": 2}}, live=True)   # 1 soft, 1 medium, 2 hard left
    sc.tyres = only_one
    t3, _ = O.optimize(sc, "BBB", "position", stops=(2,), step=8, top=20, search_sims=100, final_sims=200, jobs=1, verbose=False)
    for p in t3["plan"]:
        assert all(p.compounds[1:].count(c) <= only_one.available("BBB")[c] for c in ("SOFT", "MEDIUM", "HARD"))


def test_no_possible_strategy_is_an_error_that_says_why():
    sc = scenario()
    sc.tyres = TyreSets(dict(ALLOCATION["normal"]), {}, overrides={"BBB": {"SOFT": 0, "MEDIUM": 0, "HARD": 0}}, live=True)
    with pytest.raises(ValueError, match="no strategy that fits the tyres left"):
        O.optimize(sc, "BBB", "position", stops=(1,), step=8, search_sims=50, final_sims=100, jobs=1, verbose=False)


def test_the_real_austin_weekend_has_the_expected_usage():
    if not Path("data/2026/Austin/fp1/events.csv").exists():
        pytest.skip("Austin sessions not available")
    t = TyreSets.from_weekend(Path("data"), "2026", "Austin")
    assert t.allocation == ALLOCATION["normal"] and t.available("ANT")["SOFT"] == 0 and t.available("VER")["SOFT"] >= 2
    assert all(0 <= v <= 8 for d in t.used.values() for v in d.values())
