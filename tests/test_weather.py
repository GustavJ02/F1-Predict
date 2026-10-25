"""Weather: rain detection, the track temperature in the lap model, and what the race model does when it rains."""
import io

import numpy as np
import pandas as pd
import pytest
from rich.console import Console

from live.race import model, priors as P, watch
from live.race.pace import lap_temps
from live.race.projection import lap_time
from live.race.weather import is_wet_race, rain_rows, wet_lap_share, wet_now
from play.scenario import Scenario
from play.strategy import Plan
from f1live.paths import Session
from test_battles import T0, race
from test_projection import priors


def with_weather(h, rain=0.0, compound=None, last_rows=None):
    h = h.copy()
    h["rainfall"] = rain
    h["track_temp"] = 35.0
    if compound:
        h["compound"] = compound
    return h


def test_rain_from_the_weather_feed_and_from_the_tyres():
    dry = with_weather(race({"AAA": 90.0, "BBB": 90.5}, {"AAA": 0, "BBB": 2.0}, laps=10))
    assert rain_rows(dry) == 0 and wet_lap_share(dry) == 0 and not is_wet_race(dry) and not wet_now(dry)
    raining = with_weather(dry, rain=1.0)
    assert rain_rows(raining) == 1.0 and is_wet_race(raining) and wet_now(raining)
    inters = with_weather(dry, compound="INTERMEDIATE")
    assert wet_lap_share(inters) == 1.0 and is_wet_race(inters) and wet_now(inters)
    assert not wet_now(pd.DataFrame(columns=dry.columns)) and rain_rows(race({"AAA": 90.0}, {"AAA": 0}, laps=3)) == 0   # no weather columns


def test_a_shower_that_has_passed_is_not_raining_now():
    h = with_weather(race({"AAA": 90.0, "BBB": 90.5}, {"AAA": 0, "BBB": 2.0}, laps=10))
    h.loc[h["race_lap"] <= 2, "rainfall"] = 1.0                    # rained in the first laps, dry for the last 8 (12 minutes)
    assert is_wet_race(h)                                          # 2 of 10 laps wet: a wet race for the record books
    assert not wet_now(h)
    assert wet_now(h, window_s=10 ** 6)


def test_track_temperature_changes_tyre_wear_not_the_base_pace():
    model_ = dict(offset={c: 0.0 for c in P.COMPOUNDS}, wear={c: 0.04 for c in P.COMPOUNDS}, fuel=0.0,
                  wear_temp={"SOFT": 0.001, "MEDIUM": 0.0005, "HARD": 0.0}, temp_ref=35.0)
    comp = np.array([0, 1, 2])
    cold = lap_time(model_, 90.0, comp, 20.0, 10, temp=25.0)
    ref = lap_time(model_, 90.0, comp, 20.0, 10, temp=35.0)
    hot = lap_time(model_, 90.0, comp, 20.0, 10, temp=45.0)
    assert ref == pytest.approx(lap_time(model_, 90.0, comp, 20.0, 10))            # no temperature given = the reference
    assert (hot - ref == pytest.approx([0.2, 0.1, 0.0])) and (cold - ref == pytest.approx([-0.2, -0.1, 0.0]))
    assert lap_time(model_, 90.0, comp, 0.0, 10, temp=60.0) == pytest.approx(lap_time(model_, 90.0, comp, 0.0, 10))   # new tyres: no effect
    assert lap_time({k: v for k, v in model_.items() if k != "wear_temp"}, 90.0, comp, 20.0, 10, temp=60.0) == pytest.approx(ref)


def test_lap_temps_fall_back_where_there_is_no_reading():
    rows = pd.DataFrame({"track_temp": [30.0, 0.0, np.nan, 50.0]})
    assert list(lap_temps(rows, 37.0)) == [30.0, 37.0, 37.0, 50.0]
    assert list(lap_temps(pd.DataFrame({"x": [1, 2]}), 37.0)) == [37.0, 37.0]


def test_the_fit_recovers_a_temperature_effect_on_tyre_wear():
    rng = np.random.default_rng(3)
    items = {}
    for race_i in range(12):
        temp_race = 20 + 4 * race_i
        rows = []
        for drv in range(8):
            base = 90 + 0.1 * drv
            for stint in (1, 2):
                comp = ("SOFT", "MEDIUM")[stint - 1]
                for age in range(1, 22):
                    lap = (stint - 1) * 22 + age
                    temp = temp_race + rng.normal(0, 2)
                    t = base + 0.04 * age + 0.0012 * age * (temp - 35.0) - 0.05 * lap + (0.2 if comp == "MEDIUM" else 0) + rng.normal(0, 0.1)
                    rows.append(dict(driver=str(drv), stint=stint, compound=comp, age=age, lap=lap, lap_time_s=t, temp=temp))
        items[("2025", f"R{race_i}")] = dict(clean=pd.DataFrame(rows))
    m = P.fit_lap_model(items)
    assert m["wear_temp"]["SOFT"] == pytest.approx(0.0012, abs=0.0003) and m["wear_temp"]["MEDIUM"] == pytest.approx(0.0012, abs=0.0003)
    assert m["wear"]["SOFT"] == pytest.approx(0.04, abs=0.01) and m["temp_ref"] == pytest.approx(44, abs=3)


def test_the_projection_is_paused_while_it_rains_and_resumes_when_it_is_dry():
    dry = with_weather(race({"AAA": 90.0, "BBB": 89.5, "CCC": 92.0}, {"AAA": 0, "BBB": 6.0, "CCC": 8.0}, laps=10, total=40))
    info = dict(race_lap=10, total_laps=40, session=None)
    state = pd.DataFrame({"x": [1]})
    model._last["key"] = None
    out = model.predict(state, dry, info)
    assert "projection" in set(out["kind"]) and "projection_status" not in set(out["kind"])
    wet = with_weather(dry, rain=1.0)
    out = model.predict(state, wet, dict(info, race_lap=11))
    assert "projection" not in set(out["kind"])
    note = out[out["kind"] == "projection_status"]["note"].iloc[0]
    assert "rain" in note and "paused" in note
    out = model.predict(state, dry, dict(info, race_lap=12))                       # dry again: projects at once, same lap or not
    assert "projection" in set(out["kind"])


def test_the_dashboard_says_so_when_the_projection_is_paused_for_rain():
    h = with_weather(race({"AAA": 90.0, "BBB": 89.5}, {"AAA": 0, "BBB": 3.0}, laps=10, total=40), rain=1.0)
    preds = pd.DataFrame([dict(ts=T0, kind="projection_status", note="rain: projection paused (dry-tyre model)")])
    console = Console(width=170, height=44, file=io.StringIO(), force_terminal=False, color_system=None)
    console.print(watch.render(h, preds, "2026/Test/race", 44, None, 170))
    text = console.file.getvalue()
    assert "RAIN" in text and "projection paused" in text


def test_the_track_temperature_is_a_what_if_in_the_playground():
    pri = priors()
    pri["lap_model"].update(wear={"SOFT": 0.03, "MEDIUM": 0.03, "HARD": 0.03}, offset={c: 0.0 for c in P.COMPOUNDS},
                            wear_temp={"SOFT": 0.004, "MEDIUM": 0.004, "HARD": 0.004}, temp_ref=35.0)
    h = with_weather(race({"AAA": 90.0, "BBB": 90.2, "CCC": 92.0}, {"AAA": 0, "BBB": 1.0, "CCC": 5.0}, laps=10, total=50))
    sc = Scenario.live(Session("2026", "Test", "race"), history=h, sims=300, priors=pri)
    assert sc.opt["temp"] == pytest.approx(35.0)                                   # taken from the race as it is
    plan = {"AAA": Plan.parse("M")}                                                # AAA stays out for 40 laps
    cold, hot = sc.run(plans=plan, temp=10.0), sc.run(plans=plan, temp=60.0)
    assert hot.stats("AAA")["mean_pos"] >= cold.stats("AAA")["mean_pos"]
