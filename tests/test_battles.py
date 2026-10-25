"""Battle prediction and the race dashboard, on synthetic races where the answer is known."""
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rich.console import Console

from f1live.events import load
from live.race import model, watch
from live.race.battles import find_battles, laps_to_contact, latest
from live.race.pace import driver_pace

COLS = ["ts", "event", "driver", "tla", "position", "interval_s", "gap_to_leader_s", "laps_down", "in_pit",
        "n_pit_stops", "compound", "tyre_laps", "stint", "lap_number", "lap_time_s", "lap_kind", "track_status",
        "status_code", "race_lap", "total_laps", "best_lap_s", "message"]
T0 = pd.Timestamp("2026-10-11T12:00:00Z")


def race(paces, gaps, laps=10, total=30, track=None, skip_after=None):
    """Lap rows for cars in grid order. paces: {tla: s/lap}; gaps: {tla: gap to the car ahead at lap 0 in s}.
    track: {lap: status} overrides; skip_after: {tla: lap} cars that stop completing laps after that lap."""
    track = track or {}
    rows, gap = [], dict(gaps)
    for lap in range(1, laps + 1):
        cum = 0.0                                              # gap to the leader = the intervals down the order
        for i, (tla, pace) in enumerate(paces.items()):
            cum += 0.0 if i == 0 else gap[tla]
            if skip_after and lap > skip_after.get(tla, 10 ** 9):
                continue
            rows.append(dict(ts=T0 + pd.Timedelta(float(lap * 90 + i), unit="s"), event="lap", driver=str(10 + i), tla=tla,
                             position=i + 1, interval_s=None if i == 0 else gap[tla], gap_to_leader_s=cum, laps_down=0,
                             in_pit=False, n_pit_stops=0, compound="MEDIUM", tyre_laps=lap, stint=1, lap_number=lap + 1,
                             lap_time_s=pace + 0.01 * ((lap * 7 + i) % 3), lap_kind="racing",
                             track_status=track.get(lap, "AllClear"), status_code=1, race_lap=lap, total_laps=total,
                             best_lap_s=pace))
        for i, tla in enumerate(list(paces)[1:], 1):          # the car behind closes by the pace difference
            gap[tla] -= list(paces.values())[i - 1] - list(paces.values())[i]
    return pd.DataFrame(rows, columns=COLS)


def test_laps_to_contact():
    assert laps_to_contact(3.0, 0.5) == 4.0
    assert laps_to_contact(0.8, -0.2) == 0                       # already in range, whatever the pace
    assert laps_to_contact(3.0, 0.02) == np.inf                  # not closing
    assert laps_to_contact(30.0, 0.5) == np.inf                  # beyond the horizon


def test_pace_needs_clean_laps_on_the_current_stint():
    # rows are numbered from lap 2 like the real feed (lap 1 has no time); laps below 3 never count
    h = race({"AAA": 90.0}, {"AAA": 0}, laps=4)                  # clean: laps 3, 4, 5
    assert driver_pace(h)["pace_s"].iloc[0] == pytest.approx(90.01, abs=0.02)
    assert np.isnan(driver_pace(race({"AAA": 90.0}, {"AAA": 0}, laps=3))["pace_s"].iloc[0])      # only 3 and 4
    h["lap_kind"] = "in"                                         # no racing laps -> no pace
    assert driver_pace(h)["pace_s"].isna().all()
    sc = dict(track={4: "SCDeployed", 5: "SCDeployed", 6: "SCDeployed"})
    # row lap numbers 3, 4 are clean, 5-7 are safety car, 8 follows one, so the third clean lap is number 9
    assert np.isnan(driver_pace(race({"AAA": 90.0}, {"AAA": 0}, laps=7, **sc))["pace_s"].iloc[0])
    assert driver_pace(race({"AAA": 90.0}, {"AAA": 0}, laps=8, **sc))["n"].iloc[0] == 3


def test_battle_predicted_for_the_faster_car_behind():
    # BBB is 0.5 s/lap faster than AAA and 6 s behind at lap 1; CCC is slower than both, so no battle there
    h = race({"AAA": 90.0, "BBB": 89.5, "CCC": 92.0}, {"AAA": 0, "BBB": 6.0, "CCC": 8.0}, laps=6)
    status, battles = find_battles(h)
    assert status is None and len(battles) == 1
    b = battles[0]
    assert b["drivers"] == ["AAA", "BBB"] and b["position"] == 1 and not b["now"]
    assert 4 <= b["in_laps"] <= 6                                # gap is 3.5 s at lap 6: (3.5 - 1) / 0.5 = 5


def test_cars_in_range_now_are_a_battle_and_neighbouring_pairs_group():
    h = race({"AAA": 90.0, "BBB": 90.0, "CCC": 90.0}, {"AAA": 0, "BBB": 0.6, "CCC": 0.8}, laps=6)
    _, battles = find_battles(h)
    assert len(battles) == 1 and battles[0]["drivers"] == ["AAA", "BBB", "CCC"] and battles[0]["now"]


def test_predictions_pause_under_safety_car_and_stop_at_the_flag():
    h = race({"AAA": 90.0, "BBB": 89.5}, {"AAA": 0, "BBB": 6.0}, laps=6, track={6: "SCDeployed"})
    h["track_status"] = h["track_status"].where(h["lap_number"] < 7, "SCDeployed")
    assert find_battles(h)[0] == "SCDeployed: predictions paused"
    assert find_battles(race({"AAA": 90.0, "BBB": 89.5}, {"AAA": 0, "BBB": 6.0}, laps=6, total=6))[0] == "race finished"


def test_collecting_pace_data_early_and_stale_cars_are_out():
    assert find_battles(race({"AAA": 90.0, "BBB": 89.5}, {"AAA": 0, "BBB": 3.0}, laps=3))[0] == "collecting pace data"
    h = race({"AAA": 90.0, "BBB": 89.5, "CCC": 89.0}, {"AAA": 0, "BBB": 3.0, "CCC": 3.0}, laps=12, skip_after={"BBB": 4})
    lat = latest(h)
    assert lat.loc["11", "out"] and not lat.loc["10", "out"]     # BBB stopped completing laps


def test_model_returns_battle_rows_or_a_status_row():
    h = race({"AAA": 90.0, "BBB": 89.5}, {"AAA": 0, "BBB": 6.0}, laps=6)
    info = dict(race_lap=6, total_laps=30)
    model._last["key"] = None
    out = model.predict(pd.DataFrame({"x": [1]}), h, info)
    battle = out[out["kind"] == "battle"]
    assert list(battle["kind"]) == ["battle"] and battle["drivers"].iloc[0] == "AAA BBB" and out["race_lap"].iloc[0] == 6
    early = model.predict(pd.DataFrame({"x": [1]}), race({"AAA": 90.0, "BBB": 89.5}, {"AAA": 0, "BBB": 6.0}, laps=2), info)
    status = early[early["kind"] == "status"]
    assert list(status["kind"]) == ["status"] and "pace" in status["note"].iloc[0]
    assert model.predict(pd.DataFrame(), h, info) is None


def frame(history, preds):
    console = Console(width=140, height=40, file=io.StringIO(), force_terminal=False, color_system=None)
    console.print(watch.render(history, preds, "2026/Test/race", console.height))
    return console.file.getvalue()


def test_dashboard_shows_lap_standings_gaps_and_battles():
    h = race({"AAA": 90.0, "BBB": 89.5, "CCC": 92.0}, {"AAA": 0, "BBB": 6.0, "CCC": 8.0}, laps=6)
    preds = model.predict(pd.DataFrame({"x": [1]}), h, dict(race_lap=6, total_laps=30))
    preds.insert(0, "ts", T0)
    text = frame(h, preds)
    assert "LAP 6/30" in text and "GREEN" in text
    for tla in ("AAA", "BBB", "CCC"):
        assert tla in text
    assert "LEADER" in text and "+" in text and "Battle for 1st" in text and "in ~" in text


def test_dashboard_separates_coming_up_from_in_range_now():
    preds = pd.DataFrame([
        dict(ts=T0, kind="battle", position=4, drivers="ANT PIA NOR", gaps="0.8 3.1", laps="0 5", in_laps=0.0, now=True),
        dict(ts=T0, kind="battle", position=9, drivers="GAS BEA", gaps="2.4", laps="3", in_laps=3.0, now=False)])
    text = frame(race({"AAA": 90.0, "BBB": 89.5}, {"AAA": 0, "BBB": 6.0}, laps=6), preds)
    assert text.index("COMING UP") < text.index("IN RANGE NOW")
    assert "Battle for 9th" in text and "in ~3 laps" in text
    assert "P4  ANT 0.8 PIA 3.1s ~5L NOR" in text                  # the group already fighting, third car still to come


def test_dashboard_without_predictions_or_data():
    h = race({"AAA": 90.0, "BBB": 89.5}, {"AAA": 0, "BBB": 6.0}, laps=6)
    assert "waiting for predictions" in frame(h, None)
    status = pd.DataFrame([dict(ts=T0, kind="status", note="VSC: predictions paused")])
    assert "VSC: predictions paused" in frame(h, status)
    assert "waiting" in frame(pd.DataFrame(columns=COLS), None)


SINGAPORE = Path("data/2026/Singapore/race/events.csv")


@pytest.mark.skipif(not SINGAPORE.exists(), reason=f"{SINGAPORE} not found")
def test_real_race_mid_race():
    h = load(SINGAPORE)
    status, battles = find_battles(h[h["race_lap"] <= 31])
    assert status is None and battles
    assert all(b["in_laps"] <= 12 for b in battles) and all(len(b["drivers"]) >= 2 for b in battles)
    assert find_battles(h)[0] == "race finished"
    text = frame(h[h["race_lap"] <= 31], None)
    assert "VER" in text and "LAP 31/62" in text


def test_a_predictions_file_with_only_single_pair_battles_still_renders(tmp_path):
    """pandas reads 'gaps' and 'laps' as numbers when every battle so far is a single pair: the dashboard must cope."""
    csv = tmp_path / "predictions.csv"
    csv.write_text("ts,kind,position,drivers,numbers,gaps,laps,in_laps,now,race_lap,total_laps\n"
                   "2026-10-25T12:00:10Z,battle,7,PIA RUS,81 63,0.9,0,0.0,True,33,56\n"
                   "2026-10-25T12:00:10Z,battle,3,HAM NOR,44 1,5.7,10,9.6,False,33,56\n")
    from f1live.watch import read
    preds = read(csv)
    assert preds["gaps"].dtype == object and preds["laps"].dtype == object
    h = race({"AAA": 90.0, "BBB": 89.5}, {"AAA": 0, "BBB": 3.0}, laps=10, total=40)
    text = frame(h, preds)                                    # the old behaviour was an AttributeError here
    assert "PIA" in text and "HAM" in text and "IN RANGE NOW" in text
    numeric = pd.read_csv(csv)                                # even if a caller passes the numeric version
    assert numeric["gaps"].dtype == float and "HAM" in frame(h, numeric)


def test_rows_cut_short_by_a_half_written_file_are_skipped_not_fatal():
    h = race({"AAA": 90.0, "BBB": 89.5}, {"AAA": 0, "BBB": 3.0}, laps=10, total=40)
    preds = pd.DataFrame([
        dict(ts=T0, kind="battle", position=7, drivers="PIA RUS", numbers="81 63", gaps="0.9", laps="0", in_laps=0.0, now=True),
        dict(ts=T0, kind="battle", position=np.nan, drivers=np.nan, numbers=np.nan, gaps=np.nan, laps=np.nan, in_laps=np.nan, now=np.nan),
        dict(ts=T0, kind="projection", tla=np.nan, proj_rank=np.nan, p_win=np.nan)])
    text = frame(h, preds)
    assert "PIA" in text and "RUS" in text


def test_the_runner_writes_predictions_atomically(tmp_path):
    from f1live.runner import write_atomically
    target = tmp_path / "predictions.csv"
    write_atomically(pd.DataFrame({"a": [1, 2]}), target)
    assert pd.read_csv(target)["a"].tolist() == [1, 2] and not (tmp_path / "predictions.tmp").exists()
    write_atomically(pd.DataFrame({"a": [3]}), target)
    assert pd.read_csv(target)["a"].tolist() == [3]
