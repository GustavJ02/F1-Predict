"""pre.* on synthetic sessions written to a temp data dir (no real data needed)."""
import numpy as np
import pandas as pd
import pytest

from pre.history import available, pace_table
from pre.quali.features import Tables, ewma, fill, weekend_features
from pre.quali.predict_grid import Fit, predict_weekend, simulate

COLS = ["ts", "event", "session_part", "driver", "tla", "team", "lap_number", "lap_time_s", "lap_kind",
        "compound", "tyre_new", "tyre_laps", "track_temp", "air_temp"]
DRIVERS = [("1", "AAA", "Red"), ("2", "BBB", "Red"), ("3", "CCC", "Blue"), ("4", "DDD", "Blue"),
           ("5", "EEE", "Green"), ("6", "FFF", "Green")]


def write(root, year, race, session, start, times, deleted=()):
    """times: {tla: [lap times]}; every lap is a flying lap."""
    rows = []
    for num, tla, team in DRIVERS:
        for i, t in enumerate(times.get(tla, [])):
            rows.append(dict(ts=f"{start}:{i:02d}.000Z", event="lap", session_part=1, driver=num, tla=tla, team=team,
                             lap_number=i + 2, lap_time_s=t, lap_kind="push", compound="SOFT", tyre_new=True,
                             tyre_laps=0, track_temp=30, air_temp=20))
    for tla, t in deleted:
        num = next(d[0] for d in DRIVERS if d[1] == tla)
        rows.append(dict(ts=f"{start}:59.000Z", event="lap_deleted", driver=num, lap_time_s=t))
    d = root / str(year) / race / session
    d.mkdir(parents=True)
    pd.DataFrame(rows, columns=COLS).to_csv(d / "events.csv", index=False)


def test_pace_table_drops_deleted_laps_and_ranks(tmp_path):
    write(tmp_path, 2025, "A", "quali", "2025-03-01T10:00", {"AAA": [90.0, 91.0], "BBB": [91.8], "CCC": [90.9]},
          deleted=[("AAA", 90.0)])
    t = pace_table(tmp_path / "2025/A/quali/events.csv")
    assert t.loc["1", "best_s"] == 91.0                      # the 90.0 was deleted
    assert t["tla"].tolist() == ["CCC", "AAA", "BBB"] or t["tla"].tolist() == ["AAA", "CCC", "BBB"]
    assert t["gap_pct"].min() == 0 and t["rank"].tolist() == [1, 2, 3]


def test_ewma_weights_recent_sessions_more():
    old, new = pd.Series({"x": 0.0}), pd.Series({"x": 1.0})
    assert 0.5 < ewma([old, new], halflife=1).iloc[0] < 1.0
    assert ewma([pd.Series({"x": 0.0, "y": 2.0}), pd.Series({"x": 1.0})])["y"] == 2.0     # missing a session is fine


def test_fill_uses_weekend_percentile():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, np.nan]})
    assert fill(df, ["a"])["a"].iloc[-1] == pytest.approx(df["a"].quantile(0.8))


def test_simulate_probabilities():
    pred = pd.Series([0.0, 0.3, 0.6, 0.9], index=list("abcd"))
    res = simulate(pred, pd.Series(["t1", "t1", "t2", "t2"], index=pred.index), Fit([], pd.Series(dtype=float), 0, 0.2, 0.2, 5))
    assert res["pole"].sum() == pytest.approx(1)
    assert res["pole"].is_monotonic_decreasing and res["position"].is_monotonic_increasing
    assert res["position"].sum() == pytest.approx(10)       # positions 1..4


def synthetic_season(root, n_weekends=8, seed=0):
    rng = np.random.default_rng(seed)
    pace = {tla: 90 + 0.4 * i for i, (_, tla, _) in enumerate(DRIVERS)}          # AAA fastest
    for w in range(n_weekends):
        day = f"2025-0{w + 1}-1"
        for s, noise, when in (("fp1", 0.4, "08:00"), ("fp2", 0.3, "12:00"), ("fp3", 0.2, "08:00")):
            write(root, 2025, f"R{w}", s, f"{day}T{when}", {t: [p + rng.normal(0, noise) + 0.5] for t, p in pace.items()})
        write(root, 2025, f"R{w}", "quali", f"{day}T14:00", {t: [p + rng.normal(0, 0.1)] for t, p in pace.items()})


def test_practice_pace_predicts_the_grid(tmp_path):
    synthetic_season(tmp_path)
    tables = Tables(available(tmp_path))
    quali = tables.av[tables.av["session"] == "quali"]
    assert len(quali) == 8 and quali["start"].is_monotonic_increasing
    f = weekend_features(tables, 2025, "R7")
    assert f.loc["AAA", "form_driver"] < f.loc["FFF", "form_driver"]        # built from the 7 earlier qualis only
    assert weekend_features(tables, 2025, "R0")["form_driver"].isna().all()
    fit, res, _ = predict_weekend(tables, 2025, "R7", with_target=True)
    assert fit.n_train == 6       # R0 has no season form yet, so it can't train a model that uses form
    assert res["position"].idxmin() == "AAA" and res["pole"].idxmax() == "AAA"
    assert res["position"].corr(pd.Series(range(6), index=res.index), method="spearman") > 0.9


def test_no_prediction_without_enough_history(tmp_path):
    synthetic_season(tmp_path, n_weekends=3)
    fit, res, _ = predict_weekend(Tables(available(tmp_path)), 2025, "R2")
    assert fit is None and res is None


def test_available_skips_a_session_that_has_no_events_yet(tmp_path):
    write(tmp_path, 2025, "A", "quali", "2025-03-01T10:00", {"AAA": [90.0]})
    live = tmp_path / "2025" / "B" / "race"                       # a session being recorded: the parser has just created the file
    live.mkdir(parents=True)
    (live / "events.csv").write_text("")
    header = tmp_path / "2025" / "C" / "race"
    header.mkdir(parents=True)
    (header / "events.csv").write_text("ts,event,driver\n")
    assert list(available(tmp_path)["race"]) == ["A"]


def test_scripts_that_read_every_race_skip_a_race_being_recorded(tmp_path):
    from f1live.events import has_events
    live = tmp_path / "events.csv"
    assert not has_events(live)                          # no file yet
    live.write_text("")
    assert not has_events(live)                          # created by the parser, nothing written
    live.write_text("ts,event\n")
    assert not has_events(live)                          # header only
    live.write_text("ts,event\n2026-10-25T20:00:00Z,lap\n")
    assert has_events(live)
