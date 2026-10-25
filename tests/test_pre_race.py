"""pre.race: race pace ratings, grid blending and the pre-race simulation, on synthetic data."""
import numpy as np
import pandas as pd
import pytest

from live.race import priors as P
from pre.race import predict_result as PR
from pre.race.pace import race_pace
from test_battles import race
from test_pre import DRIVERS, write
from test_projection import priors


def test_race_pace_is_relative_to_the_field_and_ranks_faster_cars_first():
    h = race({"AAA": 90.0, "BBB": 90.5, "CCC": 91.5}, {"AAA": 0, "BBB": 3.0, "CCC": 3.0}, laps=20, total=30)
    h["team"] = h["tla"].map({"AAA": "Red", "BBB": "Blue", "CCC": "Green"})
    t = race_pace(h, priors())
    assert list(t.index[np.argsort(t["rel"])]) == ["AAA", "BBB", "CCC"]
    assert t["rel"].median() == pytest.approx(0.0, abs=1e-9)
    assert t.loc["CCC", "rel"] - t.loc["AAA", "rel"] == pytest.approx(1.5, abs=0.1)


def test_sinkhorn_gives_every_driver_and_position_a_total_of_one():
    M = PR.sinkhorn(np.random.default_rng(0).random((6, 6)) + 0.05)
    assert M.sum(axis=1) == pytest.approx(np.ones(6)) and M.sum(axis=0) == pytest.approx(np.ones(6), abs=1e-3)


def sim_table(n=6, sims=300, **opt):
    rel = pd.Series({d[1]: 0.1 * i for i, d in enumerate(DRIVERS[:n])})
    sd = pd.Series(0.2, index=rel.index)
    grid = list(rel.index)
    out, info = PR.simulate_from_grid(grid, rel, sd, priors(), "Test", 40, 90.0, sims, 0, **opt)
    return out, grid


def test_simulate_from_grid_returns_a_full_position_distribution():
    out, grid = sim_table()
    cols = [f"pos_{k}" for k in range(1, 7)]
    assert out[cols].sum(axis=1).to_numpy() == pytest.approx(np.ones(6)) and out[cols].sum(axis=0).to_numpy() == pytest.approx(np.ones(6))
    assert out["p_win"].sum() == pytest.approx(1.0) and out.iloc[0]["tla"] == grid[0]       # the fastest car on pole


def test_blend_with_grid_keeps_the_distributions_valid_and_pulls_towards_the_grid():
    out, grid = sim_table(pace_sd=2.0)                    # a very noisy simulation: the pole sitter's chance is low
    table = np.full((22, 22), 0.001)
    np.fill_diagonal(table, 1.0)                           # a table in which everybody finishes where they started
    table = table / table.sum(axis=1, keepdims=True)
    mixed = PR.blend_with_grid(out, grid, table, 0.5)
    assert mixed[[f"pos_{k}" for k in range(1, 7)]].sum(axis=1).to_numpy() == pytest.approx(np.ones(6))
    assert mixed.set_index("tla").loc[grid[0], "p_win"] > out.set_index("tla").loc[grid[0], "p_win"]
    assert list(mixed["proj_rank"]) == list(range(1, 7)) and mixed["mean_pos"].is_monotonic_increasing


def test_sparkline_and_percentiles():
    row = pd.Series({f"pos_{k}": v for k, v in enumerate([0.0, 0.1, 0.5, 0.3, 0.1, 0.0], 1)})
    assert PR.sparkline(row, 6)[2] == "█" and PR.sparkline(row, 6)[0] == " " and len(PR.sparkline(row, 6)) == 6
    assert PR.percentile_range(row, 6) == (2, 5)


def test_pace_model_recovers_how_rating_and_quali_gap_predict_race_pace():
    rng = np.random.default_rng(2)
    rows = pd.DataFrame(dict(race=["r"] * 400, tla=range(400), rating=rng.normal(0, 0.8, 400), qrel=rng.normal(0, 1.0, 400)))
    rows["rel"] = 0.9 * rows["rating"] + 0.2 * rows["qrel"] + rng.normal(0, 0.1, 400)
    coef, sd = PR.fit_pace_model(rows, ridge=0.1)["full"]
    assert coef["rating"] == pytest.approx(0.9, abs=0.05) and coef["qrel"] == pytest.approx(0.2, abs=0.05) and sd == pytest.approx(0.1, abs=0.03)
    q = pd.DataFrame({"qrel": [0.0, 1.0]}, index=["AAA", "BBB"])
    rel, _ = PR.predict_pace(PR.fit_pace_model(rows), pd.DataFrame({"rating": [0.5]}, index=["AAA"]), q)     # BBB has no rating
    assert rel["AAA"] == pytest.approx(0.9 * 0.5, abs=0.15) and not np.isnan(rel["BBB"])


def test_a_driver_without_a_valid_quali_time_starts_last(tmp_path):
    times = {"AAA": [90.0], "BBB": [90.5], "CCC": [91.0], "DDD": [91.5], "EEE": [92.0], "FFF": [93.0]}
    write(tmp_path, 2026, "Test", "quali", "2026-05-01T14:00", times, deleted=[("FFF", 93.0)])
    from pre.history import available
    q = PR.quali_table(available(tmp_path), 2026, "Test")
    assert q.loc["FFF", "rank"] == len(q) == 6 and q.loc["AAA", "rank"] == 1
    assert q.loc["FFF", "qrel"] == q["qrel"].max()


SINGAPORE = pytest.importorskip("pathlib").Path("data/2026/Austin/pre_race.csv")


@pytest.mark.skipif(not SINGAPORE.exists(), reason="run python -m pre.race.predict_result 2026 Austin first")
def test_the_austin_prediction_file_is_a_proper_distribution():
    d = pd.read_csv(SINGAPORE)
    cols = [c for c in d.columns if c.startswith("pos_")]
    assert len(d) == len(cols) == 22 and d[cols].sum(axis=1).to_numpy() == pytest.approx(np.ones(22), abs=1e-6)
    assert d["p_win"].sum() == pytest.approx(1.0, abs=1e-6) and d["p_podium"].sum() == pytest.approx(3.0, abs=1e-6)
