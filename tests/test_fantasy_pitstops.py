"""Pit stop model: team effects from lane times, and constructor pit points per scenario."""
import numpy as np
import pandas as pd

from fantasy import pitstops as P


def stops(effects, n=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for race in range(6):
        for team, e in effects.items():
            for _ in range(n // 6):
                rows.append(dict(year=2025, race=f"R{race}", start=pd.Timestamp("2025-03-01", tz="UTC") + pd.Timedelta(days=14 * race),
                                 team=team, tla="XXX", r=e + 0.9 * rng.standard_normal()))
    return pd.DataFrame(rows)


def test_team_effects_rank_the_teams_and_shrink_toward_zero():
    s = stops({"fast": -0.4, "mid": 0.0, "slow": 0.5})
    eff = P.team_effects(s, pd.Timestamp("2026-01-01", tz="UTC"))
    raw = s.groupby("team")["r"].mean()
    assert eff["fast"] < eff["mid"] < eff["slow"]
    assert (eff.abs() < raw.abs()).all()                              # shrunk toward the average team


def test_only_races_before_the_cut_count():
    s = stops({"a": 0.3})
    assert P.team_effects(s, pd.Timestamp("2020-01-01", tz="UTC")).empty


def test_points_follow_team_speed_and_the_fastest_stop_goes_to_one_team():
    m = P.PitModel(effect={"quick": -0.4, "slow": 0.4})
    rng = np.random.default_rng(0)
    n = np.full((20000, 2), 3)
    pts = m.points(rng, ["quick", "slow"], n)
    assert pts[:, 0].mean() > pts[:, 1].mean() + 3
    no_stops = m.points(rng, ["quick", "slow"], np.zeros((50, 2), int))
    assert (no_stops == 0).all()
    one = m.points(rng, ["quick", "slow"], np.array([[1, 0]] * 400))      # only one team stops: it has the quickest stop
    assert (one[:, 0] >= 5).all() and (one[:, 1] == 0).all()


def test_stop_times_are_blank_where_there_is_no_stop():
    m = P.PitModel()
    t = m.stop_times(np.random.default_rng(1), ["a", "b"], np.array([[2, 0]]))
    assert np.isfinite(t[0, 0, :2]).all() and np.isnan(t[0, 0, 2:]).all() and np.isnan(t[0, 1]).all()


def test_stop_times_follow_the_published_quartiles():
    st = {"quick": dict(low=2.15, q1=2.27, med=2.47, q3=2.87), "slow": dict(low=2.78, q1=3.0, med=3.47, q3=4.4)}
    m = P.PitModel(stats=st)
    t = m.stop_times(np.random.default_rng(0), ["quick", "slow"], np.full((50000, 2), 1))[:, :, 0]
    for k, team in enumerate(["quick", "slow"]):
        got = np.quantile(t[:, k], [0.25, 0.5, 0.75])
        assert np.allclose(got, [st[team]["q1"], st[team]["med"], st[team]["q3"]], atol=0.03)
    assert t[:, 0].min() >= 2.10                                           # a hard floor near the fastest stop of the season


def test_file_statistics_beat_the_lane_effect_and_a_missing_team_falls_back():
    m = P.PitModel(stats={"a": dict(low=2.2, q1=2.3, med=2.5, q3=2.9)}, effect={"b": 1.0})
    assert m.team_stats("a")["med"] == 2.5
    assert abs(m.team_stats("b")["med"] - (P.MEDIAN_AT_AVERAGE + P.MEDIAN_PER_LANE_S)) < 1e-9
    assert m.team_stats("unknown")["med"] == P.MEDIAN_AT_AVERAGE
