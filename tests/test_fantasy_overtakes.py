"""Counting overtakes from lap data, and the overtake model on synthetic races."""
import numpy as np
import pandas as pd

from fantasy import overtakes as O


def laps(rows):
    """rows: (lap, tla, position, kind) -> events frame with lap rows."""
    return pd.DataFrame([dict(event="lap", tla=t, position=p, lap_number=l, lap_kind=k) for l, t, p, k in rows])


def test_pass_between_laps_counts_for_the_car_that_gained():
    ev = laps([(2, "A", 2, "racing"), (2, "B", 1, "racing"), (2, "C", 3, "racing"),
               (3, "A", 3, "racing"), (3, "B", 1, "racing"), (3, "C", 2, "racing")])
    got = O.count_by_lap(ev, grid=pd.Series({"A": 1, "B": 2, "C": 3}))
    assert got.to_dict() == {"A": 0, "B": 1, "C": 1}                          # B passed A before lap 2, C passed A before lap 3


def test_pit_stops_are_not_passes():
    ev = laps([(2, "A", 1, "racing"), (2, "B", 2, "racing"), (3, "A", 3, "in"), (3, "B", 1, "racing"),
               (4, "A", 3, "out"), (4, "B", 1, "racing"), (5, "A", 2, "racing"), (5, "B", 1, "racing")])
    got = O.count_by_lap(ev, grid=pd.Series({"A": 1, "B": 2}))
    assert got.sum() == 0                                                      # A pitted and came out behind: nothing counts


def test_a_retiring_car_is_not_passed():
    ev = laps([(2, "A", 1, "racing"), (2, "B", 2, "racing"), (2, "C", 3, "racing"),
               (3, "A", 1, "racing"), (3, "B", 3, "racing"), (3, "C", 2, "racing"),    # B is slow, then stops: last timed lap
               (4, "A", 1, "racing"), (4, "C", 2, "racing")])
    got = O.count_by_lap(ev, grid=pd.Series({"A": 1, "B": 2, "C": 3}))
    assert got["C"] == 0


def synthetic_dataset(n_races=200, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_races):
        grid = np.arange(1, 21)
        finish = np.argsort(np.argsort(grid + rng.normal(0, 3, 20))) + 1
        mult = rng.gamma(8, 1 / 8)
        mu = np.exp(0.4 + 0.1 * np.clip(grid - finish, 0, None) + 0.1 * np.log(grid)) * mult
        for g, f, m in zip(grid, finish, mu):
            rows.append(dict(year=2024, race=f"R{r % 10}", session="race", tla=f"D{g}", grid=g, finish=f, classified=True,
                             ot=rng.poisson(m)))
    return pd.DataFrame(rows)


def test_model_recovers_the_effect_of_places_gained_and_samples_with_the_right_mean():
    d = synthetic_dataset()
    m = O.fit_model(d)
    assert 0.07 < m.coef["race"][1] < 0.14                                     # places gained
    rng = np.random.default_rng(1)
    grid = np.arange(1, 21)
    finish = np.tile(grid, (3000, 1))
    s = m.sample(rng, "race", grid, finish, np.zeros_like(finish, bool))
    assert s.shape == finish.shape
    assert abs(s.mean() - d["ot"].mean()) < 0.35 * d["ot"].mean()
    assert s.sum(axis=1).std() > np.sqrt(s.sum(axis=1).mean())                 # the race-wide multiplier spreads the totals


def test_not_classified_and_gains_move_the_mean():
    m = O.OvertakeModel(coef={"race": np.array([0.5, 0.1, 0.0, 0.1, -0.4])}, alpha={"race": 0.1}, race_cv={"race": 0.3})
    base = m.mean("race", 10, 10, False)
    assert m.mean("race", 10, 5, False) > base
    assert m.mean("race", 10, 10, True) < base
