"""The vectorised scoring against the scalar reference, and the pieces of the scenario simulation."""
import itertools

import numpy as np

from fantasy import scenarios as SC
from fantasy import scoring as S


def test_ladder_matches_the_scalar_tables():
    pos = np.arange(1, 23)
    assert (SC.ladder(S.RACE, pos) == [S.table_points(S.RACE, p) for p in pos]).all()
    assert (SC.ladder(S.QUALI, pos) == [S.quali_points(p) for p in pos]).all()


def test_race_points_match_the_scalar_function():
    rng = np.random.default_rng(0)
    grid, finish = rng.integers(1, 23, 400), rng.integers(1, 23, 400)
    dnf, ot = rng.random(400) < 0.15, rng.integers(0, 8, 400)
    fl, dd = rng.random(400) < 0.05, rng.random(400) < 0.05
    got = SC.race_points(grid, finish, dnf, ot, fl, dd)
    want = [S.race_points(g, f, classified=not d, overtakes=int(o), fastest_lap=bool(a), dotd=bool(b))
            for g, f, d, o, a, b in zip(grid, finish, dnf, ot, fl, dd)]
    assert (got == want).all()


def test_constructor_quali_bonus_matches_the_scalar_function_for_every_pair_of_parts():
    for a, b in itertools.product((1, 2, 3), repeat=2):
        assert SC.constructor_quali_bonus(np.array([a]), np.array([b]))[0] == S.constructor_quali_bonus((a, b))


def test_quali_part_follows_the_elimination_rule():
    parts = SC.quali_part(np.arange(1, 23), 22)
    assert list(parts) == [3] * 10 + [2] * 6 + [1] * 6
    assert list(SC.quali_part(np.arange(1, 21), 20)) == [3] * 10 + [2] * 5 + [1] * 5


def test_remap_delivers_the_target_marginals_and_keeps_retired_cars_last():
    rng = np.random.default_rng(1)
    n, cars = 6000, 6
    retired = np.zeros((n, cars), bool)
    retired[:, 5] = rng.random(n) < 0.5
    finish = (rng.random((n, cars)) + retired * 10).argsort(axis=1).argsort(axis=1) + 1   # equally likely anywhere, retired last
    target = np.full((cars, cars), 0.02)
    target[np.arange(cars), np.arange(cars)] = 0.9                                # car c finishes P(c+1) with 90 %
    target = target / target.sum(axis=1, keepdims=True)
    new = SC.remap(finish, retired, target, rng)
    assert (np.sort(new, axis=1) == np.arange(1, cars + 1)).all()                 # still a permutation in every scenario
    assert retired[:, 5].any() and (new[retired[:, 5], 5] == cars).all()
    assert (new[:, 0] == 1).mean() > 0.8                          # car 0 now wins most of the time
    assert abs(new[:, 1].mean() - 2.0) < 0.35


def test_dotd_goes_to_one_classified_driver_per_scenario():
    rng = np.random.default_rng(2)
    n, cars = 2000, 10
    finish = rng.random((n, cars)).argsort(axis=1).argsort(axis=1) + 1
    grid = np.tile(np.arange(1, cars + 1), (n, 1))
    retired = rng.random((n, cars)) < 0.2
    d = SC.dotd_draw(rng, finish, grid, retired)
    assert (d.sum(axis=1) == 1).all() and not (d & retired).any()
    won = (d * (finish == 1)).sum() / n
    assert 0.2 < won < 0.7                                                         # the winner is the most likely pick
