"""Fantasy scoring against the published rules (fantasy.formula1.com/en/game-rules)."""
from fantasy import scoring as S


def test_quali_ladder_and_no_time():
    assert [S.quali_points(p) for p in (1, 2, 10, 11, 22)] == [10, 9, 1, 0, 0]
    assert S.quali_points(None) == -5


def test_race_finish_ladder():
    assert [S.race_points(p, p) for p in (1, 2, 3, 10, 11)] == [25, 18, 15, 1, 0]


def test_race_positions_gained_lost_overtakes_bonuses():
    assert S.race_points(grid=8, finish=3) == 15 + 5                       # gained 5
    assert S.race_points(grid=2, finish=12) == 0 - 10                      # lost 10, uncapped in the race
    assert S.race_points(grid=5, finish=5, overtakes=3, fastest_lap=True, dotd=True) == 10 + 3 + 10 + 10


def test_not_classified_gets_penalty_not_lost_positions():
    assert S.race_points(grid=3, finish=None, classified=False) == -20
    assert S.sprint_points(grid=3, finish=None, classified=False) == -10


def test_sprint_scores_positions_with_a_cap_on_losses():
    assert S.sprint_points(grid=6, finish=1) == 8 + 5
    assert S.sprint_points(grid=1, finish=20) == -10                        # lost 19, capped at 10
    assert S.sprint_points(grid=4, finish=4, fastest_lap=True) == 5 + 5


def test_constructor_quali_bonus_takes_the_best_tier_only():
    assert S.constructor_quali_bonus((3, 3)) == 10
    assert S.constructor_quali_bonus((3, 2)) == 5
    assert S.constructor_quali_bonus((3, 1)) == 5
    assert S.constructor_quali_bonus((2, 2)) == 3
    assert S.constructor_quali_bonus((2, 1)) == 1
    assert S.constructor_quali_bonus((1, 1)) == -1
    assert S.constructor_quali_bonus((3, 3), disqualified=1) == 5


def test_pit_stop_points():
    assert [S.pit_stop_points(t) for t in (1.9, 2.0, 2.19, 2.2, 2.49, 2.5, 2.99, 3.0, 4.5)] == [20, 10, 10, 5, 5, 2, 2, 0, 0]
    assert S.constructor_pit_points([2.3, 2.6], fastest_of_race=True) == 5 + 2 + 5
    assert S.constructor_pit_points([1.75]) == 20 + 15


def test_transfer_penalty():
    assert [S.transfer_penalty(n) for n in range(5)] == [0, 0, 0, -10, -20]
    assert S.transfer_penalty(4, free=3) == -10
