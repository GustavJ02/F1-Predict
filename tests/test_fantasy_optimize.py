"""The optimizer against brute force on small made-up weekends."""
import itertools

import numpy as np
import pytest

from fantasy import optimize as O
from fantasy.scenarios import Scenarios


def weekend(seed=0, n_scen=400, n_drivers=8, n_cons=4):
    rng = np.random.default_rng(seed)
    drivers = [f"D{i}" for i in range(n_drivers)]
    cons = [f"C{i}" for i in range(n_cons)]
    skill = rng.uniform(0, 20, n_drivers)
    dp = skill[None, :] + rng.normal(0, 8, (n_scen, n_drivers))
    cskill = rng.uniform(0, 40, n_cons)
    cp = cskill[None, :] + rng.normal(0, 12, (n_scen, n_cons))
    sc = Scenarios(drivers=drivers, team_of={d: cons[i % n_cons] for i, d in enumerate(drivers)}, constructors=cons,
                   driver_points=dp, constructor_points=cp)
    market = O.Market(drivers={d: float(p) for d, p in zip(drivers, rng.uniform(3, 20, n_drivers).round(1))},
                      constructors={c: float(p) for c, p in zip(cons, rng.uniform(5, 30, n_cons).round(1))})
    return sc, market


def brute(sc, market, budget, mode="fresh", free=2, swaps=None, limitless=False):
    """Best mean over every team (and boost) by enumeration. -> (mean, drivers, constructors)."""
    best = (-np.inf, None, None)
    cur = set(market.team_drivers + market.team_constructors)
    nd = min(5, len(sc.drivers))
    for ds in itertools.combinations(sc.drivers, nd):
        for cs in itertools.combinations(sc.constructors, 2):
            cost = sum(market.drivers[d] for d in ds) + sum(market.constructors[c] for c in cs)
            if not limitless and cost > budget + 1e-9:
                continue
            n_new = len(set(ds + cs) - cur) if mode == "transfers" else 0
            if swaps is not None and n_new != swaps:
                continue
            pts = sc.driver_points[:, [sc.drivers.index(d) for d in ds]].mean(axis=0)
            mean = pts.sum() + pts.max() + sc.constructor_points[:, [sc.constructors.index(c) for c in cs]].mean(axis=0).sum()
            mean += S_pen(n_new, free) if mode == "transfers" else 0
            if mean > best[0]:
                best = (mean, ds, cs)
    return best


def S_pen(n, free):
    return -10 * max(0, n - free)


@pytest.fixture(autouse=True)
def small_team(monkeypatch):
    monkeypatch.setattr(O, "N_DRIVERS", 5)


def test_fresh_matches_brute_force_and_respects_the_cap():
    sc, market = weekend(1)
    cap = 60.0
    plan = O.solve(sc, market, "fresh", budget=cap)
    want, ds, cs = brute(sc, market, cap)
    assert plan.cost <= cap + 1e-9 and len(plan.drivers) == 5 and len(plan.constructors) == 2
    assert plan.mean == pytest.approx(want, abs=1e-6)
    assert plan.boost in plan.drivers


def test_limitless_ignores_the_budget():
    sc, market = weekend(2)
    capped = O.solve(sc, market, "fresh", budget=110.0)
    free = O.solve(sc, market, "fresh", limitless=True)
    assert free.mean >= capped.mean
    assert free.mean == pytest.approx(brute(sc, market, 0, limitless=True)[0], abs=1e-6)


def test_top_plans_are_different_and_in_order():
    sc, market = weekend(3)
    plans = O.top_plans(sc, market, 4, budget=70.0)
    sets = [frozenset(p.drivers + p.constructors) for p in plans]
    assert len(set(sets)) == len(sets)
    assert all(a.mean >= b.mean - 1e-9 for a, b in zip(plans, plans[1:]))


def test_transfers_charge_for_swaps_beyond_the_free_ones():
    sc, market = weekend(4)
    market.team_drivers, market.team_constructors = ["D0", "D1", "D2", "D3", "D4"], ["C0", "C1"]
    market.bank, market.free_transfers = 10.0, 2
    cap = market.team_value() + market.bank
    plan = O.solve(sc, market, "transfers")
    assert plan.swaps == len(plan.into) == len(plan.out)
    assert plan.penalty == -10 * max(plan.swaps - 2, 0)
    assert plan.mean == pytest.approx(brute(sc, market, cap, "transfers", 2)[0], abs=1e-6)
    for n in range(0, 5):
        p = O.solve(sc, market, "transfers", swaps=n)
        assert p is None or p.swaps == n
    keep = O.solve(sc, market, "transfers", swaps=0)
    assert set(keep.drivers) == set(market.team_drivers) and set(keep.constructors) == set(market.team_constructors)


def test_a_big_edge_justifies_paying_for_a_third_swap():
    sc, market = weekend(5)
    market.team_drivers, market.team_constructors = ["D0", "D1", "D2", "D3", "D4"], ["C0", "C1"]
    market.bank, market.free_transfers = 100.0, 2
    sc.driver_points[:, [5, 6, 7]] += 60.0                                        # three drivers outside the team become far better
    plan = O.solve(sc, market, "transfers")
    assert plan.swaps >= 3 and plan.penalty <= -10


def test_risk_prefers_the_steadier_team():
    rng = np.random.default_rng(6)
    n = 2000
    steady = rng.normal(20, 1, (n, 1))
    wild = np.where(rng.random((n, 1)) < 0.5, 45.0, -5.0) + rng.normal(0, 1, (n, 1))   # mean 20, huge spread
    dp = np.hstack([steady, wild] + [rng.normal(10, 1, (n, 1)) for _ in range(4)])
    cp = rng.normal(10, 1, (n, 3))
    sc = Scenarios(drivers=[f"D{i}" for i in range(6)], team_of={}, constructors=["C0", "C1", "C2"], driver_points=dp, constructor_points=cp)
    market = O.Market(drivers={f"D{i}": 5.0 for i in range(6)}, constructors={f"C{i}": 5.0 for i in range(3)})
    neutral = O.solve(sc, market, "fresh", budget=1000)
    cautious = O.solve(sc, market, "fresh", budget=1000, risk=1.0)
    assert cautious.cvar() >= neutral.cvar() - 1e-6
    assert cautious.points.std() < neutral.points.std() + 1e-9 or "D1" not in cautious.drivers + [cautious.boost]


def test_an_inactive_driver_in_the_team_is_swapped_out_on_its_own():
    sc, market = weekend(7)
    market.team_drivers, market.team_constructors = ["D0", "D1", "D2", "D3", "D4"], ["C0", "C1"]
    market.bank, market.free_transfers = 5.0, 2
    sc.driver_points[:, 2] = -25.0                                                 # D2 does not take part
    plan = O.solve(sc, market, "transfers")
    assert "D2" not in plan.drivers and "D2" in plan.out
