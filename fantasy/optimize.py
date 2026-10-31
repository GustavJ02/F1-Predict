"""The best F1 Fantasy team for a weekend, from the scenarios of fantasy.scenarios.

    python -m fantasy.optimize fresh 2026 Mexico_City                      # a new team: any 5 drivers + 2 constructors under the cap
    python -m fantasy.optimize fresh 2026 Mexico_City --limitless          # no budget cap (the Limitless chip)
    python -m fantasy.optimize transfers 2026 Mexico_City                  # should I change my team? (data/fantasy/team.json)
    python -m fantasy.optimize transfers 2026 Mexico_City --risk 0.5       # also care about the bad weekends

A team is 5 drivers and 2 constructors, one driver carries the DRS boost (his points count double). Maximised over the scenarios:

    (1 - risk) * mean points  +  risk * CVaR(alpha)  -  10 * (swaps beyond the free ones)

CVaR(alpha) is the average of the worst alpha share of the scenarios (default 10 %), so risk 0 is the plain expected points and risk 1
the team that does best in its worst weekends. It is an integer linear program (scipy's HiGHS): 22 + 11 + 22 yes/no variables, the cost
cap, and for the risk term one constraint per scenario.

Files (data/fantasy/, local): prices.json {"drivers": {"VER": 28.5, ...}, "constructors": {"Ferrari": 28.5, ...}} in $M, and for the
transfers mode team.json {"drivers": [5 codes], "constructors": [2 names], "bank": unspent $M, "free_transfers": 2 or 3}.
The budget of a transfer plan is the value of the current team at today's prices (players are sold at the current price) plus the bank.
Price changes during the week are ignored: the optimizer looks at one weekend.
"""
import argparse
import json
import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, hstack, identity, lil_matrix

from f1live.paths import DATA
from fantasy import scoring as S

N_DRIVERS, N_CONSTRUCTORS = 5, 2
FRESH_BUDGET = 100.0
ALPHA = 0.10
FANTASY_DIR = DATA / "fantasy"


@dataclass
class Market:
    drivers: dict                       # tla -> price ($M)
    constructors: dict                  # name -> price
    team_drivers: list = field(default_factory=list)
    team_constructors: list = field(default_factory=list)
    bank: float = 0.0
    free_transfers: int = S.FREE_TRANSFERS

    def team_value(self):
        return sum(self.drivers[d] for d in self.team_drivers) + sum(self.constructors[c] for c in self.team_constructors)


def load_market(prices=FANTASY_DIR / "prices.json", team=FANTASY_DIR / "team.json"):
    """Market from the two JSON files; the team file is optional (fresh mode)."""
    p = json.loads(Path(prices).read_text())
    m = Market(drivers={k: float(v) for k, v in p["drivers"].items()}, constructors={k: float(v) for k, v in p["constructors"].items()})
    if team is not None and Path(team).exists():
        t = json.loads(Path(team).read_text())
        m.team_drivers, m.team_constructors = list(t["drivers"]), list(t["constructors"])
        m.bank, m.free_transfers = float(t.get("bank", 0.0)), int(t.get("free_transfers", S.FREE_TRANSFERS))
        unknown = [d for d in m.team_drivers if d not in m.drivers] + [c for c in m.team_constructors if c not in m.constructors]
        if unknown or len(m.team_drivers) != N_DRIVERS or len(m.team_constructors) != N_CONSTRUCTORS:
            raise SystemExit(f"team.json: need {N_DRIVERS} drivers and {N_CONSTRUCTORS} constructors that are in prices.json (problem: {unknown})")
    return m


@dataclass
class Plan:
    drivers: list
    constructors: list
    boost: str
    cost: float
    swaps: int
    penalty: float                      # points lost for swaps beyond the free ones (negative or 0)
    points: np.ndarray                  # per scenario, boost and penalty included
    out: list = field(default_factory=list)
    into: list = field(default_factory=list)

    @property
    def mean(self):
        return float(self.points.mean())

    def cvar(self, alpha=ALPHA):
        k = max(int(round(alpha * len(self.points))), 1)
        return float(np.sort(self.points)[:k].mean())


def plan_points(sc, drivers, constructors, boost, penalty=0.0):
    """Points of a team in every scenario."""
    di = [sc.drivers.index(d) for d in drivers]
    ci = [sc.constructors.index(c) for c in constructors]
    return sc.driver_points[:, di].sum(axis=1) + sc.driver_points[:, sc.drivers.index(boost)] + sc.constructor_points[:, ci].sum(axis=1) + penalty


def _universe(sc, market):
    """Drivers and constructors that are both priced and simulated, with a warning for the rest."""
    drivers = [d for d in sc.drivers if d in market.drivers]
    cons = [c for c in sc.constructors if c in market.constructors]
    missing = [d for d in market.drivers if d not in sc.drivers] + [c for c in market.constructors if c not in sc.constructors]
    if missing:
        warnings.warn(f"priced but not in the weekend's simulation, left out: {missing}")
    return drivers, cons


def solve(sc, market, mode="fresh", budget=None, limitless=False, risk=0.0, alpha=ALPHA, swaps=None, exclude=()):
    """The best plan, or None if there is none. mode 'fresh': no current team, no swap penalty. 'transfers': the team of `market`,
    swaps beyond market.free_transfers cost 10 each. swaps: force exactly that many swaps (transfers mode). exclude: asset sets
    (collections of driver codes and constructor names) of plans already found; a team that shares 6 or more of 7 assets with one
    is not returned (the top-N list)."""
    drivers, cons = _universe(sc, market)
    nD, nT, S_ = len(drivers), len(cons), len(sc.driver_points)
    di = [sc.drivers.index(d) for d in drivers]
    ci = [sc.constructors.index(c) for c in cons]
    Pd, Pc = sc.driver_points[:, di], sc.constructor_points[:, ci]
    price = np.array([market.drivers[d] for d in drivers] + [market.constructors[c] for c in cons])
    use_cvar = risk > 0
    transfers = mode == "transfers"
    if transfers and not market.team_drivers:
        raise SystemExit("transfers mode needs data/fantasy/team.json")
    # variables: x (drivers in the team), y (constructors), z (boost), e (swaps beyond the free ones), then eta and u for the CVaR
    ix, iy, iz = np.arange(nD), nD + np.arange(nT), nD + nT + np.arange(nD)
    ie = nD + nT + nD
    ieta, iu0 = ie + 1, ie + 2
    n = ie + 1 + ((1 + S_) if use_cvar else 0)
    mean_d, mean_c = Pd.mean(axis=0), Pc.mean(axis=0)
    c = np.zeros(n)                                              # maximise: stored negated
    c[ix], c[iz], c[iy] = (1 - risk) * mean_d, (1 - risk) * mean_d, (1 - risk) * mean_c
    c[ie] = S.EXTRA_TRANSFER if transfers else 0.0
    if use_cvar:
        c[ieta], c[iu0:] = risk, -risk / (alpha * S_)
    rows, lo, hi = [], [], []

    def add(coefs, low, high):
        rows.append(coefs), lo.append(low), hi.append(high)

    add({**{i: 1 for i in ix}}, N_DRIVERS, N_DRIVERS)
    add({**{i: 1 for i in iy}}, N_CONSTRUCTORS, N_CONSTRUCTORS)
    add({**{i: 1 for i in iz}}, 1, 1)
    for k in range(nD):
        add({iz[k]: 1, ix[k]: -1}, -np.inf, 0)                   # the boost goes to a driver in the team
    if not limitless:
        cap = budget if budget is not None else (market.team_value() + market.bank if transfers else FRESH_BUDGET)
        add({**{ix[k]: price[k] for k in range(nD)}, **{iy[k]: price[nD + k] for k in range(nT)}}, -np.inf, cap)
    if transfers:
        new = {ix[k]: 1 for k in range(nD) if drivers[k] not in market.team_drivers}
        new.update({iy[k]: 1 for k in range(nT) if cons[k] not in market.team_constructors})
        if swaps is None:
            add({**new, ie: -1}, -np.inf, market.free_transfers)  # swaps - e <= free  (e >= 0, the objective pushes it down)
        else:
            add(new, swaps, swaps)
            add({ie: 1}, max(swaps - market.free_transfers, 0), max(swaps - market.free_transfers, 0))
    names = drivers + cons
    for prev in exclude:
        idx = [ix[drivers.index(a)] if a in drivers else iy[cons.index(a)] for a in prev if a in names]
        add({i: 1 for i in idx}, -np.inf, N_DRIVERS + N_CONSTRUCTORS - 1)
    if use_cvar:                                                 # eta - points(s) - u(s) <= 0 for every scenario s
        head = np.zeros((S_, iu0))
        head[:, ieta] = 1
        head[:, ix], head[:, iz], head[:, iy] = -Pd, -Pd, -Pc
        A = hstack([csr_matrix(head), -identity(S_, format="csr")], format="csr")
    A_rows = lil_matrix((len(rows), n))
    for r, coefs in enumerate(rows):
        for j, v in coefs.items():
            A_rows[r, j] = v
    constraints = [LinearConstraint(csr_matrix(A_rows), lo, hi)]
    if use_cvar:
        constraints.append(LinearConstraint(csr_matrix(A), -np.inf, 0))
    integrality = np.zeros(n)
    integrality[: ie] = 1
    lb, ub = np.zeros(n), np.ones(n)
    ub[ie] = N_DRIVERS + N_CONSTRUCTORS
    if use_cvar:
        lb[ieta], ub[ieta] = -np.inf, np.inf
        ub[iu0:] = np.inf
    res = milp(-c, constraints=constraints, integrality=integrality, bounds=Bounds(lb, ub))
    if not res.success:
        return None
    x = np.round(res.x)
    team_d = [drivers[k] for k in range(nD) if x[ix[k]]]
    team_c = [cons[k] for k in range(nT) if x[iy[k]]]
    boost = next(drivers[k] for k in range(nD) if x[iz[k]])
    cost = float(sum(market.drivers[d] for d in team_d) + sum(market.constructors[t] for t in team_c))
    n_swaps = penalty = 0
    out, into = [], []
    if transfers:
        into = [a for a in team_d + team_c if a not in market.team_drivers + market.team_constructors]
        out = [a for a in market.team_drivers + market.team_constructors if a not in team_d + team_c]
        n_swaps = len(into)
        penalty = S.transfer_penalty(n_swaps, market.free_transfers)
    return Plan(team_d, team_c, boost, cost, n_swaps, penalty, plan_points(sc, team_d, team_c, boost, penalty), out, into)


def top_plans(sc, market, count=5, **kw):
    """The `count` best different teams (fresh mode), best first."""
    found, plans = [], []
    for _ in range(count):
        p = solve(sc, market, exclude=found, **kw)
        if p is None:
            break
        plans.append(p)
        found.append(p.drivers + p.constructors)
    return plans


def swap_table(sc, market, max_swaps=N_DRIVERS + N_CONSTRUCTORS, **kw):
    """transfers mode: the best plan for 0, 1, 2 ... swaps (the penalty is in each plan's points), best first by net points."""
    plans = [p for n in range(max_swaps + 1) if (p := solve(sc, market, "transfers", swaps=n, **kw)) is not None]
    return plans


def describe(plan, alpha=ALPHA):
    boost = lambda d: f"{d}*" if d == plan.boost else d
    return (f"{' '.join(boost(d) for d in plan.drivers)} | {' + '.join(plan.constructors)}  "
            f"${plan.cost:.1f}M  mean {plan.mean:.1f}  sd {plan.points.std():.1f}  worst-{int(alpha * 100)}% {plan.cvar(alpha):.1f}")


# ---------------------------------------------------------------------------------------------------------------------------
def scenarios_for(year, race, cache, reuse, grid, grids, runs, seed):
    """Scenarios from fantasy.scenarios, cached to `cache` (a pickle) so that changing only the optimizer settings is instant."""
    from fantasy import scenarios as SC
    if reuse and Path(cache).exists():
        with open(cache, "rb") as f:
            return pickle.load(f)
    sc = SC.simulate_weekend(year, race, grids=grids, runs=runs, seed=seed, grid=grid, verbose=True)
    sc.parts = {}                                              # the optimizer needs only the points
    Path(cache).parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(sc, f)
    return sc


def report_fresh(sc, market, a):
    budget = None if a.limitless else (a.budget if a.budget is not None else FRESH_BUDGET)
    plans = top_plans(sc, market, a.top, budget=budget, limitless=a.limitless, risk=a.risk, alpha=ALPHA)
    cap = "no budget cap (Limitless)" if a.limitless else f"budget ${budget:.1f}M"
    print(f"\nBest new teams for {a.year} {a.race}  ({cap}, risk {a.risk:g}; * = DRS boost)\n")
    for i, p in enumerate(plans, 1):
        print(f"{i}. {describe(p)}")


def report_transfers(sc, market, a):
    drivers, cons = _universe(sc, market)
    cap = a.budget if a.budget is not None else market.team_value() + market.bank
    keep_boost = max(market.team_drivers, key=lambda d: sc.driver_points[:, sc.drivers.index(d)].mean())
    keep_pts = plan_points(sc, market.team_drivers, market.team_constructors, keep_boost)
    plans = swap_table(sc, market, budget=a.budget, limitless=a.limitless, risk=a.risk, alpha=ALPHA)
    print(f"\nTransfers for {a.year} {a.race}: your team {' '.join(market.team_drivers)} | {' + '.join(market.team_constructors)}"
          f"  (value ${market.team_value():.1f}M, bank ${market.bank:.1f}M, budget ${cap:.1f}M, {market.free_transfers} free swaps)\n")
    print(f"keep:     boost {keep_boost}  mean {keep_pts.mean():.1f}  sd {keep_pts.std():.1f}\n")
    out_w = max([len(" ".join(p.out) or "-") for p in plans] + [3])
    in_w = max([len(" ".join(p.into) or "-") for p in plans] + [2])
    print(f"swaps  net mean  vs keep  P(better)   cost  {'out':<{out_w}} -> {'in':<{in_w}}  boost")
    best = None
    for p in plans:
        diff = p.points - keep_pts
        better = "-" if p.swaps == 0 else f"{(diff > 0).mean():.0%}"
        row = f"{p.swaps:>5}  {p.mean:8.1f}  {p.mean - keep_pts.mean():+7.1f}  {better:>8}  {p.cost:5.1f}  {' '.join(p.out) or '-':<{out_w}} -> {' '.join(p.into) or '-':<{in_w}}  {p.boost}"
        print(row)
        if best is None or p.mean > best.mean:
            best = p
    if best is None or best.swaps == 0 or best.mean - keep_pts.mean() < 0.5:
        print("\nVerdict: keep your team" + (f" (set the boost on {keep_boost})." if best is None or best.swaps == 0 else "; no swap is worth it."))
    else:
        extra = f", of which {-best.penalty:.0f} points of penalty already deducted" if best.penalty else ""
        print(f"\nVerdict: make {best.swaps} swap{'s' if best.swaps > 1 else ''}: {' '.join(best.out)} -> {' '.join(best.into)}, boost {best.boost}: "
              f"+{best.mean - keep_pts.mean():.1f} expected points net{extra}.")


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["fresh", "transfers"])
    ap.add_argument("year", type=int)
    ap.add_argument("race", help="data folder of the race, e.g. Mexico_City")
    ap.add_argument("--budget", type=float, help="cap in $M (fresh: 100, transfers: team value + bank)")
    ap.add_argument("--limitless", action="store_true", help="no budget cap and no swap penalty")
    ap.add_argument("--risk", type=float, default=0.0, help="0 = maximise the mean, 1 = the worst 10 %% of weekends")
    ap.add_argument("--top", type=int, default=5, help="fresh: how many teams to list")
    ap.add_argument("--prices", default=FANTASY_DIR / "prices.json", type=Path)
    ap.add_argument("--team", default=FANTASY_DIR / "team.json", type=Path)
    ap.add_argument("--grid", help="comma-separated driver codes in starting order (default: the qualifying result or a simulated grid)")
    ap.add_argument("--grids", type=int, default=40)
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reuse", action="store_true", help="use the scenarios saved by the last run for this race")
    a = ap.parse_args()
    market = load_market(a.prices, a.team if a.mode == "transfers" else None)
    cache = FANTASY_DIR / "cache" / f"{a.year}_{a.race}.pkl"
    sc = scenarios_for(a.year, a.race, cache, a.reuse, a.grid.split(",") if a.grid else None, a.grids, a.runs, a.seed)
    (report_fresh if a.mode == "fresh" else report_transfers)(sc, market, a)


if __name__ == "__main__":
    main()
