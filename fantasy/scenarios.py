"""Monte Carlo of one race weekend scored in fantasy points: what the optimizer maximizes.

    python -m fantasy.scenarios 2026 Mexico_City                      # before qualifying: the grid is simulated too
    python -m fantasy.scenarios 2026 Mexico_City --grid VER,NOR,...   # a known grid (penalties included)

Each scenario is one possible weekend, scored with fantasy.scoring for every driver and constructor, so correlations come for free
(team-mates share a car, a safety car shuffles everybody, a DNF costs a driver and helps the cars behind him).

Pipeline
  1. grid     qualifying already run: that grid. Otherwise pre.quali.predict_grid draws `grids` qualifying outcomes (team shock + driver
              noise on the predicted gaps); the race model's pace inputs use the predicted gaps.
  2. race     for every grid, `runs` runs of the race Monte Carlo (live.race.projection, via pre.race.predict_result): finishing order,
              retirements, stops, quickest lap. The simulation alone is too random at the front (pole-sitters win two thirds of races,
              the simulation gave a quarter), so predict_result blends its finishing distribution with where cars from that grid slot
              really finish. The blend only exists as marginals; remap() brings every driver's simulated finishing position to the
              blended marginal while keeping each scenario's dependence (quantile mapping on the rank within the scenario).
  3. extras   overtakes (fantasy.overtakes), constructor pit stops (fantasy.pitstops), fastest lap (the quickest simulated clean lap)
              and Driver of the Day (below).
  4. points   fantasy.scoring, per scenario.

Driver of the Day is a fan vote and nothing recorded says how fans vote. The model is a guess: among the classified drivers the chance
is proportional to exp(weights): a win, a podium, a points finish and places gained (the usual winners of the award). It is worth
10 points to one driver, so a wrong guess moves little; DOTD holds the weights.

A driver who does not take part (withdrawn=["OCO"], --out OCO) is left out of the simulation (a reserve who drove practice for his team takes the
seat), and he is put back at -25 points (his penalties) in every scenario, so that a team that still holds him scores accordingly and
the optimizer swaps him out on its own. His constructor takes the -25 as well only if no substitute drives.

Not modelled: sprint weekends (their scoring is in fantasy.scoring, the simulation is not), time penalties in the pits, chips.
"""
import argparse
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from f1live.paths import DATA
from live.race import priors as P
from fantasy import overtakes as OT
from fantasy import pitstops as PS
from fantasy import scoring as S
from pre.history import available, race_key
from pre.quali import predict_grid as QG
from pre.quali.features import Tables
from pre.race import predict_result as PR
from pre.race.pace import RaceArchive

DOTD = dict(win=1.6, podium=0.8, points=0.4, per_place_gained=0.18, max_gained=15)
Q3_CARS = 10                       # cars in the last part of qualifying
Q1_ELIMINATED = {22: 6, 20: 5}     # cars out in Q1 (and again in Q2) by field size


@dataclass
class Scenarios:
    drivers: list                  # tla, the columns of driver_points
    team_of: dict                  # tla -> constructor
    constructors: list             # the columns of constructor_points
    driver_points: np.ndarray      # (scenarios, drivers)
    constructor_points: np.ndarray
    parts: dict = field(default_factory=dict)   # per-scenario pieces for diagnostics: grid, finish, retired, overtakes, ...
    meta: dict = field(default_factory=dict)

    def summary(self):
        """DataFrame of the drivers and constructors: mean, sd and the 10th / 90th percentile of their weekend points."""
        def stats(x, names, kind):
            return pd.DataFrame(dict(kind=kind, mean=x.mean(0), sd=x.std(0), p10=np.percentile(x, 10, axis=0),
                                     p90=np.percentile(x, 90, axis=0)), index=names)
        return pd.concat([stats(self.driver_points, self.drivers, "driver"),
                          stats(self.constructor_points, self.constructors, "constructor")])


# ---------------------------------------------------------------------------------------------------------------------------
# vectorised scoring (the scalar versions in fantasy.scoring are the reference; the tests check these against them)
# ---------------------------------------------------------------------------------------------------------------------------
def ladder(table, position):
    """Points of an array of 1-based positions in a ladder (0 below it)."""
    t = np.concatenate([table, [0]])
    return t[np.clip(np.asarray(position, int) - 1, 0, len(table))]


def quali_part(position, n_cars):
    """Last qualifying part reached from the final qualifying position: 3 = Q3, 2 = Q2, 1 = out in Q1."""
    out = Q1_ELIMINATED.get(n_cars, 5 if n_cars <= 20 else 6)
    return np.where(position <= Q3_CARS, 3, np.where(position <= n_cars - out, 2, 1))


def constructor_quali_bonus(part_a, part_b):
    """Vectorised scoring.constructor_quali_bonus for arrays of the two drivers' parts."""
    q3 = (part_a == 3).astype(int) + (part_b == 3)
    q2 = (part_a >= 2).astype(int) + (part_b >= 2)
    return np.where(q3 == 2, 10, np.where(q3 == 1, 5, np.where(q2 == 2, 3, np.where(q2 == 1, 1, -1))))


def race_points(grid, finish, retired, overtakes, fastest_lap, dotd):
    """Vectorised scoring.race_points on (scenarios, drivers) arrays. finish is ignored where retired."""
    gain = grid - finish
    pts = ladder(S.RACE, finish) + gain + overtakes + S.RACE_FASTEST_LAP * fastest_lap + S.DRIVER_OF_THE_DAY * dotd
    return np.where(retired, S.RACE_NC + overtakes, pts)


def remap(finish, retired, target, rng):
    """Bring each car's simulated finishing position to the target marginal, keeping the scenarios' dependence.

    finish: (n, cars) positions 1..cars from the simulation (retired cars last), retired: (n, cars) bool, target: (cars, cars) with
    target[c, k] = P(car c finishes k+1). Each car's position is turned into a uniform u through its own simulated distribution
    (randomised inside the step), then through the inverse of the target distribution; the scenario's cars are then ranked by that
    value, retired cars at the back. -> (n, cars) positions."""
    n, cars = finish.shape
    pos = np.arange(1, cars + 1)
    sim = np.stack([(finish[:, c, None] == pos[None, :]).mean(axis=0) for c in range(cars)])        # (cars, positions)
    cdf_sim = np.cumsum(sim, axis=1)
    cdf_tgt = np.cumsum(target / target.sum(axis=1, keepdims=True), axis=1)
    u = np.empty((n, cars))
    x = np.empty((n, cars))
    knots = np.arange(cars + 1)
    for c in range(cars):
        hi = cdf_sim[c][finish[:, c] - 1]
        lo = hi - sim[c][finish[:, c] - 1]
        u[:, c] = lo + rng.random(n) * (hi - lo)
        x[:, c] = np.interp(u[:, c], np.concatenate([[0.0], cdf_tgt[c]]), knots)
    key = x + retired * 1e3
    return key.argsort(axis=1).argsort(axis=1) + 1


def dotd_draw(rng, finish, grid, retired):
    """(scenarios, drivers) 0/1: who wins Driver of the Day, drawn among the classified drivers (see DOTD)."""
    w = DOTD
    gained = np.clip(grid - finish, 0, w["max_gained"])
    logit = (w["win"] * (finish == 1) + w["podium"] * ((finish > 1) & (finish <= 3)) + w["points"] * (finish <= 10)
             + w["per_place_gained"] * gained)
    p = np.where(retired, 0.0, np.exp(logit))
    p = p / p.sum(axis=1, keepdims=True)
    pick = (p.cumsum(axis=1) < rng.random(len(p))[:, None]).sum(axis=1).clip(0, p.shape[1] - 1)
    out = np.zeros(p.shape, int)
    out[np.arange(len(p)), pick] = 1
    return out


# ---------------------------------------------------------------------------------------------------------------------------
# the simulation
# ---------------------------------------------------------------------------------------------------------------------------
def quali_stage(tables, av, year, race, grids, seed, grid=None, withdrawn=()):
    """-> (drivers, team Series, positions (grids, drivers), q table for the pace model, was the grid known, {withdrawn driver: team})."""
    actual = PR.quali_table(av, year, race)
    if grid is not None or actual is not None:
        q = actual
        if grid is not None:
            order = [d for d in grid if d not in withdrawn]
            drivers = sorted(order)
            team = q["team"] if q is not None else None
            if q is None:
                raise SystemExit("--grid needs the qualifying session of the weekend (or use the predicted grid)")
            pos = np.array([[order.index(d) + 1 for d in drivers]])
        else:
            drivers = [d for d in sorted(q.index) if d not in withdrawn]
            pos = q.loc[drivers, "rank"].rank(method="first").to_numpy()[None, :]       # re-ranked without the withdrawn drivers
        gone = {d: q.loc[d, "team"] if d in q.index else None for d in withdrawn}
        return drivers, q.loc[drivers, "team"], pos.astype(int), q, True, gone
    fit, res, target = QG.predict_weekend(tables, year, race)
    if fit is None:
        raise SystemExit("not enough earlier weekends to predict the qualifying")
    pred = QG.predict(fit, target)
    drivers = PR.race_field(target, withdrawn)
    pred = pred.loc[drivers]
    draws = QG.draw_gaps(pred, target.loc[drivers, "team"], fit, grids, seed)
    pos = draws.argsort(axis=1).argsort(axis=1) + 1
    q = pd.DataFrame(dict(team=target.loc[drivers, "team"], gap_pct=pred - pred.min()))
    q["qrel"] = q["gap_pct"] - q["gap_pct"].median()
    q["rank"] = q["gap_pct"].rank(method="first").astype(int)
    return drivers, q["team"], pos, q, False, {d: target.loc[d, "team"] if d in target.index else None for d in withdrawn}


def simulate_weekend(year, race, grids=40, runs=100, seed=0, grid=None, laps=None, data_dir=DATA, verbose=False, withdrawn=()):
    """Scenarios for the race weekend. See the module docstring. withdrawn: driver codes who do not take part."""
    av = available(data_dir)
    priors = P.load_priors()
    archive = RaceArchive(av[av["session"] == "race"].reset_index(drop=True), priors)
    tables = Tables(av)
    drivers, team, pos, q, known, gone = quali_stage(tables, av, year, race, grids, seed, grid, tuple(withdrawn))
    n_grids, D = pos.shape
    w, why = PR.prepare_weekend(archive, av, priors, year, race, grid=None, laps=laps, q=q, fill_team=True)
    if w is None:
        raise SystemExit(why)
    table = PR.finishing_table(archive, av, w["before"])
    col = {d: i for i, d in enumerate(drivers)}
    rng = np.random.default_rng(seed + 12345)
    keep = [d for d in drivers if d in w["rel"].index]
    if len(keep) < D:
        raise SystemExit(f"no pace estimate for {sorted(set(drivers) - set(keep))}")
    S_ = n_grids * runs
    finish, retired = np.empty((S_, D), int), np.empty((S_, D), bool)
    stops, fastest, grid_pos = np.empty((S_, D)), np.empty((S_, D)), np.empty((S_, D), int)
    for k in range(n_grids):
        order = sorted(drivers, key=lambda d: pos[k, col[d]])
        out, info = PR.simulate_from_grid(order, w["rel"], w["sd"], priors, w["circuit"], w["total"], w["tlap"], sims=runs,
                                          seed=seed + k, temp=w["temp"])
        blended = PR.blend_with_grid(out, order, table)
        cols = [f"pos_{j}" for j in range(1, D + 1)]
        target = blended.set_index("tla").loc[order, cols].to_numpy()
        fin = remap(info["finish"], info["retired"], target, rng)
        sl = slice(k * runs, (k + 1) * runs)
        idx = [col[t] for t in info["tlas"]]
        finish[sl][:, idx] = fin
        retired[sl][:, idx] = info["retired"]
        stops[sl][:, idx] = info["stops"]
        fastest[sl][:, idx] = info["fastest_lap_s"]
        grid_pos[sl] = pos[k][None, :]
        if verbose:
            print(f"  grid {k + 1}/{n_grids}", end="\r")
    sc = score(year, race, drivers, team, grid_pos, finish, retired, stops, fastest, seed,
               meta=dict(grid_known=known, grids=n_grids, runs=runs, laps=w["total"], circuit=race_key(race)), av=av)
    return add_inactive(sc, gone)


def score(year, race, drivers, team, grid, finish, retired, stops, fastest, seed, meta, av):
    """Fantasy points of every scenario from the simulated outcomes (all arrays (scenarios, drivers))."""
    rng = np.random.default_rng(seed + 777)
    S_, D = finish.shape
    constructors = sorted(set(team))
    ci = np.array([constructors.index(team[d]) for d in drivers])
    q_pts = ladder(S.QUALI, grid)
    ot_model = OT.fit_model(OT.dataset(av))
    overtakes = ot_model.sample(rng, "race", grid, finish, retired, race_key(race))
    clean = np.where(np.isfinite(fastest), fastest, np.inf)
    fl = np.zeros((S_, D), int)
    has_lap = np.isfinite(clean.min(axis=1))
    fl[np.arange(S_)[has_lap], clean.argmin(axis=1)[has_lap]] = 1
    dotd = dotd_draw(rng, finish, grid, retired)
    race_ex_dotd = race_points(grid, finish, retired, overtakes, fl, np.zeros_like(dotd))
    driver_pts = q_pts + race_ex_dotd + S.DRIVER_OF_THE_DAY * dotd
    parts_q = quali_part(grid, D)
    pit_model = PS.fit_model(av)
    team_stops = np.stack([stops[:, ci == t].sum(axis=1) for t in range(len(constructors))], axis=1).astype(int)
    pit_pts = pit_model.points(rng, constructors, team_stops)
    cons = np.zeros((S_, len(constructors)))
    for t in range(len(constructors)):
        idx = np.flatnonzero(ci == t)[:2]                         # a team with one car in the field scores that car only
        part_a = parts_q[:, idx[0]]
        part_b = parts_q[:, idx[1]] if len(idx) > 1 else np.ones(S_, int)
        cons[:, t] = (q_pts[:, idx].sum(axis=1) + constructor_quali_bonus(part_a, part_b)
                      + race_ex_dotd[:, idx].sum(axis=1) + pit_pts[:, t])
    return Scenarios(drivers=drivers, team_of=dict(team), constructors=constructors, driver_points=driver_pts,
                     constructor_points=cons, meta=meta,
                     parts=dict(grid=grid, finish=finish, retired=retired, overtakes=overtakes, fastest_lap=fl, dotd=dotd,
                                pit=pit_pts, stops=stops))


def add_inactive(sc, inactive):
    """Put drivers who do not take part back into the table at their fixed score: the rules give every inactive driver the penalties
    (-5 no qualifying time, -20 not classified in the race). inactive: {driver code: constructor}.
    His constructor scores both of its drivers' points, so it takes the penalties too, unless a substitute drives: a team that still
    has two cars in the simulation (the reserve who took the seat) does not. A driver who drove none of the weekend's sessions comes
    with team None: he still scores the penalties, his constructor is left alone (a substitute drove practice in his place)."""
    pts = S.QUALI_NO_TIME + S.RACE_NC
    for code, team in inactive.items():
        cars = sum(t == team for t in sc.team_of.values())
        sc.drivers.append(code)
        sc.team_of[code] = team
        sc.driver_points = np.hstack([sc.driver_points, np.full((len(sc.driver_points), 1), float(pts))])
        if team is not None and team in sc.constructors and cars < 2:
            sc.constructor_points[:, sc.constructors.index(team)] += pts
    sc.meta["inactive"] = sorted(inactive)
    return sc


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("year", type=int)
    ap.add_argument("race", help="data folder of the race, e.g. Mexico_City")
    ap.add_argument("--grids", type=int, default=40, help="simulated qualifying outcomes (ignored with a known grid)")
    ap.add_argument("--runs", type=int, default=100, help="race simulations per grid")
    ap.add_argument("--laps", type=int)
    ap.add_argument("--grid", help="comma-separated driver codes in starting order")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="", help="comma-separated codes of drivers who do not take part")
    ap.add_argument("--data-dir", default=DATA, type=type(DATA))
    a = ap.parse_args()
    sc = simulate_weekend(a.year, a.race, a.grids, a.runs, a.seed, a.grid.split(",") if a.grid else None, a.laps, a.data_dir, verbose=True,
                           withdrawn=[d for d in a.out.split(",") if d])
    m = sc.meta
    print(f"\n{a.year} {a.race}: {sc.driver_points.shape[0]} scenarios ({'known grid' if m['grid_known'] else str(m['grids']) + ' simulated grids'}"
          f" x {m['runs']} runs), {m['laps']} laps\n")
    print(sc.summary().round(1).sort_values(["kind", "mean"], ascending=[True, False]).to_string())


if __name__ == "__main__":
    main()
