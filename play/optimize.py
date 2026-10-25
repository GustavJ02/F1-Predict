"""Find the best tyre strategy for a driver, with pit windows.

For one driver at a time, every candidate strategy (compound sequence x stop laps) is simulated with all the other cars on
their normal (stochastic) strategies, using the same random draws for every candidate so that candidates can be compared
fairly. The search is coarse-to-fine: a coarse grid of stop laps over all legal compound sequences, then a local search
around the best few, then the finalists are re-simulated with many more runs for the numbers you read. The pit window of a
stop is the range of laps whose result is within a tolerance of the best.

Tyres: a driver only has so many fresh sets of each compound for the weekend (the allocation minus the sets already used in
practice, qualifying and the sprint, or earlier in the race), so only plans that fit are ranked and the output says what
the limits cost (play/tyres.py). Say what a driver really has with --sets HAM=S0,M2,H1, rule a compound out with --ban HAM=S,
or switch the limits off with --tyres none.

What 'best' means (--objective): points (expected championship points from the finishing position), win, podium, position
(mean finishing position).

Limits worth knowing: the other cars don't react to the plan (a real rival would cover an undercut), the plan is followed
exactly (unless --sc-react), and the answer is only as good as the pace, tyre and passing assumptions in live/race/priors.json.
A strategy that wins by a few hundredths of a place is a tie: look at the windows and the error bars. The lap model has no
tyre cliff, so plans with a stint longer than the tyre has ever lasted (fewer than --min-survival of past stints on that
compound got that far) are dropped, and the 'tyre life' column shows how routine the longest stint of each plan is.

Usage (from the repo root):
    python -m play.optimize pre 2026 Austin HAM
    python -m play.optimize pre 2026 Austin HAM VER --objective win --stops 1 2
    python -m play.optimize live 2026 Austin race HAM --objective points      # from the current state of the race
"""
import argparse
import itertools
import os
from collections import Counter
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from live.race.priors import COMPOUNDS
from play.scenario import POINTS_TABLE, Scenario, compare
from play.strategy import MIN_STINT, Plan
from play.tyres import add_tyre_args, configure_tyres

OBJECTIVES = {"points": "expected points", "win": "chance of winning", "podium": "chance of a podium",
              "position": "mean finishing position (lower is better)"}
TOLERANCE = {"points": 0.15, "win": 0.01, "podium": 0.01, "position": 0.1}


def objective_value(finish, objective):
    """Score of one car's finishing positions (a vector over runs). Higher is better for every objective."""
    if objective == "points":
        return float(np.where(finish <= 10, POINTS_TABLE[np.minimum(finish, 10) - 1], 0.0).mean())
    if objective == "win":
        return float((finish == 1).mean())
    if objective == "podium":
        return float((finish <= 3).mean())
    if objective == "position":
        return float(-finish.mean())
    raise ValueError(f"unknown objective '{objective}' (choose from {', '.join(OBJECTIVES)})")


def tyre_survival(priors, circuit, compound, age):
    """Share of past stints on this compound (at this circuit if there are enough) that were still going at tyre age `age`."""
    curve = priors["stint"]["circuits"].get(circuit, {}).get(compound) or priors["stint"]["pooled"][compound]
    ages, surv = np.asarray(curve["age"]), np.asarray(curve["surv"])
    i = np.searchsorted(ages, age, side="right") - 1
    return 1.0 if i < 0 else float(surv[min(i, len(surv) - 1)])


def plan_survival(scenario, driver, plan):
    """Lowest survival share over the plan's stints: how realistic is the longest stint? (1 = routine, ~0 = never seen)."""
    row = scenario.cars.set_index("tla").loc[driver]
    start_age = float(row["age"]) if scenario.done > 0 else 0.0
    prev = scenario.done
    worst = 1.0
    for i, (compound, lap) in enumerate(zip(plan.compounds, plan.laps + (scenario.total,))):
        age = (lap - prev) + (start_age if i == 0 else 0.0)
        worst = min(worst, tyre_survival(scenario.priors, scenario.circuit, compound, age))
        prev = lap
    return worst


def sequences(n_stops, first=None, must_change=True):
    """Compound sequences with n_stops stops. `first` fixes the starting compound (a race under way); `must_change` drops
    the ones that use only one compound (illegal in a dry race)."""
    out = []
    for seq in itertools.product(COMPOUNDS, repeat=n_stops + 1):
        if first and seq[0] != first:
            continue
        if must_change and len(set(seq)) < 2:
            continue
        out.append(seq)
    return out


def lap_grids(n_stops, lo, hi, step, min_stint):
    """All increasing tuples of stop laps in [lo, hi] on a grid of `step` laps with at least min_stint laps between stops."""
    laps = range(lo, hi + 1, step)
    if n_stops == 0:
        return [()]
    if n_stops == 1:
        return [(l,) for l in laps]
    return [c for c in itertools.combinations(laps, n_stops) if all(b - a >= min_stint for a, b in zip(c, c[1:]))]


_WORKER = {}


def _init_worker(search):
    _WORKER["search"] = search


def _score_in_worker(args):
    plan, sims = args
    return _WORKER["search"].score(plan, sims)


class Search:
    """Evaluates plans for one driver in one scenario, remembering results."""
    def __init__(self, scenario, driver, objective, opt, sc_react=False):
        self.sc, self.driver, self.objective, self.opt, self.sc_react = scenario, driver, objective, opt, sc_react
        self.cache = {}

    def score(self, plan, sims=None, keep=False):
        key = (str(plan), sims)
        if key not in self.cache:
            sc = self.sc if sims is None else _with_sims(self.sc, sims)
            res = sc.run(plans={self.driver: plan}, sc_react=self.sc_react, **self.opt)
            self.cache[key] = (objective_value(res.column(self.driver), self.objective), res if keep else None)
        return self.cache[key][0]


def score_many(search, plans, sims, jobs):
    """Scores of many plans, spread over `jobs` processes (the results are the same as scoring them one by one)."""
    plans = list(plans)
    if jobs <= 1 or len(plans) < 24:
        return [search.score(p, sims) for p in plans]
    with ProcessPoolExecutor(jobs, initializer=_init_worker, initargs=(search,)) as pool:
        values = list(pool.map(_score_in_worker, [(p, sims) for p in plans], chunksize=max(len(plans) // (jobs * 4), 1)))
    for p, v in zip(plans, values):
        search.cache[(str(p), sims)] = (v, None)
    return values


def _with_sims(scenario, sims):
    """The same scenario with a different number of runs (the random draws are regenerated for the new count)."""
    clone = Scenario(scenario.cars, scenario.meta, scenario.track, scenario.priors, scenario.circuit, None, scenario.opt, sims,
                     scenario.seed, scenario.label, scenario.rel, scenario.order, tyres=scenario.tyres)
    if scenario.comp0 is not None:
        from pre.race import predict_result as PR
        clone.comp0 = PR.start_compounds(scenario.priors, sims, len(scenario.cars), scenario.seed)
    return clone


def optimize(scenario, driver, objective="points", stops=(1, 2), step=None, min_stint=None, top=8, search_sims=400,
             final_sims=3000, refine=5, opt=None, sc_react=False, verbose=True, tolerance=None, min_survival=0.1,
             jobs=None, tyres="auto"):
    """-> (DataFrame of the best strategies, info dict). See the module docstring.

    tyres: the sets each driver has left (play.tyres.TyreSets); "auto" = the scenario's own, None = no limits. Only plans that
    fit the driver's sets are ranked. If the limits keep out a better plan, info['unrestricted'] says which and what it was
    worth."""
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective '{objective}' (choose from {', '.join(OBJECTIVES)})")
    tyres = scenario.tyres if isinstance(tyres, str) and tyres == "auto" else tyres
    live = scenario.done > 0
    row = scenario.cars.set_index("tla").loc[driver]
    first = COMPOUNDS[int(row["comp"])] if live else None
    legal_any = live and not bool(row["needs_stop"])           # already used two compounds: any sequence is fine
    lo, hi = scenario.done + 1, scenario.total - 1
    min_stint = min_stint or max(MIN_STINT, 5)
    tolerance = TOLERANCE[objective] if tolerance is None else tolerance
    search = Search(scenario, driver, objective, opt or {}, sc_react)
    t0 = time.time()

    def fits(plan):
        return tyres is None or not tyres.check(driver, plan)

    if legal_any and 0 not in stops:
        stops = (0,) + tuple(stops)                          # a car that is already legal may simply stay out
    plans = []
    for n in stops:
        step_n = step or (2 if n == 1 else 5)
        grids = lap_grids(n, lo, hi, step_n, min_stint if not live else 3)
        for seq in sequences(n, first, must_change=not legal_any):
            plans += [Plan(seq, laps) for laps in grids]
    plans = [p for p in plans if plan_survival(scenario, driver, p) >= min_survival]   # stints longer than tyres ever last
    if not any(fits(p) for p in plans):
        raise ValueError(f"{driver} has no strategy that fits the tyres left ({tyres.describe(driver)}): "
                         "try --allow-used or --sets to say what he really has")
    if verbose:
        print(f"{driver}: {len(plans)} candidate strategies in the coarse search ({search_sims} runs each) ...", flush=True)
    jobs = jobs or min(os.cpu_count() or 1, 16)
    scored = {str(p): (v, p) for p, v in zip(plans, score_many(search, plans, search_sims, jobs))}

    def improve(value, plan):                                   # local search: move the stop laps around
        current, v = plan, value
        improved = True
        while improved:
            improved = False
            for i in range(len(plan.laps)):
                for d in (-3, -2, -1, 1, 2, 3):
                    laps = list(current.laps)
                    laps[i] += d
                    cand = Plan(current.compounds, tuple(laps))
                    try:
                        cand.validate(scenario.total, scenario.done)
                    except ValueError:
                        continue
                    if any(b - a < (3 if live else min_stint - 2) for a, b in zip((scenario.done,) + cand.laps, cand.laps)):
                        continue
                    if plan_survival(scenario, driver, cand) < min_survival:
                        continue
                    sv = scored.get(str(cand), (None,))[0]
                    if sv is None:
                        sv = search.score(cand, search_sims)
                        scored[str(cand)] = (sv, cand)
                    if sv > v + 1e-9:
                        current, v, improved = cand, sv, True
        scored[str(current)] = (v, current)

    ranked_all = sorted(scored.values(), key=lambda x: -x[0])
    seeds = ranked_all[:refine] + [x for x in ranked_all if fits(x[1])][:refine]       # the best overall and the best that fit
    for value, plan in {str(p): (v, p) for v, p in seeds}.values():
        improve(value, plan)

    ranked_all = sorted(scored.values(), key=lambda x: -x[0])
    ranked = [x for x in ranked_all if fits(x[1])]
    finalists = [p for _, p in ranked[:max(top * 2, 10)]]
    for n in stops:                                              # always include the best plan of each number of stops
        finalists += [p for _, p in ranked if len(p.laps) == n][:1]
    free_plan = ranked_all[0][1]
    limited = not fits(free_plan)                                # the limits keep out the best plan
    if limited:
        finalists.append(free_plan)
    finalists = list({str(p): p for p in finalists}.values())
    if verbose:
        print(f"re-simulating the {len(finalists)} best with {final_sims} runs ...", flush=True)
    sc_final = _with_sims(scenario, final_sims) if final_sims != scenario.sims else scenario
    base = sc_final.run(sc_react=sc_react, **(opt or {}))
    rows = []
    for plan in finalists:
        res = sc_final.run(plans={driver: plan}, sc_react=sc_react, **(opt or {}))
        c = compare(base, res, [driver]).iloc[0]
        st = res.stats(driver)
        rows.append(dict(strategy=str(plan), stops=len(plan.laps), life=plan_survival(scenario, driver, plan),
                         value=objective_value(res.column(driver), objective), fits=fits(plan),
                         mean_pos=st["mean_pos"], p_win=st["p_win"], p_podium=st["p_podium"], exp_points=st["exp_points"],
                         d_points=c["d_points"], se_points=c["se_points"], d_pos=c["d_pos"], se_pos=c["se_pos"], plan=plan))
    everything = pd.DataFrame(rows).sort_values("value", ascending=False).reset_index(drop=True)
    full = everything[everything["fits"]].reset_index(drop=True)
    table = full.head(top).reset_index(drop=True)
    by_stops = full.groupby("stops").head(1).sort_values("stops").reset_index(drop=True)
    best_plan = table.loc[0, "plan"]
    windows = pit_windows(sc_final, driver, best_plan, objective, tolerance, opt or {}, sc_react, jobs=jobs)
    info = dict(baseline=base.stats(driver), windows=windows, by_stops=by_stops, tolerance=tolerance, seconds=time.time() - t0,
                scenario=scenario.label, objective=objective,
                tyres=None if tyres is None else tyres.describe(driver), unrestricted=None)
    if limited:
        free = everything[everything["strategy"] == str(free_plan)].iloc[0]
        info["unrestricted"] = dict(strategy=str(free_plan), value=float(free["value"]), exp_points=float(free["exp_points"]),
                                    mean_pos=float(free["mean_pos"]), cost=float(free["value"] - table.loc[0, "value"]),
                                    missing=tyres.check(driver, free_plan))
    return table, info


def pit_windows(scenario, driver, plan, objective, tolerance, opt=None, sc_react=False, span=8, jobs=1):
    """For each stop of `plan`: (lap of the stop, first lap, last lap) of the range whose result is within `tolerance` of the
    best, moving only that stop and keeping the others where they are."""
    search = Search(scenario, driver, objective, opt or {}, sc_react)
    out = []
    for i, lap in enumerate(plan.laps):
        cands = {}
        for l in range(lap - span, lap + span + 1):
            laps = list(plan.laps)
            laps[i] = l
            cand = Plan(plan.compounds, tuple(laps))
            try:
                cand.validate(scenario.total, scenario.done)
            except ValueError:
                continue
            cands[l] = cand
        values = dict(zip(cands, score_many(search, cands.values(), None, jobs)))
        best = max(values.values())
        ok = sorted(l for l, v in values.items() if v >= best - tolerance)
        # the contiguous run of acceptable laps around the stop itself
        lo = hi = lap
        while lo - 1 in ok:
            lo -= 1
        while hi + 1 in ok:
            hi += 1
        out.append(dict(stop=i + 1, lap=lap, first=lo, last=hi, curve=values))
    return out


def print_result(table, info, driver):
    obj = info["objective"]
    base = info["baseline"]
    print(f"\n{driver} - {info['scenario']} - best strategies by {OBJECTIVES[obj]}  ({info['seconds']:.0f} s)")
    print(f"as predicted (the field's usual stop plans): mean position {base['mean_pos']:.2f}, win {base['p_win'] * 100:.1f}%, "
          f"podium {base['p_podium'] * 100:.1f}%, expected points {base['exp_points']:.2f}")
    if info.get("tyres"):
        print(f"tyres: {info['tyres']}")
    print()
    print(f" {'#':>2} {'strategy':<14} {'mean pos':>8} {'win':>6} {'podium':>7} {'E[pts]':>7} {'tyre life':>9}   vs prediction (points / places)")
    for i, r in table.iterrows():
        print(f" {i + 1:>2} {r['strategy']:<14} {r['mean_pos']:>8.2f} {r['p_win'] * 100:>5.1f}% {r['p_podium'] * 100:>6.1f}% "
              f"{r['exp_points']:>7.2f} {r['life'] * 100:>8.0f}%   {r['d_points']:+.2f} +-{r['se_points']:.2f} / {r['d_pos']:+.2f} +-{r['se_pos']:.2f}")
    if len(info["by_stops"]) > 1:
        print("\nbest by number of stops: " + "   ".join(
            f"{int(r['stops'])}-stop {r['strategy']} ({r['exp_points']:.2f} pts, {r['mean_pos']:.2f})" if r["stops"] else
            f"no stop {r['strategy']} ({r['exp_points']:.2f} pts, {r['mean_pos']:.2f})" for _, r in info["by_stops"].iterrows()))
    free = info.get("unrestricted")
    if free:
        cost = {"points": f"{free['cost']:.2f} points", "position": f"{free['cost']:.2f} places",
                "win": f"{free['cost'] * 100:.1f} points of win chance", "podium": f"{free['cost'] * 100:.1f} points of podium chance"}[obj]
        print(f"\nthe tyre limits cost {cost}: without them {free['strategy']} would be better "
              f"({free['exp_points']:.2f} pts, mean position {free['mean_pos']:.2f}), but it " + "; ".join(free["missing"]))
    if info["windows"]:
        print(f"\npit windows for {table.loc[0, 'strategy']} (laps within {info['tolerance']} of the best {OBJECTIVES[obj].split(' (')[0]}):")
        for w in info["windows"]:
            print(f"  stop {w['stop']}: lap {w['lap']}  (window {w['first']}-{w['last']})")
    else:
        print("\nthe best plan is not to stop again.")


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", choices=["pre", "live"], help="pre: from the grid before the race; live: from the current race state")
    ap.add_argument("args", nargs="+", help="pre: YEAR RACE DRIVER...   live: YEAR RACE SESSION DRIVER...")
    ap.add_argument("--objective", choices=list(OBJECTIVES), default="points")
    ap.add_argument("--stops", type=int, nargs="+", default=[1, 2], help="numbers of stops to consider")
    ap.add_argument("--step", type=int, help="lap grid of the coarse search (default 2 for one stop, 5 for more)")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--jobs", type=int, help="processes to use (default: up to 16)")
    ap.add_argument("--search-sims", type=int, default=400)
    ap.add_argument("--final-sims", type=int, default=3000)
    ap.add_argument("--tolerance", type=float, help="pit window tolerance in objective units")
    ap.add_argument("--min-survival", type=float, default=0.1,
                    help="drop plans with a stint that fewer than this share of past stints on the compound reached (default 0.1)")
    ap.add_argument("--laps", type=int, help="race distance (pre-race; default: last year's race at the circuit)")
    ap.add_argument("--grid", help="pre-race: comma-separated driver codes in starting order (penalties)")
    ap.add_argument("--sc-react", action="store_true", help="the planned car takes a free stop under a safety car")
    ap.add_argument("--set", action="append", default=[], metavar="HAM.pace=-0.1", help="tweaks applied to every candidate")
    add_tyre_args(ap)
    a = ap.parse_args()
    from f1live.paths import Session
    from play.scenario import parse_tweak
    if a.source == "pre":
        year, race, drivers = a.args[0], a.args[1], a.args[2:]
        scenario = Scenario.pre_race(year, race, laps=a.laps, grid=a.grid.split(",") if a.grid else None, sims=a.search_sims)
    else:
        year, race, session, drivers = a.args[0], a.args[1], a.args[2], a.args[3:]
        scenario = Scenario.live(Session(year, race, session), sims=a.search_sims)
    configure_tyres(scenario, a)
    tweaks, opt = {}, {}
    for text in a.set:
        drv, what, value = parse_tweak(text)
        if drv:
            tweaks.setdefault(drv, {})[what] = value
        else:
            opt[what] = value
    if tweaks:
        opt["tweaks"] = tweaks
    for driver in [d.upper() for d in drivers]:
        try:
            table, info = optimize(scenario, driver, a.objective, tuple(a.stops), a.step, None, a.top, a.search_sims, a.final_sims,
                                   opt=opt, sc_react=a.sc_react, tolerance=a.tolerance, min_survival=a.min_survival, jobs=a.jobs)
        except ValueError as e:
            raise SystemExit(f"error: {e}")
        print_result(table, info, driver)


if __name__ == "__main__":
    main()
