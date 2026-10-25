"""The whole field on its best strategies: every driver's optimal plan, and the finishing order that results.

`optimize` finds the best plan for ONE driver while the others keep the usual (stochastic) stop plans. Here every driver gets one:
going down the order, each driver's best plan is searched given the plans the others have by then, and the round is repeated
until nobody can gain more than a small tolerance by changing (a Nash equilibrium of the simulation, or close to it; the plans
interact, because a plan that works when the others do something else is worth less once they all do it). Then the race is
simulated with every driver on his plan.

In a race under way the search starts from what each car is doing now (compound, tyre age, lap, the sets left): run it again
every few laps and the plans follow the race.

What this is and is not: it answers "what happens if everybody plays this model's optimum", not "what will happen": teams are not
optimal, the model's tyre and passing assumptions are only approximate, and the other cars' reactions are only what the
equilibrium gives. Whether it predicts a race better than the usual stop plans is tested in play.backtest_field.

Usage (from the repo root):
    python -m play.field pre 2026 Austin
    python -m play.field live 2026 Austin race --passes 2
    python -m play.field live 2026 Austin race --loop --every 3      # keep strategy.csv up to date during the race (see below)

--loop re-runs the search every --every laps of the leader, starting from the plans it had (so it is quick), and writes the
result to data/<year>/<race>/<session>/strategy.csv, which the dashboard shows when you press `s` (the projected finish with
every driver on his best plan, and each driver's next stop). It uses --jobs processes (default 4): it shares the machine with the
recording, so keep that modest.

Backtest (play.backtest_field, the 16 finished 2026 races at half distance): projecting with everybody on his optimal plan is
no more accurate than the usual stop plans (rank error 1.59 against 1.62 places, podium Brier 0.044 against 0.043, points Brier
0.096 against 0.089) and more extreme, so it is a scenario to look at, not the default projection.
"""
import argparse
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from f1live.paths import DATA
from live.race.priors import COMPOUNDS
from play.optimize import OBJECTIVES, Search, candidate_plans, objective_value, score_many, _with_sims
from play.scenario import Scenario, compare
from play.strategy import Plan


def best_response(scenario, driver, objective="points", stops=(1, 2), step=None, sims=250, fixed=None, tyres="auto",
                  min_survival=0.1, jobs=1, opt=None, current=None, refine=3):
    """The driver's best plan when the other cars follow `fixed` (the rest use the usual plans).
    -> (plan, its value, the value of `current` under the same conditions or None); plan is None if no plan fits his tyres."""
    tyres = scenario.tyres if isinstance(tyres, str) and tyres == "auto" else tyres
    plans, _ = candidate_plans(scenario, driver, stops, step, None, min_survival)
    if tyres is not None:
        plans = [p for p in plans if not tyres.check(driver, p)]
    if not plans:
        return None, None, None
    sc = _with_sims(scenario, sims) if sims != scenario.sims else scenario
    search = Search(sc, driver, objective, opt or {}, False, fixed)
    values = score_many(search, plans, None, jobs)
    scored = dict(zip((str(p) for p in plans), zip(values, plans)))
    best_value, best = max(scored.values(), key=lambda x: x[0])
    for value, plan in sorted(scored.values(), key=lambda x: -x[0])[:refine]:        # nudge the stop laps of the best few
        cur, v = plan, value
        improved = True
        while improved and cur.laps:
            improved = False
            for i in range(len(cur.laps)):
                for d in (-2, -1, 1, 2):
                    laps = list(cur.laps)
                    laps[i] += d
                    cand = Plan(cur.compounds, tuple(laps))
                    try:
                        cand.validate(scenario.total, scenario.done)
                    except ValueError:
                        continue
                    if any(b - a < 3 for a, b in zip((scenario.done,) + cand.laps, cand.laps)):
                        continue
                    if str(cand) not in scored:
                        scored[str(cand)] = (search.score(cand), cand)
                    if scored[str(cand)][0] > v + 1e-9:
                        cur, v, improved = cand, scored[str(cand)][0], True
        if v > best_value:
            best_value, best = v, cur
    cur_value = None
    if current is not None:
        cur_value = scored[str(current)][0] if str(current) in scored else search.score(current)
    return best, best_value, cur_value


def optimal_field(scenario, objective="points", stops=(1, 2), step=None, sims=250, passes=3, tol=None, jobs=None, opt=None,
                  verbose=True, tyres="auto", start=None):
    """-> (plans {tla: Plan}, log [per pass: (drivers that changed his plan)]). `start` = plans to begin from."""
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective '{objective}' (choose from {', '.join(OBJECTIVES)})")
    tol = {"points": 0.1, "win": 0.01, "podium": 0.01, "position": 0.08}[objective] if tol is None else tol
    jobs = jobs or min(os.cpu_count() or 1, 16)
    order = list(scenario.cars.sort_values("position")["tla"])
    plans, log = dict(start or {}), []
    for n in range(1, passes + 1):
        changed = []
        t0 = time.time()
        for driver in order:
            fixed = {k: v for k, v in plans.items() if k != driver}
            plan, value, cur = best_response(scenario, driver, objective, stops, step, sims, fixed, tyres, jobs=jobs, opt=opt,
                                             current=plans.get(driver))
            if plan is None:
                continue
            if driver not in plans or (cur is not None and value > cur + tol):
                if driver in plans and str(plans[driver]) == str(plan):
                    continue
                plans[driver] = plan
                changed.append(driver)
        log.append(changed)
        if verbose:
            print(f"pass {n}: {len(changed)} plans changed ({time.time() - t0:.0f} s)" + (f": {', '.join(changed)}" if changed else ""),
                  flush=True)
        if not changed:
            break
    return plans, log


def field_table(scenario, plans, sims=3000, opt=None):
    """DataFrame per driver comparing the usual stop plans with the optimal ones: projected mean position, win and podium chance,
    expected points (all the cars on the usual plans against all of them on theirs), plus the plan and its stop laps."""
    sc = _with_sims(scenario, sims) if sims != scenario.sims else scenario
    usual, best = sc.run(**(opt or {})), sc.run(plans=plans, **(opt or {}))
    rows = []
    for t in scenario.cars.sort_values("position")["tla"]:
        a, b = usual.stats(t), best.stats(t)
        rows.append(dict(driver=t, now=int(scenario.cars.set_index("tla").loc[t, "position"]),
                         plan=str(plans[t]) if t in plans else "-", usual_pos=a["mean_pos"], best_pos=b["mean_pos"],
                         usual_win=a["p_win"], best_win=b["p_win"], usual_podium=a["p_podium"], best_podium=b["p_podium"],
                         usual_pts=a["exp_points"], best_pts=b["exp_points"]))
    return pd.DataFrame(rows).set_index("driver"), usual, best


STRATEGY_COLUMNS = ["ts", "race_lap", "driver", "tla", "now", "plan", "next_lap", "next_compound", "usual_pos", "best_pos",
                    "usual_win", "best_win", "best_podium", "best_pts"]


def strategy_rows(scenario, plans, table, ts):
    """The strategy.csv rows: one per driver with his plan, the lap and compound of his next stop (empty if none) and what the
    two worlds give him."""
    cars = scenario.cars.set_index("tla")
    rows = []
    for t, r in table.iterrows():
        plan = plans.get(t)
        nxt = (plan.laps[0], plan.compounds[1]) if plan is not None and plan.laps else ("", "")
        rows.append(dict(ts=ts, race_lap=scenario.done, driver=cars.loc[t, "driver"], tla=t, now=r["now"], plan=r["plan"],
                         next_lap=nxt[0], next_compound=nxt[1], usual_pos=r["usual_pos"], best_pos=r["best_pos"],
                         usual_win=r["usual_win"], best_win=r["best_win"], best_podium=r["best_podium"],
                         best_pts=r["best_pts"]))
    return pd.DataFrame(rows, columns=STRATEGY_COLUMNS)


def warm_start(plans, scenario):
    """The plans of the last round that still make sense: every stop still to come, and the first compound is what the car is on."""
    cars = scenario.cars.set_index("tla")
    out = {}
    for t, p in plans.items():
        if t in cars.index and p.compounds[0] == COMPOUNDS[int(cars.loc[t, "comp"])] and all(lap > scenario.done for lap in p.laps):
            out[t] = p
    return out


def live_loop(session, every=3, passes=2, search_sims=250, sims=1500, jobs=4, objective="points", args=None, poll=5.0):
    """Keep strategy.csv up to date while the race is being recorded. Runs until the last lap."""
    from f1live.events import load as load_events
    from play.tyres import configure_tyres
    path = session.dir / "strategy.csv"
    plans, last = {}, None
    while True:
        try:
            history = load_events(session.events)
            sc = Scenario.live(session, history=history, sims=search_sims)
            if args is not None:
                configure_tyres(sc, args)
            lap, total = sc.done, sc.total
            if lap >= total - 1:
                print("last lap: done")
                return
            if last is None or lap - last >= every:
                t0 = time.time()
                plans, log = optimal_field(sc, objective, (1, 2), None, search_sims, passes, jobs=jobs, verbose=False,
                                           start=warm_start(plans, sc))
                table, _, _ = field_table(sc, plans, sims)
                tmp = path.with_suffix(".tmp")
                strategy_rows(sc, plans, table, history["ts"].max()).to_csv(tmp, index=False)
                os.replace(tmp, path)                                  # the dashboard never sees half a file
                last = lap
                print(f"lap {lap}/{total}: {len(plans)} plans, {sum(len(x) for x in log)} changes, {time.time() - t0:.0f} s -> {path}", flush=True)
        except (FileNotFoundError, ValueError, KeyError, pd.errors.EmptyDataError) as e:
            print(f"waiting ({type(e).__name__}: {e})", flush=True)
        time.sleep(poll)


def print_field(table, scenario):
    print(f"\n{scenario.label}: the finishing order with every driver on his best plan, against the usual stop plans\n")
    print(f" {'':3}{'now':>3} {'plan':<16} {'mean pos':>16} {'win %':>14} {'podium %':>14} {'E[points]':>16}")
    for t, r in table.iterrows():
        print(f" {t:<3}{r['now']:>3} {r['plan']:<16} {r['usual_pos']:>6.2f} -> {r['best_pos']:<6.2f} "
              f"{r['usual_win'] * 100:>5.1f} -> {r['best_win'] * 100:<5.1f} {r['usual_podium'] * 100:>5.1f} -> {r['best_podium'] * 100:<5.1f} "
              f"{r['usual_pts']:>6.2f} -> {r['best_pts']:<6.2f}")


def main():
    warnings.filterwarnings("ignore")
    from f1live.paths import Session
    from play.tyres import add_tyre_args, configure_tyres
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", choices=["pre", "live"])
    ap.add_argument("args", nargs="+", help="pre: YEAR RACE    live: YEAR RACE SESSION")
    ap.add_argument("--objective", choices=list(OBJECTIVES), default="points")
    ap.add_argument("--stops", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--step", type=int)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--search-sims", type=int, default=250)
    ap.add_argument("--sims", type=int, default=3000, help="runs for the final comparison")
    ap.add_argument("--jobs", type=int)
    ap.add_argument("--laps", type=int)
    ap.add_argument("--grid")
    ap.add_argument("--data-dir", default=DATA, type=Path)
    ap.add_argument("--loop", action="store_true", help="live: keep strategy.csv up to date until the race ends")
    ap.add_argument("--every", type=int, default=3, help="with --loop: recompute every N laps of the leader")
    add_tyre_args(ap)
    a = ap.parse_args()
    if a.loop:
        if a.source != "live":
            raise SystemExit("--loop is for a race that is being recorded: python -m play.field live YEAR RACE SESSION --loop")
        live_loop(Session(a.args[0], a.args[1], a.args[2], a.data_dir), a.every, a.passes, a.search_sims, min(a.sims, 1500), a.jobs or 4,
                  a.objective, a)
        return
    if a.source == "pre":
        scenario = Scenario.pre_race(a.args[0], a.args[1], data_dir=a.data_dir, laps=a.laps, grid=a.grid.split(",") if a.grid else None,
                                     sims=a.search_sims)
    else:
        scenario = Scenario.live(Session(a.args[0], a.args[1], a.args[2], a.data_dir), sims=a.search_sims)
    configure_tyres(scenario, a)
    plans, log = optimal_field(scenario, a.objective, tuple(a.stops), a.step, a.search_sims, a.passes, jobs=a.jobs)
    table, _, _ = field_table(scenario, plans, a.sims)
    print_field(table, scenario)
    print("\nplans settled." if log and not log[-1] else f"\nstopped after {len(log)} passes; the last pass changed {len(log[-1])} plans.")


if __name__ == "__main__":
    main()
