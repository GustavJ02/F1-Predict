"""Try tyre strategies and tweaks for one or more drivers and see what they do to the finishing positions.

Everything is compared with the race as predicted, using the same random draws, so the differences are the effect of your
change (with their standard errors), not luck. Chances here come from the simulation alone (no blend with the grid table),
so read the differences more than the absolute numbers.

    python -m play.playground pre 2026 Austin --plan HAM=M-25-H
    python -m play.playground pre 2026 Austin --plan HAM=S-14-H --plan NOR=H-30-M --set HAM.pace=-0.1 --show VER
    python -m play.playground pre 2026 Austin --try HAM=S-24-H --try HAM=M-25-H --try HAM=S-12-M-34-H   # one by one
    python -m play.playground pre 2026 Austin --move HAM=2 --set sc=2 --set pass_shift=-1
    python -m play.playground pre 2026 Austin --try HAM=S-24-H --try HAM=M-25-H --set temp=55   # on a very hot track
    python -m play.playground live 2026 Austin race --plan HAM=M-38-H --sc-react
    python -m play.playground pre 2026 Austin --try ANT=S-24-H --ban ANT=S     # a plan the driver has no tyres for gets a warning    # from the current state of the race

Strategies: compound-lap-compound ..., e.g. M-25-H (mediums, stop lap 25, hards) or S-10-M-30-H. In a live race the first
compound is the one on the car now and the stop laps must be in the future.

--set NAME=VALUE changes:
    DRV.pace=-0.1    seconds per lap for that driver (negative = faster)       DRV.wear=0.8   tyre wear x0.8 (better tyre care)
    DRV.pit=1.5      extra seconds in every one of that driver's stops          sc=2           safety car / VSC rate x2
    temp=50          track temperature in C (tyres wear faster when hot)
    pass_shift=-1    passing logit shift (negative = harder to pass)            pit_scale=1.2  every pit stop x1.2
    dnf=2            retirement rate x2                                        lap_noise=0.3  seconds of noise per lap
--move DRV=N (before the race) puts the driver on grid slot N; --sc-react lets planned cars take a free stop under a safety car.

Tyres: each driver has the weekend's allocation minus the fresh sets used so far (practice, qualifying, sprint; the race so far when
live). A plan that needs sets he doesn't have is flagged "(!)" (it is still simulated, as if he did). Say what he really has with
--sets HAM=S0,M2,H1, rule compounds out with --ban HAM=S, change the allocation with --allocation H2,M3,S8, count used sets as fresh
with --allow-used S1, or switch it off with --tyres none.
"""
import argparse
import warnings

import pandas as pd
from f1live.paths import Session
from play.scenario import GLOBAL_OPTS, Scenario, compare, parse_tweak
from play.strategy import Plan
from play.tyres import add_tyre_args, configure_tyres


def parse_plan(text):
    drv, _, strat = text.partition("=")
    if not strat:
        raise ValueError(f"'{text}': expected DRIVER=STRATEGY, e.g. HAM=M-25-H")
    return drv.upper(), Plan.parse(strat)


def describe(drv, plan, total):
    legal = "" if plan.legal() else "   (!) one compound only: illegal in a dry race"
    stints = []
    prev = 0
    for c, lap in zip(plan.compounds, plan.laps + (total,)):
        stints.append(f"{c.lower()} {lap - prev} laps")
        prev = lap
    return f"{drv}: {plan}  =  " + ", ".join(stints) + legal


def format_comparison(cmp, label):
    print(f"\n{label}")
    print(f" {'':4}{'mean pos':>16} {'win %':>14} {'podium %':>14} {'points chance %':>16} {'E[points]':>20}")
    for t, r in cmp.iterrows():
        def cell(x, y, d, se, scale=1.0, fmt="{:.1f}"):
            return f"{fmt.format(x * scale)}->{fmt.format(y * scale)} ({d * scale:+.1f})".rjust(14)
        print(f" {t:<4}{r['base_pos']:>6.2f}->{r['alt_pos']:<5.2f}({r['d_pos']:+.2f}) "
              f"{cell(r['base_win'], r['alt_win'], r['d_win'], r['se_win'], 100)} {cell(r['base_podium'], r['alt_podium'], r['d_podium'], r['se_podium'], 100)} "
              f"{cell(r['base_points_chance'], r['alt_points_chance'], r['d_points_chance'], r['se_points_chance'], 100):>16} "
              f"{r['base_points']:>6.2f}->{r['alt_points']:<6.2f}({r['d_points']:+.2f} +-{r['se_points']:.2f})")


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", choices=["pre", "live"])
    ap.add_argument("args", nargs="+", help="pre: YEAR RACE    live: YEAR RACE SESSION")
    ap.add_argument("--plan", action="append", default=[], metavar="HAM=M-25-H", help="strategy for a driver (all together)")
    ap.add_argument("--try", dest="tries", action="append", default=[], metavar="HAM=S-24-H", help="strategies tried one by one")
    ap.add_argument("--set", action="append", default=[], metavar="HAM.pace=-0.1")
    ap.add_argument("--move", action="append", default=[], metavar="HAM=2", help="grid slot (before the race)")
    ap.add_argument("--show", nargs="+", default=[], help="more drivers to show")
    ap.add_argument("--top-moved", type=int, default=3, help="also show the drivers whose finish changed most")
    ap.add_argument("--sims", type=int, default=3000)
    ap.add_argument("--laps", type=int, help="race distance (pre-race; default: last year's race at the circuit)")
    ap.add_argument("--grid", help="pre-race: comma-separated driver codes in starting order (penalties)")
    ap.add_argument("--sc-react", action="store_true")
    add_tyre_args(ap)
    a = ap.parse_args()

    if a.source == "pre":
        scenario = Scenario.pre_race(a.args[0], a.args[1], laps=a.laps, grid=a.grid.split(",") if a.grid else None, sims=a.sims)
    else:
        scenario = Scenario.live(Session(a.args[0], a.args[1], a.args[2]), sims=a.sims)
    print(scenario.label)
    configure_tyres(scenario, a)

    def tyre_warnings(drv, plan):
        problems = scenario.tyres.check(drv, plan) if scenario.tyres else []
        return [f"(!) {drv}: {plan} " + "; ".join(problems) + f"   [{scenario.tyres.describe(drv)}]"] if problems else []

    plans = dict(parse_plan(t) for t in a.plan)
    tweaks, opts = {}, {}
    for text in a.set:
        drv, what, value = parse_tweak(text)
        if drv:
            if what not in ("pace", "wear", "pit"):
                raise SystemExit(f"unknown driver tweak '{what}' (pace, wear, pit)")
            tweaks.setdefault(drv, {})[what] = value
        elif what in GLOBAL_OPTS:
            opts[what] = value
        else:
            raise SystemExit(f"unknown setting '{what}'. Global settings: " + ", ".join(GLOBAL_OPTS))
    moves = {d.upper(): int(n) for d, n in (m.split("=") for m in a.move)}
    base = scenario.run(sc_react=a.sc_react)

    if plans or tweaks or opts or moves:
        for drv, plan in plans.items():
            print(describe(drv, plan, scenario.total))
            for w in tyre_warnings(drv, plan):
                print(w)
        for drv, tw in tweaks.items():
            print(f"{drv}: " + ", ".join(f"{k} {v:+g}" for k, v in tw.items()))
        if opts or moves:
            print("race: " + ", ".join([f"{k}={v:g}" for k, v in opts.items()] + [f"{d} to grid slot {n}" for d, n in moves.items()]))
        alt = scenario.run(plans=plans, tweaks=tweaks, grid=moves, sc_react=a.sc_react, **opts)
        shown = list(dict.fromkeys(list(plans) + list(tweaks) + list(moves) + [d.upper() for d in a.show]))
        format_comparison(compare(base, alt, shown, a.top_moved), "your changes vs the race as predicted:")

    rows = []
    for text in a.tries:
        drv, plan = parse_plan(text)
        alt = scenario.run(plans={drv: plan}, sc_react=a.sc_react)
        c = compare(base, alt, [drv]).iloc[0]
        rows.append(dict(driver=drv, strategy=str(plan), mean_pos=c["alt_pos"], d_pos=c["d_pos"], se_pos=c["se_pos"],
                         win=c["alt_win"] * 100, podium=c["alt_podium"] * 100, points=c["alt_points"], d_points=c["d_points"],
                         se_points=c["se_points"], legal="illegal" if not plan.legal() else
                         ("no tyres!" if tyre_warnings(drv, plan) else "")))
    if rows:
        print("\nstrategies tried one by one (vs the race as predicted):")
        print(f" {'':4}{'strategy':<16}{'mean pos':>9}{'':>16}{'win %':>7}{'podium %':>9}{'E[points]':>10}{'':>20}")
        for r in rows:
            print(f" {r['driver']:<4}{r['strategy']:<16}{r['mean_pos']:>9.2f}  ({r['d_pos']:+.2f} +-{r['se_pos']:.2f}){r['win']:>7.1f}{r['podium']:>9.1f}"
                  f"{r['points']:>10.2f}  ({r['d_points']:+.2f} +-{r['se_points']:.2f})  {r['legal']}")
    if not (plans or tweaks or opts or moves or rows):
        print("\nnothing to try: give --plan, --try, --set or --move (see --help). The race as predicted:")
        print(base.table().head(10).round(3).to_string())


if __name__ == "__main__":
    main()
