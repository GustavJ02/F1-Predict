"""A race to experiment on: the state it starts from, and a `run` that simulates it with changes.

    sc = Scenario.pre_race(2026, "Austin")                  # from the grid (before the race)
    sc = Scenario.live(Session("2026", "Austin", "race"))   # from the current state of a race that is being recorded
    base = sc.run()                                         # as predicted
    alt = sc.run(plans={"HAM": Plan.parse("S-14-H-36-M")}, tweaks={"HAM": {"pace": -0.1}})
    compare(base, alt, ["HAM"])                             # what changed for him, with error bars

Every run uses the same random draws (see live.race.projection.simulate), so a difference between two runs is the effect of
the change, not luck. Win / podium / points chances here come from the simulation alone, without the 50/50 blend with the
grid table that `pre.race.predict_result` applies: the blend would shrink exactly the effects you want to see. Expect the
absolute numbers to be somewhat more spread out than that table; read the differences.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from f1live.events import load
from f1live.paths import DATA
from live.race import priors as P
from live.race.battles import latest
from live.race.projection import car_states, simulate
from pre.history import available
from pre.race import predict_result as PR
from pre.race.pace import RaceArchive
from play.tyres import TyreSets

POINTS_TABLE = np.array([25, 18, 15, 12, 10, 8, 6, 4, 2, 1], float)
GLOBAL_OPTS = {"temp": "track temperature in C (tyres wear faster when it is hot)", "sc": "safety car rate multiplier", "pass_shift": "added to the passing logit (negative = harder to pass)",
               "pit_scale": "multiplier on the pit loss", "lap_noise": "s of noise per lap", "dnf": "retirement rate multiplier"}


@dataclass
class Result:
    out: pd.DataFrame           # per driver, sorted by mean finishing position (as live.race.projection.simulate)
    finish: np.ndarray          # runs x cars: finishing position of every car in every run
    tlas: list                  # the cars' codes, in the order of finish's columns
    info: dict = field(default_factory=dict)

    def column(self, tla):
        return self.finish[:, self.tlas.index(tla)]

    def stats(self, tla):
        f = self.column(tla)
        pts = np.where(f <= 10, POINTS_TABLE[np.minimum(f, 10) - 1], 0.0)
        return dict(mean_pos=f.mean(), p_win=(f == 1).mean(), p_podium=(f <= 3).mean(), p_points=(f <= 10).mean(),
                    exp_points=pts.mean(), stops=float(self.out.set_index("tla").loc[tla, "stops_left"]))

    def table(self):
        return pd.DataFrame({t: self.stats(t) for t in self.tlas}).T.sort_values("mean_pos")


def compare(base, alt, drivers=None, top_moved=0):
    """Paired differences alt - base per driver: mean position, win / podium / points chance, expected points, each with its
    standard error over the runs. Positive d_points / negative d_pos is an improvement."""
    drivers = list(drivers or [])
    if top_moved:
        moved = (alt.table()["mean_pos"] - base.table()["mean_pos"]).abs().sort_values(ascending=False)
        drivers += [d for d in moved.index if d not in drivers][:top_moved]
    rows = []
    n = base.finish.shape[0]
    for t in drivers:
        fb, fa = base.column(t), alt.column(t)
        pb, pa = np.where(fb <= 10, POINTS_TABLE[np.minimum(fb, 10) - 1], 0.0), np.where(fa <= 10, POINTS_TABLE[np.minimum(fa, 10) - 1], 0.0)
        r = dict(driver=t)
        for name, x, y in (("pos", fb, fa), ("win", fb == 1, fa == 1), ("podium", fb <= 3, fa <= 3),
                           ("points_chance", fb <= 10, fa <= 10), ("points", pb, pa)):
            d = y.astype(float) - x.astype(float)
            r[f"base_{name}"], r[f"alt_{name}"] = float(np.mean(x)), float(np.mean(y))
            r[f"d_{name}"], r[f"se_{name}"] = float(d.mean()), float(d.std(ddof=1) / np.sqrt(n))
        rows.append(r)
    return pd.DataFrame(rows).set_index("driver")


def parse_tweak(text):
    """'HAM.pace=-0.2' -> ('HAM', 'pace', -0.2);  'sc=2' -> (None, 'sc', 2.0)."""
    key, _, value = text.partition("=")
    if not value:
        raise ValueError(f"'{text}': expected NAME=VALUE, e.g. HAM.pace=-0.2 or sc=2")
    drv, _, what = key.rpartition(".")
    return (drv.upper() or None), what.lower(), float(value)


class Scenario:
    """The state a race simulation starts from, plus the settings shared by every run of it."""

    def __init__(self, cars, meta, track, priors, circuit, comp0=None, opt=None, sims=1000, seed=0, label="",
                 rel=None, order=None, needs_stop=None, tyres=None):
        self.cars, self.meta, self.track, self.priors, self.circuit = cars, meta, track, priors, circuit
        self.comp0, self.opt, self.sims, self.seed, self.label = comp0, dict(opt or {}), sims, seed, label
        self.rel, self.order = rel, order                       # pre-race only: lets `run(grid=...)` rebuild the grid
        self.tyres = tyres                                      # play.tyres.TyreSets: which sets each driver has left (or None)
        self.total, self.done = meta[2], meta[1]

    @property
    def drivers(self):
        return list(self.cars["tla"])

    # -- construction -----------------------------------------------------------------------------------------
    @classmethod
    def pre_race(cls, year, race, data_dir=DATA, laps=None, grid=None, sims=1000, seed=0, priors=None):
        av = available(Path(data_dir))
        priors = priors or P.load_priors()
        archive = RaceArchive(av[av["session"] == "race"].reset_index(drop=True), priors)
        w, why = PR.prepare_weekend(archive, av, priors, int(year), race, grid, laps)
        if w is None:
            raise ValueError(why)
        cars = PR.grid_cars(w["order"], w["rel"])
        opt = dict(pace_sd=PR.pace_sd_for(w["sd"], w["order"]), gap_noise=PR.GAP_NOISE, temp=PR.weekend_track_temp(av, int(year), race))
        comp0 = PR.start_compounds(priors, sims, len(cars), seed)
        return cls(cars, (w["tlap"], 0, w["total"]), "AllClear", priors, w["circuit"], comp0, opt, sims, seed,
                   f"{year} {race} (pre-race, {w['total']} laps)", rel=w["rel"], order=w["order"],
                   tyres=TyreSets.from_weekend(Path(data_dir), year, race))

    @classmethod
    def live(cls, session, history=None, sims=1000, seed=0, priors=None):
        """From the events recorded so far in `session` (a f1live.paths.Session), or from `history` if you pass the rows."""
        priors = priors or P.load_priors()
        history = load(session.events) if history is None else history
        cars, meta = car_states(history, priors)
        if cars is None:
            raise ValueError(meta)
        track = history["track_status"].dropna().iloc[-1] if history["track_status"].notna().any() else "AllClear"
        tyres = TyreSets.from_weekend(session.data_dir, session.year, session.race, history=history)
        sc = cls(cars, meta, track, priors, P.circuit_of(session.year, session.race), None, {"temp": cars.attrs.get("track_temp")}, sims, seed,
                 f"{session} (live, lap {meta[1]}/{meta[2]})", tyres=tyres)
        sc.needs_stop = dict(zip(cars["tla"], cars["needs_stop"]))
        return sc

    # -- running ----------------------------------------------------------------------------------------------
    def _check_plans(self, plans):
        for tla, plan in plans.items():
            if tla not in self.drivers:
                raise ValueError(f"{tla} is not in the race ({', '.join(self.drivers)})")
            plan.validate(self.total, self.done)

    def run(self, plans=None, tweaks=None, grid=None, sc_react=False, **opt):
        """Simulate. plans {tla: Plan}; tweaks {tla: {pace, wear, pit}}; grid {tla: new grid slot} (pre-race only);
        global options (see GLOBAL_OPTS): sc, pass_shift, pit_scale, lap_noise, dnf. -> Result."""
        plans, tweaks = dict(plans or {}), dict(tweaks or {})
        self._check_plans(plans)
        cars, comp0 = self.cars, self.comp0
        if grid:
            if self.order is None:
                raise ValueError("grid changes only make sense before the race")
            order = [d for d in self.order if d not in grid]
            for tla, slot in sorted(grid.items(), key=lambda kv: kv[1]):
                order.insert(min(slot - 1, len(order)), tla)
            cars = PR.grid_cars(order, self.rel)
            comp0 = comp0[:, [self.order.index(t) for t in order]]
        if comp0 is not None:
            comp0 = comp0.copy()
            for tla, plan in plans.items():
                comp0[:, list(cars["tla"]).index(tla)] = P.COMPOUNDS.index(plan.start)
        o = dict(self.opt)
        for key in ("sc", "dnf"):
            if key in opt:
                o[key] = float(opt.pop(key))
        o.update(opt)
        sim_opt = dict(o, plans={t: _SimPlan(p.stops) for t, p in plans.items()}, tweaks=tweaks, sc_react=sc_react)
        sc_scale, dnf_scale = sim_opt.pop("sc", 1.0), sim_opt.pop("dnf", 1.0)
        pri = self.priors
        if sc_scale != 1.0:
            pri = {**pri, "safety": {**pri["safety"], "sc_rate": pri["safety"]["sc_rate"] * sc_scale,
                                     "vsc_rate": pri["safety"]["vsc_rate"] * sc_scale,
                                     "circuits": {c: {k: v * sc_scale for k, v in d.items()} for c, d in pri["safety"]["circuits"].items()}}}
        if dnf_scale != 1.0:
            sim_opt["dnf_scale"] = dnf_scale
        if comp0 is not None:
            sim_opt["comp0"] = comp0
        out, info = simulate(cars, self.meta, self.track, pri, self.circuit, self.sims, self.seed, **sim_opt)
        return Result(out, info.pop("finish"), info.pop("tlas"), info)


class _SimPlan:
    def __init__(self, stops):
        self.stops = stops

