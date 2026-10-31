"""Count on-track overtakes from a recorded events.csv, the way F1 Fantasy scores them (1 point per pass).

Fantasy's rule: a pass counts when one driver legally passes another on track and the driver passed was not entering or in the pit
lane, or suffering a car failure or going unreasonably slow. Two counters:

  count_continuous   needs the live position feed (snapshot rows, only recorded for some 2026 races). Positions are sampled every
                     SAMPLE_S seconds; a change in the order of two cars counts when the new order lasts at least PERSIST_S (the
                     feed flickers between timing loops) and neither car is in, or within PIT_BUFFER_S of, the pit lane. Passes on
                     a car that retires within RETIRE_BUFFER_S are not counted.
  count_by_lap       works on every recorded session (the archive only has positions at the end of each lap): the order of two cars
                     at the end of consecutive laps. Misses passes that are made and repaid within a lap and the pass-and-repass
                     of the first lap (the first recorded lap is lap 2: the first interval is grid -> end of lap 2).
                     Intervals in which either car has an in- or out-lap are ignored, so a pit stop never counts as a pass.

On the one race that has both (2026 United States GP, grid = qualifying order) count_by_lap finds 76 passes and count_continuous
79, with matching per-driver counts, so the lap counter is used for the whole archive without a correction.
"""
import numpy as np
import pandas as pd

SAMPLE_S = 5
PERSIST_S = 15
PIT_BUFFER_S = 30
RETIRE_BUFFER_S = 90
PIT_KINDS = ("in", "out")


def _ahead(P):
    """P: (time, cars) positions -> (time, cars, cars) boolean: car i is ahead of car j."""
    return P[:, :, None] < P[:, None, :]


def count_by_lap(ev, grid=None):
    """Passes made per driver from end-of-lap positions. -> Series indexed by tla (cars that never passed: 0).
    grid: optional Series tla -> starting slot, used as the order before the first recorded lap."""
    laps = ev[(ev["event"] == "lap") & ev["tla"].notna() & ev["position"].notna() & ev["lap_number"].notna()]
    if laps.empty:
        return pd.Series(dtype=int)
    pos = laps.pivot_table(index="lap_number", columns="tla", values="position", aggfunc="last").sort_index()
    kind = laps.pivot_table(index="lap_number", columns="tla", values="lap_kind", aggfunc="last").reindex(pos.index)
    drivers = list(pos.columns)
    last = pos.apply(lambda c: c.last_valid_index())                       # the last lap each car was timed on
    race_end = pos.index.max()
    counts = pd.Series(0, index=drivers, dtype=int)
    prev = grid.reindex(drivers) if grid is not None else None
    prev_lap = None
    for lap in pos.index:
        cur = pos.loc[lap]
        if prev is not None:
            clean = ~kind.loc[lap].isin(PIT_KINDS)
            if prev_lap is not None:
                clean &= ~kind.loc[prev_lap].isin(PIT_KINDS)
            running = (last > lap) | (last == race_end)                     # a car timed for the last time on `lap` is retiring
            ok = (clean & running & cur.notna() & prev.notna()).to_numpy()
            before, after = prev.to_numpy(float), cur.to_numpy(float)
            passed = (before[:, None] > before[None, :]) & (after[:, None] < after[None, :])    # [i, j]: i was behind j, now ahead
            passed &= ok[:, None] & ok[None, :]
            counts += passed.sum(axis=1)
        prev, prev_lap = cur.where(cur.notna(), prev), lap
    return counts


def race_start(ev):
    """Timestamp of the start of a race or sprint from race control, or None."""
    msg = ev["message"].fillna("")
    hit = ev.loc[msg.str.contains(r"RACE START|SPRINT START"), "ts"]
    return hit.iloc[0] if len(hit) else None


def count_continuous(ev):
    """Passes made per driver from the live position feed (snapshot rows), after the race start. -> Series, or None without a feed."""
    start = race_start(ev)
    snap = ev[(ev["event"] == "snapshot") & ev["position"].notna() & ev["tla"].notna()]
    if start is None or snap.empty:
        return None
    snap = snap[snap["ts"] >= start]
    pos = snap.pivot_table(index="ts", columns="tla", values="position", aggfunc="last").ffill()
    in_pit = snap.assign(in_pit=snap["in_pit"].map(lambda v: v is True or v == "True").astype(float)) \
        .pivot_table(index="ts", columns="tla", values="in_pit", aggfunc="last").ffill().fillna(0.0).astype(bool)
    grid = pd.date_range(pos.index[0], pos.index[-1], freq=f"{SAMPLE_S}s")
    pos = pos.reindex(pos.index.union(grid)).ffill().reindex(grid)
    in_pit = in_pit.reindex(in_pit.index.union(grid)).ffill().reindex(grid).fillna(False).astype(bool)
    drivers = list(pos.columns)
    near = in_pit.copy()
    buf = pd.Timedelta(seconds=PIT_BUFFER_S)
    for r in ev[ev["event"].isin(["pit_in", "pit_out"]) & (ev["ts"] >= start) & ev["tla"].isin(drivers)].itertuples():
        near.loc[(near.index >= r.ts - buf) & (near.index <= r.ts + buf), r.tla] = True
    out_buf = pd.Timedelta(seconds=RETIRE_BUFFER_S)
    for r in ev[(ev["event"] == "retired") & ev["tla"].isin(drivers)].itertuples():
        near.loc[(near.index >= r.ts - out_buf) & (near.index <= r.ts), r.tla] = True
    ahead, near_a, n = _ahead(pos.to_numpy()), near.to_numpy(), len(drivers)
    need = max(int(round(PERSIST_S / SAMPLE_S)), 1)
    counts = np.zeros(n, int)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            s = ahead[:, i, j]
            # runs of constant order; a run shorter than `need` is a flicker and is merged into its predecessor
            change = np.flatnonzero(s[1:] != s[:-1]) + 1
            bounds = np.concatenate([[0], change, [len(s)]])
            state = s[0]
            for a, b in zip(bounds[:-1], bounds[1:]):
                if b - a < need and a != 0:
                    continue
                if a != 0 and s[a] != state:
                    if s[a] and not (near_a[a, i] or near_a[a, j]):
                        counts[i] += 1
                    state = s[a]
    return pd.Series(counts, index=drivers)


# ---------------------------------------------------------------------------------------------------------------------------
# Predicting overtakes: how many a driver will make, given where he starts and finishes (for the fantasy Monte Carlo)
#
# Per driver, count ~ NegBin(mean = exp(b . x) * circuit * race), x = places gained (up), places lost (down), log grid slot, and
# not-classified (a car that retires has fewer passes to make). Fitted by Poisson regression on every recorded race / sprint (grid =
# qualifying / sprint qualifying order, as everywhere else in this repo). Two things on top, because the total of a race swings far
# more than counting noise allows (sd 40 % of the mean): `race`, a gamma multiplier shared by all drivers of one scenario, and a
# `circuit` factor, the shrunk average of how that circuit's earlier races compared with the regression (Monaco 0.3, Las Vegas 1.4).
# ---------------------------------------------------------------------------------------------------------------------------
from dataclasses import dataclass, field

from scipy.optimize import minimize

FEATURES = {"race": ["up", "down", "lg", "dnf"], "sprint": ["up", "lg", "dnf"]}
CIRCUIT_SHRINK = 1.0            # a circuit's factor counts as this many races' worth of "no information" (factor 1)
NOT_CLASSIFIED = 0.9            # a car that completes less than this share of the winner's laps is not classified


def design(session, grid, finish, dnf):
    """Feature matrix (…, 1 + n_features) of broadcastable grid / finish / dnf arrays."""
    grid, finish, dnf = np.broadcast_arrays(np.asarray(grid, float), np.asarray(finish, float), np.asarray(dnf, float))
    gain = grid - finish
    cols = {"up": np.clip(gain, 0, None), "down": np.clip(-gain, 0, None), "lg": np.log(grid), "dnf": dnf}
    return np.stack([np.ones_like(grid)] + [cols[c] for c in FEATURES[session]], axis=-1)


def dataset(av, before=None):
    """One row per driver and race / sprint: year, race, session, tla, grid, finish, classified, ot (passes counted by lap).
    av: pre.history.available(); before: only sessions that started earlier than this timestamp."""
    from f1live.events import load
    from pre.history import pace_table
    rows = []
    sessions = av[av["session"].isin(["race", "sprint"])]
    if before is not None:
        sessions = sessions[sessions["start"] < before]
    for r in sessions.itertuples():
        quali = av[(av["year"] == r.year) & (av["race"] == r.race) & (av["session"] == ("quali" if r.session == "race" else "sq"))]
        if quali.empty:
            continue
        table = pace_table(quali["path"].iloc[0])
        ev = load(r.path)
        laps = ev[(ev["event"] == "lap") & ev["tla"].notna()]
        if table.empty or laps.empty:
            continue
        grid = table.reset_index(drop=True).set_index("tla")["rank"]
        done = laps.groupby("tla")["lap_number"].max()
        final = pd.to_numeric(ev[ev["driver"].notna()].groupby("tla").tail(1).set_index("tla")["position"], errors="coerce")
        made = count_by_lap(ev, grid)
        for tla, n in made.items():
            if tla in grid.index and pd.notna(final.get(tla)):
                rows.append(dict(year=r.year, race=r.race, session=r.session, tla=tla, grid=int(grid[tla]), finish=int(final[tla]),
                                 classified=bool(done.get(tla, 0) >= NOT_CLASSIFIED * done.max()), ot=int(n)))
    return pd.DataFrame(rows, columns=["year", "race", "session", "tla", "grid", "finish", "classified", "ot"])


def _poisson_fit(X, y):
    f = lambda b: np.sum(np.exp(X @ b) - y * (X @ b))
    g = lambda b: X.T @ (np.exp(X @ b) - y)
    return minimize(f, np.zeros(X.shape[1]), jac=g, method="BFGS").x


@dataclass
class OvertakeModel:
    coef: dict = field(default_factory=dict)        # session -> coefficients (intercept first, then FEATURES[session])
    alpha: dict = field(default_factory=dict)       # session -> negative-binomial dispersion of a driver's count within a race
    race_cv: dict = field(default_factory=dict)     # session -> spread of the race-wide multiplier
    circuit: dict = field(default_factory=dict)     # session -> {race folder: factor}

    def mean(self, session, grid, finish, dnf, circuit=None):
        """Expected passes per driver (arrays broadcast together), including the circuit factor."""
        mu = np.exp(design(session, grid, finish, dnf) @ self.coef[session])
        return mu * self.circuit.get(session, {}).get(circuit, 1.0)

    def sample(self, rng, session, grid, finish, dnf, circuit=None):
        """Simulated passes, same shape as `finish` (scenarios x cars): a multiplier shared by the scenario's row, then
        negative-binomial noise per driver."""
        mu = self.mean(session, grid, finish, dnf, circuit)
        cv2, alpha = self.race_cv[session] ** 2, self.alpha[session]
        race = rng.gamma(1 / cv2, cv2, size=(mu.shape[0],) + (1,) * (mu.ndim - 1))
        lam = mu * race * rng.gamma(1 / alpha, alpha, size=mu.shape)
        return rng.poisson(lam)


def fit_model(d):
    """OvertakeModel from dataset(). A session type with too little data is left out (sample() then raises KeyError)."""
    m = OvertakeModel()
    for session in FEATURES:
        x = d[d["session"] == session]
        if len(x) < 200:
            continue
        X = design(session, x["grid"], x["finish"], ~x["classified"])
        y = x["ot"].to_numpy(float)
        b = _poisson_fit(X, y)
        mu = np.exp(X @ b)
        x = x.assign(mu=mu, y=y)
        by_race = x.groupby(["year", "race"]).agg(y=("y", "sum"), mu=("mu", "sum"))
        ratio = (by_race["y"] / by_race["mu"]).rename("ratio")
        adj = mu * x.set_index(["year", "race"]).index.map(ratio).to_numpy()
        alpha = max(float(np.sum((y - adj) ** 2 - adj) / np.sum(adj ** 2)), 0.01)
        noise = (x.assign(adj=adj).groupby(["year", "race"])["adj"].apply(lambda a: (a + alpha * a ** 2).sum())
                 / by_race["mu"] ** 2).mean()
        m.coef[session], m.alpha[session] = b, alpha
        m.race_cv[session] = float(np.sqrt(max(ratio.var() - noise, 0.01)))
        r = ratio.reset_index()
        g = r.groupby("race")["ratio"]
        m.circuit[session] = ((g.sum() + CIRCUIT_SHRINK) / (g.count() + CIRCUIT_SHRINK)).to_dict()
    return m
