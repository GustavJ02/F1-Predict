"""Predict a race before it starts: a distribution of finishing positions for every driver.

Pace: each driver's expected race pace relative to the field = a regression on
    rating     their recency-weighted race pace over this season's earlier races (pre.race.pace.ratings), and
    quali gap  how far off the field's median they were in this weekend's qualifying (percent of a lap),
fitted on every earlier race we have (rating only exists for races with earlier races in the same season; without one the
regression uses the qualifying gap alone). The spread of its errors becomes the uncertainty of each driver's pace.

Race: the grid is the qualifying result (override with --grid for penalties). The Monte Carlo of live.race.projection runs the
whole race from the lights: starting gaps with a random spread, starting compound drawn from what cars usually start on, the
circuit's pit loss, stint lengths, passing difficulty and safety car rate.

The simulation alone makes the front too random (the pole-sitter wins about two thirds of races, the simulation gave him a
quarter): on the day track position is stickier than a lap-time model sees. So each driver's finishing distribution is
blended 50/50 with the table of where cars starting from that slot really finished in the earlier races (smoothed), and
rebalanced so that every position is taken by exactly one driver. The 50/50 weight was the best of the backtest.

Usage (from the repo root; the race folder is the data folder, e.g. Austin or United_States):
    python -m pre.race.predict_result 2026 Austin --laps 56
    python -m pre.race.predict_result 2026 Austin --grid VER,NOR,LEC,...      # actual grid order if there are penalties
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from f1live.events import load
from f1live.paths import DATA
from live.race import priors as P
from live.race.pace import clean_laps, lap_rows
from live.race.projection import simulate
from pre.history import available, pace_table, race_key
from pre.race.pace import RaceArchive, ratings

GRID_BLEND = 0.5         # weight of the empirical "starts P_g, finishes P_k" table against the simulation
GRID_SIGMA = 1.2         # smoothing of that table across finishing positions
START_GAP = 0.3          # s between grid slots
GAP_NOISE = 1.0          # s, random spread of the starting gaps (reaction at the start, first-lap shuffles)
PACE_SD_SCALE = 0.5      # the regression's residual is mostly noise in the measured pace, not in the true pace
MIN_PACE_SD = 0.10       # s
SIMS = 1000


def quali_table(av, year, race):
    """DataFrame indexed by tla: team, gap_pct (to pole), qrel (gap_pct minus the field median), rank. None if no quali."""
    row = av[(av["year"] == year) & (av["race"] == race_key(race)) & (av["session"] == "quali")]
    if row.empty:
        return None
    t = pace_table(row["path"].iloc[0])
    if t.empty:
        return None
    t = t.reset_index(drop=True).set_index("tla")
    t["qrel"] = t["gap_pct"] - t["gap_pct"].median()
    t = t[["team", "gap_pct", "qrel", "rank"]]
    # drivers who took part but have no valid time (all laps deleted, crashed) start from the back
    ev = load(row["path"].iloc[0])
    took_part = ev[ev["event"] == "lap"].dropna(subset=["tla"]).groupby("tla")["team"].last()
    for i, (tla, team) in enumerate((x for x in took_part.items() if x[0] not in t.index), 1):
        t.loc[tla] = [team, t["gap_pct"].max() + 1.0, t["qrel"].max() + 1.0, len(t) + 1]
    return t


def training_rows(archive, av, before, priors_unused=None):
    """One row per (earlier race, driver): rating (or NaN), qrel, and the pace that was actually shown (rel)."""
    rows = []
    for r in archive.races[archive.races["start"] < before].itertuples():
        q = quali_table(av, r.year, r.race)
        if q is None:
            continue
        pace = archive.pace(r.path)
        rt = ratings(archive, r.year, r.start)
        for tla, x in pace.iterrows():
            if tla in q.index:
                rows.append(dict(race=f"{r.year}/{r.race}", tla=tla, rel=x["rel"], qrel=q.loc[tla, "qrel"],
                                 rating=rt["rating"].get(tla, np.nan) if len(rt) else np.nan))
    return pd.DataFrame(rows, columns=["race", "tla", "rel", "qrel", "rating"])


def fit_pace_model(rows, ridge=1.0):
    """{'full': (coef, sd) on [1, rating, qrel], 'q': (coef, sd) on [1, qrel]} by ridge regression."""
    out = {}
    for name, cols in (("full", ["rating", "qrel"]), ("q", ["qrel"])):
        d = rows.dropna(subset=cols)
        if len(d) < 20:
            continue
        X = np.c_[np.ones(len(d)), d[cols].to_numpy()]
        pen = ridge * np.eye(X.shape[1]); pen[0, 0] = 0
        beta = np.linalg.solve(X.T @ X + pen, X.T @ d["rel"].to_numpy())
        resid = d["rel"].to_numpy() - X @ beta
        out[name] = (dict(zip(["const"] + cols, beta)), float(np.sqrt(np.mean(resid ** 2))))
    return out


def predict_pace(model, rt, q):
    """-> (Series tla -> expected pace vs the field in s/lap, sd). Uses the rating where the driver has one."""
    rel = pd.Series(index=q.index, dtype=float)
    sd = {}
    for tla in q.index:
        has = len(rt) and tla in rt.index and not pd.isna(rt.loc[tla, "rating"])
        name = "full" if has and "full" in model else "q"
        if name not in model:
            rel[tla], sd[tla] = np.nan, np.nan
            continue
        coef, s = model[name]
        rel[tla] = coef["const"] + coef["qrel"] * q.loc[tla, "qrel"] + (coef["rating"] * rt.loc[tla, "rating"] if name == "full" else 0.0)
        sd[tla] = s
    return rel, pd.Series(sd)


def race_laps_and_pace(av, year, race, laps=None):
    """(total laps, typical lap time) from the same circuit's last race we have, else the given/default values."""
    prev = av[(av["session"] == "race") & (av["race"] == race_key(race)) & (av["year"] < year)].sort_values("start")
    total, tlap = laps, 95.0
    if len(prev):
        h = load(prev["path"].iloc[-1])
        total = total or int(h["total_laps"].max())
        c = clean_laps(lap_rows(h))
        if len(c):
            tlap = float(c["lap_time_s"].median())
    return int(total or 55), tlap


def simulate_from_grid(grid, rel, sd, priors, circuit, total, tlap, sims=SIMS, seed=0, **opt):
    """grid: list of tla in starting order; rel: Series tla -> pace (s/lap vs the field); sd: Series of pace uncertainty."""
    cars = pd.DataFrame(dict(driver=grid, tla=grid, position=np.arange(1, len(grid) + 1)))
    cars["gap"] = (cars["position"] - 1) * START_GAP
    cars["comp"], cars["age"] = 1, 0.0
    cars["base"] = rel.reindex(grid).fillna(rel.median()).to_numpy()
    cars["known"], cars["needs_stop"] = True, True
    rng = np.random.default_rng(seed + 1)
    shares = np.array([priors["start"][c] for c in P.COMPOUNDS])
    comp0 = rng.choice(len(P.COMPOUNDS), size=(sims, len(grid)), p=shares / shares.sum())
    pace_sd = float(max(sd.reindex(grid).fillna(sd.max()).max() * PACE_SD_SCALE, MIN_PACE_SD))
    defaults = dict(pace_sd=pace_sd, gap_noise=GAP_NOISE, comp0=comp0)          # (an extra first-lap safety car risk backtested worse)
    return simulate(cars, (tlap, 0, total), "AllClear", priors, circuit, sims, seed, **{**defaults, **opt})


def predict_weekend(archive, av, priors, year, race, grid=None, laps=None, sims=SIMS, seed=0, before=None, use_rating=True,
                    blend=GRID_BLEND, **opt):
    """-> (DataFrame per driver with pos_1..pos_N, info) or (None, reason). `before`: only use races started before this."""
    q = quali_table(av, year, race)
    if q is None:
        return None, f"no qualifying data for {year} {race}"
    if before is None:
        done = av[(av["year"] == year) & (av["race"] == race_key(race)) & (av["session"] == "race")]
        before = done["start"].iloc[0] if len(done) else pd.Timestamp("2100-01-01T00:00:00Z").isoformat()
    rows = training_rows(archive, av, before)
    model = fit_pace_model(rows)
    if not use_rating:
        model.pop("full", None)
    if "q" not in model:
        return None, "not enough earlier races to fit the pace model"
    rt = ratings(archive, year, before)
    rel, sd = predict_pace(model, rt, q)
    order = grid or list(q.sort_values("rank").index)
    order = [d for d in order if d in rel.index]
    total, tlap = race_laps_and_pace(av, year, race, laps)
    out, info = simulate_from_grid(order, rel, sd, priors, P.circuit_of(year, race), total, tlap, sims, seed, **opt)
    if blend:
        out = blend_with_grid(out, order, finishing_table(archive, av, before), blend)
    out["grid"] = out["tla"].map({d: i + 1 for i, d in enumerate(order)})
    out["rel_pace"] = out["tla"].map(rel)
    info.update(total_laps=total, pace_model=model, n_train=len(rows), rated=int(rt["rating"].notna().sum()) if len(rt) else 0)
    return out, info


def finishing_table(archive, av, before, n=22, sigma=GRID_SIGMA):
    """Row g = distribution of the finishing position of a car that started on grid slot g+1, from races before `before`."""
    C = np.full((n, n), 0.02)
    for r in archive.races[archive.races["start"] < before].itertuples():
        q = quali_table(av, r.year, r.race)
        if q is None:
            continue
        result = archive.result(r.path)
        for tla, g in q["rank"].items():
            if tla in result.index and g <= n and result[tla] <= n:
                C[int(g) - 1, int(result[tla]) - 1] += 1
    k = np.arange(n)
    C = C @ np.exp(-0.5 * ((k[:, None] - k[None, :]) / sigma) ** 2)
    return C / C.sum(axis=1, keepdims=True)


def sinkhorn(M, iters=60):
    """Rescale rows and columns until every driver's distribution sums to 1 and every position is taken exactly once."""
    for _ in range(iters):
        M = M / M.sum(axis=1, keepdims=True)
        M = M / M.sum(axis=0, keepdims=True)
    return M / M.sum(axis=1, keepdims=True)


def blend_with_grid(out, order, table, weight=GRID_BLEND):
    """Mix each driver's simulated finishing distribution with the grid table; recompute the summary columns."""
    n = len(out)
    cols = [f"pos_{k}" for k in range(1, n + 1)]
    slot = out["tla"].map({d: i for i, d in enumerate(order)}).to_numpy()
    prior = table[np.minimum(slot, table.shape[0] - 1)][:, :n]
    prior = prior / prior.sum(axis=1, keepdims=True)
    M = sinkhorn((1 - weight) * out[cols].to_numpy() + weight * prior)
    out = out.copy()
    out[cols] = M
    cum = np.cumsum(M, axis=1)
    out["mean_pos"] = M @ np.arange(1, n + 1)
    out["proj_pos"] = (cum < 0.5).sum(axis=1) + 1
    out["p_win"], out["p_podium"], out["p_points"] = M[:, 0], M[:, :min(3, n)].sum(axis=1), M[:, :min(10, n)].sum(axis=1)
    out = out.sort_values(["mean_pos", "position"]).reset_index(drop=True)
    out["proj_rank"] = np.arange(1, n + 1)
    return out


def sparkline(row, n):
    """Distribution over finishing positions 1..n as block characters, scaled to the row's most likely position."""
    p = np.array([row[f"pos_{k}"] for k in range(1, n + 1)])
    blocks = "▁▂▃▄▅▆▇█"
    return "".join(" " if v < 0.005 else blocks[min(int(v / p.max() * (len(blocks) - 1) + 0.5), len(blocks) - 1)] for v in p)


def percentile_range(row, n, lo=0.1, hi=0.9):
    c = np.cumsum([row[f"pos_{k}"] for k in range(1, n + 1)])
    return int(np.searchsorted(c, lo) + 1), int(np.searchsorted(c, hi) + 1)


def print_table(out, info, top=None):
    n = len([c for c in out.columns if c.startswith("pos_")])
    print(f"{info['total_laps']} laps, pace model fitted on {info['n_train']} driver-races, "
          f"{info['rated']} drivers rated, chance of a safety car / VSC {info['p_sc'] * 100:.0f}%\n")
    print(f" {'':3}{'DRV':4}{'GRID':>5} {'MEAN':>5} {'P10-P90':>8} {'WIN':>5} {'PODIUM':>7} {'TOP10':>6}   finishing position 1 {'.' * (n - 4)} {n}")
    for i, (_, r) in enumerate(out.head(top or len(out)).iterrows(), 1):
        lo, hi = percentile_range(r, n)
        print(f" {i:>2} {r['tla']:4}{int(r['grid']):>5} {r['mean_pos']:>5.1f} {f'{lo}-{hi}':>8} {r['p_win'] * 100:>4.0f}% "
              f"{r['p_podium'] * 100:>6.0f}% {r['p_points'] * 100:>5.0f}%   {sparkline(r, n)}")


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("year", type=int)
    ap.add_argument("race", help="data folder of the race, e.g. Austin or United_States")
    ap.add_argument("--laps", type=int, help="race distance (default: last year's race at the circuit)")
    ap.add_argument("--grid", help="comma-separated driver codes in starting order (default: the qualifying result)")
    ap.add_argument("--sims", type=int, default=SIMS)
    ap.add_argument("--csv", type=Path, help="write the full distribution here (default: data/<year>/<race>/pre_race.csv)")
    ap.add_argument("--data-dir", default=DATA, type=Path)
    a = ap.parse_args()
    av = available(a.data_dir)
    priors = P.load_priors()
    archive = RaceArchive(av[av["session"] == "race"].reset_index(drop=True), priors)
    out, info = predict_weekend(archive, av, priors, a.year, a.race, a.grid.split(",") if a.grid else None, a.laps, a.sims)
    if out is None:
        raise SystemExit(info)
    print_table(out, info)
    path = a.csv or a.data_dir / str(a.year) / a.race / "pre_race.csv"
    out.drop(columns=["driver"]).to_csv(path, index=False)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
