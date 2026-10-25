"""Projected finishing order: a Monte Carlo simulation of the rest of the race from the current state.

Each simulated lap, every car gets a lap time from its own pace (anchored on its recent clean laps), the tyre it is on
(compound offset, wear with age) and the fuel burned (all from live.race.priors), plus noise. Then:
  pit stops   a car pits when its tyre age reaches a stint length sampled from the survival curve of past stints on that
              compound (at this circuit if we have enough); a car that hasn't used two dry compounds must stop before the end. A stop costs the
              circuit's pit loss; the next compound follows the historical switches.
  traffic     a car can't get closer than GAP_MIN to the car in front. When it is faster it passes with the probability the
              priors give for its pace advantage at this circuit (hard to pass at Monaco, easy at Abu Dhabi); otherwise it is
              held up. A car is not held up by a rival that is in the pits.
  safety car  SC / VSC deployments at the circuit's rate: the field bunches up (SC), nobody passes, stops cost much less,
              and cars due to pit mostly pit under it.
  retirements a small chance per lap.
The finishing positions over all simulations give each car's probability of winning, a podium and the points, and its
average finishing position, which orders the projection.
"""
import numpy as np
import pandas as pd
from live.race.battles import latest
from live.race.pace import clean_laps, lap_rows
from live.race.priors import COMPOUNDS, circuit_of

GAP_MIN = 1.0           # s, the closest a car follows another in dirty air (DRS range)
PACE_SD = 0.10          # s, uncertainty of each car's pace estimate
LAP_NOISE = 0.20        # s per lap
PIT_SCALE = 0.8         # share of the measured pit loss that is really lost (fresher tyres win part of it back)
DNF_PER_LAP = 0.0008
SC_LAP_FACTOR, VSC_LAP_FACTOR = 1.45, 1.30
SC_PIT_RATIO, VSC_PIT_RATIO = 0.45, 0.65     # pit loss under SC / VSC relative to green
SC_BUNCH = 0.9          # s, gap between cars once the field has bunched up behind the safety car
PACE_LAPS = 8           # laps used to anchor a car's pace
POINTS = 10
N_SIMS = 400


def lap_time(model, base, comp, age, lap):
    """Deterministic lap time. comp: array of compound indices into COMPOUNDS."""
    off = np.array([model["offset"][c] for c in COMPOUNDS])
    wear = np.array([model["wear"][c] for c in COMPOUNDS])
    return base + off[comp] + wear[comp] * age - model["fuel"] * lap


def car_states(history, priors):
    """Per running car: position, gap to the leader (s), compound index, tyre age, stops, compounds used and a pace base
    (lap time at age 0 on mediums at lap 0). Returns (DataFrame, leader lap time, laps done, total laps) or (None, reason)."""
    lat = latest(history)
    if lat.empty:
        return None, "waiting for race data"
    lap = history[history["race_lap"].notna() & history["total_laps"].notna()].tail(1)
    if lap.empty:
        return None, "waiting for race data"
    n0, total = int(lap["race_lap"].iloc[0]), int(lap["total_laps"].iloc[0])
    model = priors["lap_model"]
    laps = lap_rows(history)
    clean = clean_laps(laps)
    clean = clean[clean["compound"].isin(COMPOUNDS) & clean["tyre_laps"].notna()].copy()
    ci = clean["compound"].map({c: i for i, c in enumerate(COMPOUNDS)}).to_numpy()
    clean["resid"] = clean["lap_time_s"].to_numpy() - lap_time(model, 0.0, ci, clean["tyre_laps"].to_numpy(),
                                                              clean["lap_number"].to_numpy())
    base = clean.groupby("driver")["resid"].apply(lambda s: float(np.median(s.tail(PACE_LAPS))))
    used = laps[laps["compound"].isin(COMPOUNDS)].groupby("driver")["compound"].agg(lambda s: sorted(set(s)))
    wet = bool(laps["compound"].isin(["INTERMEDIATE", "WET"]).any())

    cars = lat[~lat["out"] & lat["position"].notna()].sort_values("position")
    if len(cars) < 2:
        return None, "waiting for race data"
    lead = cars.index[0]
    lead_pace = base.get(lead, np.nan)
    t_lead = float(np.median(clean[clean["driver"] == lead]["lap_time_s"].tail(5))) if (clean["driver"] == lead).any() else 95.0
    field_base = float(base.median()) if len(base) else 90.0
    rows, prev_g, prev_ld = [], 0.0, 0
    for k, (drv, r) in enumerate(cars.iterrows()):
        ld = int(r["laps_down"]) if pd.notna(r["laps_down"]) else 0
        if k == 0:
            g = 0.0
        elif pd.notna(r["gap_to_leader_s"]) and ld == 0:
            g = float(r["gap_to_leader_s"])
        else:
            step = float(r["interval_s"]) if pd.notna(r["interval_s"]) and ld == prev_ld else 1.0
            g = prev_g + (ld - prev_ld) * t_lead + step
        g = max(g, prev_g)
        prev_g, prev_ld = g, ld
        comp = r["compound"] if r["compound"] in COMPOUNDS else "MEDIUM"
        rows.append(dict(driver=drv, tla=r["tla"], position=int(r["position"]), gap=g,
                         comp=COMPOUNDS.index(comp), age=float(r["tyre_laps"]) if pd.notna(r["tyre_laps"]) else 0.0,
                         stops=int(r["n_pit_stops"]) if pd.notna(r["n_pit_stops"]) else 0,
                         used=len(used.get(drv, [])) , base=float(base.get(drv, field_base + 0.3)),
                         known=drv in base.index, needs_stop=(len(used.get(drv, [])) < 2) and not wet))
    return pd.DataFrame(rows), (t_lead, n0, total)


def sample_stint(rng, km, comp_idx, current_age, shape):
    """Tyre age at the next stop, sampled from the compound's survival curve given the tyre has already lasted
    current_age laps (inf = it never stops: the curve has mass beyond the longest observed stop)."""
    out = np.empty(shape)
    for c in np.unique(comp_idx):
        curve = km[COMPOUNDS[c]]
        ages, surv = np.asarray(curve["age"], float), np.asarray(curve["surv"], float)
        sel = comp_idx == c
        age = current_age[sel] if np.ndim(current_age) else np.full(int(sel.sum()), current_age)
        at = np.searchsorted(ages, age, side="right") - 1                # last curve point at or below the current age
        s_at = np.where(at >= 0, surv[np.clip(at, 0, len(ages) - 1)], 1.0)   # S(age)
        target = rng.random(int(sel.sum())) * s_at                      # draw S(X) uniformly below S(age)
        idx = np.searchsorted(-surv, -target, side="left")               # first age whose survival is <= target
        out[sel] = np.where(idx < len(ages), np.maximum(ages[np.minimum(idx, len(ages) - 1)], age + 1), np.inf)
    return out


def next_compound(rng, trans, comp_idx, used_mask):
    """Compound after a stop: historical switches; if the car still hasn't used two compounds, it must change."""
    n = len(comp_idx)
    new = np.empty(n, int)
    P = np.array([[trans[a][b] + 1.0 for b in COMPOUNDS] for a in COMPOUNDS])
    for i in range(n):
        w = P[comp_idx[i]].copy()
        if used_mask[i]:
            w[comp_idx[i]] = 0.0
        new[i] = rng.choice(len(COMPOUNDS), p=w / w.sum())
    return new


def project(history, priors, circuit, n_sims=N_SIMS, seed=0, **opt):
    """-> (DataFrame per running car, info dict) or (None, reason string).

    DataFrame columns: driver, tla, position, proj_pos (median finish), mean_pos, p_win, p_podium, p_points, stops_left.
    info: laps_left, p_sc (chance of a safety car or VSC before the flag), under_sc.

    opt (for experiments, all default to the values above): sc (False = no safety cars), dnf (False = no retirements),
    pace_sd, lap_noise, gap_min, pit_scale (multiplies the pit loss), friction (False = no traffic model).
    PACE_SD, LAP_NOISE, PIT_SCALE and GAP_MIN were set by a grid over the backtest's calibration (live.race.backtest_projection)."""
    sc_on, dnf_p = opt.get("sc", True), DNF_PER_LAP if opt.get("dnf", True) else 0.0
    pace_sd, lap_noise, gap_min = opt.get("pace_sd", PACE_SD), opt.get("lap_noise", LAP_NOISE), opt.get("gap_min", GAP_MIN)
    friction = opt.get("friction", True)
    cars, meta = car_states(history, priors)
    if cars is None:
        return None, meta
    t_lead, n0, total = meta
    R = total - n0
    if R <= 1:
        return None, "race finished" if R <= 0 else "last lap"
    track = history["track_status"].dropna().iloc[-1] if history["track_status"].notna().any() else "AllClear"
    rng = np.random.default_rng(seed)
    S, C = n_sims, len(cars)
    model, trans = priors["lap_model"], priors["transitions"]
    pit_loss = priors["pit_loss"]["circuits"].get(circuit, priors["pit_loss"]["pooled"]) * opt.get("pit_scale", PIT_SCALE)
    km = {c: priors["stint"]["circuits"].get(circuit, {}).get(c) or priors["stint"]["pooled"][c] for c in COMPOUNDS}
    pa, pb = priors["passing"]["a"], priors["passing"]["b"]
    p_eff = priors["passing"]["circuits"].get(circuit, 0.0)
    sf = priors["safety"]
    cs = sf["circuits"].get(circuit, sf)

    t = np.tile(cars["gap"].to_numpy(float), (S, 1))
    comp = np.tile(cars["comp"].to_numpy(), (S, 1))
    age = np.tile(cars["age"].to_numpy(float), (S, 1))
    used2 = np.tile(~cars["needs_stop"].to_numpy(), (S, 1))               # has used two compounds (or wet race)
    pace = cars["base"].to_numpy(float)[None, :] + pace_sd * rng.standard_normal((S, C)) \
        * np.where(cars["known"].to_numpy(), 1.0, 3.0)[None, :]
    stop_age = sample_stint(rng, km, comp.ravel(), age.ravel(), S * C).reshape(S, C)
    alive = np.ones((S, C), bool)
    sc_left, vsc_left = np.zeros(S, int), np.zeros(S, int)
    if track == "SCDeployed":
        sc_left[:] = max(int(round(sf["sc_laps"])) - 1, 1)
    elif track in ("VSCDeployed", "VSCEnding"):
        vsc_left[:] = 1
    had_sc = (sc_left > 0) | (vsc_left > 0)
    stops_done = np.zeros((S, C))

    for k in range(1, R + 1):
        lap_no = n0 + k
        # --- safety car / VSC
        start_sc = (sc_left == 0) & (vsc_left == 0) & (rng.random(S) < cs["sc_rate"] * sc_on) & (k <= R - 2)
        start_vsc = (sc_left == 0) & (vsc_left == 0) & ~start_sc & (rng.random(S) < cs["vsc_rate"] * sc_on) & (k <= R - 2)
        sc_left = np.where(start_sc, 2 + rng.poisson(max(sf["sc_laps"] - 2, 0.5), S), sc_left)
        vsc_left = np.where(start_vsc, 1 + rng.poisson(max(sf["vsc_laps"] - 1, 0.5), S), vsc_left)
        under_sc, under_vsc = sc_left > 0, (vsc_left > 0) & ~(sc_left > 0)
        first_sc = start_sc
        had_sc |= under_sc | under_vsc

        base_lt = lap_time(model, pace, comp, age, lap_no)                    # deterministic, used for pass odds
        lt = base_lt + lap_noise * rng.standard_normal((S, C))
        lt = np.where(under_sc[:, None], t_lead * SC_LAP_FACTOR, np.where(under_vsc[:, None], lt * VSC_LAP_FACTOR, lt))

        # --- pit stops
        due = (age + 1 >= stop_age) & (stop_age - age <= R - k + 1) | (~used2 & (k >= R - 1) & (stop_age > age))
        due &= (k <= R - 1)
        under_any = (under_sc | under_vsc)[:, None]
        early = under_any & ((age >= 0.5 * stop_age) & (stop_age - age < R - k + 8)) & (rng.random((S, C)) < 0.8)
        pit = (due | early) & alive
        ratio = np.where(under_sc, SC_PIT_RATIO, np.where(under_vsc, VSC_PIT_RATIO, 1.0))[:, None]
        lt = lt + np.where(pit, ratio * (pit_loss + 1.2 * rng.standard_normal((S, C))), 0.0)
        # tyre state for the next lap
        if pit.any():
            ss, cc = np.nonzero(pit)
            cur = comp[ss, cc]
            new = next_compound(rng, trans, cur, ~used2[ss, cc])
            changed = new != cur
            used2[ss, cc] = used2[ss, cc] | changed
            comp[ss, cc] = new
            age[ss, cc] = 0
            stops_done[ss, cc] += 1
            stop_age[ss, cc] = sample_stint(rng, km, new, np.zeros(len(new)), len(new))
        age = np.where(pit, 0.0, age + 1)
        # --- retirements
        alive &= ~(rng.random((S, C)) < dnf_p)
        lt = np.where(alive, lt, np.inf)

        t_new = t + lt
        # --- traffic: process cars front to back in the order they started the lap in; each one follows the last
        # car or passes it (sorting by the new times would hand every faster car its pass before the odds are checked)
        order = np.argsort(t, axis=1, kind="stable")
        ts = np.take_along_axis(t_new, order, axis=1)
        lts = np.take_along_axis(base_lt, order, axis=1)
        pit_s = np.take_along_axis(pit, order, axis=1)
        adj = ts.copy()
        front = adj[:, 0].copy()
        front_lt, front_pit = lts[:, 0].copy(), pit_s[:, 0].copy()
        for j in range(1, C):
            tj = ts[:, j]
            finite = np.isfinite(tj)
            behind = finite & (tj < front + gap_min)
            wants = behind & (tj < front)
            delta = np.clip(front_lt - lts[:, j], *priors_clip())
            p_pass = 1.0 / (1.0 + np.exp(-(pa + pb * delta + p_eff)))
            free = front_pit & ~pit_s[:, j]                     # the car ahead is in the pits and this one isn't
            success = wants & ((rng.random(S) < p_pass) | free | (not friction))
            held = behind & ~success
            adj[:, j] = np.where(held, front + gap_min, tj)
            newer = adj[:, j] > front
            front_lt = np.where(newer, lts[:, j], front_lt)
            front_pit = np.where(newer, pit_s[:, j], front_pit)
            front = np.maximum(front, np.where(finite, adj[:, j], front))
        # under a safety car nobody passes and the field bunches up behind the leader
        if under_sc.any():
            sel = under_sc
            bunch = ts[sel].copy()
            for j in range(1, C):
                bunch[:, j] = np.where(np.isfinite(ts[sel, j]), np.minimum(bunch[:, j], bunch[:, j - 1] + SC_BUNCH), np.inf)
            adj[sel] = bunch
        t = np.empty_like(t_new)
        np.put_along_axis(t, order, adj, axis=1)

    finish = t.argsort(axis=1).argsort(axis=1) + 1                       # 1 = winner; retirements (inf) last
    out = cars[["driver", "tla", "position"]].copy()
    out["mean_pos"] = finish.mean(axis=0)
    out["proj_pos"] = np.median(finish, axis=0)
    out["p_win"] = (finish == 1).mean(axis=0)
    out["p_podium"] = (finish <= 3).mean(axis=0)
    out["p_points"] = (finish <= POINTS).mean(axis=0)
    out["stops_left"] = (stops_done.mean(axis=0))
    out = out.sort_values(["mean_pos", "position"]).reset_index(drop=True)
    out["proj_rank"] = np.arange(1, len(out) + 1)
    return out, dict(laps_left=R, p_sc=float(had_sc.mean()), under_sc=track in ("SCDeployed", "VSCDeployed"))


def priors_clip():
    return (-1.0, 2.0)


def circuit_for(session):
    return circuit_of(session.year, session.race)
