"""Predict how many safety cars a race will have: the distribution of full safety cars (SC), of virtual safety cars (VSC) and of
SC or VSC deployments together.

Fitted from every race in data/ (track status, rain and tyres of each recording), no other input. Per kind (SC, VSC) the chance that
one starts on a green lap is

    rate = mu * exp(b_phase * phase) * exp(b_wet * wet lap) * season effect * circuit effect

    deployment  a change of track status to SCDeployed / VSCDeployed from anything else; its length is the laps until AllClear.
                Laps under a deployment are not at risk (nothing new starts while one is out), laps 1-2 have their own pooled rate.
    wet         a lap with rainfall, or on which more than a third of the cars are on intermediate / wet tyres. Wet laps make SCs about
                three times likelier; VSCs hardly more (crashes in the wet, not breakdowns). For a coming race the rain is a probability
                (--rain-prob; default: how often a race has had wet laps): a wet race gets a block of wet laps of a length drawn from
                the wet races so far.
    phase       how far into the season the race is (calendar, 0-1). Later races have more VSCs (power units run to the end of their
                allocation, championship fights), SCs are unchanged. Fitted, so it is what the data says, not an assumption.
    season      the season's own level (e.g. a new-regulation year with more breakdowns raises the VSC rate, not the SC rate), Gamma
                distributed around 1 with a shape fitted by maximum likelihood over the seasons. The season being predicted is judged by
                its races so far and shrunk towards 1 as far as seasons have been seen to differ; a season without races yet is the prior
                alone, a wide spread around 1.
    circuit     the same for the circuit, shrunk as far as circuits have been seen to differ (very little for VSCs).
    length      drawn from the lengths seen (SC and VSC separately)

The season and circuit effects are fitted by alternating (each judged on the exposure the other has already adjusted for). Prediction:
every simulated race draws its season and circuit effects from their posteriors (so the uncertainty of a rate seen in few races is in the
answer), then runs lap by lap. SC and VSC rates are independent.

Usage (from the repo root; the race folder is the data folder, e.g. Mexico_City):
    python -m pre.race.predict_sc 2026 Mexico_City                    # laps from the circuit's last race, rain chance from history
    python -m pre.race.predict_sc 2026 Mexico_City --rain-prob 0.1 --date 2026-11-01
    python -m pre.race.predict_sc 2026 Mexico_City --rain-prob 0      # dry
    python -m pre.race.predict_sc 2026 Mexico_City --csv out.csv       # the three distributions as a table
"""
import argparse
from datetime import date as Date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import gammaln

from f1live.paths import DATA
from live.race.priors import circuit_of
from pre.history import available

KINDS = ("sc", "vsc")
STATUS = {"sc": "SCDeployed", "vsc": "VSCDeployed"}
EARLY_LAPS = 2           # laps counted as 'the start'
MIN_RATE = 1e-6          # per lap, so a kind that never happened still has a rate to take a logarithm of
MIN_SHAPE, MAX_SHAPE = 0.2, 1e4      # Gamma shape of a random effect: large = no differences
WET_SHARE = 0.3          # share of cars on intermediates / wets that makes a lap wet
PRIOR_SD = 1.5           # sd of the weak normal prior on the log multipliers of wet and phase (keeps them finite on few events)
SIMS = 100000
SHOW = 5                 # the table lists 0 .. SHOW-1 and 'SHOW or more'
SEASON_START, SEASON_END = 60, 335      # day of year of an ordinary first and last race: phase 0 and 1


def phase_of(day_of_year):
    """How far into the season a race is, 0-1 (calendar), clipped."""
    return float(np.clip((day_of_year - SEASON_START) / (SEASON_END - SEASON_START), 0, 1))


def wet_laps(ev, total):
    """Boolean array indexed by lap (0..total): rain on that lap, or more than WET_SHARE of the cars on intermediate / wet tyres."""
    wet = np.zeros(total + 1, bool)
    lap = pd.to_numeric(ev["race_lap"], errors="coerce")
    ok = lap.notna() & (lap >= 1) & (lap <= total)
    if "rainfall" in ev:
        rain = pd.to_numeric(ev["rainfall"], errors="coerce")[ok].groupby(lap[ok].astype(int)).max()
        wet[rain.index.to_numpy()] |= (rain > 0).to_numpy()
    if "compound" in ev:
        laps = ev[(ev["event"] == "lap") & ok]
        share = laps["compound"].isin(["INTERMEDIATE", "WET"]).groupby(lap[laps.index].astype(int)).mean()
        wet[share.index.to_numpy()] |= (share > WET_SHARE).to_numpy()
    return wet


def race_record(ev, day_of_year=None):
    """What one race's events say about safety cars: sc / vsc = [(start lap, length)], the laps at risk of a new deployment
    (not under one, split into the start and the rest, dry and wet), the wet laps and the phase of the season."""
    total = int(ev["total_laps"].max())
    ts = ev[ev["event"] == "track_status"][["track_status", "race_lap"]].dropna(subset=["race_lap"])
    out = dict(laps=total, sc=[], vsc=[])
    prev, open_ = None, []                       # open_: [(kind, start lap)] still waiting for AllClear
    for status, lap in zip(ts["track_status"], ts["race_lap"].astype(int)):
        for kind in KINDS:
            if status == STATUS[kind] and prev != status:
                open_.append((kind, lap))
        if status == "AllClear" and open_:
            out_lap = lap
            for kind, start in open_:
                out[kind].append((start, max(out_lap - start, 1)))
            open_ = []
        prev = status
    for kind, start in open_:                     # still out at the flag
        out[kind].append((start, max(total - start, 1)))
    busy = np.zeros(total + 2, bool)              # a deployment of d laps blocks the d - 1 laps after the one it started on
    for kind in KINDS:
        for start, length in out[kind]:
            busy[start + 1:start + length] = True
    wet = wet_laps(ev, total)
    lap = np.arange(total + 1)
    risk = ~busy[:total + 1] & (lap >= 1)
    early = lap <= EARLY_LAPS
    out.update(dry=int((risk & ~early & ~wet).sum()), wet=int((risk & ~early & wet).sum()),
               early_dry=int((risk & early & ~wet).sum()), early_wet=int((risk & early & wet).sum()),
               wet_share=float(wet[1:].mean()), phase=phase_of(day_of_year) if day_of_year is not None else 0.5)
    return out


def collect(data_dir=DATA):
    """{(year, race): race_record} for every race recording with a lap count."""
    records = {}
    cols = ["ts", "event", "track_status", "race_lap", "total_laps", "rainfall", "compound"]
    for _, row in available(data_dir)[lambda d: d["session"] == "race"].iterrows():
        try:
            ev = pd.read_csv(row["path"], usecols=lambda c: c in cols)
            day = pd.to_datetime(ev["ts"].iloc[0], utc=True, format="ISO8601").dayofyear
            records[(str(row["year"]), row["race"])] = race_record(ev, day)
        except (ValueError, KeyError, IndexError):
            continue                              # an empty or unfinished recording
    return records


def nb_logpmf(n, shape, rate, exposure):
    """log P(count = n) when the count is Poisson(lambda * exposure) and lambda ~ Gamma(shape, rate): negative binomial."""
    p = rate / (rate + exposure)
    return gammaln(n + shape) - gammaln(shape) - gammaln(n + 1) + shape * np.log(p) + n * np.log1p(-p)


def late_events(r, kind):
    return sum(1 for start, _ in r[kind] if start > EARLY_LAPS)


def fit_fixed(items, kind):
    """(mu, b_wet, b_phase, early rate): Poisson fit of the late deployments of one kind to the laps at risk, with a weak prior on the
    two multipliers. phase enters as (phase - 0.5), so mu is the rate of a dry lap in the middle of a season."""
    n = np.array([late_events(r, kind) for r in items.values()], float)
    dry = np.array([r["dry"] for r in items.values()], float)
    wet = np.array([r["wet"] for r in items.values()], float)
    x = np.array([r["phase"] - 0.5 for r in items.values()])

    def nll(theta):
        mu = np.exp(theta[0])
        m = mu * np.exp(theta[2] * x) * (dry + np.exp(theta[1]) * wet)
        return -(n * np.log(np.maximum(m, 1e-300)) - m).sum() + 0.5 * ((theta[1] / PRIOR_SD) ** 2 + (theta[2] / PRIOR_SD) ** 2)
    start = [np.log(max(n.sum() / max(dry.sum() + wet.sum(), 1), MIN_RATE)), 0.0, 0.0]
    best = minimize(nll, start, method="BFGS")
    mu, b_wet, b_phase = float(max(np.exp(best.x[0]), MIN_RATE)), float(best.x[1]), float(best.x[2])
    cov = np.asarray(best.hess_inv)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    early_n = sum(1 for r in items.values() for start, _ in r[kind] if start <= EARLY_LAPS)
    early_exposure = sum(r["early_dry"] + np.exp(b_wet) * r["early_wet"] for r in items.values())
    return dict(mu=mu, b_wet=b_wet, b_phase=b_phase, se_wet=float(se[1]), se_phase=float(se[2]),
                early=float(max(early_n / max(early_exposure, 1), MIN_RATE)))


def fit_shape(counts, exposures):
    """Shape alpha of a Gamma(alpha, alpha) multiplier (mean 1) by maximum likelihood over groups with (count, exposure)."""
    n, e = np.asarray(counts, float), np.asarray(exposures, float)
    res = minimize_scalar(lambda t: -nb_logpmf(n, np.exp(t), np.exp(t), e).sum(), bounds=(np.log(MIN_SHAPE), np.log(MAX_SHAPE)),
                          method="bounded")
    return float(np.exp(res.x))


def base_exposure(r, fixed):
    """Expected deployments of a race under the fixed effects alone."""
    return fixed["mu"] * np.exp(fixed["b_phase"] * (r["phase"] - 0.5)) * (r["dry"] + np.exp(fixed["b_wet"]) * r["wet"])


def fit(records, exclude=(), rounds=3):
    """Fitted parameters (a JSON-able dict) from race_records; races in `exclude` ((year, race) keys) are left out."""
    items = {k: v for k, v in records.items() if k not in set(exclude)}
    out = dict(races=len(items), kinds={}, lengths={}, wet_shares=sorted(r["wet_share"] for r in items.values() if r["wet_share"] > 0),
               p_wet_race=float(np.mean([r["wet_share"] > 0 for r in items.values()])) if items else 0.0)
    for kind in KINDS:
        fixed = fit_fixed(items, kind)
        base = {k: base_exposure(r, fixed) for k, r in items.items()}
        events = {k: late_events(r, kind) for k, r in items.items()}
        circ, year = {k: circuit_of(*k) for k in items}, {k: k[0] for k in items}
        season_mean, circuit_mean = {y: 1.0 for y in set(year.values())}, {c: 1.0 for c in set(circ.values())}
        for _ in range(rounds):                   # alternate: each effect on the exposure the other has adjusted for
            groups = {}
            for k in items:
                n, e = groups.setdefault(year[k], [0, 0.0]), None
                groups[year[k]][0] += events[k]
                groups[year[k]][1] += base[k] * circuit_mean[circ[k]]
            a_season = fit_shape([g[0] for g in groups.values()], [g[1] for g in groups.values()])
            season_mean = {y: (a_season + n) / (a_season + e) for y, (n, e) in groups.items()}
            cgroups = {}
            for k in items:
                g = cgroups.setdefault(circ[k], [0, 0.0])
                g[0] += events[k]
                g[1] += base[k] * season_mean[year[k]]
            a_circuit = fit_shape([g[0] for g in cgroups.values()], [g[1] for g in cgroups.values()])
            circuit_mean = {c: (a_circuit + n) / (a_circuit + e) for c, (n, e) in cgroups.items()}
        seasons = {}
        for k in items:
            g = seasons.setdefault(year[k], [0, 0.0])
            g[0] += events[k]
            g[1] += base[k] * circuit_mean[circ[k]]
        out["kinds"][kind] = dict(fixed, a_season=a_season, a_circuit=a_circuit,
                                  seasons={y: dict(n=int(n), exposure=float(e)) for y, (n, e) in seasons.items()},
                                  circuits={c: dict(n=int(n), exposure=float(e)) for c, (n, e) in cgroups.items()})
        out["lengths"][kind] = sorted(length for r in items.values() for _, length in r[kind])
    return out


def posterior(params, kind, circuit=None, year=None):
    """(shape, rate) of the Gamma over the circuit's multiplier, or (with `year`) the season's, after what its races showed."""
    p = params["kinds"][kind]
    if year is not None:
        seen, a = p["seasons"].get(str(year), dict(n=0, exposure=0.0)), p["a_season"]
    else:
        seen, a = p["circuits"].get(circuit, dict(n=0, exposure=0.0)), p["a_circuit"]
    return a + seen["n"], a + seen["exposure"]


def simulate(params, circuit, year, laps, phase=0.5, rain_prob=None, wet=None, sims=SIMS, seed=0):
    """{'sc': counts, 'vsc': counts} (arrays of `sims` simulated races) for a race of `laps` laps at `circuit` in season `year`.
    Rain: `wet` (boolean array, index 0..laps, wet laps known) or `rain_prob` (chance of a wet race, default: how often races have been
    wet); a wet race gets a block of wet laps with a length drawn from the wet races seen."""
    rng = np.random.default_rng(seed)
    p = params["kinds"]
    mult = {k: rng.gamma(*(lambda s, r: (s, 1 / r))(*posterior(params, k, circuit=circuit)), sims)
            * rng.gamma(*(lambda s, r: (s, 1 / r))(*posterior(params, k, year=year)), sims) for k in KINDS}
    lengths = {k: np.array(params["lengths"][k] or [3]) for k in KINDS}
    if wet is not None:
        is_wet = np.tile(np.asarray(wet, bool)[None, :laps + 1], (sims, 1))
        if is_wet.shape[1] < laps + 1:
            is_wet = np.pad(is_wet, ((0, 0), (0, laps + 1 - is_wet.shape[1])))
    else:
        prob = params["p_wet_race"] if rain_prob is None else rain_prob
        shares = np.array(params["wet_shares"] or [0.3])
        wet_race = rng.random(sims) < prob
        length = np.maximum(np.round(rng.choice(shares, sims) * laps).astype(int), 1)
        first = np.floor(rng.random(sims) * np.maximum(laps - length + 1, 1)).astype(int) + 1
        lap_no = np.arange(laps + 1)[None, :]
        is_wet = wet_race[:, None] & (lap_no >= first[:, None]) & (lap_no < (first + length)[:, None])
    busy, count = np.zeros(sims, int), {k: np.zeros(sims, int) for k in KINDS}
    for lap in range(1, laps + 1):
        idle = busy == 0
        prob_lap = {}
        for k in KINDS:
            f = p[k]
            wet_mult = np.where(is_wet[:, lap], np.exp(f["b_wet"]), 1.0)
            if lap <= EARLY_LAPS:
                prob_lap[k] = np.minimum(f["early"] * wet_mult, 1.0)
            else:
                prob_lap[k] = 1 - np.exp(-f["mu"] * np.exp(f["b_phase"] * (phase - 0.5)) * mult[k] * wet_mult)
        start_sc = idle & (rng.random(sims) < prob_lap["sc"])
        start_vsc = idle & ~start_sc & (rng.random(sims) < prob_lap["vsc"])
        for kind, started in (("sc", start_sc), ("vsc", start_vsc)):
            count[kind] += started
            busy = np.where(started, rng.choice(lengths[kind], sims), busy)
        busy = np.maximum(busy - 1, 0)                    # a deployment of d laps blocks the next d - 1 laps
    return count


def distribution(counts, show=SHOW):
    """DataFrame: one row per number of deployments (0 .. show-1 and 'show+') and a column per distribution, as probabilities."""
    series = {"Full safety car (SC)": counts["sc"], "VSC only": counts["vsc"], "SC or VSC": counts["sc"] + counts["vsc"]}
    rows = {}
    for name, n in series.items():
        rows[name] = [float((n == i).mean()) for i in range(show)] + [float((n >= show).mean())]
    df = pd.DataFrame(rows, index=[str(i) for i in range(show)] + [f"{show}+"])
    df.loc["mean"] = [float(n.mean()) for n in series.values()]
    df.loc["P(at least one)"] = [float((n >= 1).mean()) for n in series.values()]
    return df


def table(dist):
    """The distribution as text: probabilities in per cent, the mean as a number."""
    shown = dist.copy().astype(object)
    for row in shown.index:
        shown.loc[row] = [f"{v:.2f}" if row == "mean" else f"{v:.1%}" for v in dist.loc[row]]
    return shown.to_string()


def circuit_laps(records, circuit):
    """Laps of the circuit's most recent race we have, or None."""
    laps = [r["laps"] for key, r in records.items() if circuit_of(*key) == circuit]
    return laps[-1] if laps else None


def describe(params, circuit, year):
    """Lines on what was fitted."""
    lines = []
    for k in KINDS:
        f = params["kinds"][k]
        lo, hi = np.exp(f["b_wet"] - 1.645 * f["se_wet"]), np.exp(f["b_wet"] + 1.645 * f["se_wet"])
        s_shape, s_rate = posterior(params, k, year=year)
        c_shape, c_rate = posterior(params, k, circuit=circuit)
        seen_s, seen_c = f["seasons"].get(str(year), dict(n=0, exposure=0)), f["circuits"].get(circuit, dict(n=0, exposure=0))
        lines.append(f"{k.upper():3}: {f['mu']:.4f}/dry mid-season lap; wet x{np.exp(f['b_wet']):.1f} (90%: {lo:.1f}-{hi:.1f}); "
                     f"season end vs start x{np.exp(f['b_phase']):.1f}; start of race {f['early']:.3f}/lap")
        lines.append(f"     {year} season effect x{s_shape / s_rate:.2f} ({seen_s['n']} so far, shape {f['a_season']:.1f}); "
                     f"circuit effect x{c_shape / c_rate:.2f} ({seen_c['n']} seen, shape "
                     f"{'-' if f['a_circuit'] >= MAX_SHAPE else '%.1f' % f['a_circuit']})")
    lines.append(f"typical length SC {np.mean(params['lengths']['sc'] or [0]):.1f} laps, VSC {np.mean(params['lengths']['vsc'] or [0]):.1f};  "
                 f"races with wet laps: {params['p_wet_race']:.0%}")
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("year")
    ap.add_argument("race", help="folder name, e.g. Mexico_City")
    ap.add_argument("--laps", type=int, help="race distance (default: the circuit's last race)")
    ap.add_argument("--rain-prob", type=float, help="chance of a wet race (default: how often races have had wet laps)")
    ap.add_argument("--date", help="race day YYYY-MM-DD, for the place in the season (default: today)")
    ap.add_argument("--sims", type=int, default=SIMS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", type=Path, help="also write the distributions to this file")
    ap.add_argument("--data-dir", default=DATA, type=Path)
    a = ap.parse_args()
    records = collect(a.data_dir)
    circuit = circuit_of(a.year, a.race)
    # this weekend's race is not run yet; if its folder somehow has a finished recording it would leak in, so leave it out
    params = fit(records, exclude=[(str(a.year), a.race)])
    laps = a.laps or circuit_laps(records, circuit)
    if not laps:
        raise SystemExit(f"no earlier race at {circuit}: give --laps")
    day = Date.fromisoformat(a.date) if a.date else Date.today()
    phase = phase_of(day.timetuple().tm_yday)
    prob = params["p_wet_race"] if a.rain_prob is None else a.rain_prob
    dist = distribution(simulate(params, circuit, a.year, laps, phase, prob, sims=a.sims, seed=a.seed))
    here = sum(1 for k in records if circuit_of(*k) == circuit and k != (str(a.year), a.race))
    print(f"{a.year} {a.race}: {laps} laps, {day} (season phase {phase:.2f}), rain chance {prob:.0%}; "
          f"fitted on {params['races']} races ({here} at this circuit)\n")
    print(table(dist))
    print()
    print("\n".join(describe(params, circuit, a.year)))
    if a.csv:
        dist.to_csv(a.csv)


if __name__ == "__main__":
    main()
