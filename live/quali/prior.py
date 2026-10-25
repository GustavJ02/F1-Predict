"""Pace prior for the live qualifying models: how fast each driver is expected to be in this part, in seconds.

Built once per part from what is known when it starts (pre.quali.parts: season form, this weekend's practice and
sprint pace, the parts before it; each for the driver and the team's better car), then turned into lap times:

  expected time = t_ref * (1 + gap/100)
  t_ref         = the part's fastest lap: the weekend's best practice lap (Q1) or the previous part's fastest lap
                  (Q2, Q3) times the median ratio seen on earlier weekends, but never above the fastest lap
                  already set in the part (the final fastest lap can only be at or below it)

Used by live.quali.predict_elimination and predict_pole as the expected lap of a driver who still has a run to come
(see expected_times), instead of "field median" / "previous part + shift".

make_predictor() wraps a live predict function so each part's prior is built once and cached.
"""
import traceback
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pre.history import available, race_key, valid_laps_from
from pre.quali import predict_grid
from pre.quali.features import Tables
from pre.quali.parts import fit_part, part_features, part_tables, tables_from_laps


@dataclass
class Prior:
    part: int
    gap: pd.Series          # expected gap to the part's fastest lap (%), indexed by tla
    sigma_pct: float        # residual spread of that gap (%)
    t_ref: float            # prior for the part's fastest lap (s), before the cap from laps already set
    n_train: int


def _ratio(tables, before, part):
    """Median (fastest lap of `part`) / (fastest lap of the part before) over earlier weekends."""
    av = tables.av
    out = []
    for r in av[(av["session"] == "quali") & (av["start"] < before)].itertuples():
        pt = part_tables(tables, r.path)
        if part in pt and part - 1 in pt:
            out.append(pt[part]["best_s"].min() / pt[part - 1]["best_s"].min())
    return float(np.median(out)) if out else np.nan


def _fp_ratio(tables, before):
    """Median (Q1 fastest lap) / (best practice lap of the weekend) over earlier weekends."""
    av, out = tables.av, []
    for r in av[(av["session"] == "quali") & (av["start"] < before)].itertuples():
        fp = _best_practice(tables, r.year, r.race)
        pt = part_tables(tables, r.path)
        if fp and 1 in pt:
            out.append(pt[1]["best_s"].min() / fp)
    return float(np.median(out)) if out else np.nan


def _best_practice(tables, year, race):
    best = [tables.session(year, race, s) for s in ("fp1", "fp2", "fp3")]
    best = [t["best_s"].min() for t in best if t is not None]
    return min(best) if best else None


def build_prior(tables, year, race, part, history):
    """Prior for `part` from the events of the session so far (only rows before the part's start are used)."""
    race = race_key(race)
    start = history.loc[history["session_part"] == part, "ts"].min()
    drivers = history[history["tla"].notna()].groupby("tla")[["team"]].last()
    done = history[history["ts"] < start] if pd.notna(start) else history
    earlier = tables_from_laps(valid_laps_from(done)) if part > 1 else {}
    target = part_features(tables, year, race, part, drivers=drivers, earlier=earlier)
    fit = fit_part(tables, target, part)
    if fit is None:
        return None
    gap = predict_grid.predict(fit, target)
    if part == 1:
        anchor, ratio = _best_practice(tables, year, race), _fp_ratio(tables, target["start"].iloc[0])
    else:
        anchor = earlier[part - 1]["best_s"].min() if part - 1 in earlier else None
        ratio = _ratio(tables, target["start"].iloc[0], part)
    if anchor is None or np.isnan(ratio):
        return None
    # the gaps are relative to the part's fastest driver: re-centre so the best expected driver sits at 0
    gap = gap - gap.min()
    return Prior(part, gap, float(fit.sigma_driver), float(anchor * ratio), fit.n_train)


# Picked with live.quali.backtest_cutoff --prior using both years (pull 0.3 tuned on 2025 alone also cut 2026's error
# by a third); a rough optimum, not a sharp one
K_PRIOR = 3.0        # t_ref: the prior counts as this many finished drivers
PULL = 0.3           # share of the improvement the prior expects beyond mu that is used, late in the part...
PULL_EARLY = 0.6     # ...rising to this with RAMP_S or more seconds left (more improvement still to come)
RAMP_S = 600


def expected_times(prior, state, mu, clock_left=None):
    """-> (center, extra_sigma): expected lap in seconds for a run still to come, and the extra spread (s) to add to the
    attempt-to-attempt spread because the driver has no time yet. Both per row of `state` (needs tla, best, pending);
    center is NaN for drivers the prior doesn't know, which keep the old rule. mu = typical first -> second attempt
    improvement.

    t_ref starts at the prior's and moves towards what the drivers who are done for the part (their time is final)
    imply for it, one prior = K_PRIOR drivers; it never goes above the fastest lap already set."""
    best = state["best"].to_numpy(dtype=float)
    gap = state["tla"].map(prior.gap).to_numpy(dtype=float)
    has = np.isfinite(best)
    t_ref = prior.t_ref
    done = has & np.isfinite(gap) & ~state["pending"].to_numpy(dtype=bool)
    if done.any():
        implied = np.median(best[done] / (1 + gap[done] / 100))
        t_ref = (done.sum() * implied + K_PRIOR * t_ref) / (done.sum() + K_PRIOR)
    if has.any():
        t_ref = min(t_ref, np.nanmin(best))
    expected = t_ref * (1 + gap / 100)
    base = best - mu
    pull = PULL if clock_left is None else PULL + (PULL_EARLY - PULL) * min(1.0, clock_left / RAMP_S)
    center = np.where(has, base - pull * np.maximum(base - expected, 0), expected)
    extra = np.where(has, 0.0, prior.sigma_pct / 100 * t_ref)
    return center, extra


def make_predictor(predict_fn, year, race, tables=None):
    """predict_fn(rows, part, as_of, verbose=False, prior=...) -> the same with each part's prior built once per
    session (at the first call for that part, from the rows up to then) and passed in."""
    tables = tables or Tables(available())
    cache = {}

    def predict(rows, part, as_of=None, verbose=True):
        known = set(rows["tla"].dropna())
        if part not in cache or not known <= cache[part][1]:       # built again only when drivers showed up since
            try:
                prior = build_prior(tables, int(year), race, part, rows)
            except Exception:
                traceback.print_exc()
                prior = None               # no prior: the old behaviour, not a dead model
            cache[part] = (prior, known)   # a None prior is cached too, or it would be retried on every call
        return predict_fn(rows, part, as_of, verbose=verbose, prior=cache[part][0])

    return predict
