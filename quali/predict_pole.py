"""Predict who takes pole: Monte Carlo over the laps still to come in the part (normally Q3).

Same ingredients as the cutoff model (quali.state): each driver's best valid lap, which drivers still have a
lap to come, the typical first -> second attempt improvement (mu, sigma). Each simulation draws the laps still
to come and records who ends up fastest.

  - driver with a time who still runs: new lap = best - mu + sigma * N(0, 1), kept only if faster
  - driver with no time yet who runs:  new lap = base + sigma * N(0, 1), no improvement (it's their first lap);
    base = their best in the previous part + the field's median change between parts (0 if < 3 drivers
    have both), falling back to the field's median best in this part
  - whether a driver runs at all uses p_run from quali.state

Usage (from the repo root):
    python -m quali.predict_pole 2026 Austin quali [--part 3] [--as-of 2026-10-24T21:55:00Z]
"""
import argparse

import numpy as np
import pandas as pd
from f1live.events import load
from f1live.paths import add_session_args, session_from_args
from quali.state import session_state

N_PARAM_PAIRS = 3     # drivers needed with a time in both parts before the part-to-part change is trusted


def simulate_pole(best, base, gain, pending, p_run, sigma, n_sims=10_000, seed=0):
    """-> (P(pole) per driver, P(nobody sets a time)). NaN in `best` = no time yet, NaN in `base` = can't run."""
    rng = np.random.default_rng(seed)
    n = len(best)
    final = np.tile(np.where(np.isnan(best), np.inf, best), (n_sims, 1))
    runs = (rng.random((n_sims, n)) < p_run) & pending
    draw = base - gain + sigma * rng.standard_normal((n_sims, n))
    draw = np.where(np.isnan(draw), np.inf, draw)
    final = np.where(runs, np.minimum(final, draw), final)
    winner = final.argmin(axis=1)
    has_time = np.isfinite(final.min(axis=1))
    probs = np.bincount(winner[has_time], minlength=n) / n_sims
    return probs, 1 - has_time.mean()


def predict(all_rows, part=3, as_of=None, verbose=True):
    q = session_state(all_rows, part, as_of)
    if q is None:
        return None
    st = q.state
    best = st['best'].to_numpy(dtype=float)

    # Base lap for drivers without a time yet (see module docstring)
    prev = session_state(all_rows, part - 1, as_of) if part > 1 else None
    base = best.copy()
    if prev is not None:
        prev_best = prev.state['best'].reindex(st.index)
        both = st['best'].notna() & prev_best.notna()
        shift = (st['best'] - prev_best)[both].median() if both.sum() >= N_PARAM_PAIRS else 0.0
        base = np.where(np.isnan(base), (prev_best + shift).to_numpy(dtype=float), base)
    if np.isnan(best).all():
        field = np.nan
    else:
        field = np.nanmedian(best)
    base = np.where(np.isnan(base), field, base)

    has_time = ~np.isnan(best)
    probs, p_none = simulate_pole(best, base, np.where(has_time, q.mu, 0.0), st['pending'].to_numpy(),
                                  st['p_run'].to_numpy(), q.sigma)

    order = np.argsort(-probs)[:3]
    top3 = [dict(driver=st.index[i], tla=st['tla'].iloc[i], prob=float(probs[i])) for i in order if probs[i] > 0]
    leader = st['best'].idxmin() if has_time.any() else None
    current = float(np.nanmin(best)) if has_time.any() else np.nan

    if verbose:
        print(f'as of {q.now}  Q{part}  clock {q.clock_left:.0f}s')
        if leader is not None:
            print(f'current pole: {st["tla"][leader]} {current:.3f}s')
        print(f'{st["pending"].sum()} drivers still to run: {", ".join(st.index[st["pending"]])}')
        print('pole probability: ' + '  '.join(f'{t["tla"]} {t["prob"]:.1%}' for t in top3))
    return dict(ts=q.now, part=part, clock=q.clock_left, current=current, current_driver=leader,
                top3=top3, n_pending=int(st['pending'].sum()), p_none=float(p_none))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    add_session_args(ap)
    ap.add_argument('--part', type=int, help='session part (default: the latest part in the data)')
    ap.add_argument('--as-of', help='pretend "now" is this moment (ISO8601); default = use everything')
    args = ap.parse_args()
    rows = load(session_from_args(args).events)
    predict(rows, args.part or int(rows['session_part'].dropna().iloc[-1]), args.as_of)
