"""Predict the Q1/Q2 cutoff time and P1 time from events.csv.

Usage (from the repo root):
    python -m quali.predict_elimination 2026 Austin quali [--part 2] [--as-of 2026-10-24T21:33:00Z]
"""
import argparse

import numpy as np
import pandas as pd
from f1live.events import load as load_and_prepare
from f1live.paths import add_session_args, session_from_args
from quali.state import session_state


def simulate_cutoff(best, pending, p_run, mu, sigma, n_advance, n_sims=10_000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(best)
    if n < n_advance or np.isnan(best).all():
        return np.nan, np.array([np.nan, np.nan])     # not enough drivers/times yet
    # Drivers without a valid time yet: base their attempt on the field's median best
    base = np.where(np.isnan(best), np.nanmedian(best), best)
    sims = np.tile(best, (n_sims, 1))

    runs = (rng.random((n_sims, n)) < p_run) & pending
    second = base - (mu + sigma * rng.standard_normal((n_sims, n)))
    sims = np.where(runs, np.fmin(sims, second), sims)

    cutoff = np.sort(sims, axis=1)[:, n_advance - 1]   # NaNs (no time) sort last
    return np.median(cutoff), np.percentile(cutoff, [5, 95])


def predict(all_rows, part, as_of=None, verbose=True):
    q = session_state(all_rows, part, as_of)
    if q is None:
        return None
    now, state, attempts, mu, sigma = q.now, q.state, q.attempts, q.mu, q.sigma
    n_advance, clock_left = q.n_advance, q.clock_left

    best_times = state['best'].dropna().sort_values()
    current = best_times.iloc[n_advance - 1] if len(best_times) >= n_advance else np.nan

    median, (lo, hi) = simulate_cutoff(
        state['best'].to_numpy(dtype=float), state['pending'].to_numpy(),
        state['p_run'].to_numpy(), mu, sigma, n_advance)

    if verbose:
        print(f'as of {now}  clock {clock_left:.0f}s')
        print(f'improvement: mu={mu:.3f}s sigma={sigma:.3f}s (from {len(attempts)} pairs)')
        print(f'current cutoff (P{n_advance}): {current:.3f}s')
        print(f'predicted final cutoff: {median:.3f}s  (90% interval {lo:.3f}-{hi:.3f})')
        print(f'{state["pending"].sum()} drivers still to run: {", ".join(state.index[state["pending"]])}')
    return dict(ts=now, clock=clock_left, current=current, pred=median, lo=lo, hi=hi,
                n_pending=int(state['pending'].sum()))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    add_session_args(ap)
    ap.add_argument('--part', type=int, help='session part to predict (default: the latest part in the data)')
    ap.add_argument('--as-of', help='pretend "now" is this moment (ISO8601); default = use everything')
    args = ap.parse_args()
    rows = load_and_prepare(session_from_args(args).events)
    part = args.part or int(rows['session_part'].dropna().iloc[-1])
    predict(rows, part, args.as_of)
