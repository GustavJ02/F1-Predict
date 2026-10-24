"""Predict the Q1/Q2 cutoff time and P1 time from events.csv.

Usage (from the repo root):  python -m quali.predict_elimination 2026 Austin quali
"""
import argparse

import numpy as np
import pandas as pd
from f1live.paths import add_session_args, session_from_args

PART = 2                           # which segment to predict
PRIOR_MU, PRIOR_SIGMA = 0.3, 0.3   # fallback until there are enough first->second pairs
#AS_OF = '2026-10-24T21:40:28.476Z' # pretend "now" is this moment; None = use everything
AS_OF = None

def load_and_prepare(path):
    rows = pd.read_csv(path)
    rows['ts'] = pd.to_datetime(rows['ts'], format='ISO8601')
    rows['driver'] = rows['driver'].astype('Int64').astype(str)
    return rows



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
    raw = all_rows
    now = raw['ts'].max()
    if as_of is not None:
        now = pd.Timestamp(as_of)
        raw = raw[raw['ts'] <= now]
    raw = raw[raw['session_part'] == part].sort_values('ts')
    if raw.empty:
        return None

    laps = raw[raw['event'] == 'lap']

    # Real attempts: push laps AND slower 'other' laps, filtered on pace
    attempt_laps = laps[laps['lap_kind'].isin(['push', 'other'])].copy()
    attempt_laps = attempt_laps[attempt_laps['lap_time_s'] < 1.025 * attempt_laps['lap_time_s'].min()]

    # ---- improvement first -> second attempt ----
    attempt_laps['attempt'] = attempt_laps.groupby('driver').cumcount() + 1
    attempts = (attempt_laps[attempt_laps['attempt'] <= 2]
                .pivot(index='driver', columns='attempt', values='lap_time_s')
                .reindex(columns=[1, 2])
                .rename(columns={1: 'first', 2: 'second'})
                .dropna())
    attempts['improvement_s'] = attempts['first'] - attempts['second']
    if len(attempts) >= 3:
        imp = attempts['improvement_s']
        mu = imp.median()
        sigma = max(1.4826 * (imp - mu).abs().median(), 0.05)
    else:
        mu, sigma = PRIOR_MU, PRIOR_SIGMA

    # ---- current state per driver ----
    deleted = raw[raw['event'] == 'lap_deleted'][['driver', 'lap_time_s']]
    valid = attempt_laps.merge(deleted.assign(deleted=True), on=['driver', 'lap_time_s'], how='left')
    valid = valid[valid['deleted'].isna()]

    state = pd.DataFrame(index=raw['driver'].dropna().unique())
    state['best'] = valid.groupby('driver')['lap_time_s'].min()
    feed_best = raw.groupby('driver')['best_in_part_s'].last()
    state['best'] = state['best'].fillna(feed_best)
    state['n_attempts'] = attempt_laps.groupby('driver').size()
    state['n_attempts'] = state['n_attempts'].fillna(0).astype(int)

    # Clock at "now", extrapolated from the latest row
    latest = raw.iloc[-1]
    n_advance = 1 if pd.isna(latest['n_advance']) else int(latest['n_advance'])
    clock_left = max(latest['clock_left_s'] - (now - latest['ts']).total_seconds(), 0)

    # A driver whose latest completed lap was an out-lap is on a flying lap right now.
    # That lap counts even if it finishes after the chequered flag.
    last_kind = laps.groupby('driver')['lap_kind'].last()
    state['on_flying_lap'] = state.index.map(last_kind).isin(['out'])
    state['pending'] = state['on_flying_lap'] | ((clock_left > 0) & (state['n_attempts'] < 2))
    state['p_run'] = np.where(state['on_flying_lap'], 0.95, 0.9)

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
    ap.add_argument('--part', type=int, default=PART, help='session part to predict')
    ap.add_argument('--as-of', help='pretend "now" is this moment (ISO8601); default = use everything', default=None)
    args = ap.parse_args()
    rows = load_and_prepare(session_from_args(ap.parse_args()).events)
    predict(rows, args.part, args.as_of)
