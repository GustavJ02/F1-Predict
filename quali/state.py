"""Per-driver state of a qualifying part, shared by the cutoff and pole predictions."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

PRIOR_MU, PRIOR_SIGMA = 0.3, 0.3   # fallback until there are enough first->second pairs


@dataclass
class QualiState:
    now: pd.Timestamp
    raw: pd.DataFrame           # rows of this part up to `now`
    attempt_laps: pd.DataFrame
    attempts: pd.DataFrame      # first/second attempt per driver (drivers with two)
    mu: float                   # median improvement first -> second attempt
    sigma: float
    state: pd.DataFrame         # per driver: tla, best, n_attempts, on_flying_lap, pending, p_run
    n_advance: int
    clock_left: float


def session_state(all_rows, part, as_of=None):
    """Current picture of `part` as of `as_of` (default: latest row). None if there are no rows yet."""
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

    state = pd.DataFrame(index=sorted(raw['driver'].dropna().unique(), key=int))   # fixed order: results don't depend on row order
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

    state['tla'] = raw.groupby('driver')['tla'].last()
    return QualiState(now, raw, attempt_laps, attempts, mu, sigma, state, n_advance, clock_left)
