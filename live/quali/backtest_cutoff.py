"""Backtest live.quali.predict_elimination: how close is the predicted Q1/Q2 cutoff to the real one?

For every session, predict every `--step` seconds from the start of the part until its last lap and score the
prediction against the cutoff the part ended with (the n_advance-th best valid lap of the final state). The "current"
cutoff (what the timing screen shows at that moment) is scored as the naive reference.

Usage (from the repo root):
    python -m live.quali.backtest_cutoff                    # every quali in data/
    python -m live.quali.backtest_cutoff 2026 --step 15 --csv backtest_cutoff.csv
"""
import argparse
import warnings

import numpy as np
import pandas as pd
from f1live.events import has_events, load
from f1live.paths import DATA
from pre.history import available
from pre.quali.features import Tables
from live.quali.predict_elimination import predict
from live.quali.prior import make_predictor
from live.quali.state import session_state

CLOCK_BINS = [-1, 0, 60, 180, 420, 10_000]
CLOCK_LABELS = ['0s (last runs)', '0-60s', '60-180s', '180-420s', '>420s']


def final_cutoff(rows, part):
    q = session_state(rows, part)
    if q is None:
        return None
    best = q.state['best'].dropna().sort_values()
    return best.iloc[q.n_advance - 1] if len(best) >= q.n_advance else None


def snapshot_times(rows, part, step):
    p = rows[rows['session_part'] == part]
    laps = p[p['event'] == 'lap']
    if laps.empty:
        return []
    return list(pd.date_range(p['ts'].min(), laps['ts'].max(), freq=f'{step}s'))


def backtest_session(name, rows, predict_fn=predict, step=30, parts=(1, 2)):
    out = []
    for part in parts:
        truth = final_cutoff(rows, part)
        if truth is None:
            continue
        for t in snapshot_times(rows, part, step):
            res = predict_fn(rows, part, t, verbose=False)
            if res is None or np.isnan(res['pred']):
                continue
            out.append(dict(session=name, part=part, ts=t, truth=truth, clock=res['clock'], n_pending=res['n_pending'],
                            pred=res['pred'], lo=res['lo'], hi=res['hi'], current=res['current'],
                            err=res['pred'] - truth, abs_err=abs(res['pred'] - truth),
                            naive_abs_err=abs(res['current'] - truth),
                            covered=res['lo'] <= truth <= res['hi'], width=res['hi'] - res['lo']))
    return out


def run(sessions, predict_fn=predict, step=30, parts=(1, 2), prior=False, data_dir=DATA):
    """prior=True: every session gets live.quali.prior's pace prior, built only from what came before it."""
    rows = []
    tables = Tables(available(data_dir)) if prior else None
    for name, path in sessions:
        fn = predict_fn
        if prior:
            fn = make_predictor(predict_fn, path.parent.parent.parent.name, path.parent.parent.name, tables)
        rows += backtest_session(name, load(path), fn, step, parts)
    return pd.DataFrame(rows)


def summarize(df):
    live = df[df['n_pending'] > 0]            # nothing left to predict once nobody is still to run
    live = live.assign(bucket=pd.cut(live['clock'], CLOCK_BINS, labels=CLOCK_LABELS))
    print(f'{live["session"].nunique()} sessions, {len(live)} predictions with a driver still to run\n')
    for part, d in [('all parts', live)] + [(f'Q{p}', g) for p, g in live.groupby('part')]:
        g = d.groupby('bucket', observed=True)
        print(part)
        print(pd.DataFrame({'n': g.size(), 'mae': g['abs_err'].mean(), 'naive mae': g['naive_abs_err'].mean(),
                            'bias': g['err'].mean(), 'cover90': g['covered'].mean(), 'width': g['width'].mean()})
              .round(3).to_string())
        print(f'  all: mae {d["abs_err"].mean():.3f}  naive {d["naive_abs_err"].mean():.3f}  '
              f'cover90 {d["covered"].mean():.3f}\n')


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    ap = argparse.ArgumentParser()
    ap.add_argument('year', nargs='?', default='*', help='e.g. 2026 (default: every year)')
    ap.add_argument('--step', type=int, default=30, help='seconds between predictions')
    ap.add_argument('--prior', action='store_true', help='use the pace prior (live.quali.prior)')
    ap.add_argument('--csv', help='write every prediction to this file')
    ap.add_argument('--data-dir', default=DATA, type=type(DATA))
    a = ap.parse_args()
    sessions = [(f'{p.parent.parent.parent.name}/{p.parent.parent.name}', p)
                for p in sorted(q for q in a.data_dir.glob(f'{a.year}/*/quali/events.csv') if has_events(q))]
    df = run(sessions, step=a.step, prior=a.prior, data_dir=a.data_dir)
    if a.csv:
        df.to_csv(a.csv, index=False)
    summarize(df)
