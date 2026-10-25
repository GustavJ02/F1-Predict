"""Backtest quali.predict_pole on recorded sessions: how often does the driver it favours get pole?

For every session, predict every `--step` seconds from the start of Q3 until the last Q3 lap and score the
probabilities against who actually took pole (best valid Q3 lap).

Usage (from the repo root):
    python -m quali.backtest_pole 2026                 # every data/2026/*/quali
    python -m quali.backtest_pole 2026 --step 15 --csv backtest.csv
"""
import argparse
import warnings

import numpy as np
import pandas as pd
from f1live.events import load
from f1live.paths import DATA
from quali.predict_pole import predict
from quali.state import session_state

CLOCK_BINS = [-1, 0, 60, 180, 420, 10_000]
CLOCK_LABELS = ['0s (last runs)', '0-60s', '60-180s', '180-420s', '>420s']


def snapshot_times(rows, step):
    q3 = rows[rows['session_part'] == 3]
    laps = q3[q3['event'] == 'lap']
    if laps.empty:
        return []
    return list(pd.date_range(q3['ts'].min(), laps['ts'].max(), freq=f'{step}s'))


def backtest_session(name, rows, predict_fn=predict, step=30):
    final = session_state(rows, 3)
    if final is None or final.state['best'].isna().all():
        return []
    winner = final.state['best'].idxmin()
    out = []
    for t in snapshot_times(rows, step):
        res = predict_fn(rows, 3, t, verbose=False)
        if res is None:
            continue
        probs = pd.Series(res['probs'])
        top = probs.idxmax()
        onehot = (probs.index == winner).astype(float)
        out.append(dict(session=name, ts=t, winner=winner, clock=res['clock'], n_pending=res['n_pending'],
                        top1=top, p_top1=probs[top], p_winner=probs.get(winner, 0.0),
                        hit=top == winner, brier=float(((probs.to_numpy() - onehot) ** 2).sum()),
                        leader_wins=res['current_driver'] == winner))
    return out


def run(sessions, predict_fn=predict, step=30):
    rows = []
    for name, path in sessions:
        rows += backtest_session(name, load(path), predict_fn, step)
    return pd.DataFrame(rows)


def summarize(df):
    df = df.assign(bucket=pd.cut(df['clock'], CLOCK_BINS, labels=CLOCK_LABELS))
    live = df[df['n_pending'] > 0]            # nothing left to predict once nobody is still to run
    g = live.groupby('bucket', observed=True)
    print(f'{live["session"].nunique()} sessions, {len(live)} predictions with a driver still to run\n')
    print(pd.DataFrame({
        'n': g.size(), 'top1 hit': g['hit'].mean(), 'mean p(top1)': g['p_top1'].mean(),
        'mean p(winner)': g['p_winner'].mean(), 'p(winner)<2%': g['p_winner'].apply(lambda s: (s < 0.02).mean()),
        'brier': g['brier'].mean(), 'leader wins': g['leader_wins'].mean()}).round(3).to_string())
    bins = pd.cut(live['p_top1'], [0, .4, .6, .8, .95, 1.0001])
    print('\ncalibration of the favourite:')
    print(live.groupby(bins, observed=True).agg(n=('hit', 'size'), predicted=('p_top1', 'mean'),
                                                actual=('hit', 'mean')).round(3).to_string())
    print(f'\nall: top1 hit {live["hit"].mean():.3f}  mean p(top1) {live["p_top1"].mean():.3f}  '
          f'mean p(winner) {live["p_winner"].mean():.3f}  brier {live["brier"].mean():.3f}  '
          f'leader wins {live["leader_wins"].mean():.3f}')


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    ap = argparse.ArgumentParser()
    ap.add_argument('year')
    ap.add_argument('--step', type=int, default=30, help='seconds between predictions')
    ap.add_argument('--csv', help='write every prediction to this file')
    ap.add_argument('--data-dir', default=DATA, type=type(DATA))
    a = ap.parse_args()
    sessions = [(p.parent.parent.name, p) for p in sorted((a.data_dir / a.year).glob('*/quali/events.csv'))]
    df = run(sessions, step=a.step)
    if a.csv:
        df.to_csv(a.csv, index=False)
    summarize(df)
