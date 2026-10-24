"""Watch a session's events.csv and re-run the cutoff prediction whenever new rows arrive.

Usage (from the repo root):
    python -m quali.watch_predictions 2026 Austin quali
    python -m quali.watch_predictions 2026 Austin quali --interval 0.5
"""
import argparse, time
import pandas as pd
from f1live.events import prepare
from f1live.paths import add_session_args, session_from_args
from f1live.tail import CsvTail
from quali.predict_elimination import predict


def watch(path, interval):
    tail, rows = CsvTail(path, prepare), None
    print(f'watching {path} ...')
    while True:
        new = tail.read()
        if isinstance(new, str):                # file was rewritten (parser restarted)
            print('file truncated, reloading from start')
            rows = None
        elif new is not None:
            rows = new if rows is None else pd.concat([rows, new], ignore_index=True)
            part = int(rows['session_part'].dropna().iloc[-1])
            print(f'\n--- {len(new)} new row(s), Q{part} ---')
            predict(rows, part)
        time.sleep(interval)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    add_session_args(ap)
    ap.add_argument('--interval', type=float, default=0.5, help='seconds between checks')
    args = ap.parse_args()
    try:
        watch(session_from_args(args).events, args.interval)
    except KeyboardInterrupt:
        pass