"""Watch laps.csv and re-run the cutoff prediction whenever new rows arrive.

Usage:
    python watch_predictions.py              # watches ./laps.csv
    python watch_predictions.py --csv q.csv --interval 0.5
"""
import argparse, io, os, time
import pandas as pd
from quali.predict_elimination import predict


def prepare(df):
    df['ts'] = pd.to_datetime(df['ts'], format='ISO8601')
    df['driver'] = df['driver'].astype('Int64').astype(str)
    return df


def watch(path, interval):
    rows, header, offset, buf = None, None, 0, ''
    print(f'watching {path} ...')
    while True:
        try:
            size = os.path.getsize(path)
        except FileNotFoundError:
            time.sleep(interval); continue

        if size < offset:                       # file was rewritten (parser restarted)
            print('file truncated, reloading from start')
            rows, header, offset, buf = None, None, 0, ''

        if size > offset:
            with open(path) as f:
                f.seek(offset)
                buf += f.read()
                offset = f.tell()
            # only use complete lines; keep any half-written line for next time
            complete, _, buf = buf.rpartition('\n')
            lines = [l for l in complete.splitlines() if l.strip()]
            if header is None and lines:
                header, lines = lines[0], lines[1:]
            if lines:
                new = prepare(pd.read_csv(io.StringIO('\n'.join([header] + lines))))
                rows = new if rows is None else pd.concat([rows, new], ignore_index=True)
                part = int(rows['session_part'].dropna().iloc[-1])
                print(f'\n--- {len(new)} new row(s), Q{part} ---')
                predict(rows, part)
        time.sleep(interval)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='laps.csv')
    ap.add_argument('--interval', type=float, default=0.5, help='seconds between checks')
    args = ap.parse_args()
    try:
        watch(args.csv, args.interval)
    except KeyboardInterrupt:
        pass