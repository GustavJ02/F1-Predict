"""Terminal viewer for a session's predictions.csv: redraws in place whenever the file changes.

Not run directly: live.quali.watch and live.race.watch call `main(render)` with a function that turns the predictions
DataFrame into lines of text.
"""
import argparse, os, shutil, sys, time
from datetime import datetime
import pandas as pd
from f1live.paths import add_session_args, session_from_args

CLEAR = "\033[H\033[J"
BOLD, DIM, RESET = ("\033[1m", "\033[2m", "\033[0m") if sys.stdout.isatty() else ("", "", "")


def fmt(v, digits=3):
    """Table cell: '-' for missing, floats rounded, everything else as is."""
    if pd.isna(v):
        return "-"
    if isinstance(v, float):
        return f"{v:.{digits}f}".rstrip("0").rstrip(".") if v != int(v) else str(int(v))
    return str(v)


def bar(p, width=30):
    return "█" * round(p * width) + "░" * (width - round(p * width))


def table(df, cols=None, digits=3):
    """Right-aligned plain-text table."""
    cols = list(cols or df.columns)
    cells = [[fmt(v, digits) for v in row] for row in df[cols].itertuples(index=False)]
    widths = [max(len(c), *(len(r[i]) for r in cells)) if cells else len(c) for i, c in enumerate(cols)]
    lines = ["  ".join(c.rjust(w) for c, w in zip(cols, widths))]
    return lines + ["  ".join(v.rjust(w) for v, w in zip(r, widths)) for r in cells]


TEXT_COLUMNS = {"drivers": str, "numbers": str, "gaps": str, "laps": str, "note": str, "plan": str}   # lists written as 'a b c'


def read(path):
    """predictions.csv -> DataFrame, or None if missing / empty / caught mid-write. Columns that hold space-separated lists
    are read as text: with only one-item lists so far pandas would turn them into numbers."""
    try:
        df = pd.read_csv(path, dtype=TEXT_COLUMNS)
        df["ts"] = pd.to_datetime(df["ts"], format="ISO8601")
    except (FileNotFoundError, pd.errors.EmptyDataError, KeyError, ValueError):
        return None
    return df if len(df) else None


def screen(title, df, render):
    head = f"{BOLD}{title}{RESET}   {DIM}{len(df) if df is not None else 0} predictions, " \
           f"updated {datetime.now():%H:%M:%S}{RESET}"
    body = ["", "waiting for predictions ..."] if df is None else ["", *render(df)]
    return "\n".join([head, *body])


def main(render, description):
    ap = argparse.ArgumentParser(description=description)
    add_session_args(ap)
    ap.add_argument("--interval", type=float, default=0.5, help="seconds between checks")
    ap.add_argument("--once", action="store_true", help="print the current state once and exit")
    args = ap.parse_args()
    sess = session_from_args(args)

    if args.once or not sys.stdout.isatty():
        print(screen(str(sess), read(sess.predictions), render))
        return
    seen = None
    try:
        while True:
            try:
                stamp = os.stat(sess.predictions)
                stamp = (stamp.st_mtime_ns, stamp.st_size)
            except FileNotFoundError:
                stamp = None
            if stamp != seen:
                seen = stamp
                text = screen(str(sess), read(sess.predictions), render)
                lines = shutil.get_terminal_size().lines
                print(CLEAR + "\n".join(text.splitlines()[:lines - 1]), flush=True)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
