"""Live view of race model predictions: shows the latest batch of rows, whatever columns the model returns.

Usage (from the repo root, in a second terminal next to ./run_session.sh):
    python -m live.race.watch 2026 Singapore race
    python -m live.race.watch 2026 Singapore replay --once      # print once and exit
"""
from f1live.watch import BOLD, RESET, main, table

MAX_ROWS = 24


def render(df):
    last = df[df["ts"] == df["ts"].max()].drop(columns="ts")
    lines = [f"{BOLD}Latest prediction{RESET}   model time {df['ts'].max():%H:%M:%S}", ""]
    lines += table(last.dropna(axis=1, how="all").head(MAX_ROWS))
    if len(last) > MAX_ROWS:
        lines.append(f"... {len(last) - MAX_ROWS} more rows")
    return lines


if __name__ == "__main__":
    main(render, __doc__)
