"""Live view of live/quali/model.py predictions: cutoff in Q1/Q2, pole probabilities in Q3.

Usage (from the repo root, in a second terminal next to ./run_session.sh):
    python -m live.quali.watch 2026 Austin quali
    python -m live.quali.watch 2026 Austin replay --once      # print once and exit
"""
import pandas as pd
from f1live.watch import BOLD, RESET, bar, fmt, main, table

HISTORY = 8     # rows of recent predictions shown below the headline


def render(df):
    last = df.iloc[-1]
    part = int(last["part"])
    lines = [f"{BOLD}Q{part}{RESET}   model time {last['ts']:%H:%M:%S}   still to run: {fmt(last['n_pending'])}", ""]
    if part == 3 and "pole1" in df and pd.notna(last.get("pole1")):
        lines += [f"{BOLD}Pole probability{RESET}   current pole time {fmt(last['current'])} s", ""]
        for i in (1, 2, 3):
            if pd.notna(last.get(f"pole{i}")):
                lines.append(f"  {last[f'pole{i}']:<4} {bar(last[f'p{i}'])} {last[f'p{i}']:6.1%}")
        recent = df[df["part"] == 3].tail(HISTORY)
        cols = ["ts", "pole1", "p1", "pole2", "p2", "pole3", "p3", "n_pending"]
    else:
        lines += [f"{BOLD}Cutoff{RESET}   now {fmt(last['current'])} s   predicted {fmt(last['pred'])} s   "
                  f"(90%: {fmt(last['lo'])} - {fmt(last['hi'])})"]
        recent = df[df["part"] == part].tail(HISTORY)
        cols = ["ts", "current", "pred", "lo", "hi", "n_pending"]
    recent = recent[[c for c in cols if c in recent]].copy()
    recent["ts"] = recent["ts"].dt.strftime("%H:%M:%S")
    if "p1" in recent:
        for c in ("p1", "p2", "p3"):
            recent[c] = recent[c].map(lambda v: f"{v:.0%}" if pd.notna(v) else None)
    return lines + ["", "recent:", *("  " + l for l in table(recent))]


if __name__ == "__main__":
    main(render, __doc__)
