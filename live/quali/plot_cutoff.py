"""Plot how the predicted Q1 / Q2 cutoff moved during qualifying: one panel per part, a dot for every projection (live.quali.model),
the 90 % band between the model's lowest and highest cutoff, the cutoff as it stands now (the n-th best time so far), and, once a
part is over, the cutoff it really ended on, so you can see how the projection converged on it.

Usage (from the repo root; works on a session that is being recorded and on a finished one):
    python -m live.quali.plot_cutoff 2026 Mexico_City quali                  # writes data/2026/Mexico_City/quali/cutoff.png
    python -m live.quali.plot_cutoff 2026 Mexico_City quali --follow 10      # redraw the same file every 10 s while the session runs
    python -m live.quali.plot_cutoff 2026 Mexico_City quali --out plot.png

Q3 has no cutoff (the model gives pole probabilities there) and is left out.
"""
import argparse
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

from f1live.events import load
from f1live.paths import add_session_args, session_from_args
from live.quali.classification import part_bests, part_sizes
from live.quali.cutoff_chart import lap_label

BAND = "#4c78a8"
PRED = "#1f3b73"
NOW = "#e45756"
FINAL = "#2ca02c"


def read_predictions(path):
    """predictions.csv of live.quali.model as a DataFrame (empty if there is none, or if it is another model's file)."""
    try:
        df = pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()
    if "part" not in df:
        return pd.DataFrame()
    df = df[df["part"].notna() & df["pred"].notna()].copy()
    df["ts"] = pd.to_datetime(df["ts"], format="ISO8601")
    df["part"] = df["part"].astype(int)
    return df


def final_cutoffs(history):
    """{part: the cutoff it ended on} for the parts that are over: the slowest time among the cars that went through."""
    if history is None or history["session_part"].notna().sum() == 0:
        return {}
    bests, sizes = part_bests(history), part_sizes(history)
    current = max(sizes)
    out = {}
    for part, (_, advance) in sizes.items():
        if part < current and advance:
            times = bests[part].dropna().sort_values()
            if len(times) >= advance:
                out[part] = float(times.iloc[advance - 1])
    return out


def draw(preds, finals, title="Qualifying cutoff"):
    """Figure with a panel for every part (1, 2) that has projections; None if there are none."""
    parts = [p for p in (1, 2) if (preds["part"] == p).any()] if len(preds) else []
    if not parts:
        return None
    fig, axes = plt.subplots(1, len(parts), figsize=(7.5 * len(parts), 4.8), squeeze=False)
    for ax, part in zip(axes[0], parts):
        d = preds[preds["part"] == part]
        ax.fill_between(d["ts"], d["lo"], d["hi"], color=BAND, alpha=0.25, linewidth=0, label="90 % band (lowest - highest cutoff)")
        ax.plot(d["ts"], d["pred"], color=PRED, linewidth=0.8, alpha=0.6)
        ax.scatter(d["ts"], d["pred"], s=9, color=PRED, zorder=3, label="projected cutoff")
        now = d[d["current"].notna()]
        if len(now):
            ax.step(now["ts"], now["current"], where="post", color=NOW, linewidth=1.4, linestyle="--", label="cutoff right now")
        if part in finals:
            ax.axhline(finals[part], color=FINAL, linewidth=1.4, label=f"final cutoff {lap_label(finals[part])}")
        low, high = d["lo"].min(), d["hi"].max()                 # zoom on the projections: the early "right now" times are far off
        if part in finals:
            low, high = min(low, finals[part]), max(high, finals[part])
        pad = 0.08 * max(high - low, 0.05)
        ax.set_ylim(low - pad, high + pad)
        ax.set_title(f"Q{part}", loc="left", fontweight="bold")
        ax.yaxis.set_major_formatter(FuncFormatter(lap_label))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        short = (d["ts"].max() - d["ts"].min()) < pd.Timedelta(minutes=20)           # a few minutes of Q2: show the seconds too
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S" if short else "%H:%M"))
        ax.grid(alpha=0.25)
        ax.set_xlabel("time (UTC)")
        ax.set_ylabel("cutoff lap time")
        ax.legend(loc="best", fontsize=8, framealpha=0.9)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def render(sess, out):
    """Draw the session's current state to `out`. -> True if a plot was written."""
    preds = read_predictions(sess.predictions)
    try:
        history = load(sess.events)
    except (FileNotFoundError, pd.errors.EmptyDataError, ValueError):
        history = None
    fig = draw(preds, final_cutoffs(history), f"{sess.year} {sess.race} {sess.name}: projected cutoff")
    if fig is None:
        return False
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return True


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_session_args(ap)
    ap.add_argument("--out", help="image file (default: cutoff.png in the session folder)")
    ap.add_argument("--follow", type=float, metavar="SECONDS", help="keep redrawing the file at this interval (Ctrl-C to stop)")
    a = ap.parse_args()
    sess = session_from_args(a)
    out = a.out or sess.dir / "cutoff.png"
    while True:
        if render(sess, out):
            print(f"{time.strftime('%H:%M:%S')} wrote {out}")
        else:
            print(f"{time.strftime('%H:%M:%S')} no cutoff projections yet in {sess.predictions} (needs MODEL=live.quali.model)")
        if not a.follow:
            break
        try:
            time.sleep(a.follow)
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
