"""Full-screen race dashboard: live standings and gaps on the left, predictions (battles) on the right.

Usage (from the repo root, in a second terminal next to ./run_session.sh; works the same for a replay):
    python -m live.race.watch 2026 Singapore race
    python -m live.race.watch 2026 Singapore replay --once      # print one frame and exit

Reads events.csv (standings, gaps, tyres) and predictions.csv (from live.race.model) of the session. The screen
redraws whenever either file changes; Ctrl-C quits.
"""
import argparse, os, sys, time
import pandas as pd
from rich.console import Console, Group
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from f1live.events import prepare
from f1live.paths import add_session_args, session_from_args
from f1live.tail import CsvTail
from f1live.watch import read
from live.race.battles import HORIZON, THRESH, latest

TYRE = {"SOFT": ("S", "red"), "MEDIUM": ("M", "yellow"), "HARD": ("H", "white"), "INTERMEDIATE": ("I", "green"),
        "WET": ("W", "blue")}
TRACK = {"AllClear": ("GREEN", "green"), "Yellow": ("YELLOW", "yellow"), "SCDeployed": ("SAFETY CAR", "bold yellow"),
         "VSCDeployed": ("VSC", "yellow"), "VSCEnding": ("VSC ENDING", "yellow"), "Red": ("RED FLAG", "bold red")}
MAX_BATTLES = 8


def lap_time(s):
    if pd.isna(s):
        return "-"
    m, sec = divmod(float(s), 60)
    return f"{int(m)}:{sec:06.3f}"


def gap_text(r):
    if r["out"]:
        return "OUT"
    if r["position"] == 1:
        return "LEADER"
    if pd.notna(r["laps_down"]) and r["laps_down"] > 0:
        return f"+{int(r['laps_down'])} L"
    return "-" if pd.isna(r["gap_to_leader_s"]) else f"+{r['gap_to_leader_s']:.1f}"


def interval_text(r):
    if r["out"] or r["position"] == 1:
        return ""
    if pd.isna(r["interval_s"]):
        return "-"
    return f"+{r['interval_s']:.1f}"


def standings_table(history):
    lat = latest(history)
    if lat.empty:
        return Text("waiting for timing data ...", style="dim")
    last_lap = history[history["event"] == "lap"].groupby("driver")["lap_time_s"].last()
    t = Table(box=None, pad_edge=False, header_style="bold dim", expand=True)
    for name, just in (("P", "right"), ("#", "right"), ("DRV", "left"), ("GAP", "right"), ("INT", "right"),
                       ("TYRE", "left"), ("PIT", "right"), ("LAST", "right"), ("BEST", "right")):
        t.add_column(name, justify=just, no_wrap=True)
    for drv, r in lat.iterrows():
        if pd.isna(r["position"]):
            continue
        tyre = TYRE.get(r["compound"])
        tyre_cell = Text(f"{tyre[0]} {int(r['tyre_laps'])}" if tyre and pd.notna(r["tyre_laps"]) else "-",
                         style=tyre[1] if tyre else "")
        close = pd.notna(r["interval_s"]) and r["interval_s"] <= THRESH and not r["out"] and r["position"] > 1
        style = "dim" if r["out"] else None
        pit = "" if pd.isna(r["n_pit_stops"]) else str(int(r["n_pit_stops"]))
        if r["in_pit"] is True or (pd.notna(r["in_pit"]) and bool(r["in_pit"])):
            pit = "PIT"
        t.add_row(str(int(r["position"])), drv, Text(str(r["tla"]), style="bold"), gap_text(r),
                  Text(interval_text(r), style="bold yellow" if close else ""), tyre_cell, pit,
                  lap_time(last_lap.get(drv)), lap_time(r["best_lap_s"]), style=style)
    return t


def ordinal(n):
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def battles_panel(preds, max_lines=30):
    """Right panel: 'Coming up' (predicted battles, one line per car) above 'In range now' (one line per battle)."""
    if preds is None or preds.empty:
        return Text("waiting for predictions ...", style="dim")
    batch = preds[preds["ts"] == preds["ts"].max()]
    status = batch[batch["kind"] == "status"]
    if len(status):
        return Text(str(status["note"].iloc[0]), style="dim italic")
    battles = batch[batch["kind"] == "battle"].sort_values(["in_laps", "position"])
    soon, now = battles[~battles["now"].astype(bool)], battles[battles["now"].astype(bool)]
    now_budget = min(len(now) + 1, 8) if len(now) else 0
    lines, used = [], 0
    if len(soon):
        lines.append(Text("COMING UP", style="bold underline"))
        used = 1
    for _, b in soon.iterrows():
        drivers, gaps, laps = b["drivers"].split(), b["gaps"].split(), b["laps"].split()
        if used + len(drivers) + 2 > max_lines - now_budget:
            break
        n = int(float(b["in_laps"]))
        head = Text(f"Battle for {ordinal(int(b['position']))}", style="bold")
        head.append(f"   in ~{n} lap{'s' if n != 1 else ''}", style="bold yellow")
        lines += [head, Text(f"  P{int(b['position']):<2} {drivers[0]}", style="bold")]
        for i, (d, g, k) in enumerate(zip(drivers[1:], gaps, laps), 1):
            row = Text(f"  P{int(b['position']) + i:<2} {d}", style="bold")
            row.append(f"   +{float(g):.1f} s  ", style="dim")
            row.append("in range" if float(k) == 0 else f"~{int(float(k))} laps", style="red" if float(k) == 0 else "yellow")
            lines.append(row)
        lines.append(Text(""))
        used += len(drivers) + 2
    if len(now):
        lines.append(Text("IN RANGE NOW", style="bold red underline"))
        for _, b in now.head(max(now_budget - 1, 1)).iterrows():
            drivers, gaps, laps = b["drivers"].split(), b["gaps"].split(), b["laps"].split()
            row = Text(f"P{int(b['position']):<2} {drivers[0]}", style="bold")
            for d, g, k in zip(drivers[1:], gaps, laps):
                inside = float(k) == 0
                row.append(f" {float(g):.1f}" + ("" if inside else f"s ~{int(float(k))}L") + " ", style="red" if inside else "yellow")
                row.append(d, style="bold")
            lines.append(row)
    return Group(*lines) if lines else Text("no battles expected", style="dim")


def header(history, preds, sess):
    lap = history[history["race_lap"].notna()].tail(1)
    race_lap = int(lap["race_lap"].iloc[0]) if len(lap) else None
    total = int(lap["total_laps"].iloc[0]) if len(lap) and pd.notna(lap["total_laps"].iloc[0]) else None
    track = history["track_status"].dropna().iloc[-1] if history["track_status"].notna().any() else None
    label, style = TRACK.get(track, (str(track or "-").upper(), ""))
    t = Text()
    t.append(f" {sess} ", style="bold reverse")
    t.append("   LAP ", style="dim")
    t.append(f"{race_lap if race_lap is not None else '-'}", style="bold")
    t.append(f"/{total}" if total else "", style="dim")
    t.append("    ")
    t.append(f" {label} ", style=f"{style} reverse" if style else "reverse")
    t.append(f"    {history['ts'].max():%H:%M:%S} UTC", style="dim")
    return t


def render(history, preds, sess="", height=40):
    """One frame as a rich renderable. history: events.csv rows, preds: predictions.csv (or None)."""
    layout = Layout()
    layout.split_column(Layout(name="head", size=1), Layout(name="body"))
    layout["body"].split_row(Layout(name="left", ratio=3), Layout(name="right", ratio=2))
    layout["head"].update(header(history, preds, sess) if history is not None and len(history) else Text(f" {sess}  waiting ..."))
    layout["left"].update(Panel(standings_table(history) if history is not None and len(history) else Text("waiting ..."),
                                title="Standings", title_align="left"))
    layout["right"].update(Panel(battles_panel(preds, max_lines=max(height - 6, 8)), title="Battles (pace-based)", title_align="left",
                                 subtitle=f"within {THRESH:.0f} s, next {HORIZON} laps", subtitle_align="left"))
    return layout


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_session_args(ap)
    ap.add_argument("--interval", type=float, default=0.5, help="seconds between file checks")
    ap.add_argument("--once", action="store_true", help="print one frame and exit")
    args = ap.parse_args()
    sess = session_from_args(args)
    tail, history = CsvTail(sess.events, prepare), None

    def poll():
        nonlocal history
        new = tail.read()
        if isinstance(new, str):
            history = None
        elif new is not None:
            history = new if history is None else pd.concat([history, new], ignore_index=True)

    if args.once or not sys.stdout.isatty():
        poll()
        console = Console(width=args.once and 140 or None, height=50 if args.once else None)
        console.print(render(history, read(sess.predictions), str(sess), console.height) if history is not None
                      else Text(f"{sess}: no events yet"))
        return
    console = Console()
    stamp = None
    from rich.live import Live
    with Live(console=console, screen=True, auto_refresh=False) as live:
        try:
            while True:
                poll()
                now = tuple(os.stat(p).st_mtime_ns if p.exists() else None for p in (sess.events, sess.predictions))
                if now != stamp or console.size != getattr(live, "_size", None):
                    stamp, live._size = now, console.size
                    live.update(render(history, read(sess.predictions), str(sess), console.height) if history is not None
                                else Text(f" {sess}: waiting for events.csv ..."), refresh=True)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
