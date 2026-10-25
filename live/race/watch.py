"""Full-screen race dashboard: live standings and gaps on the left, predictions (battles) on the right.

Usage (from the repo root, in a second terminal next to ./run_session.sh; works the same for a replay):
    python -m live.race.watch 2026 Singapore race
    python -m live.race.watch 2026 Singapore replay --once      # print one frame and exit

Reads events.csv (standings, gaps, tyres), predictions.csv (from live.race.model) and, if f1live.telemetry is
running, telemetry.csv (where each car is on track) of the session. The screen redraws whenever a file changes;
Ctrl-C quits.

Columns: standings | projected finishing order (win / podium / points chances from a Monte Carlo of the rest of the
race, live.race.projection) | battles; a narrower screen stacks the last two on the right.

Bottom strip: one line for the lap with a dot per car at its position along the lap, so the gaps between cars can be
seen (live.race.track). Without telemetry the positions are estimated from lap timing.
"""
import argparse, os, sys, time
import numpy as np
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
from live.race.pace import driver_pace
from live.race.track import TrackProgress, build_reference, load_reference, timing_progress

TYRE = {"SOFT": ("S", "red"), "MEDIUM": ("M", "yellow"), "HARD": ("H", "white"), "INTERMEDIATE": ("I", "green"),
        "WET": ("W", "blue")}
TRACK = {"AllClear": ("GREEN", "green"), "Yellow": ("YELLOW", "yellow"), "SCDeployed": ("SAFETY CAR", "bold yellow"),
         "VSCDeployed": ("VSC", "yellow"), "VSCEnding": ("VSC ENDING", "yellow"), "Red": ("RED FLAG", "bold red")}
MAX_BATTLES = 8
TEAM_COLOUR = {"Red Bull Racing": "#3671C6", "Ferrari": "#E8002D", "Mercedes": "#27F4D2", "McLaren": "#FF8000",
               "Aston Martin": "#229971", "Alpine": "#FF87BC", "Williams": "#64C4FF", "Racing Bulls": "#6692FF",
               "Kick Sauber": "#52E252", "Audi": "#F50537", "Haas F1 Team": "#B6BABD", "Cadillac": "#C8AA4B"}
TRACK_ROWS = 4          # rows of driver labels above the track line


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
    rows = preds[preds["kind"].isin(["battle", "status"])]
    if rows.empty:
        return Text("waiting for predictions ...", style="dim")
    batch = rows[rows["ts"] == rows["ts"].max()]
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


def pct(p):
    """Probability as a table cell: '-' for under half a percent, '100' for the rest of the way up."""
    if pd.isna(p) or p < 0.005:
        return "-"
    return "100" if p >= 0.995 else f"{p * 100:.0f}"


def prob_style(p, strong="green"):
    return f"bold {strong}" if p >= 0.5 else (strong if p >= 0.1 else "dim")


def projection_panel(preds, history=None, max_rows=22):
    """Projected finishing order from the Monte Carlo (kind='projection' rows): rank, change vs now, win / podium /
    points probabilities and the expected number of stops still to come."""
    rows = preds[preds["kind"] == "projection"] if preds is not None and "kind" in preds else None
    if rows is None or rows.empty:
        return Text("waiting for the first projection (needs a few clean laps) ...", style="dim")
    status = preds[preds["kind"] == "status"]
    if len(status) and status.iloc[-1]["ts"] >= rows["ts"].max() and status.iloc[-1]["note"] == "race finished":
        return Text("race finished: the standings are the result", style="dim italic")
    rows = rows[rows["ts"] == rows["ts"].max()].sort_values("proj_rank")
    now = {}
    if history is not None and len(history):
        lat = latest(history)
        run = lat[~lat["out"] & lat["position"].notna()].sort_values("position")
        now = {d: i + 1 for i, d in enumerate(run.index)}                  # rank among the cars still running
    t = Table(box=None, pad_edge=False, header_style="bold dim", expand=True)
    for name, just in (("#", "right"), ("DRV", "left"), ("NOW", "right"), ("WIN%", "right"), ("PODIUM%", "right"),
                       ("PTS%", "right"), ("STOPS", "right")):
        t.add_column(name, justify=just, no_wrap=True)
    for _, r in rows.head(max_rows).iterrows():
        cur = now.get(str(r["driver"]), None)
        move = Text("")
        if cur is not None:
            d = cur - int(r["proj_rank"])
            move = Text(f"▲{d}", style="green") if d > 0 else Text(f"▼{-d}", style="red") if d < 0 else Text("=", style="dim")
        t.add_row(str(int(r["proj_rank"])), Text(str(r["tla"]), style="bold"), move,
                  Text(pct(r["p_win"]), style=prob_style(r["p_win"])), Text(pct(r["p_podium"]), style=prob_style(r["p_podium"])),
                  Text(pct(r["p_points"]), style=prob_style(r["p_points"], "cyan")), f"{r['stops_left']:.1f}")
    first = rows.iloc[0]
    note = f"{int(first['laps_left'])} laps left  ·  safety car / VSC chance {first['p_sc'] * 100:.0f}%"
    if isinstance(first.get("note"), str) and first["note"]:
        note += f"  ·  {first['note']}"
    return Group(Text(note, style="dim"), t)


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


class TrackLine:
    """One line for the lap (start/finish at both ends) with a dot per car and its code above, staggered over
    TRACK_ROWS rows so cars close together stay readable (a label that can't be placed is dropped, its dot stays)."""
    def __init__(self, cars):
        self.cars = cars            # [(code, progress 0-1, style, is_leader)]

    def __rich_console__(self, console, options):
        W = max(options.max_width, 20)
        H = TRACK_ROWS + 2
        grid = [[(" ", "")] * W for _ in range(H)]
        line = TRACK_ROWS
        grid[line] = [("─", "dim")] * W
        grid[line][0], grid[line][-1] = ("├", "dim"), ("┤", "dim")
        for frac in (0.25, 0.5, 0.75):
            grid[line][int(frac * (W - 1))] = ("┼", "dim")
        for frac, label in ((0, "S/F"), (0.25, "25%"), (0.5, "50%"), (0.75, "75%")):
            x = int(frac * (W - 1))
            for k, ch in enumerate(label):
                if x + k < W:
                    grid[line + 1][x + k] = (ch, "dim")
        for k, ch in enumerate("S/F"):
            grid[line + 1][W - 3 + k] = (ch, "dim")
        row_end = [-2] * TRACK_ROWS
        placed = []
        for code, frac, style, leader in sorted(self.cars, key=lambda c: c[1]):
            x = min(int(frac * (W - 1)), W - 1)
            lx = min(x, W - len(code))
            r = next((r for r in range(TRACK_ROWS) if lx > row_end[r]), None)
            if r is not None:
                row_end[r] = lx + len(code)
            placed.append((code, x, lx, r, style, leader))
        for code, x, lx, r, style, leader in placed:               # stems first, labels and dots over them
            for rr in range(0 if r is None else 0, 0 if r is None else r):
                if grid[line - 1 - rr][x][0] == " ":
                    grid[line - 1 - rr][x] = ("│", "dim")
        for code, x, lx, r, style, leader in placed:
            grid[line][x] = ("◆" if leader else "●", f"bold {style}" if style else "bold")
            if r is not None:
                for k, ch in enumerate(code):
                    grid[line - 1 - r][lx + k] = (ch, f"bold {style}" if style else "bold")
        for row in grid:
            t = Text()
            for ch, st in row:
                t.append(ch, style=st or None)
            yield t


def track_cars(history, progress):
    """[(code, progress, colour, is_leader)] for the cars still running that have a progress."""
    lat = latest(history)
    cars = []
    for drv, frac in progress.items():
        if drv not in lat.index or lat.loc[drv, "out"]:
            continue
        r = lat.loc[drv]
        cars.append((str(r["tla"]), frac, TEAM_COLOUR.get(r.get("team"), "white"), r["position"] == 1))
    return cars


class Feed:
    """Follows the session's files and keeps what the dashboard needs: events, the track reference and each car's
    latest position."""
    TEL_STALE = 15          # s of event time after which telemetry no longer counts as live

    def __init__(self, sess):
        self.sess = sess
        self.events = CsvTail(sess.events, prepare)
        self.tel = CsvTail(sess.telemetry, prepare_telemetry)
        self.history = None
        self.buffer = []                  # telemetry kept only until the reference lap exists
        self.pos = {}                     # driver -> (ts, x, y)
        self.tracker = None
        ref = load_reference(sess.dir.parent / "track.csv")
        if ref is not None:
            self.tracker = TrackProgress(ref)
        self._last_try = 0.0

    def poll(self):
        new = self.events.read()
        if isinstance(new, str):
            self.history = None
        elif new is not None:
            self.history = new if self.history is None else pd.concat([self.history, new], ignore_index=True)
        fresh = self.tel.read()
        if isinstance(fresh, str):
            self.pos, self.buffer = {}, []
        elif fresh is not None:
            ok = fresh.dropna(subset=["x", "y"])
            for drv, r in ok.groupby("driver").tail(1).set_index("driver").iterrows():
                self.pos[drv] = (r["ts"], r["x"], r["y"])
            if self.tracker is None:
                self.buffer.append(ok[["ts", "driver", "x", "y"]])
        if self.tracker is None and self.buffer and self.history is not None and time.time() - self._last_try > 10:
            self._last_try = time.time()
            ref = build_reference(pd.concat(self.buffer, ignore_index=True), self.history)
            if ref is not None:
                self.tracker, self.buffer = TrackProgress(ref), []

    def progress(self):
        """-> ({driver: progress 0-1}, 'telemetry' | 'timing' | None)."""
        h = self.history
        if h is None or h.empty:
            return {}, None
        now = h["ts"].max()
        if self.tracker is not None and self.pos:
            live = {d: p for d, p in self.pos.items() if (now - p[0]).total_seconds() <= self.TEL_STALE}
            if live:
                return {d: self.tracker.progress(d, x, y) for d, (_, x, y) in live.items()}, "telemetry"
        lat = latest(h)
        if lat.empty:
            return {}, None
        pace = driver_pace(h)["pace_s"]
        last = h[h["event"] == "lap"].groupby("driver")["lap_time_s"].last()
        return timing_progress(lat, now, pace.where(pace.notna(), last)), "timing"


def prepare_telemetry(df):
    df["ts"] = pd.to_datetime(df["ts"], format="ISO8601")
    df["driver"] = df["driver"].astype(int).astype(str)
    return df


def render(history, preds, sess="", height=40, progress=None, width=140):
    """One frame as a rich renderable. history: events.csv rows, preds: predictions.csv (or None),
    progress: ({driver: 0-1 lap progress}, 'telemetry' | 'timing' | None) for the track line.
    A wide screen gets three columns (standings | projected finish | battles); otherwise the right column is split."""
    layout = Layout()
    layout.split_column(Layout(name="head", size=1), Layout(name="body"), Layout(name="foot", size=TRACK_ROWS + 4))
    ok = history is not None and len(history)
    proj_panel = Panel(projection_panel(preds, history), title="Projected finish", title_align="left",
                       subtitle="Monte Carlo of the rest of the race", subtitle_align="left")
    battle_panel = Panel(battles_panel(preds, max_lines=max(height - 6 if width >= 150 else (height - 14) // 2, 8)),
                         title="Battles (pace-based)", title_align="left",
                         subtitle=f"within {THRESH:.0f} s, next {HORIZON} laps", subtitle_align="left")
    stand_panel = Panel(standings_table(history) if ok else Text("waiting ..."), title="Standings", title_align="left")
    if width >= 150:
        layout["body"].split_row(Layout(stand_panel, name="left", ratio=5), Layout(proj_panel, name="mid", ratio=4),
                                 Layout(battle_panel, name="right", ratio=4))
    else:
        layout["body"].split_row(Layout(stand_panel, name="left", ratio=3), Layout(name="right", ratio=2))
        layout["right"].split_column(Layout(proj_panel, name="proj", ratio=3), Layout(battle_panel, name="battles", ratio=2))
    layout["head"].update(header(history, preds, sess) if ok else Text(f" {sess}  waiting ..."))
    prog, mode = progress or ({}, None)
    cars = track_cars(history, prog) if ok and prog else []
    note = {"telemetry": "from car telemetry", "timing": "estimated from lap timing"}.get(mode, "waiting for positions")
    layout["foot"].update(Panel(TrackLine(cars) if cars else Text("no positions yet", style="dim"),
                                title=f"Lap position ({note})", title_align="left",
                                subtitle="◆ leader", subtitle_align="right"))
    return layout


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_session_args(ap)
    ap.add_argument("--interval", type=float, default=0.5, help="seconds between file checks")
    ap.add_argument("--once", action="store_true", help="print one frame and exit")
    args = ap.parse_args()
    sess = session_from_args(args)
    feed = Feed(sess)

    def frame(height, width):
        return render(feed.history, read(sess.predictions), str(sess), height, feed.progress(), width)

    if args.once or not sys.stdout.isatty():
        feed.poll()
        console = Console(width=args.once and 170 or None, height=50 if args.once else None)
        console.print(frame(console.height, console.width) if feed.history is not None else Text(f"{sess}: no events yet"))
        return
    console = Console()
    stamp = None
    from rich.live import Live
    with Live(console=console, screen=True, auto_refresh=False) as live:
        try:
            while True:
                feed.poll()
                now = tuple(os.stat(p).st_mtime_ns if p.exists() else None
                            for p in (sess.events, sess.predictions, sess.telemetry))
                if now != stamp or console.size != getattr(live, "_size", None):
                    stamp, live._size = now, console.size
                    live.update(frame(console.height, console.width) if feed.history is not None
                                else Text(f" {sess}: waiting for events.csv ..."), refresh=True)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
