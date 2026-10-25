"""Full-screen race dashboard: live standings and gaps on the left, predictions (battles) on the right.

Usage (from the repo root, in a second terminal next to ./run_session.sh; works the same for a replay):
    python -m live.race.watch 2026 Singapore race
    python -m live.race.watch 2026 Singapore replay --once      # print one frame and exit

Delay: a TV broadcast is usually behind the live timing. --delay SECONDS shows the race as it was that long ago (standings,
gaps, battles, projection and the lap position line), and the keys ] [ (1 s) and } { (10 s) change it while you watch, so you can
nudge it until the screen matches the picture; 0 goes back to live. The delay is remembered for next time. The runner and the
recording are not delayed: only what is shown.

Press `s` to swap the projection for the finish if every driver follows the strategy the model finds best for him, which needs
`python -m play.field live YEAR RACE SESSION --loop` running in another terminal (it writes strategy.csv). It is a scenario,
not a forecast: in the backtest it was no more accurate than the projection with the usual stop plans.

Reads events.csv (standings, gaps, tyres), predictions.csv (from live.race.model) and, if f1live.telemetry is
running, telemetry.csv (where each car is on track) of the session. The screen redraws whenever a file changes;
Ctrl-C quits.

Columns: standings | projected finishing order (win / podium / points chances from a Monte Carlo of the rest of the
race, live.race.projection) | battles; a narrower screen stacks the last two on the right.

Bottom strip: one line for the lap with a dot per car at its position along the lap, so the gaps between cars can be
seen (live.race.track). Without telemetry the positions are estimated from lap timing.
"""
import argparse, os, sys, time
from pathlib import Path
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
from live.race.hazards import BANNERS, SpeedProfile, hazards, neutral_at, track_state
from live.race.pace import driver_pace
from live.race.track import (TrackProgress, build_reference, load_corners, load_reference, nearest_progress,
                             sector_fractions, timing_progress)
from live.race.weather import wet_now

TYRE = {"SOFT": ("S", "red"), "MEDIUM": ("M", "yellow"), "HARD": ("H", "white"), "INTERMEDIATE": ("I", "green"),
        "WET": ("W", "blue")}
TRACK = {"AllClear": ("GREEN", "green"), "Yellow": ("YELLOW", "yellow"), "SCDeployed": ("SAFETY CAR", "bold yellow"),
         "VSCDeployed": ("VSC", "yellow"), "VSCEnding": ("VSC ENDING", "yellow"), "Red": ("RED FLAG", "bold red")}
SECTOR_COLOUR = ("cyan", "magenta", "yellow")
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
    battles = batch[batch["kind"] == "battle"].dropna(subset=["drivers", "gaps", "laps", "position"])      # (a row cut short is skipped)
    battles = battles.sort_values(["in_laps", "position"])
    soon, now = battles[~battles["now"].astype(bool)], battles[battles["now"].astype(bool)]
    now_budget = min(len(now) + 1, 8) if len(now) else 0
    lines, used = [], 0
    if len(soon):
        lines.append(Text("COMING UP", style="bold underline"))
        used = 1
    for _, b in soon.iterrows():
        drivers, gaps, laps = str(b["drivers"]).split(), str(b["gaps"]).split(), str(b["laps"]).split()
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
            drivers, gaps, laps = str(b["drivers"]).split(), str(b["gaps"]).split(), str(b["laps"]).split()
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
        paused = preds[preds["kind"] == "projection_status"] if preds is not None and "kind" in preds else None
        if paused is not None and len(paused):
            return Text(str(paused.iloc[-1]["note"]), style="bold blue")
        return Text("waiting for the first projection (needs a few clean laps) ...", style="dim")
    status = preds[preds["kind"] == "status"]
    if len(status) and status.iloc[-1]["ts"] >= rows["ts"].max() and status.iloc[-1]["note"] == "race finished":
        return Text("race finished: the standings are the result", style="dim italic")
    paused = preds[preds["kind"] == "projection_status"]
    if len(paused) and paused["ts"].max() >= rows["ts"].max():
        return Text(str(paused.iloc[-1]["note"]), style="bold blue")
    rows = rows[rows["ts"] == rows["ts"].max()].dropna(subset=["tla", "proj_rank", "p_win"]).sort_values("proj_rank")
    if rows.empty:                                           # the latest rows were cut short by a half-written file
        return Text("waiting for the first projection (needs a few clean laps) ...", style="dim")
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


def strategy_panel(strategy, max_rows=22):
    """The projected finish with every driver on the model's best plan (strategy.csv from `python -m play.field live ... --loop`):
    rank, current position, plan, next stop, win / podium chances and expected points. A scenario, not the forecast."""
    if strategy is None or strategy.empty:
        return Text("no strategy.csv yet. Start it in another terminal:\n  python -m play.field live YEAR RACE SESSION --loop",
                    style="dim")
    s = strategy.sort_values("best_pos").head(max_rows)
    t = Table(box=None, pad_edge=False, header_style="bold dim", expand=True)
    for name, just in (("#", "right"), ("DRV", "left"), ("NOW", "right"), ("PLAN", "left"), ("NEXT", "left"), ("WIN%", "right"),
                       ("PODIUM%", "right"), ("PTS", "right")):
        t.add_column(name, justify=just, no_wrap=True)
    for i, (_, r) in enumerate(s.iterrows(), 1):
        tyre = TYRE.get(r["next_compound"]) if isinstance(r["next_compound"], str) else None
        nxt = Text(f"L{int(r['next_lap'])} {tyre[0]}", style=tyre[1]) if tyre and pd.notna(r["next_lap"]) else Text("no stop", style="dim")
        t.add_row(str(i), Text(str(r["tla"]), style="bold"), str(int(r["now"])), str(r["plan"]), nxt,
                  Text(pct(r["best_win"]), style=prob_style(r["best_win"])), Text(pct(r["best_podium"]), style=prob_style(r["best_podium"])),
                  f"{r['best_pts']:.1f}")
    first = s.iloc[0]
    note = f"as of lap {int(first['race_lap'])}: every car on the plan the model finds best for it"
    return Group(Text(note, style="dim"), t)


def header(history, preds, sess, delay=0.0):
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
    if wet_now(history):
        t.append("  ")
        t.append(" RAIN ", style="bold blue reverse")
    t.append(f"    {history['ts'].max():%H:%M:%S} UTC", style="dim")
    if delay:
        t.append("   ")
        t.append(f" DELAY {delay:g} s ", style="bold cyan reverse")
        t.append("  [ ] 1 s   { } 10 s   0 live", style="dim")
    return t


class TrackLine:
    """One line for the lap (start/finish at both ends) with a dot per car and its code above, staggered over
    TRACK_ROWS rows so cars close together stay readable (a label that can't be placed is dropped, its dot stays).
    The line is coloured by sector (sectors: the lap fractions where S1 and S2 end) and corners (corners:
    [(number, fraction)]) are numbered under it, as many as fit.
    A car with a flag ('stopped' or 'slow', from live.race.hazards) gets ✖ / ▼ instead of its dot and a red / yellow label,
    which blinks while `flash` is set; flagged cars are labelled first so a crowd never hides them."""
    def __init__(self, cars, corners=(), sectors=None, flash=False):
        # [(code, progress 0-1, style, is_leader[, flag])], flag None | 'stopped' | 'slow'
        self.cars = [c if len(c) == 5 else (*c, None) for c in cars]
        self.corners = corners
        self.sectors = sectors
        self.flash = flash

    def __rich_console__(self, console, options):
        W = max(options.max_width, 20)
        H = TRACK_ROWS + 2
        grid = [[(" ", "")] * W for _ in range(H)]
        line = TRACK_ROWS
        bounds = [0, *(self.sectors or []), 1]
        for x in range(W):
            sector = sum(x / (W - 1) >= b for b in bounds[1:-1]) if self.sectors else None
            grid[line][x] = ("─", SECTOR_COLOUR[sector] if self.sectors else "dim")
        for frac in self.sectors or []:
            grid[line][int(frac * (W - 1))] = ("┃", "bold")
        grid[line][0], grid[line][-1] = ("├", "dim"), ("┤", "dim")
        for k, ch in enumerate("S/F"):
            grid[line + 1][k] = (ch, "dim")
            grid[line + 1][W - 3 + k] = (ch, "dim")
        free = 4                                                # first column a corner label may use (after "S/F ")
        for label, frac in self.corners:
            x = int(frac * (W - 1))
            if grid[line][x][0] in "─":
                grid[line][x] = ("┼", grid[line][x][1])
            if x >= free and x + len(label) <= W - 4:           # numbers that would collide are dropped
                for k, ch in enumerate(label):
                    grid[line + 1][x + k] = (ch, "dim")
                free = x + len(label) + 1
        row_end = [-2] * TRACK_ROWS
        placed = []
        for code, frac, style, leader, flag in sorted(self.cars, key=lambda c: (c[4] is None, c[1])):
            x = min(int(frac * (W - 1)), W - 1)
            lx = min(x, W - len(code))
            r = next((r for r in range(TRACK_ROWS) if lx > row_end[r]), None)
            if r is not None:
                row_end[r] = lx + len(code)
            placed.append((code, x, lx, r, style, leader, flag))
        for code, x, lx, r, style, leader, flag in placed:         # stems first, labels and dots over them
            for rr in range(0 if r is None else 0, 0 if r is None else r):
                if grid[line - 1 - rr][x][0] == " ":
                    grid[line - 1 - rr][x] = ("│", "dim")
        for code, x, lx, r, style, leader, flag in placed:
            mark = "◆" if leader else "●"
            label = f"bold {style}" if style else "bold"
            if flag == "stopped":
                mark, label = "✖", "bold white on red" if self.flash else "bold red"
            elif flag == "slow":
                mark, label = "▼", "bold black on yellow" if self.flash else "bold yellow"
            grid[line][x] = (mark, label if flag else (f"bold {style}" if style else "bold"))
            if r is not None:
                for k, ch in enumerate(code):
                    grid[line - 1 - r][lx + k] = (ch, label)
        for row in grid:
            t = Text()
            for ch, st in row:
                t.append(ch, style=st or None)
            yield t


def sector_legend(sectors):
    """'S1 S2 S3' in the sector colours (empty without sector times)."""
    t = Text()
    for i, colour in enumerate(SECTOR_COLOUR if sectors else ()):
        t.append(f"S{i + 1}", style=f"bold {colour}")
        t.append("  ")
    return t


def track_cars(history, progress, flags=None):
    """[(code, progress, colour, is_leader, flag)] for the cars still running that have a progress. flags: {driver: 'stopped' |
    'slow'} (live.race.hazards); a car stopped on the track stays on the line even if it has been marked out."""
    lat = latest(history)
    flags = flags or {}
    cars = []
    for drv, frac in progress.items():
        if drv not in lat.index or (lat.loc[drv, "out"] and flags.get(drv) != "stopped"):
            continue
        r = lat.loc[drv]
        cars.append((str(r["tla"]), frac, TEAM_COLOUR.get(r.get("team"), "white"), r["position"] == 1, flags.get(drv)))
    return cars


def alert_line(history, flags, flash=False):
    """The line above the lap position: a blinking banner for a safety car / VSC / red flag, then who has stopped or crawls."""
    t = Text(no_wrap=True, overflow="ellipsis")
    state = BANNERS.get(track_state(history))
    if state:
        red = state == "RED FLAG"
        on, off = ("bold white on red", "bold red") if red else ("bold black on yellow", "bold yellow")
        t.append(f" ⚠ {state} ⚠ ", style=on if flash else off)
    lat = latest(history) if history is not None and len(history) else None
    names = {kind: [] for kind in ("stopped", "slow")}
    for drv, kind in (flags or {}).items():
        names[kind].append(str(lat.loc[drv, "tla"]) if lat is not None and drv in lat.index else drv)
    for kind, mark, label, on, off in (("stopped", "✖", "stopped on track", "bold red", "red"),
                                       ("slow", "▼", "slow on track", "bold yellow", "yellow")):
        if names[kind]:
            t.append(f"  {mark} {' '.join(sorted(names[kind]))} {label}", style=on if flash else off)
    return t


class Feed:
    """Follows the session's files and keeps what the dashboard needs: events, the track reference and each car's
    latest position."""
    TEL_STALE = 15          # s of event time after which telemetry no longer counts as live
    TRAIL_S = 300           # how far back positions are kept: the longest delay the viewer can show

    def __init__(self, sess):
        self.sess = sess
        self.events = CsvTail(sess.events, prepare)
        self.tel = CsvTail(sess.telemetry, prepare_telemetry)
        self.history = None
        self.buffer = []                  # telemetry kept only until the reference lap exists
        self.pos = {}                     # driver -> (ts, x, y, speed)
        self.trail = {}                   # driver -> [(ts, x, y, speed), ...] the last TRAIL_S seconds, to show the race as it was
        self.profile = SpeedProfile()     # what speed is normal where on the lap, to spot a car crawling (live.race.hazards)
        self.strategy, self._strategy_stamp = None, None      # strategy.csv from play.field --loop
        self.tracker = None
        self.corners = load_corners(sess.dir.parent / "corners.csv")
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
        self.poll_strategy()
        fresh = self.tel.read()
        if isinstance(fresh, str):
            self.pos, self.trail, self.buffer = {}, {}, []
        elif fresh is not None:
            ok = fresh.dropna(subset=["x", "y"])
            for drv, r in ok.groupby("driver").tail(1).set_index("driver").iterrows():
                self.pos[drv] = (r["ts"], r["x"], r["y"], r["speed"])
            for drv, g in ok.groupby("driver"):
                self.trail.setdefault(drv, []).extend(zip(g["ts"], g["x"], g["y"], g["speed"]))
            self._learn_speeds(ok)
            newest = ok["ts"].max() if len(ok) else None
            if newest is not None:
                cutoff = newest - pd.Timedelta(self.TRAIL_S, unit="s")
                for drv, samples in self.trail.items():
                    if samples and samples[0][0] < cutoff:
                        self.trail[drv] = [s for s in samples if s[0] >= cutoff]
            if self.tracker is None:
                self.buffer.append(ok[["ts", "driver", "x", "y"]])
        if self.tracker is None and self.buffer and self.history is not None and time.time() - self._last_try > 10:
            self._last_try = time.time()
            ref = build_reference(pd.concat(self.buffer, ignore_index=True), self.history)
            if ref is not None:
                self.tracker, self.buffer = TrackProgress(ref), []

    def _learn_speeds(self, rows):
        """Teach the speed profile from green-flag running (every second sample is plenty)."""
        if self.tracker is None or self.history is None:
            return
        rows = rows.iloc[::2]
        rows = rows[rows["speed"] >= 30]
        if rows.empty:
            return
        rows = rows[~neutral_at(self.history, rows["ts"])]
        if len(rows):
            for p, v in zip(nearest_progress(self.tracker.ref, rows["x"], rows["y"]), rows["speed"]):
                self.profile.add(p, v)

    def hazards(self, history, at):
        """{driver: 'stopped' | 'slow'} as of event time `at` (see live.race.hazards)."""
        if at is None or history is None:
            return {}
        progress_of = None
        if self.tracker is not None:
            ref = self.tracker.ref
            progress_of = lambda x, y: float(nearest_progress(ref, [x], [y])[0])
        return hazards(self.trail, history, at, progress_of, self.profile)

    def sectors(self, history=None):
        """[end of S1, end of S2] as lap fractions from the sector times so far, or None."""
        h = self.history if history is None else history
        return None if h is None else sector_fractions(h)

    def poll_strategy(self):
        path = self.sess.dir / "strategy.csv"
        try:
            stamp = os.stat(path).st_mtime_ns
        except FileNotFoundError:
            return
        if stamp != self._strategy_stamp:
            try:
                df = pd.read_csv(path)
                df["ts"] = pd.to_datetime(df["ts"], format="ISO8601")
                self.strategy, self._strategy_stamp = df, stamp
            except (pd.errors.EmptyDataError, ValueError, KeyError):
                pass                                          # caught mid-write: try again next time

    def head(self):
        """The newest moment we have data for (event time), or None."""
        stamps = [self.history["ts"].max()] if self.history is not None and len(self.history) else []
        stamps += [samples[-1][0] for samples in self.trail.values() if samples]
        return max(stamps) if stamps else None

    def position_at(self, driver, at):
        """(ts, x, y, speed) of the car's last sample at or before `at`, or None."""
        best = None
        for sample in self.trail.get(driver, ()):
            if sample[0] <= at:
                best = sample
            else:
                break
        return best

    def progress(self, history=None, at=None):
        """-> ({driver: progress 0-1}, 'telemetry' | 'timing' | None), as of event time `at` (default: now) from `history`
        (default: everything so far)."""
        h = self.history if history is None else history
        if h is None or h.empty:
            return {}, None
        now = h["ts"].max() if at is None else at
        if at is None:
            positions = self.pos
        else:                                      # as it was: the latest sample of each car before `at`
            positions = {d: s for d in self.trail if (s := self.position_at(d, at)) is not None}
        if self.tracker is not None and positions:
            live = {d: p for d, p in positions.items() if (now - p[0]).total_seconds() <= self.TEL_STALE}
            if live:
                return {d: self.tracker.progress(d, x, y) for d, (_, x, y, *_) in live.items()}, "telemetry"
        lat = latest(h)
        if lat.empty:
            return {}, None
        pace = driver_pace(h)["pace_s"]
        last = h[h["event"] == "lap"].groupby("driver")["lap_time_s"].last()
        return timing_progress(lat, now, pace.where(pace.notna(), last)), "timing"


def prepare_telemetry(df):
    df["ts"] = pd.to_datetime(df["ts"], format="ISO8601")
    df["driver"] = df["driver"].astype(int).astype(str)
    if "speed" not in df:
        df["speed"] = np.nan
    return df


def render(history, preds, sess="", height=40, progress=None, width=140, corners=(), sectors=None, delay=0.0, strategy=None,
           view="usual", flags=None, flash=False):
    """One frame as a rich renderable. history: events.csv rows, preds: predictions.csv (or None),
    progress: ({driver: 0-1 lap progress}, 'telemetry' | 'timing' | None) for the track line, with its corners and
    sector boundaries (Feed.corners, Feed.sectors()). delay (seconds) is only shown in the header: the caller passes the
    history, predictions and progress as they were that long ago (see delayed_view). view="optimal" replaces the projection
    with the finish if every car follows its best strategy (`strategy`, from strategy.csv).
    flags ({driver: 'stopped' | 'slow'}, see delayed_view) are marked on the track line and named, with a safety car / VSC /
    red flag banner, in a line above it; `flash` is the blink phase (the caller flips it every half second).
    A wide screen gets three columns (standings | projected finish | battles); otherwise the right column is split."""
    layout = Layout()
    layout.split_column(Layout(name="head", size=1), Layout(name="body"), Layout(name="foot", size=TRACK_ROWS + 5))
    ok = history is not None and len(history)
    if view == "optimal":
        proj_panel = Panel(strategy_panel(strategy), title="Projected finish: everyone on his best strategy", title_align="left",
                           subtitle=Text("a scenario, not a forecast · [s] projection"), subtitle_align="left")
    else:
        proj_panel = Panel(projection_panel(preds, history), title="Projected finish", title_align="left",
                           subtitle=Text("Monte Carlo · [s] best strategies"), subtitle_align="left")
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
    layout["head"].update(header(history, preds, sess, delay) if ok else Text(f" {sess}  waiting ... (delay {delay:g} s)"))
    prog, mode = progress or ({}, None)
    cars = track_cars(history, prog, flags) if ok and prog else []
    note = {"telemetry": "from car telemetry", "timing": "estimated from lap timing"}.get(mode, "waiting for positions")
    track = TrackLine(cars, corners, sectors, flash) if cars else Text("no positions yet", style="dim")
    layout["foot"].update(Panel(Group(alert_line(history, flags, flash) if ok else Text(""), track),
                                title=f"Lap position ({note})", title_align="left",
                                subtitle=Text.assemble(sector_legend(sectors), "◆ leader"), subtitle_align="right"))
    return layout


def delayed_view(feed, preds, delay):
    """The race as it was `delay` seconds ago, for matching a delayed broadcast: events, predictions and car positions up to
    that moment. -> dict(at, history, preds, progress, sectors, strategy, flags); history is None until there is data that old."""
    head = feed.head()
    if head is None or feed.history is None:
        return dict(at=None, history=None, preds=preds, progress=({}, None), sectors=None, strategy=None, flags={})
    at = head - pd.Timedelta(float(delay), unit="s")
    hist = feed.history[feed.history["ts"] <= at] if delay else feed.history
    if hist.empty:
        return dict(at=at, history=None, preds=None, progress=({}, None), sectors=None, strategy=None, flags={})
    if preds is not None and delay:
        preds = preds[preds["ts"] <= at]
        preds = preds if len(preds) else None
    strategy = feed.strategy
    if strategy is not None:                          # the latest search that had finished by then
        strategy = strategy[strategy["ts"] <= at]
        strategy = strategy[strategy["ts"] == strategy["ts"].max()] if len(strategy) else None
    return dict(at=at, history=hist, preds=preds, progress=feed.progress(hist, at), sectors=feed.sectors(hist), strategy=strategy,
                flags=feed.hazards(hist, at))


def safe_frame(build, last_good, log_path):
    """-> (renderable, last good renderable). If drawing a frame fails, the error is appended to log_path and the last good frame
    is kept under a red one-line banner: a bug in the display should never take the dashboard down in the middle of a race."""
    try:
        frame = build()
        return frame, frame
    except Exception as e:                                   # noqa: BLE001 (anything: the point is to keep running)
        import traceback
        try:
            with open(log_path, "a") as f:
                f.write(f"--- {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M:%S} UTC\n{traceback.format_exc()}\n")
        except OSError:
            pass
        banner = Text(f" display error: {type(e).__name__}: {e}   (details in {log_path}) ", style="bold white on red")
        return (Group(banner, last_good) if last_good is not None else banner), last_good


STEPS = {"]": 1.0, "[": -1.0, "}": 10.0, "{": -10.0}


def adjust_delay(delay, key, maximum=Feed.TRAIL_S):
    """New delay in seconds after a key press: ] [ change it by 1 s, } { by 10 s, 0 goes back to live. Between 0 and `maximum`."""
    if key == "0":
        return 0.0
    return min(max(delay + STEPS.get(key, 0.0), 0.0), float(maximum))


def window_title(sess, custom=None):
    """The terminal window's title: what you asked for with --title, else the session."""
    return custom or f"F1 race dashboard · {sess}"


def delay_file(data_dir):
    return Path(data_dir) / ".watch_delay"


def load_delay(data_dir):
    """The delay used last time (broadcast delay rarely changes), 0 if none was saved."""
    try:
        return min(max(float(delay_file(data_dir).read_text().strip()), 0.0), float(Feed.TRAIL_S))
    except (OSError, ValueError):
        return 0.0


def save_delay(data_dir, delay):
    try:
        delay_file(data_dir).write_text(f"{delay:g}\n")
    except OSError:
        pass


class Keys:
    """Read single key presses without waiting for Enter (a no-op unless stdin is a terminal)."""
    def __init__(self):
        self.fd = sys.stdin.fileno() if sys.stdin.isatty() else None

    def __enter__(self):
        if self.fd is not None:
            import termios, tty
            self.old = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)                 # Ctrl-C still works
        return self

    def __exit__(self, *exc):
        if self.fd is not None:
            import termios
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def read(self, timeout):
        """The keys pressed within `timeout` seconds (waits that long if none), as a string."""
        if self.fd is None:
            time.sleep(timeout)
            return ""
        import select
        if select.select([sys.stdin], [], [], timeout)[0]:
            return os.read(self.fd, 32).decode(errors="ignore")
        return ""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_session_args(ap)
    ap.add_argument("--interval", type=float, default=0.5, help="seconds between file checks")
    ap.add_argument("--delay", type=float, help="show the race as it was this many seconds ago, to match a delayed broadcast "
                                                 "(default: what you used last time)")
    ap.add_argument("--once", action="store_true", help="print one frame and exit")
    ap.add_argument("--title", help="title of the terminal window (default: 'F1 race dashboard · <session>')")
    ap.add_argument("--view", choices=["usual", "optimal"], default="usual",
                    help="start with the projection (usual) or the finish with every driver on his best strategy (optimal); "
                         "the key s switches")
    args = ap.parse_args()
    sess = session_from_args(args)
    feed = Feed(sess)
    delay = min(max(args.delay, 0.0), float(Feed.TRAIL_S)) if args.delay is not None else load_delay(sess.data_dir)

    view = args.view
    flash = lambda: int(time.time() * 2) % 2 == 0           # blink phase for alerts: on for half a second, off for half
    alerting = [False]                                      # did the last frame have anything blinking?

    def frame(height, width, preds_all):
        v = delayed_view(feed, preds_all, delay)
        if v["history"] is None:
            return Text(f" {sess}: no data from {delay:g} s ago yet" if delay else f" {sess}: waiting for events.csv ...")
        alerting[0] = bool(v["flags"]) or track_state(v["history"]) in BANNERS
        return render(v["history"], v["preds"], str(sess), height, v["progress"], width, feed.corners, v["sectors"], delay,
                      v["strategy"], view, v["flags"], flash())

    if args.once or not sys.stdout.isatty():
        feed.poll()
        console = Console(width=args.once and 170 or None, height=50 if args.once else None)
        console.print(frame(console.height, console.width, read(sess.predictions)) if feed.history is not None
                      else Text(f"{sess}: no events yet"))
        return
    console = Console()
    console.set_window_title(window_title(sess, args.title))          # restored by your shell's prompt when you quit
    stamp, drawn_at, drawn_delay, drawn_phase, last_good = None, None, None, None, None
    error_log = Path(sess.data_dir) / ".watch_error.log"
    from rich.live import Live
    with Live(console=console, screen=True, auto_refresh=False) as live, Keys() as keys:
        try:
            while True:
                feed.poll()
                for key in keys.read(args.interval):
                    if key in "sS":
                        view = "optimal" if view == "usual" else "usual"
                        drawn_delay = None                          # redraw now
                        continue
                    new = adjust_delay(delay, key)
                    if new != delay:
                        delay = new
                        save_delay(sess.data_dir, delay)
                now = tuple(os.stat(p).st_mtime_ns if p.exists() else None
                            for p in (sess.events, sess.predictions, sess.telemetry))
                head = feed.head()
                moved = head is not None and (drawn_at is None or (head - drawn_at).total_seconds() >= 0.5)
                blink = alerting[0] and flash() != drawn_phase
                if now != stamp or moved or blink or delay != drawn_delay or console.size != getattr(live, "_size", None):
                    stamp, drawn_at, drawn_delay, drawn_phase, live._size = now, head, delay, flash(), console.size
                    shown, last_good = safe_frame(
                        lambda: frame(console.height, console.width, read(sess.predictions)) if feed.history is not None
                        else Text(f" {sess}: waiting for events.csv ..."), last_good, error_log)
                    live.update(shown, refresh=True)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
