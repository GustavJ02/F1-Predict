"""Full-screen practice dashboard (FP1-FP3): time left, flags, timesheet and a circuit map with every car on it.

Usage (from the repo root, in a second terminal next to ./run_session.sh):
    python -m live.practice.watch 2026 Mexico_City fp1
    python -m live.practice.watch 2026 Mexico_City fp1 --once       # print one frame and exit

No prediction model runs in practice: this reads events.csv and (for the map) telemetry.csv, which ./run_session.sh records.
The header shows the time left, the track status (green / yellow / VSC / safety car / red), sector yellows from race control, the
pit exit and the weather. The timesheet has best lap, tyre, laps, current tyre and long-run pace, and what each car is doing now:
    PUSH  at flying-lap pace (lap time projected from the lap so far within 2 % of the driver's best; a quick
          long-run lap can look the same)
    OUT   on an out lap        SLOW  projected 20 % off his best: cool-down, in-lap or crawling (▼ on the map)       PIT  in the pit lane or garage
    STOP  stopped on track (✖, blinks)
Long-run pace is the median of a stint's clean laps (5 or more, green flag, no in/out laps), not fuel or engine-mode corrected:
compare it only between cars on the same tyre. The map shows a car as a coloured dot with its code: reverse video = push lap,
dim = out lap / slow. Cars in the pits are not drawn. Keys ] [ } { 0 delay the picture to match a delayed broadcast, as in
live.race.watch (--delay). Reads the same Feed, so it also works for a replay.
"""
import pandas as pd
from rich.console import Group
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from live.practice.status import (best_compound, car_modes, clock_left, long_runs, pit_exit, recent_messages, yellow_sectors)
from live.practice.trackmap import TrackMap
from live.race.battles import latest
from live.race.hazards import BANNERS, chequered, track_state
from live.race.watch import SECTOR_COLOUR, TEAM_COLOUR, TRACK, TYRE, lap_time, main, sector_legend, wet_now
from live.race.watch import alert_line

MODE_STYLE = {"PUSH": "bold green", "OUT": "dim", "SLOW": "yellow", "PIT": "dim", "STOPPED": "bold red", "RUN": ""}
WIDE = 130              # columns from which the map sits beside the timesheet instead of below it


def last_lap_text(seconds):
    """A last lap for the LAST column: '-' when there is none or it was more than five minutes (a lap spent in the garage)."""
    return "-" if seconds is None or pd.isna(seconds) or seconds > 300 else lap_time(seconds)


def clock_text(seconds):
    if seconds is None:
        return "-"
    h, rest = divmod(int(seconds), 3600)
    return f"{h}:{rest // 60:02d}:{rest % 60:02d}" if h else f"{rest // 60}:{rest % 60:02d}"


def header(history, sess, delay=0.0, part=None, over=None):
    """The top line. part: a label for the part of the session (Q1, Q2, Q3) shown after the session name; over: whether the chequered
    flag has been shown (default: any time in the session; a qualifying part ends with one too, so it passes its own)."""
    t = Text()
    t.append(f" {sess} ", style="bold reverse")
    if part:
        t.append(f" {part} ", style="bold black on cyan")
    over = chequered(history) if over is None else over
    t.append("   ")
    t.append(" CHEQUERED FLAG " if over else f" {clock_text(clock_left(history))} LEFT ", style="bold reverse")
    track = track_state(history)
    label, style = TRACK.get(track, (str(track or "GREEN").upper(), "green")) if track != "AllClear" else ("GREEN", "green")
    t.append("  ")
    t.append(f" {label} ", style=f"{style} reverse")
    for sector, kind in sorted(yellow_sectors(history).items()):
        t.append("  ")
        t.append(f" {'YY' if kind.startswith('DOUBLE') else 'Y'} S{sector} ", style="bold black on yellow")
    exit_ = pit_exit(history)
    if exit_:
        t.append(f"   PIT EXIT {exit_.upper()}", style="green" if exit_ == "open" else "bold red")
    if wet_now(history):
        t.append("  ")
        t.append(" RAIN ", style="bold blue reverse")
    w = history[["track_temp", "air_temp"]].dropna().tail(1)
    if len(w):
        t.append(f"   track {w['track_temp'].iloc[0]:.0f}° air {w['air_temp'].iloc[0]:.0f}°", style="dim")
    t.append(f"    {history['ts'].max():%H:%M:%S} UTC", style="dim")
    if delay:
        t.append("   ")
        t.append(f" DELAY {delay:g} s ", style="bold cyan reverse")
        t.append("  [ ] 1 s   { } 10 s   0 live", style="dim")
    return t


def timesheet(history, modes):
    lat = latest(history)
    if lat.empty:
        return Text("waiting for timing data ...", style="dim")
    lat = lat.sort_values("best_lap_s", na_position="last")            # each car's row is from its own last update: rank by time
    lat["rank"] = lat["best_lap_s"].rank(method="min")
    laps = history[history["event"] == "lap"]
    timed = laps[laps["lap_kind"].isin(["push", "other"])].groupby("driver").size()
    last_lap = laps.groupby("driver")["lap_time_s"].last()
    best_tyre, runs = best_compound(history), long_runs(history)
    t = Table(box=None, pad_edge=False, header_style="bold dim", expand=True)
    for name, just in (("P", "right"), ("#", "right"), ("DRV", "left"), ("BEST", "right"), ("GAP", "right"), ("T", "left"),
                       ("LAPS", "right"), ("NOW", "left"), ("MODE", "left"), ("LAST", "right"), ("RUN", "left"),
                       ("PACE", "right"), ("+", "right"), ("DEG", "right")):
        t.add_column(name, justify=just, no_wrap=True)
    tyre = lambda c: TYRE.get(c, ("-", ""))
    for drv, r in lat.iterrows():
        mode = modes.get(drv, "")
        run = runs.loc[drv] if drv in runs.index else None
        now = tyre(r["compound"])
        gap = "" if r["rank"] == 1 or pd.isna(r["best_lap_s"]) else f"+{r['best_lap_s'] - lat['best_lap_s'].min():.3f}"
        t.add_row("" if pd.isna(r["rank"]) else str(int(r["rank"])), drv, Text(str(r["tla"]), style=f"bold {TEAM_COLOUR.get(r['team'], 'white')}"),
                  lap_time(r["best_lap_s"]) if pd.notna(r["best_lap_s"]) else "-", gap,
                  Text(tyre(best_tyre.get(drv))[0], style=tyre(best_tyre.get(drv))[1]),
                  str(int(timed.get(drv, 0))),
                  Text(f"{now[0]} {int(r['tyre_laps'])}" if pd.notna(r["tyre_laps"]) else "-", style=now[1]),
                  Text({"STOPPED": "STOP"}.get(mode, mode), style=MODE_STYLE.get(mode, "")),
                  last_lap_text(last_lap.get(drv)),
                  Text(tyre(run["run_compound"])[0] if run is not None else "", style=tyre(run["run_compound"])[1] if run is not None else ""),
                  lap_time(run["run_pace_s"]) if run is not None else "",
                  f"{run['run_gap_s']:.2f}" if run is not None else "",
                  "" if run is None or pd.isna(run["run_deg_s"]) else f"{run['run_deg_s']:+.2f}",
                  style="dim" if mode == "PIT" else None)
    return t


def messages_panel(history):
    lines = [Text.assemble((f"{ts:%H:%M:%S} ", "dim"), text[:160]) for ts, text in recent_messages(history, 4)]
    return Panel(Group(*lines) if lines else Text("no messages yet", style="dim"), title="Race control", title_align="left")


def circuit_panel(history, progress, flags, sectors, corners, flash, ref):
    """-> (Panel with the circuit map and the legend, {driver: mode}, {driver: 'stopped'}): the map of every car in its team colour,
    styled by what it is doing (live.practice.status.car_modes). `progress` is ({driver: lap fraction}, 'telemetry' | 'timing' | None)."""
    prog, mode = progress or ({}, None)
    modes = car_modes(history, prog, flags, measured=mode == "telemetry")
    lat = latest(history)
    stopped = {d: k for d, k in (flags or {}).items() if k == "stopped"}
    cars = [(str(lat.loc[d, "tla"]), frac, TEAM_COLOUR.get(lat.loc[d, "team"], "white"), modes.get(d, "RUN"))
            for d, frac in prog.items() if d in lat.index]
    counts = pd.Series(list(modes.values())).value_counts()
    note = {"telemetry": "from car telemetry", "timing": "estimated from lap timing"}.get(mode, "waiting for positions")
    legend = Text.assemble(sector_legend(sectors), ("PUSH ", "bold green"), f"{counts.get('PUSH', 0)}  ",
                           ("OUT ○ ", "dim"), f"{counts.get('OUT', 0)}  ", ("SLOW ▼ ", "yellow"), f"{counts.get('SLOW', 0)}  ",
                           ("PIT ", "dim"), f"{counts.get('PIT', 0)}")
    if ref is not None and mode == "telemetry":
        picture = TrackMap(ref, cars, sectors, corners, flash)
    else:
        picture = Text("the map needs car telemetry (f1live.telemetry) and a track reference; waiting ..." if ref is None
                       else "waiting for car telemetry ...", style="dim")
    return Panel(picture, title=f"Circuit ({note})", title_align="left", subtitle=legend, subtitle_align="right"), modes, stopped


def render(history, preds, sess="", height=40, progress=None, width=140, corners=(), sectors=None, delay=0.0, strategy=None,
           view="usual", flags=None, flash=False, ref=None):
    """One frame. Same arguments as live.race.watch.render (preds, strategy and view are not used), plus ref: the track reference lap."""
    ok = history is not None and len(history)
    if not ok:
        return Text(f" {sess}  waiting ... (delay {delay:g} s)")
    map_panel, modes, flags = circuit_panel(history, progress, flags, sectors, corners, flash, ref)
    wide = width >= WIDE
    layout = Layout()
    layout.split_column(Layout(name="head", size=1), Layout(name="alert", size=1), Layout(name="body"),
                        Layout(messages_panel(history), name="foot", size=6))
    layout["head"].update(header(history, sess, delay))
    layout["alert"].update(alert_line(history, flags, flash))
    sheet = Panel(timesheet(history, modes), title="Timesheet", title_align="left")
    if wide:
        layout["body"].split_row(Layout(sheet, ratio=11), Layout(map_panel, ratio=9))
    else:
        layout["body"].split_column(Layout(sheet, ratio=3), Layout(map_panel, ratio=2))
    return layout


if __name__ == "__main__":
    main(render, race=False, description=__doc__, name="practice dashboard")
