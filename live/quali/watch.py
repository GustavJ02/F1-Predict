"""Full-screen qualifying dashboard: classification, the cutoff / pole predictions, flags and a circuit map with every car on it.

Usage (from the repo root, in a second terminal next to ./run_session.sh):
    python -m live.quali.watch 2026 Mexico_City quali
    python -m live.quali.watch 2026 Mexico_City quali --once       # print one frame and exit
    python -m live.quali.watch 2026 Austin replay --once          # a replay works the same

Reads events.csv (classification, tyres, flags), telemetry.csv (where each car is on track, written by f1live.telemetry) and
predictions.csv (from live.quali.model: run the session with MODEL=live.quali.model ./run_session.sh ...). Without predictions the
panel says so; the rest does not need them.

Header: session part (Q1/Q2/Q3), time left, track status, sector yellows from race control, pit exit and the weather.
Classification: the timing screen order with the best time of each part, the gap to P1 of the current part, the tyre, what the car is
doing (PUSH on a flying lap, OUT on an out lap, SLOW cool-down / in-lap / crawling, PIT, STOP stopped on track) and the last lap, with a
line at the cutoff; cars knocked out are dimmed. The map shows the same modes (see live.practice.watch); reverse video is a push lap.
Predictions: in Q1/Q2 where the cutoff is now and where it is expected to end up (90 % range), in Q3 the pole chances.

Delay: --delay SECONDS shows the session as it was that long ago, to match a delayed broadcast; the keys ] [ } { 0 change it while you
watch (see live.race.watch). It is remembered and shared with the other dashboards.
"""
import pandas as pd
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from f1live.watch import bar, fmt
from live.practice.watch import MODE_STYLE, WIDE, circuit_panel, header, last_lap_text, messages_panel
from live.quali.classification import classification
from live.race.battles import latest
from live.race.hazards import chequered
from live.race.watch import TEAM_COLOUR, TYRE, alert_line, lap_time, main

HISTORY = 5             # predictions shown as the recent trend


def prediction_lines(df):
    """Lines for the predictions panel from predictions.csv rows of live.quali.model, or a note if there are none."""
    if df is None or df.empty:
        return [Text("no predictions yet (MODEL=live.quali.model ./run_session.sh ...)", style="dim")]
    last = df.iloc[-1]
    part = int(last["part"])
    if part == 3 and "pole1" in df and pd.notna(last.get("pole1")):
        lines = [Text.assemble(("Pole probability", "bold"), f"   current pole time {fmt(last['current'])} s   still to run: {fmt(last['n_pending'])}")]
        for i in (1, 2, 3):
            if pd.notna(last.get(f"pole{i}")):
                lines.append(Text(f"  {last[f'pole{i}']:<4} {bar(last[f'p{i}'], 24)} {last[f'p{i}']:6.1%}"))
        return lines
    lines = [Text.assemble((f"Q{part} cutoff", "bold"), f"   now {fmt(last['current'])} s   predicted {fmt(last['pred'])} s   "
                           f"(90%: {fmt(last['lo'])} - {fmt(last['hi'])})   still to run: {fmt(last['n_pending'])}")]
    recent = df[df["part"] == part].tail(HISTORY)
    for _, r in recent.iterrows():
        lines.append(Text(f"  {r['ts']:%H:%M:%S}  now {fmt(r['current'])}  pred {fmt(r['pred'])}  ({fmt(r['lo'])} - {fmt(r['hi'])})", style="dim"))
    return lines


def classification_table(history, modes):
    table, current, advance = classification(history)
    if table.empty:
        return Text("waiting for timing data ...", style="dim")
    lat = latest(history)
    last_lap = history[history["event"] == "lap"].groupby("driver")["lap_time_s"].last()
    t = Table(box=None, pad_edge=False, header_style="bold dim", expand=True)
    for name, just in (("P", "right"), ("#", "right"), ("DRV", "left"), ("Q1", "right"), ("Q2", "right"), ("Q3", "right"),
                       ("GAP", "right"), ("NOW", "left"), ("MODE", "left"), ("LAST", "right")):
        t.add_column(name, justify=just, no_wrap=True)
    lead = table[f"q{current}"].min()
    for drv, r in table.iterrows():
        out = r["part"] < current                              # knocked out in an earlier part
        mode = modes.get(drv, "")
        now = TYRE.get(lat.loc[drv, "compound"], ("-", "")) if drv in lat.index else ("-", "")
        age = lat.loc[drv, "tyre_laps"] if drv in lat.index else None
        cur = r[f"q{current}"]
        gap = "" if pd.isna(cur) or cur == lead else f"+{cur - lead:.3f}"
        cells = [Text(lap_time(r[q]) if pd.notna(r[q]) else "", style="bold" if q == f"q{r['part']}" and not out else "")
                 for q in ("q1", "q2", "q3")]
        t.add_row(str(r["rank"]), drv, Text(str(r["tla"]), style=f"bold {TEAM_COLOUR.get(r['team'], 'white')}"), *cells, gap,
                  Text(f"{now[0]} {int(age)}" if pd.notna(age) and now[0] != "-" else "-", style=now[1]),
                  Text({"STOPPED": "STOP"}.get(mode, mode), style=MODE_STYLE.get(mode, "")),
                  last_lap_text(last_lap.get(drv)), style="dim" if out or mode == "PIT" else None)
        if advance and r["rank"] == advance:
            t.add_row(*[Text("─" * w, style="red") for w in (2, 2, 3, 8, 8, 8, 7, 4, 4, 8)])
    return t


def render(history, preds, sess="", height=40, progress=None, width=140, corners=(), sectors=None, delay=0.0, strategy=None,
           view="usual", flags=None, flash=False, ref=None):
    """One frame. Same arguments as live.race.watch.render (strategy and view are not used), plus ref: the track reference lap;
    preds are the predictions.csv rows (from live.quali.model) as of the moment shown."""
    if history is None or not len(history):
        return Text(f" {sess}  waiting ... (delay {delay:g} s)")
    map_panel, modes, stopped = circuit_panel(history, progress, flags, sectors, corners, flash, ref)
    table, current, advance = classification(history)
    layout = Layout()
    layout.split_column(Layout(name="head", size=1), Layout(name="alert", size=1), Layout(name="body"),
                        Layout(messages_panel(history), name="foot", size=6))
    starts = history.loc[history["event"] == "session_part_start", "ts"]            # each part ends with its own chequered flag
    this_part = history[history["ts"] >= starts.max()] if len(starts) else history
    layout["head"].update(header(history, sess, delay, part=f"Q{current}" if current else None, over=chequered(this_part)))
    layout["alert"].update(alert_line(history, stopped, flash))
    sheet = Panel(classification_table(history, modes), title="Classification", title_align="left")
    pred = Panel(Text("\n").join(prediction_lines(preds)), title="Predictions", title_align="left")
    pred_size = (len(prediction_lines(preds)) + 2)
    if width >= WIDE:
        layout["body"].split_row(Layout(sheet, ratio=11), Layout(name="right", ratio=9))
        layout["right"].split_column(Layout(pred, size=pred_size), Layout(map_panel))
    else:
        layout["body"].split_column(Layout(sheet, ratio=4), Layout(pred, size=pred_size), Layout(map_panel, ratio=3))
    return layout


if __name__ == "__main__":
    main(render, race=False, description=__doc__, name="qualifying dashboard")
