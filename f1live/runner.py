"""Watch events.csv, keep a clean per-driver state table, and call your model on every update.

Usage (from the repo root):
    python -m f1live.runner 2026 Austin race            # events.csv -> live/race/model.py -> predictions.csv
    python -m f1live.runner 2026 Austin race --tick 5   # also re-run every 5 s between rows
    python -m f1live.runner 2026 Austin quali --model live.quali.my_model

Your model is a module with one function:

    def predict(state, history, info): ...

    state    DataFrame, one row per driver (index = car number), latest known value of every column
    history  DataFrame, every event row so far (the full events.csv)
    info     dict: now, session_type, session_part, race_lap, total_laps, track_status,
             clock_left_s, session (the Session: year, race, name), new_rows (DataFrame of rows that triggered this
             call; empty on a tick),
             telemetry (last --tel-window s of telemetry.csv: ts, driver, speed, rpm, gear, throttle, brake,
             x, y, z, pos_status; empty if f1live.telemetry isn't running)

state also gets car_speed, car_gear, car_throttle, car_brake, car_x, car_y: the latest telemetry sample.
Telemetry rows never trigger a model call by themselves; they ride along on the next event or tick.

Return a dict (one prediction row), a list of dicts / DataFrame (several rows), or None.
Rows are timestamped and written to predictions.csv. The model file is reloaded automatically
when you save it, and an exception in the model is printed without stopping the runner.
"""
import argparse, importlib, importlib.util, io, os, sys, time, traceback
from datetime import datetime, timezone
import pandas as pd
from f1live.events import prepare
from f1live.paths import ROOT, add_session_args, session_from_args
from f1live.tail import CsvTail

# Columns that describe one event rather than the driver's ongoing state
EVENT_ONLY = ["event", "message", "pit_lane_s"]


TEL_COLS = ["speed", "rpm", "gear", "throttle", "brake", "x", "y"]


def prepare_telemetry(df):
    df["ts"] = pd.to_datetime(df["ts"], format="ISO8601")
    df["driver"] = df["driver"].astype(int).astype(str)
    return df


def add_telemetry(state, tel):
    """Latest sample per driver as car_* columns."""
    if state.empty or tel.empty:
        return state
    last = tel.sort_values("ts").groupby("driver").tail(1).set_index("driver")[TEL_COLS]   # latest row, not last non-null
    return state.join(last.add_prefix("car_").drop(columns="car_rpm"))


def build_state(history):
    rows = history[history["driver"].notna()]
    if rows.empty:
        return pd.DataFrame()
    state = rows.drop(columns=EVENT_ONLY).groupby("driver").last()   # last non-null per column
    last_lap = rows[rows["event"] == "lap"].groupby("driver")[["lap_time_s", "lap_kind"]].last()
    state["last_lap_s"] = last_lap["lap_time_s"]
    state["last_lap_kind"] = last_lap["lap_kind"]
    state["retired"] = state.index.isin(rows.loc[rows["event"] == "retired", "driver"])
    state["position"] = pd.to_numeric(state["position"], errors="coerce")
    return state.sort_values("position")


class ModelLoader:
    def __init__(self, name):
        sys.path.insert(0, str(ROOT))
        self.name, self.mod, self.mtime = name, None, None

    def get(self):
        path = importlib.util.find_spec(self.name).origin
        mtime = os.path.getmtime(path)
        if mtime != self.mtime:
            try:
                self.mod = importlib.reload(self.mod) if self.mod else importlib.import_module(self.name)
                print(f"[runner] {'re' if self.mtime else ''}loaded model '{self.name}'")
            except Exception:
                traceback.print_exc()
            self.mtime = mtime
        return self.mod


def main():
    ap = argparse.ArgumentParser()
    add_session_args(ap)
    ap.add_argument("--model", default="live.race.model", help="dotted module name of your model (live/race/model.py)")
    ap.add_argument("--tick", type=float, default=0, help="also call the model every N s (0 = only on new rows)")
    ap.add_argument("--poll", type=float, default=0.5)
    ap.add_argument("--tel-window", type=float, default=120, help="seconds of telemetry kept for the model")
    args = ap.parse_args()

    sess = session_from_args(args)
    tail, loader = CsvTail(sess.events, prepare), ModelLoader(args.model)
    tel_tail, tel = CsvTail(sess.telemetry, prepare_telemetry), pd.DataFrame()
    history, preds = None, []
    last_row_wall = last_tick = None
    print(f"[runner] watching {sess.events}")

    while True:
        new = tail.read()
        if isinstance(new, str):                     # reset
            print("[runner] events file rewritten, reloading")
            history = None
            continue
        fresh = tel_tail.read()
        if isinstance(fresh, str):
            tel = pd.DataFrame()
        elif fresh is not None:
            tel = pd.concat([tel, fresh], ignore_index=True) if len(tel) else fresh
            tel = tel[tel["ts"] >= tel["ts"].max() - pd.Timedelta(args.tel_window, unit="s")]   # bounded: never the whole race
        wall = time.time()
        tick_due = args.tick and history is not None and last_tick and wall - last_tick >= args.tick

        if new is not None or tick_due:
            if new is not None:
                history = new if history is None else pd.concat([history, new], ignore_index=True)
                last_row_wall = wall
            # "now" = latest event time, extrapolated by wall time since it arrived.
            # Works live and for real-speed replays (not for sped-up replays).
            now = history["ts"].max() + pd.Timedelta(wall - last_row_wall, unit="s")
            latest = history.iloc[-1]
            info = dict(now=now, session=sess, session_type=latest.get("session_type"),
                        session_part=latest.get("session_part"), race_lap=latest.get("race_lap"),
                        total_laps=latest.get("total_laps"), track_status=latest.get("track_status"),
                        clock_left_s=latest.get("clock_left_s"),
                        new_rows=new if new is not None else history.iloc[0:0], telemetry=tel)
            model = loader.get()
            if model is not None:
                try:
                    out = model.predict(add_telemetry(build_state(history), tel), history, info)
                except Exception:
                    traceback.print_exc()
                    out = None
                if out is not None:
                    df = out if isinstance(out, pd.DataFrame) else pd.DataFrame(out if isinstance(out, list) else [out])
                    df.insert(0, "ts", now)
                    preds.append(df)
                    pd.concat(preds, ignore_index=True).to_csv(sess.predictions, index=False)
            last_tick = wall
        time.sleep(args.poll)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
