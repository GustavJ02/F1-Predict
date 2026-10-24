"""Race model template. The runner calls predict() on every new event (and every --tick seconds).

Useful columns in `state` (one row per driver, index = car number):
    tla, team, position, lap_number, gap_to_leader_s, laps_down, interval_s, catching,
    in_pit, n_pit_stops, stint, compound, tyre_new, tyre_laps, best_lap_s,
    last_lap_s, last_lap_kind ('racing' / 'in' / 'out'), retired
`history` has every event row (filter on history.event == 'lap', 'pit_out', 'track_status', ...).
`info` has now, session_type, race_lap, total_laps, track_status, clock_left_s, new_rows.
"""
import pandas as pd


def predict(state, history, info):
    if state.empty:
        return None

    # Only print/predict when something relevant happened (a lap, a stop, SC...), not on every row
    new = info["new_rows"]
    if not new.empty and not new["event"].isin(["lap", "pit_out", "track_status", "retired"]).any():
        return None

    # --- baseline: finishing order = current order (replace with your model) ---
    running = state[~state["retired"]].sort_values("position")
    pred = pd.DataFrame({
        "driver": running.index,
        "tla": running["tla"].values,
        "current_position": running["position"].values,
        "predicted_position": range(1, len(running) + 1),
    })

    print(f"[model] lap {info['race_lap']}/{info['total_laps']}  {info['track_status']}  "
          f"top 3: {' '.join(pred['tla'].head(3).fillna('?'))}")
    return pred
