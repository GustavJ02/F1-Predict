"""Race model for the runner: predicts battles (see live.race.battles) and writes them to predictions.csv.

Use it with:  ./run_session.sh 2026 Singapore race        (it is the default MODEL)

Every call returns one batch of rows sharing the same ts, so a viewer can show "the latest batch":
    kind='battle'  position (of the front car), drivers ('ANT PIA NOR'), numbers, gaps (s, one per neighbouring pair),
                   laps (one per pair, 0 = within range now), in_laps (soonest pair), now (any pair in range)
    kind='status'  note: why there is nothing to show (pace data still being collected, safety car, no battles)
plus race_lap and total_laps on every row.
"""
import pandas as pd
from live.race.battles import HORIZON, find_battles


def predict(state, history, info):
    if state.empty:
        return None
    status, battles = find_battles(history)
    base = dict(race_lap=info["race_lap"], total_laps=info["total_laps"])
    if status is None and not battles:
        status = f"no battles expected in the next {HORIZON} laps"
    if status is not None:
        return dict(kind="status", note=status, **base)
    return pd.DataFrame([dict(kind="battle", position=b["position"], drivers=" ".join(b["drivers"]),
                              numbers=" ".join(b["numbers"]), gaps=" ".join(f"{g:.1f}" for g in b["gaps"]),
                              laps=" ".join(f"{l:.0f}" for l in b["laps"]), in_laps=b["in_laps"], now=b["now"], **base)
                         for b in battles])
