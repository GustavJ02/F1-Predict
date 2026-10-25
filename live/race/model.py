"""Race model for the runner: battle predictions and the projected finishing order, written to predictions.csv.

Use it with:  ./run_session.sh 2026 Singapore race        (it is the default MODEL)

Every call returns the rows of one batch (all sharing a ts):
    kind='battle'      position (of the front car), drivers ('ANT PIA NOR'), numbers, gaps (s, one per neighbouring pair),
                       laps (one per pair, 0 = within range now), in_laps (soonest pair), now (any pair in range)
    kind='status'      note: why there are no battles to show (collecting pace data, safety car, none expected)
    kind='projection'  one row per running car: driver (number), tla, position (now), proj_rank (projected finish),
                       mean_pos, p_win, p_podium, p_points, stops_left (expected), laps_left, p_sc (chance of a
                       safety car/VSC before the flag), note (race-level text)
plus race_lap and total_laps on every row. The projection is a Monte Carlo of the rest of the race (live.race.projection)
and is recomputed once per leader lap or when the track status changes; calls in between return only the battle rows.
"""
import pandas as pd
from live.race import priors as P
from live.race.battles import HORIZON, find_battles
from live.race.projection import project

_priors = None
_last = {"key": None}


def _load_priors():
    global _priors
    if _priors is None:
        _priors = P.load_priors()
    return _priors


def projection_rows(history, info, force=False):
    """Projection rows when the leader's lap or the track status changed since the last one, else None."""
    key = (info["race_lap"], history["track_status"].dropna().iloc[-1] if history["track_status"].notna().any() else None)
    if not force and key == _last["key"]:
        return None
    session = info.get("session")
    circuit = P.circuit_of(session.year, session.race) if session is not None else ""
    out, meta = project(history, _load_priors(), circuit)
    if out is None:
        return None
    _last["key"] = key
    rows = out.rename(columns={"proj_rank": "proj_rank"})[["driver", "tla", "position", "proj_rank", "mean_pos", "p_win",
                                                          "p_podium", "p_points", "stops_left"]].copy()
    rows.insert(0, "kind", "projection")
    rows["laps_left"], rows["p_sc"] = meta["laps_left"], meta["p_sc"]
    rows["note"] = "under safety car / VSC" if meta["under_sc"] else ""
    return rows


def predict(state, history, info):
    if state.empty:
        return None
    status, battles = find_battles(history)
    base = dict(race_lap=info["race_lap"], total_laps=info["total_laps"])
    if status is None and not battles:
        status = f"no battles expected in the next {HORIZON} laps"
    frames = []
    if status is not None:
        frames.append(pd.DataFrame([dict(kind="status", note=status)]))
    else:
        frames.append(pd.DataFrame([dict(kind="battle", position=b["position"], drivers=" ".join(b["drivers"]),
                                         numbers=" ".join(b["numbers"]), gaps=" ".join(f"{g:.1f}" for g in b["gaps"]),
                                         laps=" ".join(f"{l:.0f}" for l in b["laps"]), in_laps=b["in_laps"], now=b["now"])
                                    for b in battles]))
    proj = projection_rows(history, info)
    if proj is not None:
        frames.append(proj)
    return pd.concat(frames, ignore_index=True).assign(**base)
