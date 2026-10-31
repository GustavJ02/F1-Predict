"""The qualifying classification as the timing screen shows it, from events.csv.

Every driver's best time in each part comes from the feed's own per-part best (best_in_part_s, so deleted laps are already gone).
The order is built part by part: Q1 by Q1 time; in Q2 the cars that went through (the first n_entries of the part) are sorted by
their Q2 time, those without one yet after those with one, in the order they had (a car that has a time in the part is in it even
if the part's size says otherwise, e.g. when the recording started late); the cars knocked out keep their place below;
and the same again in Q3. Cars that haven't set a time in the current part keep their previous order among themselves.
"""
import numpy as np
import pandas as pd


def part_bests(history):
    """DataFrame indexed by driver, columns 1, 2, 3: the best lap of each part (NaN where none)."""
    rows = history[history["driver"].notna() & history["session_part"].notna() & history["best_in_part_s"].notna()]
    best = rows.groupby(["driver", rows["session_part"].astype(int)])["best_in_part_s"].last().unstack()
    return best.reindex(columns=[1, 2, 3])


def part_sizes(history):
    """{part: (cars in the part, cars that go through)}; the second is None in the last part or when unknown."""
    rows = history[history["session_part"].notna()]
    out = {}
    for part, g in rows.groupby(rows["session_part"].astype(int)):
        entries, advance = g["n_entries"].dropna(), g["n_advance"].dropna()
        out[part] = (int(entries.max()) if len(entries) else None, int(advance.iloc[-1]) if len(advance) else None)
    return out


def classification(history):
    """-> (DataFrame in classification order: driver (index), tla, team, q1, q2, q3, part (the last part the car is in), rank;
    the current part number; cars that go through). Empty frame if there is no timing yet."""
    drivers = history[history["driver"].notna() & history["tla"].notna()].groupby("driver")[["tla", "team"]].last()
    if drivers.empty or history["session_part"].notna().sum() == 0:
        return pd.DataFrame(), None, None
    best = part_bests(history).reindex(drivers.index)
    sizes = part_sizes(history)
    current = max(sizes)
    order = list(best[1].fillna(np.inf).sort_values(kind="stable").index)             # Q1: by time, no time last
    reached = {d: 1 for d in order}
    for part in range(2, current + 1):
        entries = sizes[part][0] or int(history[history["session_part"] == part]["driver"].nunique())
        timed = [d for d in order if pd.notna(best.loc[d, part])]            # a car with a time is in the part, whatever came before
        inside = timed + [d for d in order if d not in timed][:max(entries - len(timed), 0)]
        times = best[part].reindex(inside).fillna(np.inf)
        inside = sorted(inside, key=lambda d: (times[d], order.index(d)))
        order = inside + [d for d in order if d not in inside]
        for d in inside:
            reached[d] = part
    out = drivers.reindex(order)
    out[["q1", "q2", "q3"]] = best.reindex(order)[[1, 2, 3]].to_numpy()
    out["part"] = [reached[d] for d in order]
    out["rank"] = np.arange(1, len(out) + 1)
    return out, current, sizes[current][1]
