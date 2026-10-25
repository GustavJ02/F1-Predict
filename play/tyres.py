"""Which tyres a driver still has: the weekend's allocation minus the fresh sets already used.

A driver gets a fixed number of tyre sets per compound for the whole weekend: 2 hard, 3 medium and 8 soft on a normal
weekend, 2 / 4 / 6 on a sprint weekend (what the 41 weekends of 2025-26 show: the most any driver used of each). A set that
has been used in practice, sprint or qualifying is no longer fresh, so every stint that began on a new set (tyre_new) in
FP1-FP3, sprint qualifying, the sprint or qualifying (and, in a race under way, the race so far) takes one off.

The strategy optimizer only suggests plans that fit the sets that are left: a plan with two soft stints needs two soft sets.
Used sets can in fact be used again (teams start on used softs from qualifying), but how worn they are isn't known and the
simulation can't start a stint on a worn tyre, so by default only fresh sets count; `allow_used` adds that many extra sets of a
compound that are treated as fresh (optimistic). Anything can be overridden by hand with `overrides`.

    tyres = TyreSets.from_weekend(Path("data"), "2026", "Austin")
    tyres.available("HAM")           # {'SOFT': 1, 'MEDIUM': 2, 'HARD': 1}
    tyres.check("HAM", Plan.parse("S-12-S-30-H"))     # -> ['needs 2 SOFT sets, 1 available']
"""
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from f1live.events import has_events, load
from f1live.paths import Session
from live.race.priors import COMPOUNDS

WEEKEND_SESSIONS = ("fp1", "fp2", "fp3", "sq", "sprint", "quali")      # sessions that use up sets before the race
ALLOCATION = {"normal": {"HARD": 2, "MEDIUM": 3, "SOFT": 8}, "sprint": {"HARD": 2, "MEDIUM": 4, "SOFT": 6}}
LETTER = {"S": "SOFT", "M": "MEDIUM", "H": "HARD"}


def parse_counts(text):
    """'H2,M3,S8' -> {'HARD': 2, 'MEDIUM': 3, 'SOFT': 8} (compounds not mentioned are left out)."""
    out = {}
    for part in text.replace(" ", "").upper().split(","):
        if not part:
            continue
        if part[0] not in LETTER or not part[1:].isdigit():
            raise ValueError(f"'{part}': expected a compound letter (S, M, H) and a number, e.g. S2")
        out[LETTER[part[0]]] = int(part[1:])
    return out


def fresh_sets(history):
    """Counter {(tla, compound): fresh sets started} over the stints of one session's events (a stint that begins with
    tyre_new = True began on a new set)."""
    if not {"tyre_new", "tla", "compound", "stint"} <= set(history.columns):
        return Counter()                                           # a recording without tyre data: usage unknown
    laps = history[(history["event"] == "lap") & history["driver"].notna() & history["compound"].isin(COMPOUNDS)]
    out = Counter()
    for (_, _), g in laps.sort_values("ts").groupby(["driver", "stint"]):
        if g["tyre_new"].iloc[0] is True or g["tyre_new"].iloc[0] == True:      # noqa: E712 (numpy bool / object column)
            out[(g["tla"].iloc[-1], g["compound"].iloc[-1])] += 1
    return out


def is_sprint_weekend(data_dir, year, race):
    return any(has_events(Session(str(year), race, s, Path(data_dir)).events) for s in ("sq", "sprint"))


@dataclass
class TyreSets:
    allocation: dict
    used: dict = field(default_factory=dict)          # tla -> {compound: fresh sets used so far}
    overrides: dict = field(default_factory=dict)     # tla -> {compound: sets available} set by hand
    allow_used: dict = field(default_factory=dict)    # compound -> extra (used) sets treated as fresh
    live: bool = False                                # a race under way: the set on the car is not a set to choose

    @classmethod
    def from_weekend(cls, data_dir, year, race, allocation=None, history=None, sessions=WEEKEND_SESSIONS):
        """Usage over the weekend's sessions in data_dir (those that exist), plus the race so far if `history` is given."""
        used = Counter()
        for s in sessions:
            path = Session(str(year), race, s, Path(data_dir)).events
            if has_events(path):
                used += fresh_sets(load(path))
        if history is not None:
            used += fresh_sets(history)
        table = {}
        for (tla, compound), n in used.items():
            table.setdefault(tla, {})[compound] = n
        alloc = allocation or ALLOCATION["sprint" if is_sprint_weekend(data_dir, year, race) else "normal"]
        return cls(dict(alloc), table, live=history is not None)

    def available(self, tla):
        """{compound: fresh sets left} for one driver (never below 0), with overrides applied."""
        used = self.used.get(tla, {})
        out = {c: max(self.allocation.get(c, 0) - used.get(c, 0), 0) + self.allow_used.get(c, 0) for c in COMPOUNDS}
        out.update({c: n for c, n in self.overrides.get(tla, {}).items()})
        return out

    def needed(self, plan):
        """Sets a plan takes: one per stint (a race under way doesn't count the stint it is already in)."""
        return Counter(plan.compounds[1:] if self.live else plan.compounds)

    def check(self, tla, plan):
        """Problems with running `plan`: a list of strings, empty if the driver has the sets for it."""
        avail = self.available(tla)
        return [f"needs {n} {c} set{'s' if n != 1 else ''}, {avail[c]} available"
                for c, n in sorted(self.needed(plan).items()) if n > avail[c]]

    def describe(self, tla):
        avail, used = self.available(tla), self.used.get(tla, {})
        left = "  ".join(f"{c.lower()} {avail[c]}" for c in ("SOFT", "MEDIUM", "HARD"))
        spent = ", ".join(f"{used[c]} {c.lower()}" for c in ("SOFT", "MEDIUM", "HARD") if used.get(c)) or "none"
        note = "  (set by hand)" if tla in self.overrides else ""
        return f"fresh sets left: {left}   [allocation {self.alloc_text()}, used so far: {spent}]{note}"

    def alloc_text(self):
        return "".join(f"{c[0]}{self.allocation.get(c, 0)}" for c in ("HARD", "MEDIUM", "SOFT"))


def add_tyre_args(ap):
    """The command-line options shared by play.optimize and play.playground."""
    ap.add_argument("--tyres", choices=["auto", "none"], default="auto",
                    help="auto: limit strategies to the fresh sets each driver has left; none: no limits")
    ap.add_argument("--allocation", metavar="H2,M3,S8", help="sets per driver for the weekend (default: 2/3/8, or 2/4/6 at a sprint weekend)")
    ap.add_argument("--sets", action="append", default=[], metavar="HAM=S0,M2,H1",
                    help="the sets a driver really has available (overrides the count), repeatable")
    ap.add_argument("--ban", action="append", default=[], metavar="HAM=S", help="compounds a driver won't use, e.g. HAM=SM, repeatable")
    ap.add_argument("--allow-used", metavar="S1,M1", help="treat this many extra used sets of a compound as fresh (optimistic)")


def configure_tyres(scenario, args):
    """Apply the options of add_tyre_args to the scenario's TyreSets (or switch them off)."""
    if args.tyres == "none" or scenario.tyres is None:
        scenario.tyres = None
        return
    t = scenario.tyres
    if args.allocation:
        t.allocation = parse_counts(args.allocation)
    for text in args.sets:
        drv, _, counts = text.partition("=")
        t.overrides.setdefault(drv.upper(), {}).update(parse_counts(counts))
    for text in args.ban:
        drv, _, letters = text.partition("=")
        for letter in letters.upper():
            if letter not in LETTER:
                raise ValueError(f"'{text}': ban compounds with S, M or H")
            t.overrides.setdefault(drv.upper(), {})[LETTER[letter]] = 0
    if args.allow_used:
        t.allow_used = parse_counts(args.allow_used)
