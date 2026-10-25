"""A tyre strategy as text: compound, stop lap, compound, stop lap, compound ...

    M-18-H-40-S     start on mediums, stop on lap 18 for hards, stop on lap 40 for softs
    S-14-H          one stop: softs, stop on lap 14, hards to the end
    H               no stop (illegal in a dry race: two different compounds are required)

Compounds are S, M, H (soft, medium, hard). A stop lap is the race lap on which the car comes in. In a race that is already
under way the starting compound is the one the car is on, and every stop lap must lie after the current lap.
"""
from dataclasses import dataclass
from live.race.priors import COMPOUNDS

LETTERS = {"S": "SOFT", "M": "MEDIUM", "H": "HARD"}
SHORT = {v: k for k, v in LETTERS.items()}
MIN_STINT = 3


@dataclass(frozen=True)
class Plan:
    compounds: tuple        # one compound name per stint, e.g. ('MEDIUM', 'HARD', 'SOFT')
    laps: tuple             # the lap of each stop, len(compounds) - 1 of them

    @classmethod
    def parse(cls, text):
        parts = [p for p in text.strip().upper().replace(" ", "").split("-") if p]
        compounds, laps = [], []
        for i, p in enumerate(parts):
            if i % 2 == 0:
                if p not in LETTERS:
                    raise ValueError(f"'{p}' is not a compound (use S, M or H) in strategy '{text}'")
                compounds.append(LETTERS[p])
            else:
                if not p.isdigit():
                    raise ValueError(f"'{p}' is not a lap number in strategy '{text}'")
                laps.append(int(p))
        if len(compounds) != len(laps) + 1:
            raise ValueError(f"strategy '{text}' must alternate compound and stop lap and end with a compound")
        return cls(tuple(compounds), tuple(laps))

    def __str__(self):
        out = [SHORT[self.compounds[0]]]
        for lap, c in zip(self.laps, self.compounds[1:]):
            out += [str(lap), SHORT[c]]
        return "-".join(out)

    @property
    def stops(self):
        """[(lap, compound index)] as the simulation wants them."""
        return [(lap, COMPOUNDS.index(c)) for lap, c in zip(self.laps, self.compounds[1:])]

    @property
    def start(self):
        return self.compounds[0]

    def legal(self):
        """Dry-race rule: at least two different compounds."""
        return len(set(self.compounds)) >= 2

    def validate(self, total, done=0):
        """Raise ValueError if the stop laps don't fit the race (after lap `done`, before the last lap, in order)."""
        prev = done
        for lap in self.laps:
            if lap <= prev:
                raise ValueError(f"stop on lap {lap} must come after lap {prev} in {self}")
            if lap >= total:
                raise ValueError(f"stop on lap {lap} is not before the last lap ({total}) in {self}")
            prev = lap
        return self
