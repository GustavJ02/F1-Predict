"""F1 Fantasy scoring, as published on https://fantasy.formula1.com/en/game-rules (read 2026-10-31).

Everything here is a pure function of one driver's (or constructor's) weekend, so the simulation can score every scenario with it
and the tests can check each rule on its own. Chips and the DRS boost multiply these numbers elsewhere (fantasy.optimize).

Rules worth knowing (they differ from several summaries on the web):
  * the sprint DOES score positions gained / lost (1 per position, at most -10 lost) and overtakes, besides the finish points;
  * positions gained/lost compare the official starting grid with the classification, not the qualifying result;
  * a driver who is not classified (DNF, DSQ) gets the -20 (sprint -10) penalty instead of any lost positions;
  * constructors score the sum of their drivers' points (race: without the Driver of the Day bonus) plus the bonuses below.
"""

QUALI = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]            # points for P1..P10, 0 below
QUALI_NO_TIME = -5                                  # NC / DSQ / no time set in Q1

RACE = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]
SPRINT = [8, 7, 6, 5, 4, 3, 2, 1]

RACE_NC, SPRINT_NC = -20, -10
RACE_FASTEST_LAP, SPRINT_FASTEST_LAP = 10, 5
DRIVER_OF_THE_DAY = 10                              # driver only, not the constructor
SPRINT_MAX_LOST = 10                                # positions lost are capped at 10 in the sprint
EXTRA_TRANSFER = -10
FREE_TRANSFERS = 2

# pit stop of a constructor, seconds -> points (per stop of either driver); the fastest stop of the race adds 5,
# a new world record (under 1.8 s, set by McLaren in Qatar 2023) adds 15
PIT = [(2.0, 20), (2.2, 10), (2.5, 5), (3.0, 2)]    # (stop is under this many seconds, points); 3.0 s and over: 0
PIT_FASTEST, PIT_RECORD, RECORD_TIME = 5, 15, 1.8


def table_points(table, position):
    """Points of a 1-based finishing position in a ladder; 0 below the last scoring place."""
    return table[position - 1] if 1 <= position <= len(table) else 0


def quali_points(position):
    """Driver, qualifying. position: classified grid slot from qualifying (1 = pole), None = no time / DSQ."""
    return QUALI_NO_TIME if position is None else table_points(QUALI, position)


def constructor_quali_bonus(parts, disqualified=0):
    """Constructor bonus for qualifying. parts: the last qualifying part each of the two drivers reached (1, 2 or 3).
    Only the best tier applies: both Q3 10, one Q3 5, both Q2 3, one Q2 1, neither -1; -5 per disqualified driver."""
    reached_q3 = sum(p == 3 for p in parts)
    reached_q2 = sum(p >= 2 for p in parts)
    tier = {2: 10, 1: 5}.get(reached_q3) or {2: 3, 1: 1}.get(reached_q2, -1)
    return tier - 5 * disqualified


def _movement(grid, finish, cap=None):
    """Positions gained (+) / lost (-) between the official grid slot and the classification."""
    delta = grid - finish
    return max(delta, -cap) if cap is not None else delta


def race_points(grid, finish, classified=True, overtakes=0, fastest_lap=False, dotd=False):
    """Driver, race. grid: official starting slot, finish: classification (ignored if not classified)."""
    if not classified:
        return RACE_NC + overtakes
    return (table_points(RACE, finish) + _movement(grid, finish) + overtakes
            + (RACE_FASTEST_LAP if fastest_lap else 0) + (DRIVER_OF_THE_DAY if dotd else 0))


def sprint_points(grid, finish, classified=True, overtakes=0, fastest_lap=False):
    """Driver, sprint."""
    if not classified:
        return SPRINT_NC + overtakes
    return (table_points(SPRINT, finish) + _movement(grid, finish, SPRINT_MAX_LOST) + overtakes
            + (SPRINT_FASTEST_LAP if fastest_lap else 0))


def pit_stop_points(seconds):
    """One pit stop's points for the constructor (without the fastest-stop and record bonuses)."""
    for limit, points in PIT:
        if seconds < limit:
            return points
    return 0


def constructor_pit_points(stop_times, fastest_of_race=False):
    """All of a team's stops in a race; fastest_of_race: this team made the quickest stop of the race."""
    pts = sum(pit_stop_points(t) for t in stop_times)
    pts += PIT_RECORD * sum(t < RECORD_TIME for t in stop_times)
    return pts + (PIT_FASTEST if fastest_of_race else 0)


def transfer_penalty(n_transfers, free=FREE_TRANSFERS):
    """Points lost for `n_transfers` swaps (a swap is one asset out and one in), given the free allowance."""
    return EXTRA_TRANSFER * max(0, n_transfers - free)
