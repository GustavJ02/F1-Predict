# F1 live-timing kit

```
record (FastF1)  ->  raw.txt  ->  f1live.parse  ->  events.csv  ->  f1live.runner  ->  <session>/model.py  ->  predictions.csv
```

## Layout

```
f1live/              shared plumbing (used by everything below)
  feed.py            decoding, delta merging, file tailing, value parsing
  paths.py           data/<year>/<race>/<session>/ locations (Session)
  events.py          events.csv -> DataFrame with consistent dtypes
  telemetry.py       raw.txt (CarData + Position) -> telemetry.csv: speed, rpm, gear, throttle, brake, x, y per driver, 1 row/s
  parse.py           raw.txt -> events.csv (race, sprint and qualifying)
  tail.py            CsvTail: read rows another process appends to a CSV
  runner.py          tails events.csv, builds per-driver state, calls a model, writes predictions.csv
  replay.py          re-plays one session's raw.txt into another session in (scaled) real time
  watch.py           terminal viewer for predictions.csv (used by the live.*.watch scripts)
  sync.py            rsync data/ with the central store (rpi:/srv/F1-data): status / pull / push
  archive.py         downloads a finished session from F1's archive into raw.txt (--telemetry: raw_telemetry.txt instead)
  fetch_season.py    downloads (and parses) a whole season, or selected sessions, into data/<year>/<race>/<session>/
live/                predictions run DURING a session
  quali/             state.py (shared per-driver state), predict_elimination.py (Q1/Q2 cutoff),
                     predict_pole.py (Q3 pole probabilities), model.py (runner hook for both),
                     watch.py (live view), backtest_pole.py (score pole predictions on past sessions),
                     prior.py (pace prior for Q1/Q2 from earlier sessions + parts), backtest_cutoff.py (score the cutoff)
  race/              model.py (race model template, "current order" baseline - yours to edit),
                     watch.py (live view of the latest prediction, any columns)
pre/                 predictions made BEFORE a session, from past sessions
  history.py         pace tables for any session in data/ (best valid lap per driver, gap to the fastest in %)
  quali/             parts.py (pace of each of Q1/Q2/Q3 from earlier sessions and parts), features.py (form, practice,
                     sprint-quali inputs), predict_grid.py (pole probabilities and
                     expected grid), backtest.py (score it on every past quali, with ablations)
  race/              (not built yet)
run_session.sh       starts recorder + parser + runner together and restarts any that crash
data/                gitignored
  <year>/<race>/<session>/
    raw.txt          FastF1 recording, the one file that must not be lost
    raw_telemetry.txt  archive CarData/Position (only from `archive --telemetry`; live recordings carry them in raw.txt)
    telemetry.csv    parsed car telemetry (f1live.telemetry; a race is ~100k rows at the default 1 s)
    events.csv       parsed events (rebuilt from raw.txt whenever the parser restarts)
    predictions.csv  runner output
```

Session folders are named however you like (`quali`, `race`, `sprint`, `replay`, ...); all
commands take `year race session` and are run from the repo root.

## Central data store

The master copy of `data/` lives on the Pi (`rpi:/srv/F1-data`, same layout); `data/` here is a working copy that
can be small or empty. Set `F1_REMOTE` for another store and `F1_DATA` for another local directory.

```bash
python -m f1live.sync status                       # what differs, per file type (transfers nothing)
python -m f1live.sync pull                         # fetch missing events.csv files (~32 MB for 2025-26): enough for pre.* and backtests
python -m f1live.sync pull 2026 Austin quali --what raw telemetry     # more files of one session, e.g. to replay it
python -m f1live.sync pull --what all 2025         # everything of a year (raw, telemetry, ...)
python -m f1live.sync push                         # send new/grown files to the store (also after fetch_season)
```

- `pull` only fetches files you don't have, so it can never overwrite a recording that hasn't been pushed yet.
- `push` appends to files that grew and never replaces a longer file on the store with a shorter local one.
  Nothing is ever deleted on either side. `--dry-run` shows what would move.
- `run_session.sh` pushes its session folder every 30 s while recording and once on exit, so a live session is
  backed up on the Pi as it happens. If the Pi is unreachable it keeps going and retries (`SYNC=0` turns it off;
  replays are never pushed). Everything is written locally first: a network problem can't cost live data.
- File types for `--what`: `events` (default for pull), `telemetry` (~6 GB), `raw`, `raw_telemetry`, `predictions`, `all`.

## Race day

```bash
python -m fastf1 auth f1tv --status          # check the F1TV login before the session
./run_session.sh 2026 Austin race            # Ctrl-C stops everything
```

Edit `live/race/model.py` while it runs; the runner reloads it on save and survives exceptions.
Use another model with `MODEL=live.quali.my_model ./run_session.sh ...` (a module with `predict(state, history, info)`).

## Qualifying

```bash
MODEL=live.quali.model ./run_session.sh 2026 Austin quali   # record + parse + cutoff prediction in predictions.csv
python -m live.quali.predict_elimination 2026 Austin quali --part 2 --as-of 2026-10-24T21:33:00Z   # one-off run
python -m live.quali.predict_pole 2026 Austin quali --as-of 2026-10-24T22:00:30Z            # pole probabilities (Q3)
python -m live.quali.backtest_pole 2026                                                  # score pole predictions on every data/2026/*/quali
```

## Watching predictions live

In a second terminal while `run_session.sh` runs (works the same for a replay):

```bash
python -m live.quali.watch 2026 Austin quali     # cutoff in Q1/Q2, pole probabilities in Q3
python -m live.race.watch 2026 Singapore race    # latest batch of rows from the race model
python -m live.quali.watch 2026 Austin replay --once   # print once and exit
```

## Before a session (pre.quali)

```bash
python -m f1live.fetch_season 2025                                   # all sessions of a season (resumable, ~4 s each)
python -m f1live.fetch_season 2026 --sessions fp1 fp2 fp3 sq sprint --alias United_States=Austin
python -m pre.quali.predict_grid 2026 Austin                         # pole probabilities + expected grid
python -m pre.quali.backtest                                         # score the model on every quali we have
```

Session folders from the archive are `fp1 fp2 fp3 sq sprint quali race`; the race folder is the meeting name
without "Grand Prix", spaces as underscores (`Australian`, `Saudi_Arabian`, `United_States`, ...); a name with spaces
or accents is accepted too and converted. `Austin` is an alias of `United_States`.

## Testing without a live session

```bash
REPLAY=2026/Austin/quali SPEED=10 ./run_session.sh 2026 Austin replay
python -m f1live.archive --list 2026
python -m f1live.archive 2026 Singapore race            # -> data/2026/Singapore/race/raw.txt
REPLAY=2026/Singapore/race SPEED=20 ./run_session.sh 2026 Singapore replay
```

Replays always go to a different session folder than their source. With `SPEED` > 1 the runner's
`info["now"]` drifts between rows (it extrapolates in wall time).

## events.csv

One row per event, `event` is one of: `lap`, `pit_in`, `pit_out`, `lap_deleted`, `retired`,
`track_status`, `race_control`, `session_part_start`, `snapshot` (every `SNAPSHOT` s per driver).

Race laps have `lap_kind` `racing` / `in` / `out`; qualifying keeps `push` / `other` / `in` / `out`.
`gap_to_leader_s` is NaN for lapped cars (see `laps_down`). `race_lap` is the leader's lap,
`lap_number` the driver's. `pit_lane_s` on `pit_out` is the time from pit-lane entry to exit as
seen in the timing feed.

## Car telemetry

Live, `run_session.sh` starts `f1live.telemetry` next to the parser. The runner passes the last 120 s as
`info["telemetry"]` and the latest sample per driver as `car_speed/gear/throttle/brake/x/y` columns in `state`;
telemetry rows never trigger a model call. Archive-fetched sessions have no telemetry until you backfill:

```bash
python -m f1live.archive 2026 Singapore quali --telemetry            # one session -> raw_telemetry.txt
python -m f1live.fetch_season 2026 --telemetry                        # every session that has no telemetry.csv yet
python -m f1live.telemetry 2026 Singapore quali --replay             # (re)build telemetry.csv
```

## Pace prior for the cutoff (Q1/Q2)

`live.quali.model` gives each driver who still has a run to come an expected lap from what is known before the part:
season form, this weekend's practice and sprint pace, and the parts before it (`pre/quali/parts.py`, one ridge per part,
trained only on earlier weekends; `live/quali/prior.py` turns the gap into seconds). `USE_PRIOR = False` in the model
switches it off. Q3 (pole) doesn't use it: it did not improve the pole probabilities.

```bash
python -m pre.quali.parts                                # how well each part's pace is predicted, vs simple baselines
python -m live.quali.backtest_cutoff                     # baseline cutoff prediction on every quali in data/
python -m live.quali.backtest_cutoff --prior --csv x.csv # with the prior
```
Backtest, 42 qualis, one prediction every 30 s, same snapshots: mean absolute error of the predicted cutoff 0.629 s
-> 0.348 s, 90% interval coverage 51% -> 66%; most of the gain is early in a part (>7 min left: 1.36 s -> 0.67 s). The
prior also gives a prediction before anyone has set a time (Q2: 0.31 s error; Q1: 1.3 s, biased fast by ~0.9 s). The
pull of the prior (`live/quali/prior.py`) was picked using both years, so these numbers are fitted, not held out.
