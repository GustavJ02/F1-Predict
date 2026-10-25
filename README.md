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
  parse.py           raw.txt -> events.csv (race, sprint and qualifying)
  tail.py            CsvTail: read rows another process appends to a CSV
  runner.py          tails events.csv, builds per-driver state, calls a model, writes predictions.csv
  replay.py          re-plays one session's raw.txt into another session in (scaled) real time
  watch.py           terminal viewer for predictions.csv (used by the live.*.watch scripts)
  archive.py         downloads a finished session from F1's archive into raw.txt
  fetch_season.py    downloads (and parses) a whole season, or selected sessions, into data/<year>/<race>/<session>/
live/                predictions run DURING a session
  quali/             state.py (shared per-driver state), predict_elimination.py (Q1/Q2 cutoff),
                     predict_pole.py (Q3 pole probabilities), model.py (runner hook for both),
                     watch.py (live view), backtest_pole.py (score pole predictions on past sessions)
  race/              model.py (race model template, "current order" baseline - yours to edit),
                     watch.py (live view of the latest prediction, any columns)
pre/                 predictions made BEFORE a session, from past sessions
  history.py         pace tables for any session in data/ (best valid lap per driver, gap to the fastest in %)
  quali/             features.py (form, practice, sprint-quali inputs), predict_grid.py (pole probabilities and
                     expected grid), backtest.py (score it on every past quali, with ablations)
  race/              (not built yet)
run_session.sh       starts recorder + parser + runner together and restarts any that crash
data/                gitignored
  <year>/<race>/<session>/
    raw.txt          FastF1 recording, the one file that must not be lost
    events.csv       parsed events (rebuilt from raw.txt whenever the parser restarts)
    predictions.csv  runner output
```

Session folders are named however you like (`quali`, `race`, `sprint`, `replay`, ...); all
commands take `year race session` and are run from the repo root.

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
python -m f1live.fetch_season 2026 --sessions fp1 fp2 fp3 sq sprint --alias "United States=Austin"
python -m pre.quali.predict_grid 2026 Austin                         # pole probabilities + expected grid
python -m pre.quali.backtest                                         # score the model on every quali we have
```

Session folders from the archive are `fp1 fp2 fp3 sq sprint quali race`; the race folder is the meeting name
without "Grand Prix" (`Australian`, `Saudi Arabian`, `United States`, ...). `Austin` is an alias of `United States`.

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
