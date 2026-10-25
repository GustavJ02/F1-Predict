#!/usr/bin/env bash
# Start recording, parsing and the model runner together; restart any of them if it dies.
# Everything lives in data/<year>/<race>/<session>/ (raw.txt, events.csv, predictions.csv).
# Usage:  ./run_session.sh 2026 Austin race            (live session, needs F1TV login for the recorder)
#         REPLAY=2026/Austin/quali SPEED=10 ./run_session.sh 2026 Austin replay   (test with a recording)
# Env:    MODEL=live.quali.my_model (default live.race.model)  SNAPSHOT=5  TICK=5
set -u
cd "$(dirname "$0")"
[[ $# -eq 3 ]] || { sed -n '2,6p' "$0"; exit 1; }
YEAR=$1 RACE=$2 SESSION=$3
DIR=data/$YEAR/$RACE/$SESSION
RAW=$DIR/raw.txt
SNAPSHOT=${SNAPSHOT:-5}
TICK=${TICK:-5}
MODEL=${MODEL:-live.race.model}
if [[ -n "${REPLAY:-}" && ! -f "data/$REPLAY/raw.txt" ]]; then
  echo "REPLAY source data/$REPLAY/raw.txt does not exist" >&2; exit 1
fi
mkdir -p "$DIR"    # the recorder won't create it

restart() {   # restart <name> <cmd...> : run forever, restarting after crashes
  local name=$1; shift
  while true; do
    "$@"
    echo "[$name] exited ($?), restarting in 2 s"; sleep 2
  done
}

trap 'trap - INT TERM; echo stopping; kill 0' INT TERM
pids=()
if [[ -n "${REPLAY:-}" ]]; then
  python -m f1live.replay "$REPLAY" "$YEAR/$RACE/$SESSION" --speed "${SPEED:-1}" & pids+=($!)
else
  # --append so a restarted recorder continues the same file instead of overwriting it
  restart recorder python -m fastf1.livetiming save --append "$RAW" & pids+=($!)
fi
sleep 2
restart parser python -m f1live.parse "$YEAR" "$RACE" "$SESSION" --snapshot "$SNAPSHOT" --quiet & pids+=($!)
sleep 1
restart runner python -m f1live.runner "$YEAR" "$RACE" "$SESSION" --model "$MODEL" --tick "$TICK" & pids+=($!)

wait
