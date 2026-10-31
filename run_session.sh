#!/usr/bin/env bash
# Start recording, parsing and the model runner together; restart any of them if it dies.
# Everything lives in data/<year>/<race>/<session>/ (raw.txt, events.csv, predictions.csv).
# While recording, the session folder is also pushed to the central store (F1_REMOTE, default rpi:/srv/F1-data)
# every SYNC_EVERY seconds and once more on exit; if the store is unreachable the session carries on and retries.
# Usage:  ./run_session.sh 2026 Austin race            (live session, needs F1TV login for the recorder)
#         REPLAY=2026/Austin/quali SPEED=10 ./run_session.sh 2026 Austin replay   (test with a recording)
# Env:    MODEL=live.quali.my_model (default live.race.model; fp1/fp2/fp3 run no model unless MODEL is set)  SNAPSHOT=5  TICK=5
#         SYNC=0 (don't push)  SYNC_EVERY=30  F1_DATA=<local data dir, default data>  F1_REMOTE=<host:path>
set -u
cd "$(dirname "$0")"
[[ $# -eq 3 ]] || { sed -n '2,6p' "$0"; exit 1; }
YEAR=$1 RACE=$2 SESSION=$3
DATA=${F1_DATA:-data}
DIR=$DATA/$YEAR/$RACE/$SESSION
RAW=$DIR/raw.txt
SNAPSHOT=${SNAPSHOT:-5}
TICK=${TICK:-5}
# practice has nothing to predict: no model runner for fp1/fp2/fp3 unless MODEL is set (watch it with python -m live.practice.watch)
RUN_MODEL=1; [[ $SESSION =~ ^fp[123]$ && -z ${MODEL:-} ]] && RUN_MODEL=0
MODEL=${MODEL:-live.race.model}
if [[ -n "${REPLAY:-}" && ! -f "$DATA/$REPLAY/raw.txt" ]]; then
  echo "REPLAY source $DATA/$REPLAY/raw.txt does not exist" >&2; exit 1
fi
mkdir -p "$DIR"    # the recorder won't create it

restart() {   # restart <name> <cmd...> : run forever, restarting after crashes
  local name=$1; shift
  while true; do
    "$@"
    echo "[$name] exited ($?), restarting in 2 s"; sleep 2
  done
}

push() { python -m f1live.sync push "$YEAR" "$RACE" "$SESSION" --quiet; }
# replays are throw-away test data: never pushed
SYNC=${SYNC:-1}; [[ -n "${REPLAY:-}" ]] && SYNC=0

cleanup() {
  trap '' INT TERM            # survive our own kill 0 so the final push can run
  echo stopping; kill 0; wait 2>/dev/null
  [[ $SYNC == 1 ]] && { echo "final push to the central store"; push || echo "[sync] final push failed, run: python -m f1live.sync push $YEAR $RACE $SESSION"; }
  exit 0
}
trap cleanup INT TERM
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
restart telemetry python -m f1live.telemetry "$YEAR" "$RACE" "$SESSION" & pids+=($!)
[[ $RUN_MODEL == 1 ]] && { restart runner python -m f1live.runner "$YEAR" "$RACE" "$SESSION" --model "$MODEL" --tick "$TICK" & pids+=($!); }
if [[ $SYNC == 1 ]]; then
  ( while true; do sleep "${SYNC_EVERY:-30}"; push || echo "[sync] push failed, retrying in ${SYNC_EVERY:-30} s"; done ) & pids+=($!)
fi

wait
