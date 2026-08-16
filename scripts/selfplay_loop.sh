#!/bin/bash
# Runs self-play training repeatedly, each round resuming from the previous
# round's final checkpoint. Usage: ./selfplay_loop.sh [rounds] [timesteps_per_round] [refresh_interval]
set -u
cd "$(dirname "$0")"
VENV_PY=../.venv/bin/python
ROUNDS=${1:-10}
TIMESTEPS=${2:-5000000}
REFRESH=${3:-250000}
LOG=../checkpoints/train_selfplay_loop.log

for i in $(seq 1 "$ROUNDS"); do
  LATEST=$(ls -t ../checkpoints/ppo_selfplay_2*.zip 2>/dev/null | grep -v snap | head -1)
  if [ -z "$LATEST" ]; then
    echo "[loop] no checkpoint found, aborting" | tee -a "$LOG"
    exit 1
  fi
  echo "[loop] round $i/$ROUNDS, resuming from $LATEST" | tee -a "$LOG"
  $VENV_PY train_selfplay.py --timesteps "$TIMESTEPS" --refresh-interval "$REFRESH" --n-envs 8 \
    --init-from "$LATEST" --out ppo_selfplay >> "$LOG" 2>&1
done
echo "[loop] all $ROUNDS rounds complete" | tee -a "$LOG"
