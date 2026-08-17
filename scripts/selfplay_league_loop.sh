#!/bin/bash
# Alternates training P1 ('a') and P2 ('b') against each other's latest frozen
# snapshot -- AlphaGo-style league self-play. Each side only ever needs to be good
# at its own fixed physical role, sidestepping the mirror-generalization problem a
# single shared policy kept running into.
#
# Whichever side trains SECOND each round faces a strictly fresher (just-updated)
# opponent than whoever trained first -- a real, structural "second-mover" edge
# that would otherwise always favor the same side every round and compound instead
# of cancelling out. So which side goes first alternates by round parity too.
#
# Usage: ./selfplay_league_loop.sh [rounds] [timesteps_per_round] p1_init p2_init
set -u
cd "$(dirname "$0")"
VENV_PY=../.venv/bin/python
ROUNDS=${1:-20}
TIMESTEPS=${2:-1000000}
P1_INIT=${3:?usage: selfplay_league_loop.sh [rounds] [timesteps] p1_init_checkpoint p2_init_checkpoint}
P2_INIT=${4:?usage: selfplay_league_loop.sh [rounds] [timesteps] p1_init_checkpoint p2_init_checkpoint}
LOG=../checkpoints/league_loop.log

p1_latest="$P1_INIT"
p2_latest="$P2_INIT"

train_p1() {
  local round="$1"
  echo "[league] round $round/$ROUNDS: P1 (a) vs frozen P2 ($p2_latest)" | tee -a "$LOG"
  $VENV_PY train_league.py --side a --timesteps "$TIMESTEPS" --n-envs 8 \
    --init-from "$p1_latest" --opponent-from "$p2_latest" --out ppo_p1 >> "$LOG" 2>&1
  local new_p1
  new_p1=$(ls -t ../checkpoints/ppo_p1_2*.zip 2>/dev/null | grep -v snap | head -1)
  if [ -z "$new_p1" ]; then
    echo "[league] P1 round failed, aborting" | tee -a "$LOG"
    exit 1
  fi
  p1_latest="$new_p1"
}

train_p2() {
  local round="$1"
  echo "[league] round $round/$ROUNDS: P2 (b) vs frozen P1 ($p1_latest)" | tee -a "$LOG"
  $VENV_PY train_league.py --side b --timesteps "$TIMESTEPS" --n-envs 8 \
    --init-from "$p2_latest" --opponent-from "$p1_latest" --out ppo_p2 >> "$LOG" 2>&1
  local new_p2
  new_p2=$(ls -t ../checkpoints/ppo_p2_2*.zip 2>/dev/null | grep -v snap | head -1)
  if [ -z "$new_p2" ]; then
    echo "[league] P2 round failed, aborting" | tee -a "$LOG"
    exit 1
  fi
  p2_latest="$new_p2"
}

for i in $(seq 1 "$ROUNDS"); do
  if [ $((i % 2)) -eq 1 ]; then
    train_p1 "$i"
    train_p2 "$i"
  else
    train_p2 "$i"
    train_p1 "$i"
  fi
done
echo "[league] all $ROUNDS rounds complete. P1=$p1_latest P2=$p2_latest" | tee -a "$LOG"
