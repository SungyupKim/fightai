#!/bin/bash
# Relaunches the combat viewer (watch.py) every time a league round advances, using the
# newest P1/P2 checkpoints -- so the viewer always shows the most recent matchup without
# manual relaunching. Exits once the league log reports the whole run finished.
#
# Usage: ./league_viewer_loop.sh [poll_seconds]
set -u
cd "$(dirname "$0")"
VENV_PY=../.venv/bin/python
LOG=../checkpoints/league_loop.log
VIEWER_LOG=../checkpoints/watch.log
POLL=${1:-30}
PID_FILE=../checkpoints/league_viewer.pid

current_round() {
  grep -o "\[league\] round [0-9]*/[0-9]*" "$LOG" | tail -1
}

launch_viewer() {
  local p1 p2
  p1=$(ls -t ../checkpoints/ppo_p1_2026*.zip | head -1)
  p2=$(ls -t ../checkpoints/ppo_p2_2026*.zip | head -1)
  if [ -f "$PID_FILE" ]; then
    kill "$(cat "$PID_FILE")" 2>/dev/null
    sleep 1
  fi
  DISPLAY=:1 setsid nohup $VENV_PY watch.py "$p1" --opponent "$p2" >> "$VIEWER_LOG" 2>&1 < /dev/null &
  echo $! > "$PID_FILE"
  disown
  echo "[viewer] $(date +%H:%M) round=$1 P1=$p1 P2=$p2 pid=$(cat "$PID_FILE")"
}

last=""
while true; do
  now=$(current_round)
  if [ -n "$now" ] && [ "$now" != "$last" ]; then
    launch_viewer "$now"
    last="$now"
  fi
  if tail -n 3 "$LOG" | grep -q "all .* rounds complete" && [ -n "$last" ]; then
    # the run's final round is already showing; leave that viewer open and stop polling
    break
  fi
  sleep "$POLL"
done
