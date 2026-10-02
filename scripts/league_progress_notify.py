"""Periodically parses selfplay_league_loop.sh's combined log (round announcements +
train_league.py's SB3 table output, all appended to the same file) and sends a KakaoTalk
progress summary -- same pattern as locomotion_progress_notify.py, for watching a league
run without staying at the terminal. Standalone loop, independent of the training process.

Usage: league_progress_notify.py [--interval-seconds 2700] [--log PATH] [--rounds N]
"""
import argparse
import pathlib
import re
import time

from kakao_notify import send_message

SCRIPTS = pathlib.Path(__file__).resolve().parent
DEFAULT_LOG = SCRIPTS.parent / "checkpoints" / "league_loop.log"

FIELDS = ["total_timesteps", "ep_rew_mean", "b_ep_rew_mean", "r_strike", "r_engage", "std"]
ROUND_RE = re.compile(r"\[league\] round (\d+)/(\d+): (\w+) \(\w\) vs frozen \w+")


def parse_latest_block(log_path):
    """league_loop.log accumulates across EVERY league run this project has ever done (months
    of history), and a round announcement line is tiny compared to a full SB3 table block --
    right after a fresh round starts, the most recent `--timesteps` worth of training may not
    have printed a single table yet. Scanning backwards past the round-start boundary in that
    gap would silently pick up stale metrics from an old, unrelated run instead of reporting
    "no stats yet" -- so FIELDS are only searched in lines AFTER the latest round announcement,
    never before it."""
    with open(log_path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 40_000))
        tail = f.read().decode(errors="ignore")
    lines = tail.splitlines()
    round_idx = None
    round_info = None
    for i in range(len(lines) - 1, -1, -1):
        m = ROUND_RE.match(lines[i].strip())
        if m:
            round_idx = i
            round_info = (int(m.group(1)), int(m.group(2)), m.group(3))
            break
    values = {}
    search_lines = lines[round_idx:] if round_idx is not None else lines
    for line in reversed(search_lines):
        m = re.match(r"\|\s*(\w+)\s*\|\s*([-+\d.eE]+)\s*\|", line.strip())
        if m and m.group(1) in FIELDS and m.group(1) not in values:
            values[m.group(1)] = float(m.group(2))
        if len(values) == len(FIELDS):
            break
    return values, round_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval-seconds", type=int, default=2700)  # 45 min
    parser.add_argument("--log", type=str, default=str(DEFAULT_LOG))
    args = parser.parse_args()
    log_path = pathlib.Path(args.log)

    while True:
        time.sleep(args.interval_seconds)
        try:
            v, round_info = parse_latest_block(log_path)
            if not v:
                continue
            round_line = (f"라운드: {round_info[0]}/{round_info[1]} ({round_info[2]} 학습 중)"
                          if round_info else "라운드: ?")
            steps_m = v.get("total_timesteps", 0) / 1_000_000
            msg = (
                f"[fightai] 리그 진행 상황\n"
                f"{round_line}\n"
                f"스텝: {steps_m:.2f}M\n"
                f"std: {v.get('std', float('nan')):.3f}\n"
                f"ep_rew_mean: {v.get('ep_rew_mean', float('nan')):.1f}\n"
                f"b_ep_rew_mean: {v.get('b_ep_rew_mean', float('nan')):.1f}\n"
                f"r_strike: {v.get('r_strike', float('nan')):.1f}  "
                f"r_engage: {v.get('r_engage', float('nan')):.1f}"
            )
            send_message(msg)
            print(f"[league-progress-notify] sent: {msg}", flush=True)
        except Exception as e:
            print(f"[league-progress-notify] failed: {e}", flush=True)


if __name__ == "__main__":
    main()
