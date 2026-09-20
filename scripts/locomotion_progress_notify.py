"""Periodically parses the locomotion training log's latest rollout block and sends a
KakaoTalk progress summary -- for watching the overnight run (docs 10.33) without staying
at the terminal. Standalone loop, independent of any training process so it keeps
reporting even if a training run gets restarted underneath it.

Usage: locomotion_progress_notify.py [--interval-seconds 2700] [--log PATH]
"""
import argparse
import pathlib
import re
import time

from kakao_notify import send_message

SCRIPTS = pathlib.Path(__file__).resolve().parent
DEFAULT_LOG = SCRIPTS.parent / "checkpoints" / "locomotion_train_v2.log"

FIELDS = ["fall_rate", "r_velocity", "r_height", "r_stability", "total_timesteps", "std", "ep_len_mean"]


def parse_latest_block(log_path):
    with open(log_path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 20_000))
        tail = f.read().decode(errors="ignore")
    values = {}
    for line in reversed(tail.splitlines()):
        m = re.match(r"\|\s*(\w+)\s*\|\s*([-+\d.eE]+)\s*\|", line.strip())
        if m and m.group(1) in FIELDS and m.group(1) not in values:
            values[m.group(1)] = float(m.group(2))
        if len(values) == len(FIELDS):
            break
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval-seconds", type=int, default=2700)  # 45 min
    parser.add_argument("--log", type=str, default=str(DEFAULT_LOG))
    parser.add_argument("--total-timesteps", type=float, default=None,
                         help="denominator for the step-progress line; omit to just show "
                              "the raw step count with no /target (was hardcoded '/40M' "
                              "regardless of the actual run's target -- docs 10.43)")
    args = parser.parse_args()
    log_path = pathlib.Path(args.log)

    while True:
        time.sleep(args.interval_seconds)
        try:
            v = parse_latest_block(log_path)
            if not v:
                continue
            steps_m = v.get("total_timesteps", 0) / 1_000_000
            steps_line = (f"스텝: {steps_m:.1f}M / {args.total_timesteps / 1_000_000:.0f}M"
                          if args.total_timesteps else f"스텝: {steps_m:.1f}M")
            msg = (
                f"[fightai] 이동 모델 진행 상황\n"
                f"{steps_line}\n"
                f"fall_rate: {v.get('fall_rate', float('nan')):.3f}\n"
                f"std: {v.get('std', float('nan')):.3f}\n"
                f"r_velocity: {v.get('r_velocity', float('nan')):.1f}\n"
                f"r_height: {v.get('r_height', float('nan')):.1f}  "
                f"r_stability: {v.get('r_stability', float('nan')):.1f}\n"
                f"ep_len_mean: {v.get('ep_len_mean', float('nan')):.0f}"
            )
            send_message(msg)
            print(f"[progress-notify] sent: {msg}", flush=True)
        except Exception as e:
            print(f"[progress-notify] failed: {e}", flush=True)


if __name__ == "__main__":
    main()
