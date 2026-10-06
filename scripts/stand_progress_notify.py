"""Sends a KakaoTalk progress summary for a stand-only locomotion run (train_locomotion.py --stand-only),
every --interval-minutes, plus a final message when the run saves its model. Reads the SB3 log
the run writes to; no effect on training.

Usage: stand_progress_notify.py --log ../checkpoints/locomotion_train_stand_v5.log --total-timesteps 80000000
"""
import argparse
import pathlib
import re
import time

from kakao_notify import send_message

FIELDS = {
    "total_timesteps": "누적 스텝",
    "fall_rate": "낙상률(회복 실패)",
    "ep_len_mean": "평균 버틴 스텝",
    "r_height": "머리 높이 보상",
    "r_stability": "안정성 보상",
    "r_com": "무게중심 보상",
    "r_brace": "다리 펴기 보상",
    "std": "행동 흔들림(std)",
}
LINE_RE = re.compile(r"\|\s*([\w/]+)\s*\|\s*([-+\d.eE]+)\s*\|")


def latest_values(log_path):
    with open(log_path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 60_000))
        tail = f.read().decode(errors="ignore").splitlines()
    values = {}
    for line in reversed(tail):
        m = LINE_RE.search(line)
        if not m:
            continue
        key = m.group(1).split("/")[-1]
        if key in FIELDS and key not in values:
            values[key] = float(m.group(2))
        if len(values) == len(FIELDS):
            break
    return values


def finished(log_path):
    with open(log_path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 4000))
        return "saved model to" in f.read().decode(errors="ignore")


def format_message(v, total):
    steps = v.get("total_timesteps", 0)
    pct = 100 * steps / total if total else 0
    seconds = v.get("ep_len_mean", 0) * 0.02
    lines = [
        "[fightai] 서기 학습 경과",
        f"{FIELDS['total_timesteps']}: {steps/1e6:.1f}M / {total/1e6:.0f}M ({pct:.0f}%)",
        f"{FIELDS['fall_rate']}: {v.get('fall_rate', float('nan')):.2f}  (낮을수록 좋음)",
        f"{FIELDS['ep_len_mean']}: {v.get('ep_len_mean', float('nan')):.0f}스텝 (약 {seconds:.1f}초)",
        f"{FIELDS['r_height']}: {v.get('r_height', float('nan')):.0f}",
        f"{FIELDS['r_com']}: {v.get('r_com', float('nan')):.1f}",
        f"{FIELDS['r_brace']}: {v.get('r_brace', float('nan')):.1f}",
        f"{FIELDS['r_stability']}: {v.get('r_stability', float('nan')):.1f}",
        f"{FIELDS['std']}: {v.get('std', float('nan')):.3f}",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--total-timesteps", type=float, required=True)
    parser.add_argument("--interval-minutes", type=float, default=30)
    args = parser.parse_args()
    log_path = pathlib.Path(args.log)

    last_steps = None
    while True:
        try:
            v = latest_values(log_path)
            if v and v.get("total_timesteps") != last_steps:
                last_steps = v.get("total_timesteps")
                send_message(format_message(v, args.total_timesteps))
                print(f"[stand-notify] sent at {last_steps:.0f} steps", flush=True)
            if finished(log_path):
                if v:
                    send_message(format_message(v, args.total_timesteps) + "\n\n학습 완료 (모델 저장됨)")
                print("[stand-notify] finished", flush=True)
                return
        except Exception as e:
            print(f"[stand-notify] failed: {e}", flush=True)
        time.sleep(args.interval_minutes * 60)


if __name__ == "__main__":
    main()
