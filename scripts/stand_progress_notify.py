"""Sends a KakaoTalk progress summary for a stand-only locomotion run (train_locomotion.py --stand-only),
every --interval-minutes, plus a final message when the run saves its model.

The headline number is NOT read from the SB3 training log. That log's fall_rate/ep_len_mean are
measured with stochastic (exploration-noise) actions during rollout collection, and fall_rate
specifically means "never recovered within DOWN_RECOVERY_STEPS" (a_out), not "stayed balanced" --
both of those made it a misleading proxy before (docs: v5 looked like it went 0.98->0.6 fall_rate
while the real fixed-seed/deterministic 3-second-survival gate stayed at 0%). So every interval,
this script instead loads the latest autosave checkpoint and runs the actual gate_eval-style
measurement itself: N fixed seeds, deterministic actions, "did the torso stay above FALL_HEIGHT
for 3 straight seconds". That's the number reported first; the training-log fields are kept
underneath only as secondary/reference, explicitly labeled as possibly distorted.

Usage: stand_progress_notify.py --log ../checkpoints/locomotion_train_stand_v6.log \
    --ckpt-glob '../checkpoints/autosave/ppo_stand_v6_strongankle_*_steps.zip' --total-timesteps 80000000
"""
import argparse
import glob
import os
import pathlib
import re
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from kakao_notify import send_message

GATE_N = 30
GATE_WIN = 150  # 3 seconds at 0.02s/step

FIELDS = {
    "total_timesteps": "누적 스텝",
    "calm_frac": "커리큘럼(calm 시작 비율)",
    "fall_rate": "낙상률(학습 로그, 참고용)",
    "ep_len_mean": "평균 버틴 스텝(학습 로그, 참고용)",
    "r_com": "무게중심 보상(참고용)",
    "r_brace": "다리 펴기 보상(참고용)",
    "r_stand_streak": "연속 서기 보상(참고용)",
    "max_stand_streak_steps": "최대 연속 유지 스텝(2단계 전환 기준: 500)",
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


def latest_checkpoint(ckpt_glob):
    files = glob.glob(ckpt_glob)
    return max(files, key=os.path.getmtime) if files else None


def _run_gate(env, model):
    ok = 0
    from env import FALL_HEIGHT
    for seed in range(GATE_N):
        obs, _ = env.reset(seed=seed)
        fell = False
        for _ in range(GATE_WIN):
            action, _ = model.predict(obs, deterministic=True)
            obs, _r, term, trunc, _info = env.step(action)
            if env.data.xpos[env.a_torso_id][2] < FALL_HEIGHT:
                fell = True
                break
            if term or trunc:
                break
        if not fell:
            ok += 1
    return ok / GATE_N


def gate_survival_rate(ckpt_path):
    """Real measurement: fixed seeds 0..GATE_N-1, deterministic actions, does the torso stay
    above FALL_HEIGHT for a full 3 seconds -- on the env's DEFAULT calm/harsh mix. Loaded on
    CPU so it doesn't fight the training run for the GPU."""
    from stable_baselines3 import PPO

    from locomotion_env import LocomotionEnv

    model = PPO.load(ckpt_path, device="cpu")
    env = LocomotionEnv(stand_only=True)
    return _run_gate(env, model)


def gate_survival_rate_calm_only(ckpt_path):
    """Same gate, but every reset is forced calm (near-standing, severity 0..0.3, no push) --
    matches phase 1 of the curriculum (CurriculumGateCallback, train_locomotion.py). The
    default gate_survival_rate() always mixes in harsh/collapse starts, so during phase 1 it
    stays near 0% regardless of how well phase 1 is actually going (harsh starts dominate the
    failures) -- this one isolates the thing phase 1 is actually being measured on."""
    from stable_baselines3 import PPO

    from locomotion_env import LocomotionEnv

    model = PPO.load(ckpt_path, device="cpu")
    env = LocomotionEnv(stand_only=True)
    env.set_calm_frac(1.0)
    return _run_gate(env, model)


def format_message(v, total, gate_rate, gate_ckpt, calm_gate_rate):
    steps = v.get("total_timesteps", 0)
    pct = 100 * steps / total if total else 0
    calm_frac = v.get("calm_frac")
    if calm_frac is None:
        stage_line = None
    elif calm_frac >= 0.8:
        stage_line = f"커리큘럼: 1단계 - 서기 유지 중심 (calm {calm_frac:.0%})"
    elif calm_frac > 0.5:
        stage_line = f"커리큘럼: 전환 중 (calm {calm_frac:.0%} -> 45%로 낮아지는 중)"
    else:
        stage_line = f"커리큘럼: 2단계 - 전체 분포 (calm {calm_frac:.0%}, 쓰러진 자세 포함)"

    lines = [
        "[fightai] 서기 학습 경과",
        f"{FIELDS['total_timesteps']}: {steps/1e6:.1f}M / {total/1e6:.0f}M ({pct:.0f}%)",
    ]
    if stage_line:
        lines.append(stage_line)
    lines.append("")
    if gate_rate is None:
        lines.append("실제 3초 생존율: 아직 체크포인트 없음")
    else:
        lines.append(f"실제 3초 생존율 (전체 분포, 고정 시드 {GATE_N}개): {gate_rate:.0%}")
        lines.append(f"실제 3초 생존율 (calm만, 1단계 기준): {calm_gate_rate:.0%}")
        lines.append(f"  기준 체크포인트: {os.path.basename(gate_ckpt)}")
    lines.append("")
    lines.append("-- 아래는 학습 로그 수치, 왜곡될 수 있어 참고만 --")
    lines.append(f"{FIELDS['fall_rate']}: {v.get('fall_rate', float('nan')):.2f}")
    lines.append(f"{FIELDS['ep_len_mean']}: {v.get('ep_len_mean', float('nan')):.0f}스텝")
    lines.append(f"{FIELDS['r_com']}: {v.get('r_com', float('nan')):.1f}")
    lines.append(f"{FIELDS['r_brace']}: {v.get('r_brace', float('nan')):.1f}")
    lines.append(f"{FIELDS['r_stand_streak']}: {v.get('r_stand_streak', float('nan')):.1f}")
    lines.append(f"{FIELDS['max_stand_streak_steps']}: {v.get('max_stand_streak_steps', float('nan')):.0f}")
    lines.append(f"{FIELDS['std']}: {v.get('std', float('nan')):.3f}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--ckpt-glob", required=True,
                         help="glob for this run's autosave checkpoints, e.g. "
                              "'../checkpoints/autosave/ppo_stand_v6_strongankle_*_steps.zip'")
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
                ckpt = latest_checkpoint(args.ckpt_glob)
                gate_rate = gate_survival_rate(ckpt) if ckpt else None
                calm_gate_rate = gate_survival_rate_calm_only(ckpt) if ckpt else None
                send_message(format_message(v, args.total_timesteps, gate_rate, ckpt, calm_gate_rate))
                print(f"[stand-notify] sent at {last_steps:.0f} steps "
                      f"(gate={gate_rate if gate_rate is not None else 'n/a'}, "
                      f"calm_gate={calm_gate_rate if calm_gate_rate is not None else 'n/a'})", flush=True)
            if finished(log_path):
                if v:
                    ckpt = latest_checkpoint(args.ckpt_glob)
                    gate_rate = gate_survival_rate(ckpt) if ckpt else None
                    calm_gate_rate = gate_survival_rate_calm_only(ckpt) if ckpt else None
                    send_message(format_message(v, args.total_timesteps, gate_rate, ckpt, calm_gate_rate)
                                 + "\n\n학습 완료 (모델 저장됨)")
                print("[stand-notify] finished", flush=True)
                return
        except Exception as e:
            print(f"[stand-notify] failed: {e}", flush=True)
        time.sleep(args.interval_minutes * 60)


if __name__ == "__main__":
    main()
