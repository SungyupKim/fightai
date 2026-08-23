"""Waits for a training PID to finish, validates the resulting checkpoint's fall rate
vs the scripted opponent, and if it clears FALL_RATE_THRESHOLD, bootstraps P2 and
launches the dynamic league loop -- sending a KakaoTalk update at each phase so this
can run unattended. If validation fails, keeps extending training (same reward/physics,
just more steps) and re-checking. Stops when improvement plateaus (PLATEAU_PATIENCE
iterations in a row under PLATEAU_MIN_IMPROVEMENT) rather than a fixed iteration count
-- MAX_ITERATIONS is just a hard safety cap so it can't loop forever.
"""
import argparse
import json
import pathlib
import subprocess
import time
from collections import Counter

from stable_baselines3 import PPO

from env import Fighter2DEnv, FALL_HEIGHT
from kakao_notify import send_message

FALL_RATE_THRESHOLD = 0.30      # a_down_timeout fraction vs scripted bot -- must beat the best so far (0.33, scratch9)
PLATEAU_MIN_IMPROVEMENT = 0.03  # an iteration must shave at least 3 points off the best-so-far rate to count as progress
PLATEAU_PATIENCE = 2            # stop after this many iterations in a row with no real improvement
MAX_ITERATIONS = 10             # hard safety cap regardless of plateau detection
EXTEND_TIMESTEPS = 2_000_000
CKPT_DIR = pathlib.Path("../checkpoints")


def wait_for_pid(pid):
    while True:
        try:
            import os
            os.kill(pid, 0)
        except OSError:
            return
        time.sleep(15)


def fall_rate_vs_scripted(checkpoint, episodes=30):
    model = PPO.load(checkpoint, device="cpu")
    env = Fighter2DEnv()
    reasons = []
    obs, info = env.reset()
    while len(reasons) < episodes:
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            a_z = env.data.xpos[env.a_torso_id][2]
            b_z = env.data.xpos[env.b_torso_id][2]
            reason = (
                "a_ko" if info["health_a"] <= 0 else
                "b_ko" if info["health_b"] <= 0 else
                "a_down_timeout" if a_z < FALL_HEIGHT else
                "b_down_timeout" if b_z < FALL_HEIGHT else "truncated"
            )
            reasons.append(reason)
            obs, info = env.reset()
    counts = Counter(reasons)
    a_falls = counts.get("a_down_timeout", 0)
    return a_falls / episodes, dict(counts)


def latest_ckpt(prefix):
    return str(max(CKPT_DIR.glob(f"{prefix}_2*.zip"), key=lambda p: p.stat().st_mtime))


def extend_training(checkpoint, out_name):
    subprocess.run(
        ["../.venv/bin/python", "train.py", "--timesteps", str(EXTEND_TIMESTEPS), "--n-envs", "8",
         "--resume-from", checkpoint, "--out", out_name],
        cwd=".", check=True,
    )
    return latest_ckpt(out_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--out-prefix", required=True, help="--out prefix the training run will produce")
    parser.add_argument("--label", default="checkpoint")
    args = parser.parse_args()

    send_message(f"[fightai] {args.label} 학습 대기 시작 (PID {args.wait_pid})")
    wait_for_pid(args.wait_pid)

    checkpoint = latest_ckpt(args.out_prefix)
    rate, counts = fall_rate_vs_scripted(checkpoint)
    send_message(f"[fightai] {args.label} 검증: 낙상률 {rate*100:.0f}% ({counts})")

    iteration = 0
    best_rate = rate
    best_checkpoint = checkpoint
    no_improve_streak = 0
    while rate > FALL_RATE_THRESHOLD:
        iteration += 1
        if iteration > MAX_ITERATIONS:
            send_message(f"[fightai] 안전 상한({MAX_ITERATIONS}회) 도달, 자동 진행 중단 (최선 "
                         f"{best_rate*100:.0f}%) -- 확인 필요")
            return
        send_message(f"[fightai] 기준 미달, {iteration}차 추가 학습 시작 ({EXTEND_TIMESTEPS//1_000_000}M스텝)")
        out_name = f"{args.out_prefix}_ext{iteration}"
        checkpoint = extend_training(checkpoint, out_name)
        rate, counts = fall_rate_vs_scripted(checkpoint)
        send_message(f"[fightai] {iteration}차 추가 학습 검증: 낙상률 {rate*100:.0f}% ({counts})")

        if best_rate - rate >= PLATEAU_MIN_IMPROVEMENT:
            best_rate, best_checkpoint = rate, checkpoint
            no_improve_streak = 0
        else:
            no_improve_streak += 1
            if rate < best_rate:  # still track the best checkpoint even if improvement was too small to reset the streak
                best_rate, best_checkpoint = rate, checkpoint
            if no_improve_streak >= PLATEAU_PATIENCE:
                send_message(f"[fightai] {PLATEAU_PATIENCE}회 연속 유의미한 개선 없음 (최선 "
                             f"{best_rate*100:.0f}%), 정체로 판단해 자동 진행 중단 -- 확인 필요")
                return

    send_message(f"[fightai] 기준 통과 (낙상률 {rate*100:.0f}%), P2 부트스트랩 시작")
    p2_out = "ppo_p2_bootstrap_auto"
    subprocess.run(
        ["../.venv/bin/python", "train_league.py", "--side", "b", "--timesteps", "1000000",
         "--n-envs", "8", "--init-from", checkpoint, "--opponent-from", checkpoint,
         "--out", p2_out],
        cwd=".", check=True,
    )
    p2_ckpt = latest_ckpt(p2_out)

    matchup = subprocess.run(
        ["../.venv/bin/python", "matchup_eval.py", checkpoint, p2_ckpt, "--episodes", "30"],
        cwd=".", capture_output=True, text=True, check=True,
    )
    result = json.loads(matchup.stdout.strip().splitlines()[-1])
    send_message(f"[fightai] P2 부트스트랩 완료. 초기 매치업 a_losses={result['a_losses']} "
                 f"b_losses={result['b_losses']}. 리그 시작합니다.")

    subprocess.Popen(
        ["../.venv/bin/python", "selfplay_league_dynamic.py", "--rounds", "30",
         "--timesteps", "1000000", "--eval-episodes", "30", "--margin", "3",
         "--max-consecutive", "15", "--p1-init", checkpoint, "--p2-init", p2_ckpt],
        cwd=".", stdout=open("../checkpoints/league_dynamic_auto.log", "a"),
        stderr=subprocess.STDOUT,
    )


if __name__ == "__main__":
    main()
