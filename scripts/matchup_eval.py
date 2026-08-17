"""Runs N episodes of checkpoint_a (as 'a') vs checkpoint_b (as 'b', via the mirrored
self-play observation, same as diag_engage.py) and prints a single-line JSON summary
to stdout. Used by the dashboard's "대결 지표" panel to pull live matchup stats for
whichever two checkpoints are selected, instead of running diag_engage.py by hand.
"""
import argparse
import json
from collections import Counter

import numpy as np
from stable_baselines3 import PPO

from env import Fighter2DEnv, FALL_HEIGHT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_a")
    parser.add_argument("checkpoint_b")
    parser.add_argument("--episodes", type=int, default=60)
    args = parser.parse_args()

    model = PPO.load(args.checkpoint_a, device="cpu")
    env = Fighter2DEnv(opponent_policy_path=args.checkpoint_b)

    end_reasons = []
    lengths = []
    health_a_end = []
    health_b_end = []

    obs, info = env.reset()
    ep_len = 0
    while len(end_reasons) < args.episodes:
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_len += 1
        if terminated or truncated:
            a_z = env.data.xpos[env.a_torso_id][2]
            b_z = env.data.xpos[env.b_torso_id][2]
            if info["health_a"] <= 0 and info["health_b"] <= 0:
                reason = "mutual_fall"
            elif info["health_a"] <= 0:
                reason = "a_ko"
            elif info["health_b"] <= 0:
                reason = "b_ko"
            elif a_z < FALL_HEIGHT:
                reason = "a_down_timeout"
            elif b_z < FALL_HEIGHT:
                reason = "b_down_timeout"
            else:
                reason = "truncated"
            end_reasons.append(reason)
            lengths.append(ep_len)
            health_a_end.append(info["health_a"])
            health_b_end.append(info["health_b"])
            ep_len = 0
            obs, info = env.reset()

    counts = dict(Counter(end_reasons))
    a_losses = counts.get("a_down_timeout", 0) + counts.get("a_ko", 0)
    b_losses = counts.get("b_down_timeout", 0) + counts.get("b_ko", 0)
    result = {
        "episodes": args.episodes,
        "checkpoint_a": args.checkpoint_a,
        "checkpoint_b": args.checkpoint_b,
        "end_reason_counts": counts,
        "a_losses": a_losses,
        "b_losses": b_losses,
        "mean_length": float(np.mean(lengths)),
        "median_length": float(np.median(lengths)),
        "mean_health_a_end": float(np.mean(health_a_end)),
        "mean_health_b_end": float(np.mean(health_b_end)),
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
