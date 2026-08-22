"""Runs N episodes of checkpoint_a (as 'a') vs checkpoint_b (as 'b', via the mirrored
self-play observation, same as diag_engage.py) and prints a single-line JSON summary
to stdout. Used by the dashboard's "대결 지표" panel to pull live matchup stats for
whichever two checkpoints are selected, instead of running diag_engage.py by hand.

Runs --n-envs copies in parallel subprocesses (same pattern as training) instead of one
episode at a time -- end-reason classification used to need direct env.data access
(torso height) which only works with a single in-process env, so this was serial. Now
that env.py's info dict exposes a_out/b_out directly (the same booleans it uses
internally for termination), classification only needs the info dict, which survives
the VecEnv subprocess boundary -- so this can run fully parallel.
"""
import argparse
import json
from collections import Counter

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from env import Fighter2DEnv, STANDING_HEAD_HEIGHT


def _make_env(checkpoint_b):
    def _init():
        return Fighter2DEnv(opponent_policy_path=checkpoint_b)
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_a")
    parser.add_argument("checkpoint_b")
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--n-envs", type=int, default=8)
    args = parser.parse_args()

    model = PPO.load(args.checkpoint_a, device="cpu")
    n_envs = min(args.n_envs, args.episodes)
    vec_env = SubprocVecEnv([_make_env(args.checkpoint_b) for _ in range(n_envs)])

    end_reasons = []
    lengths = []
    health_a_end = []
    health_b_end = []
    head_ratios_a = []  # a_head_z / STANDING_HEAD_HEIGHT, sampled every step for free --
    head_ratios_b = []  # reuses this same rollout instead of a separate posture pass
    hits_a = 0  # steps where 'a' landed a new hit on 'b' -- inferred from the sign of the
    hits_b = 0  # strike reward term (nonzero only on a new weapon-target contact), also free

    obs = vec_env.reset()
    ep_len = np.zeros(n_envs, dtype=int)
    while len(end_reasons) < args.episodes:
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, done, infos = vec_env.step(action)
        ep_len += 1
        for info in infos:
            head_ratios_a.append(info["a_head_z"] / STANDING_HEAD_HEIGHT)
            head_ratios_b.append(info["b_head_z"] / STANDING_HEAD_HEIGHT)
            strike = info["reward_breakdown"]["strike"]
            if strike > 0:
                hits_a += 1
            elif strike < 0:
                hits_b += 1
        for i, (d, info) in enumerate(zip(done, infos)):
            if not d:
                continue
            if info["health_a"] <= 0 and info["health_b"] <= 0:
                reason = "mutual_fall"
            elif info["health_a"] <= 0:
                reason = "a_ko"
            elif info["health_b"] <= 0:
                reason = "b_ko"
            elif info["a_out"]:
                reason = "a_down_timeout"
            elif info["b_out"]:
                reason = "b_down_timeout"
            else:
                reason = "truncated"
            end_reasons.append(reason)
            lengths.append(int(ep_len[i]))
            health_a_end.append(info["health_a"])
            health_b_end.append(info["health_b"])
            ep_len[i] = 0

    vec_env.close()

    # trim to exactly --episodes in case the last parallel step over-shot it
    end_reasons = end_reasons[:args.episodes]
    lengths = lengths[:args.episodes]
    health_a_end = health_a_end[:args.episodes]
    health_b_end = health_b_end[:args.episodes]

    counts = dict(Counter(end_reasons))
    a_losses = counts.get("a_down_timeout", 0) + counts.get("a_ko", 0)
    b_losses = counts.get("b_down_timeout", 0) + counts.get("b_ko", 0)
    result = {
        "episodes": len(end_reasons),
        "checkpoint_a": args.checkpoint_a,
        "checkpoint_b": args.checkpoint_b,
        "end_reason_counts": counts,
        "a_losses": a_losses,
        "b_losses": b_losses,
        "mean_length": float(np.mean(lengths)),
        "median_length": float(np.median(lengths)),
        "mean_health_a_end": float(np.mean(health_a_end)),
        "mean_health_b_end": float(np.mean(health_b_end)),
        "mean_head_ratio_a": float(np.mean(head_ratios_a)),
        "mean_head_ratio_b": float(np.mean(head_ratios_b)),
        "hits_a": hits_a,
        "hits_b": hits_b,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
