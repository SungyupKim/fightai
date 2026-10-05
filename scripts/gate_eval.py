"""Deterministic, seed-paired evaluation of a combat checkpoint against a fixed opponent.

Every episode uses a fixed reset seed and deterministic actions for both fighters, so the same
checkpoint always gives the same numbers, and two checkpoints are compared on identical starts.
(Stochastic evaluation swung the same checkpoint's knockdown rate by ~15 points between runs.)

Usage: gate_eval.py <checkpoint> <opponent> [--episodes 120] [--perturb]
"""
import argparse

import numpy as np
from stable_baselines3 import PPO

import env as env_module
from env import Fighter2DEnv, FALL_HEIGHT, STANDING_HEAD_HEIGHT


def evaluate(checkpoint, opponent, episodes, perturb):
    env_module.START_PERTURB_PROB = 0.3 if perturb else 0.0
    env_module.START_PUSH_PROB = 0.2 if perturb else 0.0
    model = PPO.load(checkpoint, device="cpu")
    env = Fighter2DEnv(opponent_policy_path=opponent, getup_curriculum_prob=0.0)
    opp_predict = env.opponent_policy.predict
    env.opponent_policy.predict = lambda obs, **kw: opp_predict(obs, deterministic=True)

    knocked, step_down, head_ratio, wins, losses = [], [], [], 0, 0
    for seed in range(episodes):
        obs, info = env.reset(seed=seed)
        down, heads = [], []
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            heads.append(info["a_head_z"] / STANDING_HEAD_HEIGHT)
            down.append(env.data.xpos[env.a_torso_id][2] < FALL_HEIGHT)
            done = terminated or truncated
        down = np.array(down)
        knocked.append(bool(down.any()))
        step_down.append(float(down.mean()))
        head_ratio.append(float(np.mean(heads)))
        if info.get("a_out"):
            losses += 1
        elif info.get("b_out"):
            wins += 1
    return {
        "knockdown_episodes": float(np.mean(knocked)),
        "step_down": float(np.mean(step_down)),
        "head_ratio": float(np.median(head_ratio)),
        "wins": wins,
        "losses": losses,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("opponent")
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--perturb", action="store_true", help="start with stance perturbation + shove")
    args = parser.parse_args()
    m = evaluate(args.checkpoint, args.opponent, args.episodes, args.perturb)
    print(f"knockdown_episodes={m['knockdown_episodes']:.3f} step_down={m['step_down']:.3f} "
          f"head_ratio={m['head_ratio']:.3f} wins={m['wins']} losses={m['losses']} "
          f"(n={args.episodes}, perturb={args.perturb})")


if __name__ == "__main__":
    main()
