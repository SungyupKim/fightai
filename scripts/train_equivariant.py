"""Paired self-play training with MirrorEquivariantPolicy: the action/value output is
exactly mirror-equivariant by construction (see equivariant_policy.py), instead of just
being encouraged toward symmetry via training data (mirror augmentation, paired
self-play) -- which measurably wasn't enough on its own; the a/b asymmetry kept
reappearing after full convergence even with both of those in place.

Bootstraps a fresh MirrorEquivariantPolicy by transplanting weights from an existing
(ordinary ActorCriticPolicy) checkpoint, so combat competency built up over many
previous training rounds isn't thrown away -- only the forward pass changes, no new
learnable parameters are added.
"""
import argparse
import pathlib
import time
from collections import deque

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback

from env import Fighter2DEnv
from equivariant_policy import MirrorEquivariantPolicy, build_mirror_vectors, transplant_weights
from selfplay_vec_env import PairedSelfPlayVecEnv
from train_selfplay_paired import BreakdownCallback

MODELS_DIR = pathlib.Path(__file__).resolve().parent.parent / "checkpoints"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out", type=str, default="ppo_equivariant")
    parser.add_argument("--init-from", type=str, required=True,
                         help="ordinary ActorCriticPolicy checkpoint to transplant weights from")
    parser.add_argument("--save-freq", type=int, default=25_000)
    parser.add_argument("--ent-coef", type=float, default=None)
    args = parser.parse_args()

    MODELS_DIR.mkdir(exist_ok=True)
    ckpt_dir = MODELS_DIR / "autosave"
    ckpt_dir.mkdir(exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.out}_{run_id}"

    old_model = PPO.load(args.init_from, device="cpu")
    obs_mirror, action_mirror = build_mirror_vectors()

    vec_env = PairedSelfPlayVecEnv(args.n_envs, Fighter2DEnv)

    ent_coef = args.ent_coef if args.ent_coef is not None else old_model.ent_coef
    model = PPO(
        MirrorEquivariantPolicy, vec_env, device=args.device,
        n_steps=old_model.n_steps, batch_size=old_model.batch_size,
        learning_rate=old_model.learning_rate, gamma=old_model.gamma,
        gae_lambda=old_model.gae_lambda, ent_coef=ent_coef,
        policy_kwargs={"obs_mirror": obs_mirror, "action_mirror": action_mirror},
        verbose=1, tensorboard_log=str(MODELS_DIR / "tb"),
    )
    transplant_weights(model.policy, old_model.policy.state_dict())
    print(f"[equivariant] transplanted weights from {args.init_from}, ent_coef={ent_coef}", flush=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.save_freq // vec_env.num_envs, 1),
        save_path=str(ckpt_dir),
        name_prefix=run_name,
    )
    callbacks = CallbackList([checkpoint_callback, BreakdownCallback()])

    try:
        model.learn(total_timesteps=args.timesteps, callback=callbacks, reset_num_timesteps=True)
    finally:
        out_path = MODELS_DIR / run_name
        model.save(str(out_path))
        print(f"saved model to {out_path}.zip")


if __name__ == "__main__":
    main()
