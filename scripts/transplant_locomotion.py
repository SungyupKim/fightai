"""Transplants a LocomotionEnv-trained policy (67-dim obs: the 66 Fighter2DEnv dims + a
trailing target_vx command) into a fresh Fighter2DEnv-shaped policy (66-dim obs) usable as
--init-from for train_league.py.

LocomotionEnv._obs() is `concatenate([Fighter2DEnv._obs(), [target_vx]])` -- the first 66
dims are IDENTICAL in meaning and order to the combat env's own obs, so the transplant is
just: build a fresh 66-dim policy with the same architecture, copy every weight straight
across except the first Linear layer of policy_net/value_net, where the extra 67th input
column (target_vx) is dropped.

Usage: transplant_locomotion.py <locomotion_checkpoint.zip> [--out NAME]
"""
import argparse
import pathlib

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from env import Fighter2DEnv

MODELS_DIR = pathlib.Path(__file__).resolve().parent.parent / "checkpoints"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--out", type=str, default="ppo_locomotion_transplant")
    args = parser.parse_args()

    src = PPO.load(args.checkpoint, device="cpu")
    src_obs_dim = src.observation_space.shape[0]

    vec_env = make_vec_env(Fighter2DEnv, n_envs=1, env_kwargs={"opponent_policy_path": None})
    # verbose=1 so downstream training (e.g. train_league.py) actually logs progress -- a
    # transplant built without it silently carries verbose=0 into every round that descends
    # from it, since PPO.load() otherwise just restores whatever was saved.
    dst = PPO("MlpPolicy", vec_env, device="cpu", verbose=1)
    dst_obs_dim = dst.observation_space.shape[0]

    assert src_obs_dim == dst_obs_dim + 1, (
        f"expected source obs dim ({src_obs_dim}) to be exactly one more than the combat env's "
        f"({dst_obs_dim}) -- LocomotionEnv's obs layout may have changed, re-check before transplanting"
    )
    assert src.action_space.shape == dst.action_space.shape, "action space mismatch"

    src_state = src.policy.state_dict()
    dst_state = dst.policy.state_dict()
    for key in dst_state:
        src_tensor = src_state[key]
        if src_tensor.shape == dst_state[key].shape:
            dst_state[key] = src_tensor.clone()
        else:
            # only the first Linear layer of each mlp_extractor sub-net should differ, by
            # exactly one input column (the trailing target_vx dim)
            assert key.endswith(".0.weight"), f"unexpected shape mismatch at {key}"
            assert src_tensor.shape == (dst_state[key].shape[0], dst_state[key].shape[1] + 1), (
                f"unexpected shape at {key}: src={tuple(src_tensor.shape)} "
                f"dst={tuple(dst_state[key].shape)}"
            )
            dst_state[key] = src_tensor[:, :-1].clone()
            print(f"[transplant] dropped trailing input column at {key}: "
                  f"{tuple(src_tensor.shape)} -> {tuple(dst_state[key].shape)}")

    dst.policy.load_state_dict(dst_state)

    # sanity check: a forward pass on the transplanted policy runs and produces the right shape
    obs = vec_env.reset()
    action, _ = dst.predict(obs, deterministic=True)
    print(f"[transplant] sanity check ok -- action shape {action.shape}")

    MODELS_DIR.mkdir(exist_ok=True)
    out_path = MODELS_DIR / args.out
    dst.save(str(out_path))
    print(f"saved transplanted model to {out_path}.zip")


if __name__ == "__main__":
    main()
