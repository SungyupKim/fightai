"""Trains fighter 'a' with PPO against the scripted shadow-boxing opponent 'b'."""
import argparse
import pathlib

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from env import Fighter2DEnv

MODELS_DIR = pathlib.Path(__file__).resolve().parent.parent / "checkpoints"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out", type=str, default="ppo_vs_scripted")
    parser.add_argument("--save-freq", type=int, default=5_000,
                         help="env steps between checkpoints (divided across n-envs internally)")
    parser.add_argument("--resume-from", type=str, default=None,
                         help="path to a .zip checkpoint to resume training from")
    args = parser.parse_args()

    MODELS_DIR.mkdir(exist_ok=True)
    ckpt_dir = MODELS_DIR / "autosave"
    ckpt_dir.mkdir(exist_ok=True)

    vec_env = make_vec_env(Fighter2DEnv, n_envs=args.n_envs, vec_env_cls=SubprocVecEnv)

    if args.resume_from:
        model = PPO.load(args.resume_from, env=vec_env, device=args.device)
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            device=args.device,
            n_steps=512,
            batch_size=1024,
            learning_rate=3e-4,
            verbose=1,
            tensorboard_log=str(MODELS_DIR / "tb"),
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.save_freq // args.n_envs, 1),
        save_path=str(ckpt_dir),
        name_prefix=args.out,
    )

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=checkpoint_callback,
            reset_num_timesteps=args.resume_from is None,
        )
    finally:
        out_path = MODELS_DIR / args.out
        model.save(str(out_path))
        print(f"saved model to {out_path}.zip")


if __name__ == "__main__":
    main()
