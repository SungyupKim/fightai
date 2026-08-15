"""Self-play training: 'a' trains against periodically-refreshed snapshots of its
own policy (playing 'b', mirrored) instead of the fixed scripted opponent. Every
--refresh-interval steps, 'a's current weights are snapshotted and the opponent
in every parallel env is reloaded from that snapshot, so 'b' keeps pace with 'a's
own improvement instead of being a fixed, eventually-exploitable pattern.
"""
import argparse
import pathlib
import shutil
import time

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from env import Fighter2DEnv

MODELS_DIR = pathlib.Path(__file__).resolve().parent.parent / "checkpoints"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=5_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out", type=str, default="ppo_selfplay")
    parser.add_argument("--init-from", type=str, required=True,
                         help="checkpoint to bootstrap 'a' and the initial 'b' opponent from")
    parser.add_argument("--refresh-interval", type=int, default=250_000,
                         help="env steps between opponent snapshot refreshes")
    parser.add_argument("--save-freq", type=int, default=25_000,
                         help="env steps between autosave checkpoints")
    args = parser.parse_args()

    MODELS_DIR.mkdir(exist_ok=True)
    ckpt_dir = MODELS_DIR / "autosave"
    ckpt_dir.mkdir(exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.out}_{run_id}"

    snapshot_path = MODELS_DIR / f"selfplay_opponent_{run_id}.zip"
    shutil.copy(args.init_from, snapshot_path)

    vec_env = make_vec_env(
        Fighter2DEnv, n_envs=args.n_envs, vec_env_cls=SubprocVecEnv,
        env_kwargs={"opponent_policy_path": str(snapshot_path)},
    )

    model = PPO.load(args.init_from, env=vec_env, device=args.device)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.save_freq // args.n_envs, 1),
        save_path=str(ckpt_dir),
        name_prefix=run_name,
    )

    steps_done = 0
    try:
        while steps_done < args.timesteps:
            chunk = min(args.refresh_interval, args.timesteps - steps_done)
            model.learn(total_timesteps=chunk, callback=checkpoint_callback, reset_num_timesteps=False)
            steps_done += chunk

            snapshot_out = MODELS_DIR / f"{run_name}_snap{steps_done}"
            model.save(str(snapshot_out))
            shutil.copy(f"{snapshot_out}.zip", snapshot_path)
            vec_env.env_method("set_opponent", str(snapshot_path))
            print(f"[selfplay] opponent refreshed at {steps_done}/{args.timesteps} steps", flush=True)
    finally:
        out_path = MODELS_DIR / run_name
        model.save(str(out_path))
        print(f"saved model to {out_path}.zip")


if __name__ == "__main__":
    main()
