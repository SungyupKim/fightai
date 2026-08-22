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
from collections import deque

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from env import Fighter2DEnv

MODELS_DIR = pathlib.Path(__file__).resolve().parent.parent / "checkpoints"


class OpponentRewardCallback(BaseCallback):
    """Tracks 'b's mirrored episode reward (info['reward_b']) alongside 'a's, so
    rollout/b_ep_rew_mean shows up next to the usual ep_rew_mean. Since 'b' is a
    frozen snapshot of a recent 'a', this is a much more stable progress signal
    than ep_rew_mean alone -- raw reward scale drifts every time the reward
    function changes, but a - b tells you whether 'a' is actually outplaying its
    own recent past, which is the whole point of self-play.
    """

    def __init__(self, window=100):
        super().__init__()
        self._running = None
        self._recent = deque(maxlen=window)

    def _on_training_start(self):
        self._running = np.zeros(self.training_env.num_envs)

    def _on_step(self):
        for i, info in enumerate(self.locals["infos"]):
            if "reward_b" not in info:
                continue
            self._running[i] += info["reward_b"]
            if self.locals["dones"][i]:
                self._recent.append(self._running[i])
                self._running[i] = 0.0
        if self._recent:
            self.logger.record("rollout/b_ep_rew_mean", float(np.mean(self._recent)))
        return True


class BreakdownCallback(BaseCallback):
    """Logs the per-episode average of each reward_breakdown term (effort, engage,
    strike, etc.) as its own rollout/r_* metric, so a swing in total ep_rew_mean can
    be traced back to which specific term moved instead of guessing from a one-off
    diagnostic script."""

    KEYS = ["strike", "engage", "progress", "height", "stability", "stance"]

    def __init__(self, window=100):
        super().__init__()
        self._running = None
        self._recent = {k: deque(maxlen=window) for k in self.KEYS}

    def _on_training_start(self):
        n = self.training_env.num_envs
        self._running = {k: np.zeros(n) for k in self.KEYS}

    def _on_step(self):
        for i, info in enumerate(self.locals["infos"]):
            bd = info.get("reward_breakdown")
            if not bd:
                continue
            for k in self.KEYS:
                self._running[k][i] += bd.get(k, 0.0)
            if self.locals["dones"][i]:
                for k in self.KEYS:
                    self._recent[k].append(self._running[k][i])
                    self._running[k][i] = 0.0
        for k in self.KEYS:
            if self._recent[k]:
                self.logger.record(f"rollout/r_{k}", float(np.mean(self._recent[k])))
        return True


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
    parser.add_argument("--ent-coef", type=float, default=None,
                         help="override the loaded checkpoint's entropy coefficient (e.g. 0.01) to "
                              "reopen exploration if the policy's action std has collapsed")
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

    custom_objects = {"ent_coef": args.ent_coef} if args.ent_coef is not None else None
    model = PPO.load(args.init_from, env=vec_env, device=args.device, custom_objects=custom_objects)
    if args.ent_coef is not None:
        print(f"[selfplay] overriding ent_coef -> {args.ent_coef}", flush=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.save_freq // args.n_envs, 1),
        save_path=str(ckpt_dir),
        name_prefix=run_name,
    )
    callbacks = CallbackList([checkpoint_callback, OpponentRewardCallback(), BreakdownCallback()])

    steps_done = 0
    try:
        while steps_done < args.timesteps:
            chunk = min(args.refresh_interval, args.timesteps - steps_done)
            model.learn(total_timesteps=chunk, callback=callbacks, reset_num_timesteps=False)
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
