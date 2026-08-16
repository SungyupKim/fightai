"""True simultaneous self-play training: unlike train_selfplay.py (which trains 'a'
against a periodically-refreshed frozen snapshot of itself), here both sides of every
match are driven by the SAME live policy every step via PairedSelfPlayVecEnv, and both
perspectives' transitions feed the same PPO rollout buffer. This removes the "'a' is
always the one currently exploring, 'b' is always a frozen reference" asymmetry that
persisted across every self-play run so far (including from a from-scratch checkpoint),
since there's no longer a distinct frozen 'b' -- every sample updates the one policy.
"""
import argparse
import pathlib
import time
from collections import deque

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback

from env import Fighter2DEnv
from selfplay_vec_env import PairedSelfPlayVecEnv

MODELS_DIR = pathlib.Path(__file__).resolve().parent.parent / "checkpoints"


class BreakdownCallback(BaseCallback):
    """Logs the per-episode average of each reward_breakdown term, same as
    train_selfplay.py's -- now averaged uniformly across every slot (both the 'a' and
    'b' perspective of every match count as regular envs here, no a/b split needed)."""

    KEYS = ["strike", "effort", "jerk", "engage", "down", "knockdown_entry", "balance", "ground", "knee", "terminal"]

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
                    # most of these are signed a-vs-b terms, and every match now contributes
                    # both its 'a' slot (+X) and 'b' slot (-X) to this same pooled average --
                    # without abs() they'd cancel to ~0 regardless of how big the swings are,
                    # since that's just the zero-sum design working as intended, not "no signal"
                    self._recent[k].append(abs(self._running[k][i]))
                    self._running[k][i] = 0.0
        for k in self.KEYS:
            if self._recent[k]:
                self.logger.record(f"rollout/r_{k}", float(np.mean(self._recent[k])))
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=5_000_000)
    parser.add_argument("--n-envs", type=int, default=8,
                         help="number of physical MuJoCo sims (each contributes 2 training "
                              "samples per step -- an 'a' view and a 'b' view of the same match)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out", type=str, default="ppo_paired")
    parser.add_argument("--init-from", type=str, required=True,
                         help="checkpoint to bootstrap the single shared policy from")
    parser.add_argument("--save-freq", type=int, default=25_000,
                         help="env steps between autosave checkpoints")
    parser.add_argument("--ent-coef", type=float, default=None,
                         help="override the loaded checkpoint's entropy coefficient")
    args = parser.parse_args()

    MODELS_DIR.mkdir(exist_ok=True)
    ckpt_dir = MODELS_DIR / "autosave"
    ckpt_dir.mkdir(exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.out}_{run_id}"

    vec_env = PairedSelfPlayVecEnv(args.n_envs, Fighter2DEnv)

    custom_objects = {"ent_coef": args.ent_coef} if args.ent_coef is not None else None
    model = PPO.load(args.init_from, env=vec_env, device=args.device, custom_objects=custom_objects)
    if args.ent_coef is not None:
        print(f"[paired] overriding ent_coef -> {args.ent_coef}", flush=True)

    # save_freq counts total env steps across all 2*n_envs slots, same convention as
    # CheckpointCallback's internal (save_freq // num_envs) division
    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.save_freq // vec_env.num_envs, 1),
        save_path=str(ckpt_dir),
        name_prefix=run_name,
    )
    callbacks = CallbackList([checkpoint_callback, BreakdownCallback()])

    try:
        model.learn(total_timesteps=args.timesteps, callback=callbacks, reset_num_timesteps=False)
    finally:
        out_path = MODELS_DIR / run_name
        model.save(str(out_path))
        print(f"saved model to {out_path}.zip")


if __name__ == "__main__":
    main()
