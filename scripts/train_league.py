"""Trains ONE side (P1='a' or P2='b') against a fixed, independently-specified frozen
opponent checkpoint -- the building block for league-style alternating self-play
(selfplay_league_loop.sh), where P1 and P2 are two separately-specialized policies
that keep training against each other's latest snapshot, each only ever needing to
be good at its own fixed role. This sidesteps the persistent a/b asymmetry that kept
showing up when a single shared policy had to generalize across both mirror frames
(self-play, mirror augmentation, paired self-play, and a provably mirror-equivariant
policy all measurably failed to fully fix it -- see docs/기술문서 section 4).

Unlike train_selfplay.py, the opponent here does NOT refresh mid-run -- one call =
one round against one fixed snapshot; the alternation happens between calls.
"""
import argparse
import pathlib
import time
from collections import deque

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from env import Fighter2DEnv, Fighter2DEnvForB

MODELS_DIR = pathlib.Path(__file__).resolve().parent.parent / "checkpoints"

# log_std has no ceiling in this project (ent_coef=0, no entropy-bonus push-back), so it just
# decays every round -- measured per-dimension on a real checkpoint, some joints (mostly legs)
# had already fallen to std~0.03-0.05 while others sat at 0.4+, well before the AGGREGATE std
# logged during training looked alarming. That per-dimension collapse tracked directly with
# fall-rate regressions across rounds (docs: std ~0.15->0.02 lined up with a 63%->81% fall-rate
# swing). Tried fixing this with an entropy bonus once (ent_coef=0.01) -- it overshot into a
# runaway positive-feedback loop instead (std -> 1.11e16 over 100 rounds, no ceiling of its own).
# Clamping log_std directly after every round sidesteps needing that delicate a balance: a floor
# guarantees a minimum of exploration/adaptability always survives, and a ceiling is just cheap
# insurance against a repeat of the same runaway-growth failure mode.
LOG_STD_MIN = np.log(0.05)   # std floor ~0.05 -- never fully deterministic
LOG_STD_MAX = np.log(0.6)    # std ceiling ~0.6 -- guards against unbounded growth


class OpponentRewardCallback(BaseCallback):
    """Same as train_selfplay.py's -- tracks the frozen opponent's mirrored episode
    reward (info['reward_b']) as rollout/b_ep_rew_mean, a stable "am I outplaying my
    own recent past" signal independent of reward-function rescaling."""

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
    parser.add_argument("--side", choices=["a", "b"], required=True,
                         help="which physical role this run's policy controls")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out", type=str, default="ppo_p1")
    parser.add_argument("--init-from", type=str, required=True,
                         help="checkpoint to continue THIS side's own policy from")
    parser.add_argument("--opponent-from", type=str, required=True,
                         help="frozen checkpoint for the OTHER side (fixed for this whole run)")
    parser.add_argument("--save-freq", type=int, default=25_000)
    parser.add_argument("--ent-coef", type=float, default=None)
    args = parser.parse_args()

    MODELS_DIR.mkdir(exist_ok=True)
    ckpt_dir = MODELS_DIR / "autosave"
    ckpt_dir.mkdir(exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.out}_{run_id}"

    env_cls = Fighter2DEnv if args.side == "a" else Fighter2DEnvForB
    vec_env = make_vec_env(
        env_cls, n_envs=args.n_envs, vec_env_cls=SubprocVecEnv,
        env_kwargs={"opponent_policy_path": args.opponent_from},
    )

    custom_objects = {"ent_coef": args.ent_coef} if args.ent_coef is not None else None
    model = PPO.load(args.init_from, env=vec_env, device=args.device, custom_objects=custom_objects)
    if args.ent_coef is not None:
        print(f"[league] overriding ent_coef -> {args.ent_coef}", flush=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.save_freq // args.n_envs, 1),
        save_path=str(ckpt_dir),
        name_prefix=run_name,
    )
    callbacks = CallbackList([checkpoint_callback, OpponentRewardCallback(), BreakdownCallback()])

    try:
        model.learn(total_timesteps=args.timesteps, callback=callbacks, reset_num_timesteps=False)
    finally:
        with torch.no_grad():
            before = model.policy.log_std.data.clone()
            model.policy.log_std.data.clamp_(min=LOG_STD_MIN, max=LOG_STD_MAX)
            n_clamped = int((model.policy.log_std.data != before).sum())
        if n_clamped:
            print(f"[league] clamped log_std on {n_clamped}/{before.numel()} action dims "
                  f"into [{np.exp(LOG_STD_MIN):.3f}, {np.exp(LOG_STD_MAX):.3f}] std range", flush=True)
        out_path = MODELS_DIR / run_name
        model.save(str(out_path))
        print(f"saved model to {out_path}.zip")


if __name__ == "__main__":
    main()
