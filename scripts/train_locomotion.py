"""Trains the solo mobility/recovery policy (locomotion_env.LocomotionEnv) -- no opponent,
no --side, no --opponent-from. See locomotion_env.py for what this is testing.

Usage: train_locomotion.py [--timesteps N] [--n-envs N] [--init-from CKPT] [--out NAME]
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

from locomotion_env import LocomotionEnv

MODELS_DIR = pathlib.Path(__file__).resolve().parent.parent / "checkpoints"

# same clamp as train_league.py (docs 10.20) -- keeps exploration alive without needing an
# entropy bonus (which overshot into runaway growth the one time it was tried).
LOG_STD_MIN = np.log(0.05)
LOG_STD_MAX = np.log(0.6)


class BreakdownCallback(BaseCallback):
    KEYS = ["height", "stability", "stance", "knee_avoid", "recovery", "jerk", "velocity"]

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


class LogStdClampCallback(BaseCallback):
    """Periodic version of the once-at-the-end clamp below -- a v3fresh run (docs 10.34)
    regressed hard: fair-condition fall rate 0%->12.5% and worst-case 35%->55.3% over its
    last ~9M steps, with std quietly narrowing the whole time (0.98->0.57) and no mid-run
    floor/ceiling to stop it, unlike train_league.py's per-round clamp. Clamping every
    `freq` steps instead of only in the `finally` block keeps a minimum exploration floor
    alive throughout training, not just in the final saved snapshot."""

    def __init__(self, freq=50_000):
        super().__init__()
        self.freq = freq
        self._last_clamp_step = 0

    def _on_step(self):
        if self.num_timesteps - self._last_clamp_step >= self.freq:
            with torch.no_grad():
                self.model.policy.log_std.data.clamp_(min=LOG_STD_MIN, max=LOG_STD_MAX)
            self._last_clamp_step = self.num_timesteps
        return True


class LinearScaffoldCallback(BaseCallback):
    """Shared machinery for annealing a LocomotionEnv scaffold parameter from `start_val` to
    `end_val` over `relax_frac` of THIS CALL's own timestep budget, then holding at `end_val`
    for good -- used by both KneeScaffoldCallback (docs 10.37/10.38) and
    VelocityAssistCallback (docs 10.45). Subclass and implement `apply(value)` to broadcast
    the annealed value (via env_method) and pick a `log_key` for the logger.

    docs 10.44/10.45: originally computed progress as num_timesteps / (relax_frac *
    total_timesteps) -- num_timesteps is the model's CUMULATIVE step count (carried over by a
    warm start), so on any --init-from continuation this either (a) massively overshot the
    denominator for an already-relaxed parameter being continued, re-loosening a scaffold that
    had already finished relaxing (the exact v8b->v9 knee bug), or (b) for a BRAND NEW
    parameter added mid-project (velocity assist), the huge inherited num_timesteps instantly
    saturated frac to 1.0, skipping the ramp before a single step of this run had happened.
    Both were the same root mistake: measuring progress against absolute step count instead of
    this call's own elapsed steps. Fixed by tracking elapsed = num_timesteps - (num_timesteps
    at the start of THIS training_start), i.e. always 0 at the beginning of any given call
    regardless of warm-start history. already_relaxed=True skips straight to end_val from the
    first step -- pass it explicitly on a continuation of a schedule that already finished
    ramping in a prior run, instead of trying to re-derive an absolute step count."""

    log_key = "rollout/scaffold_value"

    def __init__(self, total_timesteps, start_val, end_val, relax_frac=0.6, freq=50_000,
                 already_relaxed=False):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.start_val = start_val
        self.end_val = end_val
        self.relax_frac = relax_frac
        self.freq = freq
        self.already_relaxed = already_relaxed
        self._start_step = None
        self._last_set_step = -freq  # force a set on the very first _on_step

    def apply(self, value):
        raise NotImplementedError

    def _on_training_start(self):
        self._start_step = self.num_timesteps

    def _on_step(self):
        if self.num_timesteps - self._last_set_step >= self.freq:
            if self.already_relaxed:
                frac = 1.0
            else:
                elapsed = self.num_timesteps - self._start_step
                frac = min(1.0, elapsed / (self.relax_frac * self.total_timesteps))
            value = self.start_val + (self.end_val - self.start_val) * frac
            self.apply(value)
            self.logger.record(self.log_key, value)
            self._last_set_step = self.num_timesteps
        return True


class KneeScaffoldCallback(LinearScaffoldCallback):
    """Anneals LocomotionEnv's per-calm-episode knee floor (docs 10.37) from `start_deg`
    (restrictive -- physically can't retreat into a deep crouch) to `end_deg`, then holds.

    v6 (docs 10.38) relaxed all the way to the joint's real limit (-140, fully unconstrained)
    and regressed hard as it did -- knee-straighter-than-90deg dropped 27%->12.5% and mean
    |vx| dropped 0.174->0.131 m/s between the 3.75M and 9.25M checkpoints, tracking the floor
    loosening from -78 to -119. Calm episodes are explicitly the "nothing threatening is
    happening" case -- there's no real reason they'd ever need a full-depth crouch (that's
    what harsh/collapse episodes are for, and they keep their own unconstrained range
    regardless of this schedule). So end_deg now stops at a permanent partial floor instead
    of releasing back to the full range -- matches the standard curriculum-RL mitigation for
    this exact failure mode (mix difficulty levels / cap the relaxation rather than fully
    removing scaffolding, since fully removing it invites reverting to the pre-scaffold
    policy)."""

    log_key = "rollout/knee_floor_deg"

    def __init__(self, total_timesteps, start_deg=-50.0, end_deg=-80.0, **kwargs):
        super().__init__(total_timesteps, start_deg, end_deg, **kwargs)

    def apply(self, value):
        self.training_env.env_method("set_min_knee_deg", value)


class VelocityAssistCallback(LinearScaffoldCallback):
    """Anneals LocomotionEnv's external push-toward-target-velocity force (docs 10.45) from
    `start_scale` down to a small permanent `end_scale` (not 0 -- the knee scaffold's lesson
    from 10.38/10.40 was that fully removing a scaffold invites reverting).

    Measured mean |actual vx| stuck at ~0.13-0.17 m/s across five straight reward/curriculum
    changes (v2 through v8b), while a pure-physics check (max root_x thrust, no other joints)
    showed the actuator can reach 1.5 m/s alone -- it's not an actuator power problem, it's
    that any real attempt at speed risks falling with no leg coordination yet, so the policy
    converges to barely moving (same shape of problem as the knee habit). The assist is an
    external nudge in qfrc_applied proportional to target_vx, calm episodes only -- gives the
    policy sustained experience of being in motion (and what balance recovery looks like at
    speed) without asking it to generate 100% of the propulsion from scratch."""

    log_key = "rollout/assist_scale"

    def __init__(self, total_timesteps, start_scale=30.0, end_scale=8.0, **kwargs):
        super().__init__(total_timesteps, start_scale, end_scale, **kwargs)

    def apply(self, value):
        self.training_env.env_method("set_assist_scale", value)


class FallRateCallback(BaseCallback):
    """Fraction of completed episodes that ended via a_out (failed to recover within
    DOWN_RECOVERY_STEPS) rather than truncation at MAX_STEPS -- the direct metric this
    experiment is about, logged live instead of only via a separate eval pass."""

    def __init__(self, window=200):
        super().__init__()
        self._recent = deque(maxlen=window)

    def _on_step(self):
        for i, info in enumerate(self.locals["infos"]):
            if self.locals["dones"][i]:
                self._recent.append(1.0 if info.get("a_out") else 0.0)
        if self._recent:
            self.logger.record("rollout/fall_rate", float(np.mean(self._recent)))
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=5_000_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out", type=str, default="ppo_locomotion")
    parser.add_argument("--init-from", type=str, default=None,
                         help="checkpoint to warm-start from (any Fighter2DEnv-compatible "
                              "policy, e.g. a combat checkpoint) -- default: train from scratch")
    parser.add_argument("--save-freq", type=int, default=25_000)
    parser.add_argument("--ent-coef", type=float, default=None)
    parser.add_argument("--knee-already-relaxed", action="store_true",
                         help="skip the knee scaffold straight to its permanent floor from "
                              "step 0 (docs 10.44/10.45) -- pass this on any --init-from "
                              "continuation of a run whose knee schedule already finished")
    parser.add_argument("--assist-already-relaxed", action="store_true",
                         help="same as --knee-already-relaxed, for the velocity assist scaffold")
    args = parser.parse_args()

    MODELS_DIR.mkdir(exist_ok=True)
    ckpt_dir = MODELS_DIR / "autosave"
    ckpt_dir.mkdir(exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.out}_{run_id}"

    vec_env = make_vec_env(LocomotionEnv, n_envs=args.n_envs, vec_env_cls=SubprocVecEnv)

    if args.init_from:
        custom_objects = {"ent_coef": args.ent_coef} if args.ent_coef is not None else None
        model = PPO.load(args.init_from, env=vec_env, device=args.device, custom_objects=custom_objects)
        print(f"[locomotion] warm-starting from {args.init_from}", flush=True)
    else:
        kwargs = {}
        if args.ent_coef is not None:
            kwargs["ent_coef"] = args.ent_coef
        model = PPO("MlpPolicy", vec_env, device=args.device, verbose=1, **kwargs)
        print("[locomotion] training from scratch", flush=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.save_freq // args.n_envs, 1),
        save_path=str(ckpt_dir),
        name_prefix=run_name,
    )
    callbacks = CallbackList([checkpoint_callback, BreakdownCallback(), FallRateCallback(),
                               LogStdClampCallback(),
                               KneeScaffoldCallback(args.timesteps, already_relaxed=args.knee_already_relaxed),
                               VelocityAssistCallback(args.timesteps, already_relaxed=args.assist_already_relaxed)])

    try:
        model.learn(total_timesteps=args.timesteps, callback=callbacks,
                    reset_num_timesteps=(args.init_from is None))
    finally:
        with torch.no_grad():
            before = model.policy.log_std.data.clone()
            model.policy.log_std.data.clamp_(min=LOG_STD_MIN, max=LOG_STD_MAX)
            n_clamped = int((model.policy.log_std.data != before).sum())
        if n_clamped:
            print(f"[locomotion] clamped log_std on {n_clamped}/{before.numel()} action dims "
                  f"into [{np.exp(LOG_STD_MIN):.3f}, {np.exp(LOG_STD_MAX):.3f}] std range", flush=True)
        out_path = MODELS_DIR / run_name
        model.save(str(out_path))
        print(f"saved model to {out_path}.zip")


if __name__ == "__main__":
    main()
