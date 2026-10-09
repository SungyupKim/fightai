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

from locomotion_env import CALM_EPISODE_FRAC, LocomotionEnv

MODELS_DIR = pathlib.Path(__file__).resolve().parent.parent / "checkpoints"

# same clamp as train_league.py (docs 10.20) -- keeps exploration alive without needing an
# entropy bonus (which overshot into runaway growth the one time it was tried).
LOG_STD_MIN = np.log(0.05)
LOG_STD_MAX = np.log(0.6)


class BreakdownCallback(BaseCallback):
    KEYS = ["height", "stability", "stance", "knee_avoid", "recovery", "jerk", "velocity", "gait", "brace", "com",
            "stand_streak", "knee_straight"]

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


class HeightRewardScaffoldCallback(LinearScaffoldCallback):
    """Anneals the stand-only head-height reward scale from `start_scale` (6.0 -- the ambient
    baseline every LocomotionEnv gets from a module-level override left by an earlier
    locomotion-mode experiment, docs 10.33-10.35, NOT env.py's bare 3.0 default) up to
    `end_scale`, instead of a one-off manual constant edit.

    Measured directly (2026-10-09, qfrc_bias at a hand-set upright pose) that holding full leg
    extension against gravity costs only ~1-5% of each joint's max torque -- not a capability
    problem. Yet checkpoints kept settling into a stable but LOW equilibrium (torso_z
    0.10-0.63, std~0 -- genuinely stable, not falling) even with a dedicated knee-straightness
    reward added. The fix (raising this scale so "stand tall" clearly beats "sit low and
    stable" in the policy's own value estimate) only showed its need well into a training run,
    and required manually stopping, editing a constant, and warm-starting every time it came
    up. Baked in as a scaffold instead so every future run gets it automatically and (like the
    other scaffolds) resumes its own ramp correctly across an --init-from continuation."""

    log_key = "rollout/height_reward_scale"

    def __init__(self, total_timesteps, start_scale=6.0, end_scale=10.0, **kwargs):
        super().__init__(total_timesteps, start_scale, end_scale, **kwargs)

    def apply(self, value):
        self.training_env.env_method("set_height_reward_scale", value)


class CurriculumGateCallback(LinearScaffoldCallback):
    """Advances LocomotionEnv's calm/harsh episode-start curriculum (docs 2026-10-09) from
    phase 1 (100% calm -- just learn to hold a stand) to phase 2 (the original combat-tuned
    CALM_EPISODE_FRAC mix, harsh/collapse starts included) only once phase 1 actually looks
    solved, instead of on a fixed timer. A fixed timer can't tell whether phase 1 finished
    early (wasting steps waiting) or is still unsolved when the deadline hits (forcing harsh
    starts onto a policy that hasn't nailed the easy case yet) -- gating on real performance
    avoids both. Trigger: the training-log fall_rate drops to <= FALL_RATE_GATE AND the mean
    longest continuous stand per episode reaches >= STREAK_GATE_STEPS (10s), both sustained
    over `window` consecutive episodes. Once triggered, phase 2 is still phased in gradually
    (the usual LinearScaffoldCallback anneal, starting from the trigger point) rather than
    switched on instantly, to soften the transition.

    "Stand still and hold it" and "get up off the ground" are different skills (the latter
    still has its own dedicated `recovery` reward term) -- this doesn't claim the second is
    free once the first works, just that learning it staged may be easier than both at once."""

    log_key = "rollout/calm_frac"
    FALL_RATE_GATE = 0.1
    STREAK_GATE_STEPS = 500  # 10s at 0.02s/step

    def __init__(self, total_timesteps, end_frac=CALM_EPISODE_FRAC, window=200,
                 post_trigger_relax_frac=0.15, **kwargs):
        super().__init__(total_timesteps, 1.0, end_frac, relax_frac=post_trigger_relax_frac, **kwargs)
        self._fall_recent = deque(maxlen=window)
        self._streak_recent = deque(maxlen=window)
        self._streak_running = None
        self._triggered = False

    def apply(self, value):
        self.training_env.env_method("set_calm_frac", value)

    def _on_training_start(self):
        super()._on_training_start()
        self._streak_running = np.zeros(self.training_env.num_envs)
        if self.already_relaxed:
            self._triggered = True  # continuation of a run whose phase-1 gate already passed

    def _on_step(self):
        for i, info in enumerate(self.locals["infos"]):
            self._streak_running[i] = max(self._streak_running[i], info.get("stand_streak_steps", 0))
            if self.locals["dones"][i]:
                self._fall_recent.append(1.0 if info.get("a_out") else 0.0)
                self._streak_recent.append(self._streak_running[i])
                self._streak_running[i] = 0.0
        if self._streak_recent:
            self.logger.record("rollout/max_stand_streak_steps", float(np.mean(self._streak_recent)))

        if not self._triggered and len(self._fall_recent) >= self._fall_recent.maxlen:
            if (np.mean(self._fall_recent) <= self.FALL_RATE_GATE
                    and np.mean(self._streak_recent) >= self.STREAK_GATE_STEPS):
                self._triggered = True
                self._start_step = self.num_timesteps  # the anneal's own clock starts now
                print(f"[curriculum] phase-1 gate passed at step {self.num_timesteps} "
                      f"(fall_rate={np.mean(self._fall_recent):.3f}, "
                      f"streak={np.mean(self._streak_recent):.0f}) -- starting phase-2 anneal",
                      flush=True)

        if self._triggered:
            return super()._on_step()
        if self.num_timesteps - self._last_set_step >= self.freq:
            self.apply(self.start_val)
            self.logger.record(self.log_key, self.start_val)
            self._last_set_step = self.num_timesteps
        return True


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
    parser.add_argument("--stand-only", action="store_true",
                         help="zero the velocity and gait rewards -- learn standing from any posture alone")
    parser.add_argument("--assist-already-relaxed", action="store_true",
                         help="same as --knee-already-relaxed, for the velocity assist scaffold")
    parser.add_argument("--calm-already-relaxed", action="store_true",
                         help="same as --knee-already-relaxed, for the stand-only calm-episode-"
                              "fraction curriculum (only meaningful with --stand-only)")
    parser.add_argument("--height-already-relaxed", action="store_true",
                         help="same as --knee-already-relaxed, for the stand-only head-height "
                              "reward scaffold (only meaningful with --stand-only)")
    args = parser.parse_args()

    MODELS_DIR.mkdir(exist_ok=True)
    ckpt_dir = MODELS_DIR / "autosave"
    ckpt_dir.mkdir(exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.out}_{run_id}"

    vec_env = make_vec_env(LocomotionEnv, n_envs=args.n_envs, vec_env_cls=SubprocVecEnv,
                           env_kwargs={"stand_only": args.stand_only})

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
    callback_list = [checkpoint_callback, BreakdownCallback(), FallRateCallback(),
                      LogStdClampCallback(),
                      KneeScaffoldCallback(args.timesteps, already_relaxed=args.knee_already_relaxed),
                      VelocityAssistCallback(args.timesteps, already_relaxed=args.assist_already_relaxed)]
    if args.stand_only:
        callback_list.append(
            CurriculumGateCallback(args.timesteps, already_relaxed=args.calm_already_relaxed))
        callback_list.append(
            HeightRewardScaffoldCallback(args.timesteps, already_relaxed=args.height_already_relaxed))
    callbacks = CallbackList(callback_list)

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
