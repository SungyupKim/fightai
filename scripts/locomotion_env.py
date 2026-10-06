"""Single-fighter mobility/recovery environment.

'a' trains to (1) regain a stable standing state from an arbitrary starting body
configuration (anywhere from a light shove to fully collapsed on the ground) and (2)
track a commanded x-velocity while doing it, with 'b' teleported off to the side and held
at zero control -- physically present in the model (it always has two bodies) but never
interacted with.

Validates a hypothesis raised after the 10.29-10.32 fall investigation: the persistent
"clean fall" plateau (85% of falls, no preceding stagger, no clear single cause -- docs
10.31) might be a single combat policy splitting its capacity between locomotion/mobility
and combat objectives, rather than a locomotion skill that just hasn't been trained
enough. Training balance/recovery/mobility in total isolation, with a much broader
curriculum than the existing get-up curriculum (GETUP_CURRICULUM_PROB, docs 10.23)
covers, is a cheap way to check: if a solo policy is dramatically more fall-resistant
than the combat-trained one at a comparable step budget, that's real evidence for a
hierarchical (locomotion + combat) architecture instead of one shared policy learning
everything at once.

v2 (docs 10.33): added velocity-command tracking -- v1 only rewarded NOT falling, with no
signal at all for actually moving, so it had no reason to learn real mobility (the
`root_x` actuator was available but unrewarded). Bumps the observation space by 1 dim
(the current target velocity), so this is NOT warm-start-compatible with v1 checkpoints.
"""
import mujoco
import numpy as np
from gymnasium import spaces

import env as env_module
from env import Fighter2DEnv, GETUP_SETTLE_STEPS

# Combat reward terms are meaningless with 'b' parked far away and inert. Zeroing these at
# module level is per-process only (SubprocVecEnv gives each worker its own copy of the
# module), so this doesn't affect env.py's other users. Left un-zeroed, ENGAGE_PENALTY_SCALE
# in particular would apply a large constant penalty (foot_dist stays huge once 'b' is
# parked) that has nothing to do with the actual task.
env_module.ENGAGE_PENALTY_SCALE = 0.0
env_module.PROGRESS_REWARD_SCALE = 0.0
env_module.DAMAGE_REWARD_SCALE = 0.0

# v2 fair-eval (docs 10.34) found the trained policy crouches at ~knee limit (-140deg)
# essentially always, even starting from a normal standing pose -- not just during actual
# recovery. RECOVERY_REWARD only engages below FALL_HEIGHT (a deep knee bend alone doesn't
# drop the torso that far, so it never fires here), and velocity-tracking reward is
# height-agnostic (crouched or standing, matching the target speed pays the same), so there
# was very little pull back to full standing once a knee bend became "good enough" to be
# stable. Doubling both height and knee-avoid pressure specifically for this task -- pure
# reward-constant changes, so this continues from the existing checkpoint instead of
# retraining from scratch.
env_module.HEAD_HEIGHT_REWARD_SCALE = 6.0   # was 3.0
env_module.KNEE_AVOID_SCALE = 1.0           # was 0.25, then 0.5 (docs 10.34) -- still measured
                                             # knee_r straighter than -90deg only 6% of the time
                                             # and mean |actual vx| only 0.158 m/s despite targets
                                             # up to 1.0 m/s (docs 10.35), so the permanent crouch
                                             # is winning even at 2x. Pushing further.

B_PARK_X = 6.0  # far enough that a/b geoms can never contact regardless of b's pose

# ---- calm/harsh episode split (docs 10.36/10.5x) ----
# v13curriculum changed BOTH CALM_EPISODE_FRAC (0.6->0.45) and HARSH_SEVERITY_LO (0->0.5) at
# once: deep-collapse a_out improved (67.5-72.5% -> 57.5-67.5%) but calm a_out got WORSE
# (20% -> 40%). v13b isolated the severity-floor half alone (CALM_EPISODE_FRAC reverted to
# 0.6, only HARSH_SEVERITY_LO=0.5 kept): calm a_out went right back to 20%, but deep-collapse
# a_out did NOT improve (70-75%, same as baseline within noise) -- the floor change contributed
# nothing on its own. So the deep-collapse gain in v13curriculum came from the OTHER half
# (more harsh-episode exposure via a lower CALM_EPISODE_FRAC), which is the real calm-vs-harsh
# tradeoff, not a free win. This run isolates that other half: CALM_EPISODE_FRAC lowered,
# severity floor reverted to 0 (original uniform [0, severity_hi]).
CALM_EPISODE_FRAC = 0.45       # was 0.6 -- see above; the half of v13curriculum not yet
                                # tested alone
CALM_SEVERITY_HI = 0.3         # calm episodes still get SOME light random perturbation
                                # (severity sampled in [0, this]), just nothing collapse-grade
HARSH_SEVERITY_LO = 0.0        # reverted -- measured to contribute nothing on its own (see
                                # above), original uniform [0, severity_hi] draw

# ---- velocity command ----
TARGET_VX_RANGE = 1.0          # m/s, sampled uniform in [-range, +range] (0 = "hold position"
                                # is just as likely to be sampled as any other value, not a
                                # separate special case)
RESAMPLE_EVERY_STEPS = None    # was 200 (docs 10.41): re-rolling mid-episode meant no single
                                # command was ever committed to for long, so "twitch briefly
                                # toward whatever's asked" could already capture most of the
                                # achievable reward -- None now means only reset() rolls a new
                                # target, so a whole episode (up to MAX_STEPS=1000) means
                                # actually committing to one sustained walk, not several stabs.
VELOCITY_REWARD_SCALE = 5.0    # was 2.0 (docs 10.35) -- measured mean |actual vx| only 0.158
                                # m/s despite targets sampled up to 1.0 m/s, i.e. the policy
                                # barely tries to move at all. 2.0 wasn't enough to outweigh
                                # the safety of standing (crouched) still; pushing to 5.0 so
                                # actually matching a fast command is worth clearly more than
                                # the other per-step rewards combined.
VELOCITY_SIGMA = 0.3           # was 0.5, then 0.7 (docs 10.35) -- the widen-it attempt was
                                # backwards: even after the knee/crouch problem got fixed
                                # (v7, docs 10.40), mean |actual vx| was STILL only 0.129 m/s,
                                # unmoved by any of the crouch fixes -- a separate problem.
                                # The wide sigma was giving real "free" credit for NOT moving:
                                # against a 1.0 m/s target, standing still (vx=0) scored
                                # exp(-1.0^2/(2*0.7^2)) = 36% of max reward doing nothing.
                                # Narrowing to 0.3 drops that to ~4%, so standing still against
                                # a real command stops being a viable safe default.
                                # gaussian shape (always >= 0, same cliff-free family as the
                                # existing height/stability rewards) rather than a raw squared
                                # penalty, for the same unbounded-blowup reasons documented at
                                # env.py's height-penalty history (docs, "Direct height PENALTY")

# ---- gait shaping (docs 10.46/10.47/10.48) ----
# The outcome reward (velocity, above) never got the policy to actually try leg-stepping --
# mean |actual vx| stuck at ~0.13-0.17 m/s across seven straight reward/curriculum/assist
# changes (v2 through v9). Tried shaping directly on the PROCESS instead of the outcome:
# -hip_r_vel*hip_l_vel (positive when the two hips rotate in OPPOSITE directions, clamped at 0
# otherwise), meant to reward the core signature of a stepping gait without being satisfiable
# by e.g. vibrating both legs in sync. Wasn't strict enough on its own -- v10 (10.47) measured
# r_gait climbing hard (130->340 over 8M steps) while mean |actual vx| and its correlation
# with target_vx both went DOWN, and fall_rate crept up too: marching/scissoring in place
# (real anti-phase hip motion, zero net displacement) satisfied this term just as well as
# actual walking. v11 (10.48) re-gates it by multiplying with velocity_credit (below) -- can
# now only pay out while ALSO tracking target_vx reasonably well, closing the loophole
# directly instead of hoping a bigger/stricter version of the same ungated shape works.
#
# v12fresh (docs 10.50): the raw product -hip_r_vel*hip_l_vel is UNBOUNDED -- measured up to
# 83 mid-training (p90=37, p99=64), versus every other reward term in this env being
# explicitly capped (height's ratio, stability's max(0,1-x)^2, velocity's gaussian). fall_rate
# hit 0.96 and r_gait exploded to 5000+/episode: the policy found that flailing both legs as
# violently as possible in opposite directions paid far more than anything else, fall risk be
# damned. Saturating it the same cliff-free way the rest of the reward function already does
# (1-exp(-x/ref), same family as height/stability/knee_avoid) removes the incentive to
# escalate past a reasonable pace -- max payout is now the same regardless of whether the hips
# move at a brisk-walk speed or a flailing one.
GAIT_REWARD_SCALE = 3.0
GAIT_REF = 2.0                 # raw -(hip_r_vel*hip_l_vel) at which the saturating curve is
                                # ~63% of its ceiling -- picked well below the flailing values
                                # (p90=37) that caused the runaway, in the plausible range for
                                # a deliberate, controlled step rather than a wild swing

# ---- fall brace reflex (docs 10.5x) ----
# While the torso is dropping fast, reward straightening the knees -- the policy otherwise
# buckles like a folding stick when it falls (hips and knees both bend further as it goes
# down, measured on the stand-only checkpoint). Gated on downward torso speed so it only pays
# out during an actual fall, and bounded (0..BRACE_REWARD_SCALE per step) like the other terms.
BRACE_REWARD_SCALE = 1.0
BRACE_VZ_REF = 1.0             # m/s downward torso speed at which the fall gate saturates


class LocomotionEnv(Fighter2DEnv):
    def __init__(self, render_mode=None, stand_only=False):
        self._stand_only = stand_only
        if stand_only:
            # standing has to be learned from the legs and arms alone -- the always-on torso PD
            # assist (env.py BALANCE_KP/KD) would otherwise carry the balance (docs 10.5x: with it
            # removed, the stand-only policy never kept its footing in an episode)
            env_module.BALANCE_KP = 0.0
            env_module.BALANCE_KD = 0.0
        super().__init__(render_mode=render_mode, opponent_policy_path=None, getup_curriculum_prob=0.0)
        self._b_root_x_qpos = self.model.joint("b_root_x").qposadr[0]
        self._b_root_x_dof = self.model.joint("b_root_x").dofadr[0]
        self._a_root_x_dof = self.model.joint("a_root_x").dofadr[0]
        self._zero_action = np.zeros(self.action_space.shape[0], dtype=np.float32)
        self._target_vx = 0.0
        self._knee_r_qpos = self.model.joint("a_knee_r").qposadr[0]
        self._knee_l_qpos = self.model.joint("a_knee_l").qposadr[0]
        self._knee_r_dof = self.model.joint("a_knee_r").dofadr[0]
        self._knee_l_dof = self.model.joint("a_knee_l").dofadr[0]
        self._hip_r_dof = self.model.joint("a_hip_r").dofadr[0]
        self._hip_l_dof = self.model.joint("a_hip_l").dofadr[0]
        # scaffolding (docs 10.37): even a forced-standing, high-target-velocity reset still
        # crouched back to ~-136deg median within 1s (measured on a mid-training v5
        # checkpoint) -- reward-side pressure (10.34/10.35) repeatedly failed to out-argue the
        # policy's own experience that extending is risky. Instead of persuading, remove the
        # escape hatch: during calm episodes only (harsh/collapse episodes still need the full
        # range to recover from genuine collapse), clamp the knee no deeper than this floor.
        # None = no clamp (full range) -- the default until a training run opts in via
        # set_min_knee_deg(), so this class stays a no-op change for anything not using it.
        self._min_knee_deg = None
        self._episode_knee_floor = None  # this episode's floor in radians, or None
        self._is_calm = False
        # velocity assist (docs 10.45): a pure physics check (max root_x thrust, no other
        # joints) showed the actuator has plenty of power -- 1.5 m/s reachable -- but the
        # torso outruns its own base of support and falls by ~6.5s with no leg coordination.
        # Real mean |actual vx| stayed ~0.13-0.17 m/s across v2/v4/v5/v6/v7/v8b regardless of
        # reward changes: likely the same root cause as the knee habit (10.37) -- any
        # committed attempt at speed risks a fall, so the policy converges to barely moving.
        # This is an external qfrc_applied nudge toward target_vx (calm episodes only, same
        # gating as the knee floor) -- like someone lightly pushing you the right way while
        # you find your own coordination, not doing the walking for you. 0.0 = no-op default.
        self._assist_scale = 0.0

        base_dim = self.observation_space.shape[0]
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(base_dim + 1,), dtype=np.float32)

    def set_min_knee_deg(self, deg):
        """deg=None disables the clamp entirely. Called externally (KneeScaffoldCallback)
        to anneal the floor over training -- SB3's env_method() broadcasts this to every
        SubprocVecEnv worker."""
        self._min_knee_deg = None if deg is None else np.deg2rad(deg)

    def set_assist_scale(self, scale):
        """Called externally (VelocityAssistCallback) to anneal the external push-toward-
        target-velocity force. 0.0 disables it entirely."""
        self._assist_scale = scale

    def _obs(self):
        return np.concatenate([super()._obs(), [self._target_vx]]).astype(np.float32)

    def _resample_target_vx(self):
        self._target_vx = float(self.np_random.uniform(-TARGET_VX_RANGE, TARGET_VX_RANGE))

    def _randomize_a_state(self, severity_lo, severity_hi):
        """Broader than Fighter2DEnv._collapse_pose: interpolates between the current
        (near-standing) pose and a widely-sampled target by a random severity in
        [0, severity_hi], covering everything from a light perturbation to a full collapse
        in one curriculum instead of a binary standing/collapsed split. Includes arms and
        torso pitch too, not just legs/waist -- a real stagger disturbs the whole body, not
        just the base."""
        severity = float(self.np_random.uniform(severity_lo, severity_hi))
        legs_waist = (("hip_r", 0.6), ("hip_l", 0.6), ("waist", 0.6))
        for j, frac in legs_waist:
            qpos = self.model.joint(f"a_{j}").qposadr[0]
            lo, hi = self.model.joint(f"a_{j}").range
            target = self.np_random.uniform(lo * frac, hi * frac)
            self.data.qpos[qpos] = severity * target + (1.0 - severity) * self.data.qpos[qpos]
        for j in ("knee_r", "knee_l"):
            qpos = self.model.joint(f"a_{j}").qposadr[0]
            lo, _hi = self.model.joint(f"a_{j}").range
            target = self.np_random.uniform(lo * 0.7, 0.0)
            self.data.qpos[qpos] = severity * target + (1.0 - severity) * self.data.qpos[qpos]
        for j in ("ankle_r", "ankle_l", "shoulder_r", "shoulder_l", "elbow_r", "elbow_l"):
            qpos = self.model.joint(f"a_{j}").qposadr[0]
            lo, hi = self.model.joint(f"a_{j}").range
            target = self.np_random.uniform(lo, hi)
            self.data.qpos[qpos] = severity * target + (1.0 - severity) * self.data.qpos[qpos]
        self.data.qpos[self.a_ry_qpos] += severity * self.np_random.uniform(-0.6, 0.6)
        return severity

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)

        self.data.qpos[self._b_root_x_qpos] = B_PARK_X
        # Two-tier curriculum (docs 10.36): measured that falls happen mostly during
        # knee-extended moments, not crouched ones (knee ~-52deg median 1s before a fall vs
        # ~-138deg overall) -- the policy has correctly learned, from its OWN experience,
        # that standing/extending is genuinely riskier under a curriculum where every single
        # episode starts with some random collapse severity and a 40% chance of a shove.
        # Reward-scale bumps (10.34/10.35) couldn't out-argue that real experience. Instead,
        # give it a lot more practice at "nothing threatening is happening, standing is
        # fine" by making most episodes calm (low severity, no push) and only a minority
        # keep the full hard curriculum -- closer to how a real fighter is only crouched
        # defensively some of the time, not permanently braced.
        is_calm = self.np_random.random() < CALM_EPISODE_FRAC
        severity = self._randomize_a_state(0.0 if is_calm else HARSH_SEVERITY_LO,
                                            CALM_SEVERITY_HI if is_calm else 1.0)
        # only calm episodes get the knee floor -- harsh/collapse episodes need the full
        # range to actually recover from a genuine collapse (docs 10.37)
        self._episode_knee_floor = self._min_knee_deg if is_calm else None
        self._is_calm = is_calm
        mujoco.mj_forward(self.model, self.data)

        saved_ctrl = self.data.ctrl.copy()
        self.data.ctrl[:] = 0.0
        for _ in range(int(GETUP_SETTLE_STEPS * severity)):
            mujoco.mj_step(self.model, self.data)
        self.data.ctrl[:] = saved_ctrl

        # push probability follows the same calm/harsh split -- a calm episode should mean
        # genuinely no threat, not just a smaller starting bend.
        # docs 10.54: tried raising this to 0.7 (v13d) and 0.45 (v13e), both warm-started from
        # v13c -- both improved calm and the push sub-case itself, but both regressed the
        # plain-collapse (no-push) case back to ~75% (from v13c's 47.5%), regardless of how
        # much the probability moved. Root cause: push/no-push split the SAME harsh-episode
        # budget, so raising push_prob necessarily steals training volume from no-push -- a
        # zero-sum tradeoff within the harsh branch, not a volume-of-exposure problem solvable
        # by tuning this constant. Reverted to the original 0.4 (the v13c value, still the best
        # weighted result found: calm 22.5%, push 67.5%, no-push 47.5%). The push sub-case
        # itself remains unsolved -- would need a different lever (e.g. its own scaffold)
        # rather than this split ratio.
        push_prob = 0.0 if is_calm else 0.4
        if self.np_random.random() < push_prob:
            self.data.qvel[self._a_root_x_dof] += self.np_random.uniform(-2.5, 2.5)

        self._resample_target_vx()
        mujoco.mj_forward(self.model, self.data)
        return self._obs(), info

    def step(self, action):
        if self._stand_only:
            # the horizontal thrust actuator (root_x) isn't a leg or arm -- a real body can't push
            # itself sideways from nothing, and the trained policy saturated it on ~half of steps
            action = np.array(action, dtype=np.float32, copy=True)
            action[0] = 0.0
        if self._is_calm and self._assist_scale and not self._stand_only:
            self.data.qfrc_applied[self._a_root_x_dof] = self._assist_scale * self._target_vx
        else:
            self.data.qfrc_applied[self._a_root_x_dof] = 0.0

        obs, reward, terminated, truncated, info = super().step(action, b_full_action=self._zero_action)
        # 'b' is scripted to be irrelevant, but down_steps['b'] can still creep up over time
        # (no actuation holding it up) and trip a_out's `a_out or b_out` termination even
        # though 'a' itself is fine -- only 'a' failing to recover should end an episode here.
        if info.get("b_out") and not info.get("a_out"):
            terminated = False
        self.data.qpos[self._b_root_x_qpos] = B_PARK_X
        self.data.qvel[self._b_root_x_dof] = 0.0

        if self._episode_knee_floor is not None:
            clamped = False
            for qadr, dadr in ((self._knee_r_qpos, self._knee_r_dof), (self._knee_l_qpos, self._knee_l_dof)):
                if self.data.qpos[qadr] < self._episode_knee_floor:
                    self.data.qpos[qadr] = self._episode_knee_floor
                    self.data.qvel[dadr] = max(0.0, self.data.qvel[dadr])  # keep any extending motion
                    clamped = True
            if clamped:
                mujoco.mj_forward(self.model, self.data)

        actual_vx = self.data.qvel[self._a_root_x_dof]
        velocity_credit = np.exp(-((actual_vx - self._target_vx) ** 2) / (2.0 * VELOCITY_SIGMA ** 2))
        velocity_reward = 0.0 if self._stand_only else VELOCITY_REWARD_SCALE * velocity_credit
        hip_r_vel = self.data.qvel[self._hip_r_dof]
        hip_l_vel = self.data.qvel[self._hip_l_dof]
        # docs 10.47: v10 measured r_gait climbing hard while mean|actual vx| and its target
        # correlation both got WORSE -- the anti-phase condition alone was satisfiable by
        # marching/scissoring in place, unrelated to whether the torso was actually moving.
        # Gating by velocity_credit (the same match-quality term velocity_reward already
        # uses) closes that loophole directly: this can only pay out while ALSO tracking
        # target_vx reasonably well, so marching against a real (nonzero) target now earns
        # both terms' worth of nothing instead of gait_reward alone for free.
        gait_raw = max(0.0, -(hip_r_vel * hip_l_vel))
        gait_reward = 0.0 if self._stand_only else GAIT_REWARD_SCALE * (1.0 - np.exp(-gait_raw / GAIT_REF)) * velocity_credit
        vz_down = -self.data.qvel[self.a_z_dof]
        fall_gate = min(1.0, max(0.0, vz_down / BRACE_VZ_REF))
        knee_mean = 0.5 * (self.data.qpos[self._knee_r_qpos] + self.data.qpos[self._knee_l_qpos])
        knee_straight = min(1.0, max(0.0, (knee_mean + np.deg2rad(140.0)) / np.deg2rad(140.0)))
        brace_reward = BRACE_REWARD_SCALE * fall_gate * knee_straight
        reward = reward + velocity_reward + gait_reward + brace_reward
        info["reward_breakdown"]["velocity"] = velocity_reward
        info["reward_breakdown"]["gait"] = gait_reward
        info["reward_breakdown"]["brace"] = brace_reward
        info["target_vx"] = self._target_vx

        if RESAMPLE_EVERY_STEPS is not None and self.step_count % RESAMPLE_EVERY_STEPS == 0:
            self._resample_target_vx()

        return self._obs(), reward, terminated, truncated, info
