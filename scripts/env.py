"""Gymnasium environment: fighter 'a' (learned policy) vs fighter 'b'. 'b' is either
the scripted shadow-boxing opponent from view.py, or -- if opponent_policy_path is
given -- a frozen snapshot of a trained policy controlling 'b' from its own mirrored
point of view (self-play). Reward is driven by contact forces between 'weapon' geoms
(fists/feet) and 'target' geoms (torso/head), plus a fall penalty.
"""
import pathlib

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from view import JOINTS, get_ctrl

MODEL_PATH = pathlib.Path(__file__).resolve().parent.parent / "models" / "fighter2d.xml"

# ---- episode / physics basics ----
FRAME_SKIP = 4
MAX_STEPS = 1000               # longer cap so recoverable knockdowns have room to play out
FALL_HEIGHT = 0.55

# ---- knockdown / fall mechanics ----
FALL_PENALTY = 50.0            # solo KO at episode end: winner +, loser -
MUTUAL_FALL_PENALTY = 20.0     # both KO'd simultaneously -- discourages trading a knockdown blow
                                # instead of sustained sparring
DOWN_RECOVERY_STEPS = 150      # ~3s (FRAME_SKIP*timestep=0.02s/step) to get back up before it's a real KO
KNOCKDOWN_ENTRY_PENALTY = 30.0 # one-time cost the *instant* you go down -- discourages ever diving in
                                # for a hit, independent of how long you stay down
DOWN_PENALTY_SCALE = 2.0       # ongoing per-step cost while down, proportional to how far below
                                # FALL_HEIGHT the torso is -- discourages *staying* down once you are.
                                # (a flat per-step cost gave no incentive to push back up: standing
                                # still paid the same penalty as trying, for less effort cost)

# ---- damage / combat ----
FORCE_TO_DAMAGE = 0.007        # scales contact normal force -> HP damage. Was 0.4, which sounds small
                                # but a diagnostic (diag_kick_cap.py) showed 94-97% of real punch/kick
                                # contacts already had raw force*0.4 >> MAX_STEP_DAMAGE, so almost every
                                # landed strike was clipped down to the same 8.0 -- a weak graze and a
                                # full-power (e.g. rotational) kick scored identically. Rescaled so the
                                # cap now binds only on the hardest ~10% of hits (p90 raw force ~1154),
                                # letting strike power actually show up in HP/reward across the bulk of
                                # the distribution instead of saturating immediately.
MAX_STEP_DAMAGE = 8.0          # per-body-part cap so one impact can't insta-kill
TARGET_HEALTH_MULT = {"head": 1.6, "torso": 1.0, "leg": 0.5}   # HP damage multiplier by target part
HEAD_REWARD_BONUS = 1.5        # extra reward-only multiplier for landing headshots, on top of HP mult
KICK_REWARD_BONUS = 2.0        # extra reward-only multiplier for shin-landed damage, to outweigh the
                                # balance risk of committing a leg instead of just punching
DAMAGE_REWARD_SCALE = 1.0      # raised again (0.35 -> 0.7 -> 1.0, back to unscaled). The 0.7 attempt
                                # looked like it hurt contact frequency, but we later found "no contact"
                                # episodes are ~80% actually self-destabilized falls before ever closing
                                # the distance, not a lack of aggression -- so that read was confounded.
                                # Worth pushing scoring weight further now that it's better isolated.
                                # (does not affect actual HP loss, reward-only)

# ---- movement / control shaping ----
EFFORT_COST = 0.01             # back to the original value. Raised in steps (0.01 -> 0.05 -> 0.15)
                                # chasing self-destabilized falls, but measured NO effect on mean action
                                # magnitude at any of those values -- meanwhile a reward_breakdown check
                                # showed it was ~-250/episode, over 75% of total reward and dwarfing the
                                # +5/episode strike signal. It never worked and was drowning out the
                                # actual combat signal, so reverting rather than tuning it further.
JERK_PENALTY_SCALE = 0.02      # cost on action change frame-to-frame, discourages full-power reversals
                                # (e.g. root thrust +1 -> -1 in one step)
ENGAGE_PENALTY_SCALE = 0.3     # cost on log1p(foot distance) every step -- always some gradient to
                                # close in (no free zone), steepest near contact range and flattening
                                # out at long range
B_APPROACH_GAIN = 1.5          # scripted opponent's proportional gain for walking toward 'a'

A_MIRROR_AUGMENT_PROB = 0.5    # fraction of episodes where 'a's own observation/action is routed
                                # through the same left-right mirror transform used for 'b'. Only 'a'
                                # ever receives gradient updates -- 'b' is always a frozen snapshot --
                                # so without this, the policy only ever trains on the raw (unmirrored)
                                # frame. A diagnostic (diag_engage.py) found the trained policy is NOT
                                # actually mirror-equivariant despite the physics/geometry being fully
                                # symmetric: in mirror self-play matches, 'a' self-destabilized in
                                # ~38/40 episodes vs 'b' in ~2/40, and the gap survived every structural
                                # ablation tried (swapping physical side, swapping MuJoCo body/geom
                                # compile order, forcing both sides through an identical predict() call
                                # structure) -- so it's specifically the network's behavior under the
                                # mirrored observation, not any of those. Randomly mirroring 'a's own
                                # view/action during training gives the mirrored frame direct gradient
                                # exposure instead of leaving it entirely unoptimized.

# ---- ground contact (footwork / collapse) ----
# rewards keeping feet planted (a wider base of support is more stable) and penalizes a knee
# touching the ground (a sign of the leg buckling/collapsing, distinct from a controlled kick
# where the foot lifts but the knee doesn't drop). Both are signed a-vs-b terms like down/balance,
# so they're ~0 (no signal) whenever both fighters are equally grounded -- the common case -- and
# only kick in when there's an actual gap, rather than taxing every step regardless of relevance.
GROUND_CONTACT_SCALE = 1.0     # per-step, scaled by fraction of feet planted (0 / 0.5 / 1.0 each side)
KNEE_CONTACT_PENALTY = 15.0    # per-step, per knee touching the ground -- meant to be rare, like
                                # BALANCE_PENALTY_SCALE, so scaled higher than the ever-present terms

# ---- balance assist (physics, not reward) ----
# root_ry (torso pitch) has no RL-controlled actuator; a small always-on PD assist drives it back
# toward upright instead. Verified by direct simulation: passive damping/stiffness alone can't
# stabilize a standing biped (it's an inverted pendulum) -- even stiffness=200 still toppled from
# a small 0.4 rad/s nudge -- but this PD assist recovers from that same nudge while still losing
# to a real hit (1+ rad/s), and incidentally fixed knockdown recovery too (0 -> ~50% success rate).
BALANCE_KP = 25.0
BALANCE_KD = 10.0

# reward-side balance penalty, separate from the physics-side PD assist above: gives the RL policy
# a direct incentive to choose techniques that don't fight the PD assist too hard, instead of only
# ever hearing about instability indirectly (after already falling, via down/knockdown_entry). Only
# penalizes lean *beyond* BALANCE_FREE_ZONE, not raw |root_ry| -- a diagnostic showed normal standing
# already wobbles around ~0.02-0.03 rad continuously (the PD assist constantly micro-correcting), so
# a flat per-radian cost would tax that constant background noise instead of actual danger, the same
# trap EFFORT_COST fell into. Squared beyond the free zone so it escalates sharply as a real fall
# approaches (measured tail: p99 ~0.28 rad, max ~0.81 rad) rather than a gentle linear slope.
BALANCE_FREE_ZONE = 0.15       # rad (~8.6 deg); no penalty for lean within this range
BALANCE_PENALTY_SCALE = 100.0

# ---- stagger (posture loss from leg damage) ----
# reduces control authority + adds control noise for the staggered fighter, so losing your legs
# makes it genuinely harder to keep standing rather than just an arbitrary penalty.
LEG_STAGGER_GAIN = 0.03
STAGGER_DECAY = 0.94           # per control step; ~1-2s to recover from a full stagger
STAGGER_CONTROL_LOSS = 0.6     # at stagger=1.0, actuators only keep 40% authority
STAGGER_NOISE_STD = 0.4        # extra action noise, scaled by stagger

# crude force-length curve for the arm actuators: full torque near mid-range, tapering
# toward a fully extended or fully folded arm (like a real muscle, weakest at the ends
# of its range of motion). ARM_MIN_POWER is the fraction of max torque kept at the
# extremes; power ramps up to 1.0 at the midpoint of each joint's own range.
ARM_JOINTS = ["shoulder_r", "elbow_r", "shoulder_l", "elbow_l"]
ARM_MIN_POWER = 0.4

# same force-length curve for the legs: a diagnostic found both hips being driven to
# ~+1.5 rad (near their +90 deg range limit) and getting stuck pinned there -- full
# torque available even at the extreme let the policy slam the joint into its limit
# and hold it there (torso folds forward at the hip, height drops below FALL_HEIGHT,
# read as a "fall" even though root_ry/balance never actually went unstable). Tapering
# power at the extremes, same as the arms already do, should make holding that pinned
# position costlier than easing off.
LEG_JOINTS = ["hip_r", "knee_r", "hip_l", "knee_l"]
LEG_MIN_POWER = 0.4


class Fighter2DEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None, opponent_policy_path=None):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.opponent_policy = None
        self._viewer = None

        self.a_act = np.array([self.model.actuator(f"a_{j}").id for j in JOINTS])
        self.b_act = np.array([self.model.actuator(f"b_{j}").id for j in JOINTS])

        def power_curve_setup(prefix, joints):
            # positions within a_act/b_act (index into JOINTS) for this limb's joints,
            # plus each one's qpos address and [min, max] range for the power curve
            act_idx = np.array([JOINTS.index(j) for j in joints])
            qpos_idx = np.array([self.model.joint(f"{prefix}{j}").qposadr[0] for j in joints])
            ranges = np.array([self.model.joint(f"{prefix}{j}").range for j in joints])
            return act_idx, qpos_idx, ranges

        self.a_arm_act_idx, self.a_arm_qpos, self.a_arm_range = power_curve_setup("a_", ARM_JOINTS)
        self.b_arm_act_idx, self.b_arm_qpos, self.b_arm_range = power_curve_setup("b_", ARM_JOINTS)
        self.a_leg_act_idx, self.a_leg_qpos, self.a_leg_range = power_curve_setup("a_", LEG_JOINTS)
        self.b_leg_act_idx, self.b_leg_qpos, self.b_leg_range = power_curve_setup("b_", LEG_JOINTS)

        self.a_root_act = self.model.actuator("a_root_x").id
        self.b_root_act = self.model.actuator("b_root_x").id
        self.a_balance_act = self.model.actuator("a_root_ry").id
        self.b_balance_act = self.model.actuator("b_root_ry").id
        self.a_ry_qpos = self.model.joint("a_root_ry").qposadr[0]
        self.a_ry_dof = self.model.joint("a_root_ry").dofadr[0]
        self.b_ry_qpos = self.model.joint("b_root_ry").qposadr[0]
        self.b_ry_dof = self.model.joint("b_root_ry").dofadr[0]
        self.a_feet = [self.model.site("a_foot_r").id, self.model.site("a_foot_l").id]
        self.b_feet = [self.model.site("b_foot_r").id, self.model.site("b_foot_l").id]

        def qidx(prefix):
            names = [f"{prefix}root_x", f"{prefix}root_z", f"{prefix}root_ry"] + \
                    [f"{prefix}{j}" for j in JOINTS]
            qpos = np.array([self.model.joint(n).qposadr[0] for n in names])
            qvel = np.array([self.model.joint(n).dofadr[0] for n in names])
            return qpos, qvel

        self.a_qpos_idx, self.a_qvel_idx = qidx("a_")
        self.b_qpos_idx, self.b_qvel_idx = qidx("b_")
        # every joint here (root_ry + all 9 limb joints) rotates about the Y axis, same
        # as root_x's translation axis is X. Under a true left-right mirror reflection
        # (X -> -X), root_x AND every Y-axis rotation flip sign -- only root_z (height)
        # doesn't. [root_x, root_z, root_ry, <9 joints>] -> only index 1 stays +1.
        self._qpos_mirror = np.array([-1.0, 1.0] + [-1.0] * (1 + len(JOINTS)))
        # the action vector is [root_x thrust, <9 joint torques>] -- no height-equivalent
        # unactuated slot, so every entry flips under the same reflection.
        self._action_mirror = -np.ones(len(JOINTS) + 1)

        self.a_torso_id = self.model.body("a_torso").id
        self.b_torso_id = self.model.body("b_torso").id

        self.a_punch_weapons = {self.model.geom(f"a_{g}").id for g in ("forearm_r", "forearm_l")}
        self.a_kick_weapons = {self.model.geom(f"a_{g}").id for g in ("shin_r", "shin_l")}
        self.b_punch_weapons = {self.model.geom(f"b_{g}").id for g in ("forearm_r", "forearm_l")}
        self.b_kick_weapons = {self.model.geom(f"b_{g}").id for g in ("shin_r", "shin_l")}

        # ground-contact tracking (foot-plant reward / knee-down penalty): the shin capsule
        # spans from the knee (its own body origin, since shin_r/l is positioned exactly at
        # the knee joint) down to the foot site -- so a shin-floor contact is classified as
        # "knee" or "foot" by whichever end its contact point is closer to, no extra geoms needed.
        self.floor_id = self.model.geom("floor").id

        def leg_setup(prefix):
            shin_geoms = {self.model.geom(f"{prefix}{g}").id: self.model.body(f"{prefix}{g}").id
                          for g in ("shin_r", "shin_l")}
            foot_sites = {self.model.geom(f"{prefix}{g}").id: self.model.site(f"{prefix}foot_{g[-1]}").id
                          for g in ("shin_r", "shin_l")}
            return shin_geoms, foot_sites

        self.a_shin_geoms, self.a_foot_sites = leg_setup("a_")
        self.b_shin_geoms, self.b_foot_sites = leg_setup("b_")

        def targets_by_part(prefix):
            return {
                "head": {self.model.geom(f"{prefix}head").id},
                "torso": {self.model.geom(f"{prefix}torso").id},
                "leg": {self.model.geom(f"{prefix}{g}").id for g in ("thigh_r", "thigh_l", "shin_r", "shin_l")},
            }

        self.a_targets = targets_by_part("a_")
        self.b_targets = targets_by_part("b_")

        obs_dim = len(self.a_qpos_idx) + len(self.a_qvel_idx) + \
                  len(self.b_qpos_idx) + len(self.b_qvel_idx) + 6
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        # actions: [root_x thrust] + one entry per JOINTS
        self.action_space = spaces.Box(-1.0, 1.0, shape=(len(JOINTS) + 1,), dtype=np.float32)

        self.step_count = 0
        self.health = {"a": 100.0, "b": 100.0}
        self.stagger = {"a": 0.0, "b": 0.0}
        self.down_steps = {"a": 0, "b": 0}
        self._prev_contact_pairs = {"a_punch": set(), "a_kick": set(), "b_punch": set(), "b_kick": set()}
        self._prev_a_action = np.zeros(self.action_space.shape[0], dtype=np.float32)
        self._prev_b_action = np.zeros(self.action_space.shape[0], dtype=np.float32)
        self._a_mirror = False
        self._t = 0.0

        if opponent_policy_path:
            self.set_opponent(opponent_policy_path)

    def set_opponent(self, path):
        """Load (or reload) the frozen policy that controls 'b', for self-play.
        Pass None/empty to fall back to the scripted shadow-boxing opponent."""
        if not path:
            self.opponent_policy = None
            return
        from stable_baselines3 import PPO
        self.opponent_policy = PPO.load(path, device="cpu")

    def _obs_view(self, self_qpos_idx, self_qvel_idx, opp_qpos_idx, opp_qvel_idx,
                  self_torso_id, opp_torso_id, self_health, opp_health,
                  self_stagger, opp_stagger, mirror):
        """Builds an observation from one fighter's own point of view -- shared by both
        _obs() (a, optionally mirrored for augmentation) and _obs_for_b() (b, always
        mirrored). With mirror=True this is a true left-right mirror reflection (X ->
        -X) of the raw physical state: root_x flips (translation along the flipped
        axis), root_z doesn't (height is unaffected by an X-reflection), and root_ry +
        every limb joint flip too, since they're all rotations about the Y axis and
        reflection reverses the sign of any rotation about an axis lying in the mirror
        plane. Without the joint half of this, b's punches/kicks used the same
        absolute-direction convention as a's whether or not that was correct for b's
        actual position -- verified this measurably weakened (not eliminated) b's
        offense (658.9 dealt by a vs 116.8 by b over 40 episodes)."""
        sx, sz = self.data.xpos[self_torso_id][[0, 2]]
        ox, oz = self.data.xpos[opp_torso_id][[0, 2]]
        sign = -1.0 if mirror else 1.0
        extra = np.array([
            sign * (ox - sx), oz - sz,
            self_health / 100.0, opp_health / 100.0,
            self_stagger, opp_stagger,
        ])
        m = self._qpos_mirror if mirror else 1.0
        self_qpos = self.data.qpos[self_qpos_idx] * m
        self_qvel = self.data.qvel[self_qvel_idx] * m
        opp_qpos = self.data.qpos[opp_qpos_idx] * m
        opp_qvel = self.data.qvel[opp_qvel_idx] * m
        return np.concatenate([self_qpos, self_qvel, opp_qpos, opp_qvel, extra]).astype(np.float32)

    def _obs_for_b(self):
        """'b's point of view: 'b' is self, 'a' is the opponent. 'b' starts on the
        opposite side from 'a', so this is always the mirrored view (see _obs_view)."""
        return self._obs_view(
            self.b_qpos_idx, self.b_qvel_idx, self.a_qpos_idx, self.a_qvel_idx,
            self.b_torso_id, self.a_torso_id,
            self.health["b"], self.health["a"], self.stagger["b"], self.stagger["a"],
            mirror=True,
        )

    def _contact_damage_by_part(self, weapons, targets_by_part, prev_pairs):
        """Returns ({part_name: damage}, current_pairs). Damage only counts for
        weapon-target geom pairs making *new* contact this step (a persisting press,
        e.g. resting on a downed opponent, deals no extra damage) -- one touch, one hit."""
        damage = {part: 0.0 for part in targets_by_part}
        current_pairs = set()
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            for part, geoms in targets_by_part.items():
                pair = None
                if c.geom1 in weapons and c.geom2 in geoms:
                    pair = (c.geom1, c.geom2)
                elif c.geom2 in weapons and c.geom1 in geoms:
                    pair = (c.geom2, c.geom1)
                if pair is None:
                    continue
                current_pairs.add(pair)
                if pair not in prev_pairs:
                    force6 = np.zeros(6)
                    mujoco.mj_contactForce(self.model, self.data, i, force6)
                    damage[part] += abs(force6[0])
        for part in damage:
            damage[part] = min(damage[part] * FORCE_TO_DAMAGE, MAX_STEP_DAMAGE)
        return damage, current_pairs

    def _leg_ground_contacts(self, shin_geoms, foot_sites):
        """Counts, out of this fighter's 2 legs, how many have the foot planted on the
        floor vs. a knee down. Each shin-floor contact is classified by whichever end
        (knee = the shin body's own origin, foot = the foot site) the contact point is
        closer to. Returns (feet_planted, knees_down), each in [0, 2]."""
        feet_planted, knees_down = 0, 0
        for shin_geom_id, shin_body_id in shin_geoms.items():
            knee_pos = self.data.xpos[shin_body_id]
            foot_pos = self.data.site_xpos[foot_sites[shin_geom_id]]
            touched, nearest_is_foot = False, True
            for i in range(self.data.ncon):
                c = self.data.contact[i]
                if not ((c.geom1 == self.floor_id and c.geom2 == shin_geom_id)
                        or (c.geom2 == self.floor_id and c.geom1 == shin_geom_id)):
                    continue
                touched = True
                d_foot = np.linalg.norm(c.pos - foot_pos)
                d_knee = np.linalg.norm(c.pos - knee_pos)
                nearest_is_foot = d_foot <= d_knee
                if nearest_is_foot:
                    break  # foot contact takes priority if this leg has both
            if touched:
                feet_planted += int(nearest_is_foot)
                knees_down += int(not nearest_is_foot)
        return feet_planted, knees_down

    def _obs(self):
        """'a's point of view. Mirrored (self._a_mirror, re-rolled each reset) on a
        random fraction of episodes -- see A_MIRROR_AUGMENT_PROB."""
        return self._obs_view(
            self.a_qpos_idx, self.a_qvel_idx, self.b_qpos_idx, self.b_qvel_idx,
            self.a_torso_id, self.b_torso_id,
            self.health["a"], self.health["b"], self.stagger["a"], self.stagger["b"],
            mirror=self._a_mirror,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        noise = self.np_random.uniform(-0.05, 0.05, size=self.model.nq)
        self.data.qpos[:] += noise
        mujoco.mj_forward(self.model, self.data)
        self.step_count = 0
        self.health = {"a": 100.0, "b": 100.0}
        self.stagger = {"a": 0.0, "b": 0.0}
        self.down_steps = {"a": 0, "b": 0}
        self._prev_contact_pairs = {"a_punch": set(), "a_kick": set(), "b_punch": set(), "b_kick": set()}
        self._prev_a_action = np.zeros(self.action_space.shape[0], dtype=np.float32)
        self._prev_b_action = np.zeros(self.action_space.shape[0], dtype=np.float32)
        self._a_mirror = bool(self.np_random.random() < A_MIRROR_AUGMENT_PROB)
        self._t = 0.0
        return self._obs(), {}

    def _power_scale(self, qpos_idx, ranges, min_power):
        """1.0 at the midpoint of each joint's range of motion, tapering down to
        min_power at either extreme -- a crude stand-in for a muscle force-length
        curve, so a fully outstretched or fully folded limb pushes noticeably softer
        than one swinging through the middle of its range."""
        angles = self.data.qpos[qpos_idx]
        lo, hi = ranges[:, 0], ranges[:, 1]
        frac = np.clip((angles - lo) / (hi - lo), 0.0, 1.0)
        return min_power + (1.0 - min_power) * np.sin(np.pi * frac)

    def step(self, action, b_full_action=None):
        """b_full_action: pass a real action (in b's own mirrored-frame convention, same
        as _obs_for_b()'s caller would produce) to drive 'b' directly instead of querying
        self.opponent_policy -- used by the paired self-play VecEnv, where 'b' is the SAME
        live policy being trained rather than a frozen snapshot."""
        action = np.clip(action, -1.0, 1.0)
        jerk_penalty = JERK_PENALTY_SCALE * np.sum((action - self._prev_a_action) ** 2)
        self._prev_a_action = action.copy()
        # 'action' is in whatever frame _obs() presented to the policy this episode (see
        # A_MIRROR_AUGMENT_PROB) -- flip it back to real ctrl-space before applying, same
        # as 'b's action is always flipped back below.
        real_action = action * self._action_mirror if self._a_mirror else action
        a_root_action, a_joint_action = real_action[0], real_action[1:]

        if b_full_action is None and self.opponent_policy is not None:
            b_full_action, _ = self.opponent_policy.predict(self._obs_for_b(), deterministic=False)

        if b_full_action is not None:
            b_full_action = np.clip(b_full_action, -1.0, 1.0)
            # the whole action (root thrust + every joint torque) comes back in the mirrored
            # frame -- flip all of it to get 'b's real ctrl (see _obs_for_b for why)
            b_real_action = b_full_action * self._action_mirror
            b_root_action_fixed, b_ctrl = b_real_action[0], b_real_action[1:]
            b_jerk_penalty = JERK_PENALTY_SCALE * np.sum((b_full_action - self._prev_b_action) ** 2)
            self._prev_b_action = b_full_action.copy()
        else:
            b_root_action_fixed = None
            b_ctrl = get_ctrl(self._t, phase=np.pi)
            b_jerk_penalty = None

        for _ in range(FRAME_SKIP):
            a_authority = 1.0 - STAGGER_CONTROL_LOSS * self.stagger["a"]
            b_authority = 1.0 - STAGGER_CONTROL_LOSS * self.stagger["b"]
            a_noise = self.np_random.normal(0.0, STAGGER_NOISE_STD * self.stagger["a"], size=a_joint_action.shape)
            b_noise = self.np_random.normal(0.0, STAGGER_NOISE_STD * self.stagger["b"], size=b_ctrl.shape)
            self.data.ctrl[self.a_act] = np.clip(a_joint_action * a_authority + a_noise, -1.0, 1.0)
            self.data.ctrl[self.b_act] = np.clip(b_ctrl * b_authority + b_noise, -1.0, 1.0)
            self.data.ctrl[self.a_act[self.a_arm_act_idx]] *= self._power_scale(self.a_arm_qpos, self.a_arm_range, ARM_MIN_POWER)
            self.data.ctrl[self.b_act[self.b_arm_act_idx]] *= self._power_scale(self.b_arm_qpos, self.b_arm_range, ARM_MIN_POWER)
            self.data.ctrl[self.a_act[self.a_leg_act_idx]] *= self._power_scale(self.a_leg_qpos, self.a_leg_range, LEG_MIN_POWER)
            self.data.ctrl[self.b_act[self.b_leg_act_idx]] *= self._power_scale(self.b_leg_qpos, self.b_leg_range, LEG_MIN_POWER)

            ax = self.data.xpos[self.a_torso_id][0]
            bx = self.data.xpos[self.b_torso_id][0]
            self.data.ctrl[self.a_root_act] = np.clip(a_root_action * a_authority, -1.0, 1.0)
            if b_root_action_fixed is not None:
                b_root_action = b_root_action_fixed
            else:
                b_root_action = np.clip((ax - bx) * B_APPROACH_GAIN, -1.0, 1.0)
            self.data.ctrl[self.b_root_act] = np.clip(b_root_action * b_authority, -1.0, 1.0)

            a_ry, a_ry_vel = self.data.qpos[self.a_ry_qpos], self.data.qvel[self.a_ry_dof]
            b_ry, b_ry_vel = self.data.qpos[self.b_ry_qpos], self.data.qvel[self.b_ry_dof]
            a_balance = -BALANCE_KP * a_ry - BALANCE_KD * a_ry_vel
            b_balance = -BALANCE_KP * b_ry - BALANCE_KD * b_ry_vel
            self.data.ctrl[self.a_balance_act] = np.clip(a_balance * a_authority, -1.0, 1.0)
            self.data.ctrl[self.b_balance_act] = np.clip(b_balance * b_authority, -1.0, 1.0)

            mujoco.mj_step(self.model, self.data)
            self._t += self.model.opt.timestep

        pp = self._prev_contact_pairs
        dmg_to_b_punch, pp["a_punch"] = self._contact_damage_by_part(self.a_punch_weapons, self.b_targets, pp["a_punch"])
        dmg_to_b_kick, pp["a_kick"] = self._contact_damage_by_part(self.a_kick_weapons, self.b_targets, pp["a_kick"])
        dmg_to_a_punch, pp["b_punch"] = self._contact_damage_by_part(self.b_punch_weapons, self.a_targets, pp["b_punch"])
        dmg_to_a_kick, pp["b_kick"] = self._contact_damage_by_part(self.b_kick_weapons, self.a_targets, pp["b_kick"])

        dmg_to_b = {p: dmg_to_b_punch[p] + dmg_to_b_kick[p] for p in dmg_to_b_punch}
        dmg_to_a = {p: dmg_to_a_punch[p] + dmg_to_a_kick[p] for p in dmg_to_a_punch}

        def health_damage(parts):
            return sum(parts[p] * TARGET_HEALTH_MULT[p] for p in parts)

        def reward_damage(parts):
            return (parts["torso"] * TARGET_HEALTH_MULT["torso"]
                    + parts["leg"] * TARGET_HEALTH_MULT["leg"]
                    + parts["head"] * TARGET_HEALTH_MULT["head"] * HEAD_REWARD_BONUS)

        self.health["b"] = max(0.0, self.health["b"] - health_damage(dmg_to_b))
        self.health["a"] = max(0.0, self.health["a"] - health_damage(dmg_to_a))

        # leg hits build stagger (harder to hold posture); it decays back to 0 each step
        self.stagger["a"] = min(1.0, self.stagger["a"] * STAGGER_DECAY + dmg_to_a["leg"] * LEG_STAGGER_GAIN)
        self.stagger["b"] = min(1.0, self.stagger["b"] * STAGGER_DECAY + dmg_to_b["leg"] * LEG_STAGGER_GAIN)

        a_z = self.data.xpos[self.a_torso_id][2]
        b_z = self.data.xpos[self.b_torso_id][2]
        a_down_now = a_z < FALL_HEIGHT
        b_down_now = b_z < FALL_HEIGHT
        self.down_steps["a"] = self.down_steps["a"] + 1 if a_down_now else 0
        self.down_steps["b"] = self.down_steps["b"] + 1 if b_down_now else 0

        # a knockdown is recoverable (get back above FALL_HEIGHT) unless it drags on
        # too long or health hits zero, either of which counts as being knocked out
        a_out = self.health["a"] <= 0.0 or self.down_steps["a"] >= DOWN_RECOVERY_STEPS
        b_out = self.health["b"] <= 0.0 or self.down_steps["b"] >= DOWN_RECOVERY_STEPS

        a_foot_xs = [self.data.site_xpos[s][0] for s in self.a_feet]
        b_foot_xs = [self.data.site_xpos[s][0] for s in self.b_feet]
        foot_dist = min(abs(bf - af) for af in a_foot_xs for bf in b_foot_xs)
        # log form: no free zone (always some gradient to close in), but steep only near contact
        # range and flattening out at long range, instead of an unbounded linear penalty
        engage_penalty = np.log1p(foot_dist) * ENGAGE_PENALTY_SCALE
        a_down_depth = max(0.0, FALL_HEIGHT - a_z)
        b_down_depth = max(0.0, FALL_HEIGHT - b_z)
        down_penalty = DOWN_PENALTY_SCALE * (a_down_depth - b_down_depth)
        knockdown_entry = KNOCKDOWN_ENTRY_PENALTY * (int(self.down_steps["a"] == 1) - int(self.down_steps["b"] == 1))
        a_lean_excess = max(0.0, abs(a_ry) - BALANCE_FREE_ZONE)
        b_lean_excess = max(0.0, abs(b_ry) - BALANCE_FREE_ZONE)
        balance_penalty = BALANCE_PENALTY_SCALE * (a_lean_excess ** 2 - b_lean_excess ** 2)

        a_feet_planted, a_knees_down = self._leg_ground_contacts(self.a_shin_geoms, self.a_foot_sites)
        b_feet_planted, b_knees_down = self._leg_ground_contacts(self.b_shin_geoms, self.b_foot_sites)
        ground_bonus = GROUND_CONTACT_SCALE * ((a_feet_planted - b_feet_planted) / 2.0)
        knee_penalty = KNEE_CONTACT_PENALTY * (a_knees_down - b_knees_down)

        strike_reward = (reward_damage(dmg_to_b_punch) + reward_damage(dmg_to_b_kick) * KICK_REWARD_BONUS) \
            - (reward_damage(dmg_to_a_punch) + reward_damage(dmg_to_a_kick) * KICK_REWARD_BONUS)

        # each entry is this step's actual contribution to `reward` -- kept as the single source of
        # truth (reward = sum of these) so the breakdown in `info` can never drift from the total.
        reward_terms = {
            "strike": strike_reward * DAMAGE_REWARD_SCALE,
            "effort": -EFFORT_COST * np.sum(action ** 2),
            "jerk": -jerk_penalty,
            "engage": -engage_penalty,
            "down": -down_penalty,
            "knockdown_entry": -knockdown_entry,
            "balance": -balance_penalty,
            "ground": ground_bonus,
            "knee": -knee_penalty,
            "terminal": 0.0,
        }

        # mirrored reward from 'b's own point of view (self-play only, since the scripted
        # opponent's "score" isn't a meaningful training signal). engage_penalty is symmetric
        # (a shared distance, not an a-vs-b difference) so it carries over unchanged; down_penalty,
        # knockdown_entry, balance_penalty, ground_bonus and knee_penalty are all signed a-vs-b
        # differences, so b's version is just their negation.
        if b_full_action is not None:
            reward_terms_b = {
                "strike": -strike_reward * DAMAGE_REWARD_SCALE,
                "effort": -EFFORT_COST * np.sum(b_full_action ** 2),
                "jerk": -b_jerk_penalty,
                "engage": -engage_penalty,
                "down": down_penalty,
                "knockdown_entry": knockdown_entry,
                "balance": balance_penalty,
                "ground": -ground_bonus,
                "knee": knee_penalty,
                "terminal": 0.0,
            }
        else:
            reward_terms_b = None

        terminated = False
        if a_out and b_out:
            reward_terms["terminal"] = -MUTUAL_FALL_PENALTY
            if reward_terms_b is not None:
                reward_terms_b["terminal"] = -MUTUAL_FALL_PENALTY
            terminated = True
        elif a_out:
            reward_terms["terminal"] = -FALL_PENALTY
            if reward_terms_b is not None:
                reward_terms_b["terminal"] = FALL_PENALTY
            terminated = True
        elif b_out:
            reward_terms["terminal"] = FALL_PENALTY
            if reward_terms_b is not None:
                reward_terms_b["terminal"] = -FALL_PENALTY
            terminated = True

        reward = sum(reward_terms.values())

        self.step_count += 1
        truncated = self.step_count >= MAX_STEPS
        info = {
            "health_a": self.health["a"], "health_b": self.health["b"],
            "stagger_a": self.stagger["a"], "stagger_b": self.stagger["b"],
            "reward_breakdown": reward_terms,
        }
        if reward_terms_b is not None:
            info["reward_b"] = sum(reward_terms_b.values())
            info["reward_breakdown_b"] = reward_terms_b

        if self.render_mode == "human":
            self.render()

        return self._obs(), reward, terminated, truncated, info

    def render(self):
        if self._viewer is None:
            import mujoco.viewer
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self._viewer.sync()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None


class Fighter2DEnvForB(Fighter2DEnv):
    """Same physics/reward as Fighter2DEnv, but the RL action space controls 'b'
    instead of 'a' -- 'a' becomes the frozen opponent (loaded via set_opponent(),
    inherited unchanged from the base class). This trains a policy with DIRECT
    gradient exposure to playing from b's mirrored frame, instead of relying on a
    single shared policy to generalize there via _obs_for_b() alone -- see
    docs/기술문서 section 4: forcing the same weights to handle both frames measurably
    kept producing a persistent a/b asymmetry (self-play, mirror augmentation, paired
    self-play, and even a provably mirror-equivariant policy all failed to fully fix
    it). Training two independently-specialized policies, alternating which one is
    frozen each round (see selfplay_league_loop.sh), sidesteps the generalization
    problem entirely -- each side only ever needs to be good at its own role."""

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        return self._obs_for_b(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        if self.opponent_policy is None:
            raise RuntimeError("Fighter2DEnvForB requires a frozen 'a' opponent -- call set_opponent() first")
        a_action, _ = self.opponent_policy.predict(self._obs(), deterministic=False)
        a_action = np.clip(a_action, -1.0, 1.0)
        _, reward_a, terminated, truncated, info = super().step(a_action, b_full_action=action)
        obs_b = self._obs_for_b()
        reward_b = info["reward_b"]
        info_b = {
            "health_a": info["health_a"], "health_b": info["health_b"],
            "stagger_a": info["stagger_a"], "stagger_b": info["stagger_b"],
            "reward_breakdown": info["reward_breakdown_b"],
            "reward_b": reward_a,  # the frozen opponent's score, for the same
            # "am I outplaying my own recent past" logging OpponentRewardCallback does
        }
        return obs_b, reward_b, terminated, truncated, info_b
