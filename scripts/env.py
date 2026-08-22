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
DAMAGE_REWARD_SCALE = 3.0      # 0.35 -> 0.7 -> 1.0 -> 3.0 -> 5.0 -> back to 3.0 -> 10.0 -> 5.0 ->
                                # back to 3.0. Both 5.0 (scratch28) and 10.0 (scratch27) got kicks
                                # landing (0->15/0->36 contacts per 6000 steps) but the adversarial
                                # fall rate jumped to ~37-40% at BOTH scales (vs 3.0's ~23%) -- a
                                # cliff, not a gradual tradeoff -- while engagement/posture degraded
                                # more gradually with scale. No usable middle ground found; reverted
                                # to 3.0 (scratch23) as the league base. If kicks are worth chasing
                                # again, prefer raising KICK_REWARD_BONUS specifically over this
                                # scale, so punches don't get swept up too. Reward-only scale,
                                # doesn't change actual HP loss per hit (that's FORCE_TO_DAMAGE).

# ---- movement / control shaping ----
EFFORT_COST = 0.01             # back to the original value. Raised in steps (0.01 -> 0.05 -> 0.15)
                                # chasing self-destabilized falls, but measured NO effect on mean action
                                # magnitude at any of those values -- meanwhile a reward_breakdown check
                                # showed it was ~-250/episode, over 75% of total reward and dwarfing the
                                # +5/episode strike signal. It never worked and was drowning out the
                                # actual combat signal, so reverting rather than tuning it further.
JERK_PENALTY_SCALE = 0.02      # cost on action change frame-to-frame, discourages full-power reversals
                                # (e.g. root thrust +1 -> -1 in one step)
ENGAGE_PENALTY_SCALE = 0.5     # cost on log1p(foot distance) every step -- always some gradient to
                                # close in (no free zone), steepest near contact range and flattening
                                # out at long range. Was 0.3 -> 5.0 -> 2.0, tuning this scale alone kept
                                # trading falls for engagement and back (2.0 was still 60% falls vs the
                                # 33% balance-only baseline). Spawn distance (build_model.py) was cut
                                # from 1.8 to 1.2 instead -- less ground to cover, so backed the scale
                                # back down too rather than stacking both fixes at full strength.
PROGRESS_REWARD_SCALE = 2.0    # bonus for REDUCING foot_dist this step (previous - current), on top
                                # of engage_penalty's static "how far apart are we" cost. Standing
                                # still and not falling only slowly loses under engage_penalty alone
                                # (confirmed: reward fell from 114->50 over training at a large spawn
                                # distance as episodes got longer without closing in) -- this gives a
                                # direct, immediate signal for the ACT of closing distance, meant to
                                # help a distance curriculum (build_model.py) actually learn each step
                                # up instead of just tolerating the larger static cost.
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
BALANCE_PENALTY_SCALE = 1000.0  # was 100 -> 300 -> 600 -> 1000 -- 600 got real matches (P1 vs a
# dedicated P2) down to a fairer ~50:50 split, but still falls in roughly half of all
# episodes by eye, so raised again (still squared beyond BALANCE_FREE_ZONE, so normal
# standing wobble is untaxed)

# Tried gating root_x thrust efficiency by leg extension (skeleton-sled-style moment-of-inertia
# argument) to fight the crouch/"sitting" exploit, before the head-height reward existed. Removed
# once the height reward solved that cleanly on its own (0% crouching in scratch23) -- the gate's
# average-both-knees formula was an unintended tax on kicking too (bending one leg to kick pulled
# the average down and cost mobility), and measured kick contacts were 0/3000 steps with it in
# place vs 345 punch contacts in the same sample.

# Direct height PENALTY was tried 3x (uncapped, capped-through-fall, zeroed-at-fall then scaled
# up) and abandoned -- the "recover from a knockdown" process (up to DOWN_RECOVERY_STEPS)
# necessarily passes back UP through the same z-height range a penalty-below-a-threshold approach
# taxes, so it can't distinguish "legitimately standing back up" from "ducking as a balance-
# avoidance exploit". Scaling it enough to fight the crouch exploit also punishes every real
# recovery attempt and destabilizes training (ep_rew_mean collapsed to -13,000, -7,500, then
# -113,000 at higher scales). A pure REWARD for height instead -- always >= 0, accelerating
# (squared) as head height approaches standing -- has no cliff/threshold to double-count against:
# a knockdown recovery just earns less bonus while still low, never an escalating penalty, so it
# can be tuned aggressively without the same blowup risk.
STANDING_HEAD_HEIGHT = 1.29     # a_head z at spawn, standing straight
HEAD_HEIGHT_REWARD_SCALE = 3.0
# Uncapped ratio rewarded stretching taller than standing forever -- "maximally rigid/tall" was
# strictly better than a natural, slightly-bent ready stance, since more height always meant more
# reward. Capping the ratio at 1.0 makes reaching normal standing height the ceiling; the downward
# gradient (recover from a knockdown) is unaffected, only the "stretch past normal" incentive is
# removed.
HEAD_HEIGHT_RATIO_CAP = 1.0

# Stability bonus -- originally keyed on |root_ry angular velocity| (how fast the torso is
# tipping), but a direct measurement of 22 fall events (docs 10.7) showed 0/22 were rotational
# tipping: |root_ry| stayed ~0.02-0.05 rad (near zero) all the way to the fall, while torso z
# dropped ~0.16m in the same window. root_z has NO actuator (pure slide joint, build_model.py) --
# it's a passive consequence of the legs' ground reaction force, so once support fails the torso
# just sinks straight down with no direct way for the policy to arrest it via root_ry. So an
# ry-velocity-based bonus was measuring the wrong axis entirely and (correctly) had zero effect on
# fall rate across 79 rounds. Rewritten to reward low DOWNWARD root_z velocity instead -- the
# actual signature of an in-progress collapse. Measured on a real checkpoint: normal
# standing/bobbing has downward z-vel median~0.035, p90~0.26 m/s; real falls spike to ~1.5-2.7.
# Z_VEL_REF=1.0 gives normal movement a strong partial bonus while an in-progress fast collapse
# gets ~0 (only penalizes downward speed, not upward/jumping motion). Always >= 0, same
# cliff-free shape as height reward. Scale kept modest like before (per-episode magnitude checked
# against strike before training -- see docs 9.8/10.3 for why that matters).
STABILITY_REWARD_SCALE = 0.15
Z_VEL_REF = 1.0

# The z-velocity stability bonus above didn't move the needle on fall rate over a full 100-round
# run (stayed flat ~68%, docs 10.8) -- the reward target was right (docs 10.7) but root_z has no
# actuator, so the only lever the policy has is indirect (hip/knee -> ground reaction force).
# Trying a more direct lever: reward a wider front-back stance (the only kind of "wide base" this
# sagittal-plane-only biped can form -- see docs 10.2, legs can't spread sideways, only stagger
# fore/aft), on the theory that a bigger base of support gives more margin before ground reaction
# fails and root_z starts free-falling. Measured foot x-gap on a real checkpoint: median~0.025m
# (feet together most of the time), p90~0.68m (only during kicks/punches, not sustained). Capped
# at ratio 1.0 like height, so it stops rewarding past a normal stance width instead of pushing
# toward an ever-wider (eventually anatomically absurd) split.
STANCE_REWARD_SCALE = 0.15
STANCE_GAP_REF = 0.4

# Reward for keeping knees off the ground. First version used the existing contact-based
# knees_down count (0/1/2, from _leg_ground_contacts) -- but that's a binary post-hoc signal: it
# only fires at the instant of contact, giving no gradient warning beforehand the way height/
# stability do. Switched to a continuous proxy instead: each knee joint's own world-frame height
# above the floor (xanchor z), same cliff-free capped-ratio shape as height reward, so the bonus
# smoothly fades as a knee sinks toward the ground well before it actually touches. Measured
# normal standing/moving min-knee-height on a real checkpoint: median~0.41m, occasionally
# dipping to ~0.07m during kicks (not a problem, that's brief); KNEE_HEIGHT_REF=0.4 puts normal
# stance near the ratio-1 ceiling. Uses the MINIMUM of the two knees (not the average) so one
# knee sinking can't be masked by the other staying up -- matches the visual "kneeling" cue,
# where even one knee down is the thing to avoid.
KNEE_AVOID_SCALE = 0.15
KNEE_HEIGHT_REF = 0.4

# Tried a flat per-step "alive cost" to break passive standoffs (nothing in strike/engage/
# progress/height pushes toward actually attacking once already close). Both scales tried (0.1,
# 0.03) made things worse, not better -- engagement dropped (66.7% -> 56.7% -> 43.3%) and height
# fell back toward the crouch (0.92 -> 0.62 -> 0.56) instead of improving. Reverted; passive
# standoffs are a known open issue.

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

# Tried gating root_x thrust by leg stride split (force stepping instead of a free torso
# slide) -- measured it actually made the vs-scripted-bot fall rate WORSE (43% -> 57%,
# regressing back to pre-balance-penalty levels), likely because the required leg split
# fights the balance penalty's incentive to stay upright/symmetric. Reverted.


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
        self.a_z_dof = self.model.joint("a_root_z").dofadr[0]
        self.b_z_dof = self.model.joint("b_root_z").dofadr[0]
        self.a_knee_r_id = self.model.joint("a_knee_r").id
        self.a_knee_l_id = self.model.joint("a_knee_l").id
        self.b_knee_r_id = self.model.joint("b_knee_r").id
        self.b_knee_l_id = self.model.joint("b_knee_l").id
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
        self.a_head_id = self.model.body("a_head").id
        self.b_head_id = self.model.body("b_head").id

        self.a_punch_weapons = {self.model.geom(f"a_{g}").id for g in ("forearm_r", "forearm_l")}
        self.a_kick_weapons = {self.model.geom(f"a_{g}").id for g in ("shin_r", "shin_l")}
        self.b_punch_weapons = {self.model.geom(f"b_{g}").id for g in ("forearm_r", "forearm_l")}
        self.b_kick_weapons = {self.model.geom(f"b_{g}").id for g in ("shin_r", "shin_l")}

        # ground-contact tracking (foot-plant reward / knee-down penalty): each leg now has a
        # dedicated foot geom (a heel-to-toe capsule at the shin's bottom, added for a real
        # fore-aft base of support instead of balancing on the shin's rounded tip) -- a
        # foot-geom-floor contact is unambiguously "foot planted", a shin-geom-floor contact
        # (the shin capsule itself, above the foot) is unambiguously "knee down".
        self.floor_id = self.model.geom("floor").id

        def leg_setup(prefix):
            shin_geoms = {self.model.geom(f"{prefix}shin_{s}").id for s in ("r", "l")}
            foot_geoms = {self.model.geom(f"{prefix}foot_{s}").id for s in ("r", "l")}
            return shin_geoms, foot_geoms

        self.a_shin_geoms, self.a_foot_geoms = leg_setup("a_")
        self.b_shin_geoms, self.b_foot_geoms = leg_setup("b_")

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

    def _leg_ground_contacts(self, shin_geoms, foot_geoms):
        """Counts, out of this fighter's 2 legs, how many have the foot planted on the
        floor vs. a knee down. Each leg now has a dedicated foot geom (below/separate
        from the shin capsule), so a foot-geom-floor contact is unambiguously a plant
        and a shin-geom-floor contact is unambiguously a knee-down. Returns
        (feet_planted, knees_down), each in [0, 2]."""
        feet_planted, knees_down = 0, 0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            other = c.geom2 if c.geom1 == self.floor_id else c.geom1 if c.geom2 == self.floor_id else None
            if other is None:
                continue
            if other in foot_geoms:
                feet_planted += 1
            elif other in shin_geoms:
                knees_down += 1
        return min(feet_planted, 2), min(knees_down, 2)

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
        a_foot_xs = [self.data.site_xpos[s][0] for s in self.a_feet]
        b_foot_xs = [self.data.site_xpos[s][0] for s in self.b_feet]
        self._prev_foot_dist = min(abs(bf - af) for af in a_foot_xs for bf in b_foot_xs)
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
        a_stance_gap = abs(a_foot_xs[0] - a_foot_xs[1])
        b_stance_gap = abs(b_foot_xs[0] - b_foot_xs[1])
        a_stance_reward = STANCE_REWARD_SCALE * min(1.0, a_stance_gap / STANCE_GAP_REF) ** 2
        b_stance_reward = STANCE_REWARD_SCALE * min(1.0, b_stance_gap / STANCE_GAP_REF) ** 2
        # log form: no free zone (always some gradient to close in), but steep only near contact
        # range and flattening out at long range, instead of an unbounded linear penalty
        engage_penalty = np.log1p(foot_dist) * ENGAGE_PENALTY_SCALE
        # reward for the ACT of closing distance this step, not just the state of being close --
        # positive when foot_dist shrank since last step. Symmetric for a/b (both get credit for
        # the same shared distance closing), unlike engage_penalty being on both sides too.
        progress_reward = (self._prev_foot_dist - foot_dist) * PROGRESS_REWARD_SCALE
        self._prev_foot_dist = foot_dist
        a_down_depth = max(0.0, FALL_HEIGHT - a_z)
        b_down_depth = max(0.0, FALL_HEIGHT - b_z)
        down_penalty = DOWN_PENALTY_SCALE * (a_down_depth - b_down_depth)
        knockdown_entry = KNOCKDOWN_ENTRY_PENALTY * (int(self.down_steps["a"] == 1) - int(self.down_steps["b"] == 1))
        a_lean_excess = max(0.0, abs(a_ry) - BALANCE_FREE_ZONE)
        b_lean_excess = max(0.0, abs(b_ry) - BALANCE_FREE_ZONE)
        # absolute, NOT signed a-vs-b like down/knockdown_entry/ground/knee below -- those are
        # genuinely relative (what matters in a knockdown is WHO is down), but balance isn't:
        # both fighters being equally unstable at once is still bad for both of them, and a
        # signed-difference version nets to ~0 in exactly that case, killing the gradient
        # toward staying upright whenever the opponent is just as off-balance.
        a_balance_penalty = BALANCE_PENALTY_SCALE * a_lean_excess ** 2
        b_balance_penalty = BALANCE_PENALTY_SCALE * b_lean_excess ** 2

        a_head_z = self.data.xpos[self.a_head_id][2]
        b_head_z = self.data.xpos[self.b_head_id][2]
        a_height_reward = HEAD_HEIGHT_REWARD_SCALE * min(HEAD_HEIGHT_RATIO_CAP, max(0.0, a_head_z / STANDING_HEAD_HEIGHT)) ** 2
        b_height_reward = HEAD_HEIGHT_REWARD_SCALE * min(HEAD_HEIGHT_RATIO_CAP, max(0.0, b_head_z / STANDING_HEAD_HEIGHT)) ** 2

        a_z_down_vel = max(0.0, -self.data.qvel[self.a_z_dof])
        b_z_down_vel = max(0.0, -self.data.qvel[self.b_z_dof])
        a_stability_reward = STABILITY_REWARD_SCALE * max(0.0, 1.0 - a_z_down_vel / Z_VEL_REF) ** 2
        b_stability_reward = STABILITY_REWARD_SCALE * max(0.0, 1.0 - b_z_down_vel / Z_VEL_REF) ** 2

        a_min_knee_z = min(self.data.xanchor[self.a_knee_r_id][2], self.data.xanchor[self.a_knee_l_id][2])
        b_min_knee_z = min(self.data.xanchor[self.b_knee_r_id][2], self.data.xanchor[self.b_knee_l_id][2])
        a_knee_avoid_reward = KNEE_AVOID_SCALE * min(1.0, max(0.0, a_min_knee_z / KNEE_HEIGHT_REF)) ** 2
        b_knee_avoid_reward = KNEE_AVOID_SCALE * min(1.0, max(0.0, b_min_knee_z / KNEE_HEIGHT_REF)) ** 2

        strike_reward = (reward_damage(dmg_to_b_punch) + reward_damage(dmg_to_b_kick) * KICK_REWARD_BONUS) \
            - (reward_damage(dmg_to_a_punch) + reward_damage(dmg_to_a_kick) * KICK_REWARD_BONUS)

        # Simplified to 4 terms (strike, engage, progress, height) -- effort/jerk/down/
        # knockdown_entry/balance/ground/knee/terminal all dropped. balance is subsumed by height
        # (leaning also drops head height, so the accelerating height reward already discourages
        # it); falling is now only discouraged indirectly, by forfeiting height+strike reward
        # while down, not via a direct terminal penalty -- kept as the single source of truth
        # (reward = sum of these) so the breakdown in `info` can never drift from the total.
        reward_terms = {
            "strike": strike_reward * DAMAGE_REWARD_SCALE,
            "engage": -engage_penalty,
            "progress": progress_reward,
            "height": a_height_reward,
            "stability": a_stability_reward,
            "stance": a_stance_reward,
            "knee_avoid": a_knee_avoid_reward,
        }

        # mirrored reward from 'b's own point of view (self-play only, since the scripted
        # opponent's "score" isn't a meaningful training signal). engage_penalty/progress_reward
        # are symmetric (a shared distance, not an a-vs-b difference) so they carry over unchanged.
        if b_full_action is not None:
            reward_terms_b = {
                "strike": -strike_reward * DAMAGE_REWARD_SCALE,
                "engage": -engage_penalty,
                "progress": progress_reward,
                "height": b_height_reward,
                "stability": b_stability_reward,
                "stance": b_stance_reward,
                "knee_avoid": b_knee_avoid_reward,
            }
        else:
            reward_terms_b = None

        terminated = a_out or b_out

        reward = sum(reward_terms.values())

        self.step_count += 1
        truncated = self.step_count >= MAX_STEPS
        info = {
            "health_a": self.health["a"], "health_b": self.health["b"],
            "stagger_a": self.stagger["a"], "stagger_b": self.stagger["b"],
            "reward_breakdown": reward_terms,
            "a_out": a_out, "b_out": b_out,
            "a_head_z": a_head_z, "b_head_z": b_head_z,
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
