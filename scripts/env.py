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

FRAME_SKIP = 4
MAX_STEPS = 1000               # longer cap so recoverable knockdowns have room to play out
FALL_HEIGHT = 0.55
FALL_PENALTY = 50.0
MUTUAL_FALL_PENALTY = 20.0     # discourage trading a knockdown blow instead of sustained sparring
DOWN_RECOVERY_STEPS = 150      # ~3s (at FRAME_SKIP*timestep=0.02s/step) to get back up before it's a knockout
DOWN_PENALTY_PER_STEP = 0.15   # extra cost per step spent below FALL_HEIGHT, pushes toward getting up fast
KNOCKDOWN_ENTRY_PENALTY = 30.0 # one-time punitive cost the instant you go down, so diving in for a hit isn't worth it
FORCE_TO_DAMAGE = 0.4          # scales contact normal force -> health points
MAX_STEP_DAMAGE = 8.0          # clip per-step damage so one huge impact can't insta-kill
EFFORT_COST = 0.01
ALIVE_BONUS = 0.05
ENGAGE_DIST = 0.9              # beyond this, small shaping reward to close distance
B_APPROACH_GAIN = 1.5          # scripted opponent's proportional gain for walking toward 'a'

# per-bodypart health multipliers: head strikes hurt more, leg strikes least
TARGET_HEALTH_MULT = {"head": 1.6, "torso": 1.0, "leg": 0.5}
HEAD_REWARD_BONUS = 1.5        # extra reward multiplier for landing headshots, on top of health mult
KICK_REWARD_BONUS = 2.0        # extra reward multiplier for shin-landed damage, to outweigh the balance
                                # risk of committing a leg instead of just punching (reward only, not health)
DAMAGE_REWARD_SCALE = 0.35     # downweight raw contact-force reward so KO/knockdown outcomes dominate
                                # over just landing one hard hit (doesn't affect actual health loss)

# leg damage builds "stagger" (posture loss): reduces control authority + adds control
# noise for the staggered fighter, so losing your legs makes it genuinely harder to
# keep standing rather than just an arbitrary penalty.
LEG_STAGGER_GAIN = 0.03
STAGGER_DECAY = 0.94           # per control step; ~1-2s to recover from a full stagger
STAGGER_CONTROL_LOSS = 0.6     # at stagger=1.0, actuators only keep 40% authority
STAGGER_NOISE_STD = 0.4        # extra action noise, scaled by stagger


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
        self.a_root_act = self.model.actuator("a_root_x").id
        self.b_root_act = self.model.actuator("b_root_x").id

        def qidx(prefix):
            names = [f"{prefix}root_x", f"{prefix}root_z", f"{prefix}root_ry"] + \
                    [f"{prefix}{j}" for j in JOINTS]
            qpos = np.array([self.model.joint(n).qposadr[0] for n in names])
            qvel = np.array([self.model.joint(n).dofadr[0] for n in names])
            return qpos, qvel

        self.a_qpos_idx, self.a_qvel_idx = qidx("a_")
        self.b_qpos_idx, self.b_qvel_idx = qidx("b_")

        self.a_torso_id = self.model.body("a_torso").id
        self.b_torso_id = self.model.body("b_torso").id

        self.a_punch_weapons = {self.model.geom(f"a_{g}").id for g in ("forearm_r", "forearm_l")}
        self.a_kick_weapons = {self.model.geom(f"a_{g}").id for g in ("shin_r", "shin_l")}
        self.b_punch_weapons = {self.model.geom(f"b_{g}").id for g in ("forearm_r", "forearm_l")}
        self.b_kick_weapons = {self.model.geom(f"b_{g}").id for g in ("shin_r", "shin_l")}

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

    def _obs_for_b(self):
        """Same shape/ordering convention as _obs(), but from 'b's point of view:
        'b' is self, 'a' is the opponent. 'b' starts on the opposite side from 'a',
        so root_x position/velocity (index 0 of each qpos/qvel block) are mirrored
        (negated) -- otherwise a policy trained as 'a' (positive thrust = approach)
        would see the same relative-distance sign but push the wrong physical
        direction when applied to 'b's real, un-mirrored actuator."""
        ax, az = self.data.xpos[self.a_torso_id][[0, 2]]
        bx, bz = self.data.xpos[self.b_torso_id][[0, 2]]
        extra = np.array([
            -(ax - bx), az - bz,
            self.health["b"] / 100.0, self.health["a"] / 100.0,
            self.stagger["b"], self.stagger["a"],
        ])
        b_qpos = self.data.qpos[self.b_qpos_idx].copy()
        b_qvel = self.data.qvel[self.b_qvel_idx].copy()
        a_qpos = self.data.qpos[self.a_qpos_idx].copy()
        a_qvel = self.data.qvel[self.a_qvel_idx].copy()
        b_qpos[0] *= -1
        b_qvel[0] *= -1
        a_qpos[0] *= -1
        a_qvel[0] *= -1
        return np.concatenate([b_qpos, b_qvel, a_qpos, a_qvel, extra]).astype(np.float32)

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

    def _obs(self):
        ax, az = self.data.xpos[self.a_torso_id][[0, 2]]
        bx, bz = self.data.xpos[self.b_torso_id][[0, 2]]
        extra = np.array([
            bx - ax, bz - az,
            self.health["a"] / 100.0, self.health["b"] / 100.0,
            self.stagger["a"], self.stagger["b"],
        ])
        return np.concatenate([
            self.data.qpos[self.a_qpos_idx], self.data.qvel[self.a_qvel_idx],
            self.data.qpos[self.b_qpos_idx], self.data.qvel[self.b_qvel_idx],
            extra,
        ]).astype(np.float32)

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
        self._t = 0.0
        return self._obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        a_root_action, a_joint_action = action[0], action[1:]

        if self.opponent_policy is not None:
            b_full_action, _ = self.opponent_policy.predict(self._obs_for_b(), deterministic=False)
            b_full_action = np.clip(b_full_action, -1.0, 1.0)
            # root thrust comes back in the mirrored frame -- negate to get 'b's real ctrl direction
            b_root_action_fixed, b_ctrl = -b_full_action[0], b_full_action[1:]
        else:
            b_root_action_fixed = None
            b_ctrl = get_ctrl(self._t, phase=np.pi)

        for _ in range(FRAME_SKIP):
            a_authority = 1.0 - STAGGER_CONTROL_LOSS * self.stagger["a"]
            b_authority = 1.0 - STAGGER_CONTROL_LOSS * self.stagger["b"]
            a_noise = self.np_random.normal(0.0, STAGGER_NOISE_STD * self.stagger["a"], size=a_joint_action.shape)
            b_noise = self.np_random.normal(0.0, STAGGER_NOISE_STD * self.stagger["b"], size=b_ctrl.shape)
            self.data.ctrl[self.a_act] = np.clip(a_joint_action * a_authority + a_noise, -1.0, 1.0)
            self.data.ctrl[self.b_act] = np.clip(b_ctrl * b_authority + b_noise, -1.0, 1.0)

            ax = self.data.xpos[self.a_torso_id][0]
            bx = self.data.xpos[self.b_torso_id][0]
            self.data.ctrl[self.a_root_act] = np.clip(a_root_action * a_authority, -1.0, 1.0)
            if b_root_action_fixed is not None:
                b_root_action = b_root_action_fixed
            else:
                b_root_action = np.clip((ax - bx) * B_APPROACH_GAIN, -1.0, 1.0)
            self.data.ctrl[self.b_root_act] = np.clip(b_root_action * b_authority, -1.0, 1.0)

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

        dx = self.data.xpos[self.b_torso_id][0] - self.data.xpos[self.a_torso_id][0]
        engage_penalty = max(0.0, abs(dx) - ENGAGE_DIST) * 0.02
        down_penalty = DOWN_PENALTY_PER_STEP * (int(a_down_now) - int(b_down_now))
        knockdown_entry = KNOCKDOWN_ENTRY_PENALTY * (int(self.down_steps["a"] == 1) - int(self.down_steps["b"] == 1))

        strike_reward = (reward_damage(dmg_to_b_punch) + reward_damage(dmg_to_b_kick) * KICK_REWARD_BONUS) \
            - (reward_damage(dmg_to_a_punch) + reward_damage(dmg_to_a_kick) * KICK_REWARD_BONUS)

        reward = strike_reward * DAMAGE_REWARD_SCALE \
            - EFFORT_COST * np.sum(action ** 2) + ALIVE_BONUS - engage_penalty - down_penalty \
            - knockdown_entry

        terminated = False
        if a_out and b_out:
            reward -= MUTUAL_FALL_PENALTY
            terminated = True
        elif a_out:
            reward -= FALL_PENALTY
            terminated = True
        elif b_out:
            reward += FALL_PENALTY
            terminated = True

        self.step_count += 1
        truncated = self.step_count >= MAX_STEPS
        info = {
            "health_a": self.health["a"], "health_b": self.health["b"],
            "stagger_a": self.stagger["a"], "stagger_b": self.stagger["b"],
        }

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
