"""Diagnostic: how often do kick impacts hit the MAX_STEP_DAMAGE cap?

Monkeypatches _contact_damage_by_part to also record the *pre-cap* force-based
damage for kick weapon sets (a's and b's shins), so we can see the raw force
distribution instead of the clipped one. Does not modify env.py.
"""
import sys

import numpy as np
from stable_baselines3 import PPO

from env import Fighter2DEnv, FORCE_TO_DAMAGE, MAX_STEP_DAMAGE

CHECKPOINT = sys.argv[1] if len(sys.argv) > 1 else "../checkpoints/ppo_selfplay_20260816_211053.zip"
N_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 20000

model = PPO.load(CHECKPOINT, device="cpu")
env = Fighter2DEnv(opponent_policy_path=CHECKPOINT)  # mirror self-play match

kick_events = []  # raw pre-cap damage values for every NEW kick contact this run
punch_events = []

orig = env._contact_damage_by_part


def patched(weapons, targets_by_part, prev_pairs):
    is_kick = weapons is env.a_kick_weapons or weapons is env.b_kick_weapons
    is_punch = weapons is env.a_punch_weapons or weapons is env.b_punch_weapons
    damage, current_pairs = ({part: 0.0 for part in targets_by_part}, set())
    import mujoco
    for i in range(env.data.ncon):
        c = env.data.contact[i]
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
                mujoco.mj_contactForce(env.model, env.data, i, force6)
                raw_damage = abs(force6[0]) * FORCE_TO_DAMAGE
                damage[part] += raw_damage
                if is_kick and raw_damage > 0:
                    kick_events.append(raw_damage)
                elif is_punch and raw_damage > 0:
                    punch_events.append(raw_damage)
    for part in damage:
        damage[part] = min(damage[part], MAX_STEP_DAMAGE)
    return damage, current_pairs


env._contact_damage_by_part = patched

obs, info = env.reset()
for _ in range(N_STEPS):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

kick_events = np.array(kick_events)
print(f"checkpoint: {CHECKPOINT}")
print(f"steps: {N_STEPS}")
print(f"total kick contact events: {len(kick_events)}")
if len(kick_events):
    capped = (kick_events >= MAX_STEP_DAMAGE).sum()
    print(f"cap = {MAX_STEP_DAMAGE}")
    print(f"events hitting/exceeding cap: {capped} ({100*capped/len(kick_events):.1f}%)")
    print(f"raw pre-cap damage: mean={kick_events.mean():.2f} median={np.median(kick_events):.2f} "
          f"p90={np.percentile(kick_events,90):.2f} max={kick_events.max():.2f}")
else:
    print("no kick contacts recorded")

punch_events = np.array(punch_events)
print()
print(f"total punch contact events: {len(punch_events)}")
if len(punch_events):
    capped = (punch_events >= MAX_STEP_DAMAGE).sum()
    print(f"events hitting/exceeding cap: {capped} ({100*capped/len(punch_events):.1f}%)")
    print(f"raw pre-cap damage: mean={punch_events.mean():.2f} median={np.median(punch_events):.2f} "
          f"p90={np.percentile(punch_events,90):.2f} max={punch_events.max():.2f}")
