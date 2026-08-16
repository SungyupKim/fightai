"""Loads a trained PPO checkpoint and watches fighter 'a' play against 'b' in the
interactive MuJoCo viewer. 'b' is the scripted shadow-boxing opponent by default,
or another (or the same) trained policy if --opponent is given, mirrored to
control 'b' from its own point of view -- same as self-play training.
"""
import argparse
import time

import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO

from env import FRAME_SKIP, Fighter2DEnv

HIT_FLASH_STEPS = 8            # how many frames a hit marker stays visible
HEALTH_BAR_WIDTH = 0.7
HEALTH_BAR_HEIGHT = 0.05
HEALTH_BAR_Z_OFFSET = 0.65     # above the head


def _add_geom(scn, gtype, size, pos, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, gtype, np.asarray(size, dtype=np.float64),
                         np.asarray(pos, dtype=np.float64), np.eye(3).flatten(), np.asarray(rgba, dtype=np.float32))
    scn.ngeom += 1


def _draw_health_bar(scn, torso_x, torso_z, health_frac, rgba):
    # anchored on the left so the bar visibly shrinks toward that side as health drops
    left = torso_x - HEALTH_BAR_WIDTH / 2
    width = HEALTH_BAR_WIDTH * max(0.0, health_frac)
    center_x = left + width / 2
    _add_geom(scn, mujoco.mjtGeom.mjGEOM_BOX,
              [max(width / 2, 1e-4), 0.02, HEALTH_BAR_HEIGHT / 2],
              [center_x, 0.0, torso_z + HEALTH_BAR_Z_OFFSET], rgba)


def _draw_hit_marker(scn, torso_x, torso_z, age, rgba):
    # a brief expanding, fading spark at the fighter that just got hit
    t = age / HIT_FLASH_STEPS
    radius = 0.12 + 0.10 * t
    faded_rgba = [rgba[0], rgba[1], rgba[2], rgba[3] * (1.0 - t)]
    _add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [radius, 0, 0],
              [torso_x, 0.0, torso_z + 0.3], faded_rgba)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str, help="path to a saved .zip model, controls 'a'")
    parser.add_argument("--opponent", type=str, default=None,
                         help="path to a checkpoint to control 'b' (defaults to the scripted opponent; "
                              "pass the same path as checkpoint to watch a self-play mirror match)")
    args = parser.parse_args()

    model = PPO.load(args.checkpoint, device="cpu")
    env = Fighter2DEnv(opponent_policy_path=args.opponent)
    obs, info = env.reset()
    prev_health = {"a": info.get("health_a", 100.0), "b": info.get("health_b", 100.0)}
    hit_flash = {"a": 0, "b": 0}  # frames remaining for each fighter's "just got hit" marker
    A_RGBA = (0.85, 0.2, 0.2, 1.0)
    B_RGBA = (0.2, 0.4, 0.85, 1.0)

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.lookat[:] = [0, 0, 1.0]
        viewer.cam.distance = 4.5
        viewer.cam.azimuth = -90
        viewer.cam.elevation = -10

        while viewer.is_running():
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            if info["health_a"] < prev_health["a"]:
                hit_flash["a"] = HIT_FLASH_STEPS
            if info["health_b"] < prev_health["b"]:
                hit_flash["b"] = HIT_FLASH_STEPS
            prev_health["a"], prev_health["b"] = info["health_a"], info["health_b"]

            viewer.user_scn.ngeom = 0
            a_x, a_z = env.data.xpos[env.a_torso_id][[0, 2]]
            b_x, b_z = env.data.xpos[env.b_torso_id][[0, 2]]
            _draw_health_bar(viewer.user_scn, a_x, a_z, info["health_a"] / 100.0, A_RGBA)
            _draw_health_bar(viewer.user_scn, b_x, b_z, info["health_b"] / 100.0, B_RGBA)
            if hit_flash["a"] > 0:
                _draw_hit_marker(viewer.user_scn, a_x, a_z, HIT_FLASH_STEPS - hit_flash["a"], (1.0, 1.0, 0.2, 0.9))
                hit_flash["a"] -= 1
            if hit_flash["b"] > 0:
                _draw_hit_marker(viewer.user_scn, b_x, b_z, HIT_FLASH_STEPS - hit_flash["b"], (1.0, 1.0, 0.2, 0.9))
                hit_flash["b"] -= 1

            viewer.sync()
            time.sleep(env.model.opt.timestep * FRAME_SKIP)

            if terminated or truncated:
                print(f"episode ended: health={info}")
                obs, info = env.reset()
                prev_health = {"a": info.get("health_a", 100.0), "b": info.get("health_b", 100.0)}
                hit_flash = {"a": 0, "b": 0}


if __name__ == "__main__":
    main()
