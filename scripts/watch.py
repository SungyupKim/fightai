"""Loads a trained PPO checkpoint and watches fighter 'a' play against the
scripted opponent 'b' in the interactive MuJoCo viewer.
"""
import argparse
import time

import mujoco.viewer
from stable_baselines3 import PPO

from env import FRAME_SKIP, Fighter2DEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str, help="path to a saved .zip model")
    args = parser.parse_args()

    model = PPO.load(args.checkpoint, device="cpu")
    env = Fighter2DEnv()
    obs, info = env.reset()

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.lookat[:] = [0, 0, 1.0]
        viewer.cam.distance = 4.5
        viewer.cam.azimuth = -90
        viewer.cam.elevation = -10

        while viewer.is_running():
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            viewer.sync()
            time.sleep(env.model.opt.timestep * FRAME_SKIP)

            if terminated or truncated:
                print(f"episode ended: health={info}")
                obs, info = env.reset()


if __name__ == "__main__":
    main()
