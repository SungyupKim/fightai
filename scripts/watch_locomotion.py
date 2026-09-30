"""Watches the solo mobility/recovery policy (locomotion_env.LocomotionEnv) in the
interactive MuJoCo viewer -- 'b' is parked off to the side and irrelevant, no health
bars/hit markers since there's no combat. Resets are triggered on a timer as well as on
episode end, so a checkpoint that's gotten good at NOT falling still gets shown a fresh
random perturbation regularly instead of just standing still forever.
"""
import argparse
import time

import mujoco.viewer
from stable_baselines3 import PPO

from env import FRAME_SKIP
from locomotion_env import LocomotionEnv

RESET_EVERY_SECONDS = 20.0     # matches env.py's MAX_STEPS=1000 (*0.02s/step) -- 6.0 was
                                # cutting episodes off at <40% of the ~700-750 step ep_len_mean
                                # measured at the end of training, hiding whatever happens after
                                # perturbation-recovery (docs 10.5x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    args = parser.parse_args()

    model = PPO.load(args.checkpoint, device="cpu")
    env = LocomotionEnv()
    obs, info = env.reset()
    last_reset = time.time()

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.lookat[:] = [0, 0, 1.0]
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = -90
        viewer.cam.elevation = -10

        while viewer.is_running():
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            viewer.sync()
            time.sleep(env.model.opt.timestep * FRAME_SKIP)

            if terminated or truncated or (time.time() - last_reset > RESET_EVERY_SECONDS):
                print(f"reset (terminated={terminated} truncated={truncated} a_out={info.get('a_out')})")
                obs, info = env.reset()
                last_reset = time.time()


if __name__ == "__main__":
    main()
