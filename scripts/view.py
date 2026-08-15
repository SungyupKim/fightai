"""Opens the interactive MuJoCo viewer and drives both fighters with a scripted
sine-wave 'shadow boxing' motion so you can see the ragdoll model move.

This is just a visual smoke test -- no learning happens here. RL training
comes later and will replace get_ctrl() with a trained policy.
"""
import pathlib
import time

import mujoco
import mujoco.viewer
import numpy as np

MODEL_PATH = pathlib.Path(__file__).resolve().parent.parent / "models" / "fighter2d.xml"

JOINTS = ["head", "shoulder_r", "elbow_r", "shoulder_l", "elbow_l",
          "hip_r", "knee_r", "hip_l", "knee_l"]


def get_ctrl(t, phase=0.0):
    """Hand-scripted rhythmic motion, one value per joint in JOINTS order."""
    punch = 0.9 * max(0.0, np.sin(2.0 * t + phase))
    step = 0.5 * np.sin(1.3 * t + phase)
    return np.array([
        0.15 * np.sin(2.0 * t + phase),      # head bob
        punch,                                # shoulder_r swings out to punch
        -punch,                               # elbow_r extends with the punch
        -0.3 * np.sin(2.0 * t + phase + np.pi),  # shoulder_l guard sway
        -0.6,                                 # elbow_l stays bent (guard)
        step,                                  # hip_r
        -0.4 * max(0.0, np.sin(1.3 * t + phase)),  # knee_r
        -step,                                 # hip_l
        -0.4 * max(0.0, np.sin(1.3 * t + phase + np.pi)),  # knee_l
    ])


def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    a_ids = [model.actuator(f"a_{j}").id for j in JOINTS]
    b_ids = [model.actuator(f"b_{j}").id for j in JOINTS]

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0, 0, 1.0]
        viewer.cam.distance = 4.5
        viewer.cam.azimuth = -90
        viewer.cam.elevation = -10

        start = time.time()
        while viewer.is_running():
            t = time.time() - start
            data.ctrl[a_ids] = get_ctrl(t, phase=0.0)
            data.ctrl[b_ids] = get_ctrl(t, phase=np.pi)  # opposite phase

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(max(0.0, model.opt.timestep - (time.time() - start - t)))


if __name__ == "__main__":
    main()
