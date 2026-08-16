"""Opens the interactive MuJoCo viewer for fighter3d.xml and drives both fighters
with a simple scripted motion (root thrust toward each other + arm swinging) so
you can eyeball the 3D model and its balance/movement before env3d.py exists.
"""
import pathlib
import time

import mujoco
import mujoco.viewer
import numpy as np

MODEL_PATH = pathlib.Path(__file__).resolve().parent.parent / "models" / "fighter3d.xml"


def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0, 0, 1.0]
        viewer.cam.distance = 5.5
        viewer.cam.azimuth = -110
        viewer.cam.elevation = -15

        start = time.time()
        while viewer.is_running():
            t = time.time() - start
            ax = data.xpos[model.body("a_torso").id][0]
            bx = data.xpos[model.body("b_torso").id][0]

            data.ctrl[model.actuator("a_root_x").id] = np.clip((bx - ax) * 1.5, -1, 1)
            data.ctrl[model.actuator("b_root_x").id] = np.clip((ax - bx) * 1.5, -1, 1)
            data.ctrl[model.actuator("a_root_y").id] = 0.15 * np.sin(0.7 * t)
            data.ctrl[model.actuator("b_root_y").id] = 0.15 * np.sin(0.7 * t + np.pi)

            punch = 0.6 * max(0.0, np.sin(2.0 * t))
            data.ctrl[model.actuator("a_shoulder_r").id] = punch
            data.ctrl[model.actuator("a_elbow_r").id] = -punch
            data.ctrl[model.actuator("b_shoulder_r").id] = -punch
            data.ctrl[model.actuator("b_elbow_r").id] = -punch

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
