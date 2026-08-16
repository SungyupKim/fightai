"""Generates a full-3D ragdoll fighter MJCF model with two characters facing off.

Root translation (x, y) is directly actuated, same philosophy as the 2D model --
skip learning locomotion from scratch. Height (z) and all orientation (a ball
joint) are left fully passive/physical, so balance and knockdowns stay a real
physics outcome rather than something scripted.
"""

FIGHTER_TEMPLATE = """
    <body name="{p}torso" pos="{x} {y} 1.06">
      <joint name="{p}root_x" type="slide" axis="1 0 0" limited="false" damping="0.5"/>
      <joint name="{p}root_y" type="slide" axis="0 1 0" limited="false" damping="0.5"/>
      <joint name="{p}root_z" type="slide" axis="0 0 1" limited="false" damping="0.5"/>
      <joint name="{p}root_orient" type="ball" damping="0.5"/>
      <geom name="{p}torso" type="capsule" fromto="0 0 -0.22 0 0 0.22" size="0.10" rgba="{color}"/>

      <body name="{p}head" pos="0 0 0.22">
        <joint name="{p}head" type="hinge" axis="0 1 0" range="-40 40" damping="1" armature="0.01"/>
        <geom name="{p}head" type="sphere" size="0.13" pos="0 0 0.13" rgba="{color}"/>
      </body>

      <body name="{p}upper_arm_r" pos="0 -0.07 0.20">
        <joint name="{p}shoulder_r" type="hinge" axis="0 1 0" range="-160 160" damping="1" armature="0.01"/>
        <geom name="{p}upper_arm_r" type="capsule" fromto="0 0 0 0 0 -0.28" size="0.045" rgba="{color}"/>
        <body name="{p}forearm_r" pos="0 0 -0.28">
          <joint name="{p}elbow_r" type="hinge" axis="0 1 0" range="-150 0" damping="1" armature="0.01"/>
          <geom name="{p}forearm_r" type="capsule" fromto="0 0 0 0 0 -0.25" size="0.04" rgba="{color}"/>
        </body>
      </body>

      <body name="{p}upper_arm_l" pos="0 0.07 0.20">
        <joint name="{p}shoulder_l" type="hinge" axis="0 1 0" range="-160 160" damping="1" armature="0.01"/>
        <geom name="{p}upper_arm_l" type="capsule" fromto="0 0 0 0 0 -0.28" size="0.045" rgba="{color}"/>
        <body name="{p}forearm_l" pos="0 0 -0.28">
          <joint name="{p}elbow_l" type="hinge" axis="0 1 0" range="-150 0" damping="1" armature="0.01"/>
          <geom name="{p}forearm_l" type="capsule" fromto="0 0 0 0 0 -0.25" size="0.04" rgba="{color}"/>
        </body>
      </body>

      <body name="{p}thigh_r" pos="0 -0.09 -0.22">
        <joint name="{p}hip_r" type="hinge" axis="0 1 0" range="-120 90" damping="2" armature="0.02"/>
        <geom name="{p}thigh_r" type="capsule" fromto="0 0 0 0 0 -0.42" size="0.07" rgba="{color}"/>
        <body name="{p}shin_r" pos="0 0 -0.42">
          <joint name="{p}knee_r" type="hinge" axis="0 1 0" range="-140 0" damping="2" armature="0.02"/>
          <geom name="{p}shin_r" type="capsule" fromto="0 0 0 0 0 -0.42" size="0.055" rgba="{color}"/>
        </body>
      </body>

      <body name="{p}thigh_l" pos="0 0.09 -0.22">
        <joint name="{p}hip_l" type="hinge" axis="0 1 0" range="-120 90" damping="2" armature="0.02"/>
        <geom name="{p}thigh_l" type="capsule" fromto="0 0 0 0 0 -0.42" size="0.07" rgba="{color}"/>
        <body name="{p}shin_l" pos="0 0 -0.42">
          <joint name="{p}knee_l" type="hinge" axis="0 1 0" range="-140 0" damping="2" armature="0.02"/>
          <geom name="{p}shin_l" type="capsule" fromto="0 0 0 0 0 -0.42" size="0.055" rgba="{color}"/>
        </body>
      </body>
    </body>
"""

JOINTS = [
    "head", "shoulder_r", "elbow_r", "shoulder_l", "elbow_l",
    "hip_r", "knee_r", "hip_l", "knee_l",
]

ACTUATOR_GEAR = {
    "head": 20, "shoulder_r": 45, "elbow_r": 30, "shoulder_l": 45, "elbow_l": 30,
    "hip_r": 60, "knee_r": 40, "hip_l": 60, "knee_l": 40,
}

ROOT_GEAR = 90


def make_fighter(prefix, x, y, color):
    return FIGHTER_TEMPLATE.format(p=prefix, x=x, y=y, color=color)


def make_actuators(prefix):
    lines = [
        f'    <motor name="{prefix}root_x" joint="{prefix}root_x" gear="{ROOT_GEAR}" ctrlrange="-1 1"/>',
        f'    <motor name="{prefix}root_y" joint="{prefix}root_y" gear="{ROOT_GEAR}" ctrlrange="-1 1"/>',
    ]
    for j in JOINTS:
        lines.append(
            f'    <motor name="{prefix}{j}" joint="{prefix}{j}" '
            f'gear="{ACTUATOR_GEAR[j]}" ctrlrange="-1 1"/>'
        )
    return "\n".join(lines)


def build():
    fighters = make_fighter("a_", -0.9, 0.0, "0.85 0.2 0.2 1") + make_fighter("b_", 0.9, 0.0, "0.2 0.4 0.85 1")
    actuators = make_actuators("a_") + "\n" + make_actuators("b_")

    return f"""<mujoco model="fighter3d">
  <compiler angle="degree"/>
  <option gravity="0 0 -9.81" timestep="0.005"/>

  <visual>
    <headlight ambient="0.4 0.4 0.4"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.6 0.7 0.9" rgb2="0.1 0.1 0.2" width="256" height="256"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.3 0.3 0.3" rgb2="0.4 0.4 0.4" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.1"/>
  </asset>

  <worldbody>
    <light pos="0 -2 3" dir="0 0.5 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="6 6 0.1" material="grid" friction="1.0 0.01 0.01"/>
    <camera name="side" pos="0 -4.5 1.1" xyaxes="1 0 0 0 0 1"/>
    <camera name="quarter" pos="-3 -3.5 2.2" xyaxes="0.7 -0.7 0 0.3 0.3 0.9"/>
{fighters}
  </worldbody>

  <actuator>
{actuators}
  </actuator>
</mujoco>
"""


if __name__ == "__main__":
    import pathlib
    out = pathlib.Path(__file__).resolve().parent.parent / "models" / "fighter3d.xml"
    out.write_text(build())
    print(f"wrote {out}")
