"""Generates a 2D (XZ-plane) ragdoll fighter MJCF model with two characters facing off."""

FIGHTER_TEMPLATE = """
    <body name="{p}torso" pos="{x} 0 1.06">
      <joint name="{p}root_x" type="slide" axis="1 0 0" limited="false" damping="0.5"/>
      <joint name="{p}root_z" type="slide" axis="0 0 1" limited="false" damping="0.5"/>
      <joint name="{p}root_ry" type="hinge" axis="0 1 0" limited="false" damping="0.5"/>
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
        <!-- damping raised 2->3 alongside the hip/knee gear bump (60->90/40->60, see
             ACTUATOR_GEAR) -- gear alone made falls worse (68%->86% over 10 league rounds),
             stronger torque with unchanged damping likely made motion twitchier/harder to
             control rather than more stable. Scaling damping by the same 1.5x tests whether
             that's actually the cause. -->
        <joint name="{p}hip_r" type="hinge" axis="0 1 0" range="-105 105" damping="3" armature="0.02"/>
        <geom name="{p}thigh_r" type="capsule" fromto="0 0 0 0 0 -0.42" size="0.07" rgba="{color}"/>
        <body name="{p}shin_r" pos="0 0 -0.42">
          <joint name="{p}knee_r" type="hinge" axis="0 1 0" range="-140 0" damping="3" armature="0.02"/>
          <geom name="{p}shin_r" type="capsule" fromto="0 0 0 0 0 -0.42" size="0.055" rgba="{color}"/>
          <!-- foot: tried a flat box sole for a genuine contact patch instead of a rolling line
               contact, but it measurably made falls worse (68%->90% over ~2100 episodes) -- a
               box has corners, and a corner catching the ground at even a slight foot angle
               creates a sudden torque spike a capsule's smooth curve never would (same reason
               robots use rounded feet, not flat-edged ones). Reverted to a capsule but fattened
               the radius 2x (0.035->0.07) for a bigger, more forgiving contact tolerance while
               keeping the smooth rolling contact. -->
          <geom name="{p}foot_r" type="capsule" fromto="-0.05 0 -0.42 0.12 0 -0.42" size="0.07" rgba="{color}"/>
          <site name="{p}foot_r" pos="0 0 -0.42" size="0.02"/>
        </body>
      </body>

      <body name="{p}thigh_l" pos="0 0.09 -0.22">
        <joint name="{p}hip_l" type="hinge" axis="0 1 0" range="-105 105" damping="3" armature="0.02"/>
        <geom name="{p}thigh_l" type="capsule" fromto="0 0 0 0 0 -0.42" size="0.07" rgba="{color}"/>
        <body name="{p}shin_l" pos="0 0 -0.42">
          <joint name="{p}knee_l" type="hinge" axis="0 1 0" range="-140 0" damping="3" armature="0.02"/>
          <geom name="{p}shin_l" type="capsule" fromto="0 0 0 0 0 -0.42" size="0.055" rgba="{color}"/>
          <geom name="{p}foot_l" type="capsule" fromto="-0.05 0 -0.42 0.12 0 -0.42" size="0.07" rgba="{color}"/>
          <site name="{p}foot_l" pos="0 0 -0.42" size="0.02"/>
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
    # hip/knee raised 60->90 / 40->60 (50%): fattening the foot capsule fixed the "sudden
    # ground-support loss" fall mode (68%->44%), but a follow-up measurement showed a second
    # mode remains -- knees now visibly buckle (bend further) during the ~20 steps before a
    # fall, instead of staying static. Testing whether more leg torque lets the policy actually
    # arrest a stumble instead of getting overpowered by torso weight/combat impacts.
    "hip_r": 90, "knee_r": 60, "hip_l": 90, "knee_l": 60,
}

ROOT_GEAR = 90
# root_ry is not part of the RL action space -- env.py drives it with a small always-on
# PD balance assist (like a person's reflexes), since a standing biped is an inverted
# pendulum that plain passive damping/stiffness can't stabilize (verified: even stiffness=200
# still toppled from a small 0.4 rad/s nudge). A strong hit can still overwhelm the assist.
BALANCE_GEAR = 250


def make_fighter(prefix, x, color):
    return FIGHTER_TEMPLATE.format(p=prefix, x=x, color=color)


def make_actuators(prefix):
    lines = [
        f'    <motor name="{prefix}root_x" joint="{prefix}root_x" gear="{ROOT_GEAR}" ctrlrange="-1 1"/>',
        f'    <motor name="{prefix}root_ry" joint="{prefix}root_ry" gear="{BALANCE_GEAR}" ctrlrange="-1 1"/>',
    ]
    for j in JOINTS:
        lines.append(
            f'    <motor name="{prefix}{j}" joint="{prefix}{j}" '
            f'gear="{ACTUATOR_GEAR[j]}" ctrlrange="-1 1"/>'
        )
    return "\n".join(lines)


def build():
    # Retested -0.9/0.9 (1.8) again after fixing the zero-sum balance_penalty bug --
    # still only 6.7% engagement (worse than the pre-fix 10%), so the far-spawn problem
    # is real and independent of that bug. -0.6/0.6 (1.2) is confirmed the right call.
    fighters = make_fighter("a_", -0.6, "0.85 0.2 0.2 1") + make_fighter("b_", 0.6, "0.2 0.4 0.85 1")
    actuators = make_actuators("a_") + "\n" + make_actuators("b_")

    return f"""<mujoco model="fighter2d">
  <compiler angle="degree"/>
  <option gravity="0 0 -9.81" timestep="0.005"/>

  <visual>
    <headlight ambient="0.4 0.4 0.4"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.6 0.7 0.9" rgb2="0.1 0.1 0.2" width="256" height="256"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.3 0.3 0.3" rgb2="0.4 0.4 0.4" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="6 6" reflectance="0.1"/>
  </asset>

  <worldbody>
    <light pos="0 -2 3" dir="0 0.5 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="5 3 0.1" material="grid" friction="1.0 0.01 0.01"/>
    <camera name="side" pos="0 -4.5 1.1" xyaxes="1 0 0 0 0 1"/>
{fighters}
  </worldbody>

  <actuator>
{actuators}
  </actuator>
</mujoco>
"""


if __name__ == "__main__":
    import pathlib
    out = pathlib.Path(__file__).resolve().parent.parent / "models" / "fighter2d.xml"
    out.write_text(build())
    print(f"wrote {out}")
