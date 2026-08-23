"""Generates a 2D (XZ-plane) ragdoll fighter MJCF model with two characters facing off."""

FIGHTER_TEMPLATE = """
    <body name="{p}torso" pos="{x} 0 1.06">
      <joint name="{p}root_x" type="slide" axis="1 0 0" limited="false" damping="0.5"/>
      <joint name="{p}root_z" type="slide" axis="0 0 1" limited="false" damping="0.5"/>
      <joint name="{p}root_ry" type="hinge" axis="0 {ay} 0" limited="false" damping="0.5"/>
      <geom name="{p}pelvis" type="capsule" fromto="0 0 -0.22 0 0 0" size="0.10" rgba="{color}"/>

      <!-- waist: previously the whole torso (pelvis to shoulders) was one rigid capsule, so
           root_ry (hip-level lean, always-on PD balance assist, not RL-controlled) was the
           only way the WHOLE body could pitch -- no way to lean/twist the upper body
           independently while keeping the legs/pelvis planted, the way a real punch or dodge
           does. Splits the old single torso geom into pelvis (root, legs attach here) + chest
           (child, head/arms attach here) with an RL-actuated hinge between them. Biased toward
           forward flexion (-20 back, 45 forward) like a real spine. -->
      <body name="{p}chest" pos="0 0 0">
        <joint name="{p}waist" type="hinge" axis="0 {ay} 0" range="-20 45" damping="2" armature="0.02"/>
        <geom name="{p}chest" type="capsule" fromto="0 0 0 0 0 0.22" size="0.10" rgba="{color}"/>

        <body name="{p}head" pos="0 0 0.22">
          <joint name="{p}head" type="hinge" axis="0 {ay} 0" range="-40 40" damping="1" armature="0.01"/>
          <geom name="{p}head" type="sphere" size="0.13" pos="0 0 0.13" rgba="{color}"/>
        </body>

        <body name="{p}upper_arm_r" pos="0 -0.07 0.20">
          <joint name="{p}shoulder_r" type="hinge" axis="0 {ay} 0" range="-160 160" damping="1" armature="0.01"/>
          <geom name="{p}upper_arm_r" type="capsule" fromto="0 0 0 0 0 -0.28" size="0.045" rgba="{color}"/>
          <body name="{p}forearm_r" pos="0 0 -0.28">
            <joint name="{p}elbow_r" type="hinge" axis="0 {ay} 0" range="-150 0" damping="1" armature="0.01"/>
            <geom name="{p}forearm_r" type="capsule" fromto="0 0 0 0 0 -0.25" size="0.04" rgba="{color}"/>
          </body>
        </body>

        <body name="{p}upper_arm_l" pos="0 0.07 0.20">
          <joint name="{p}shoulder_l" type="hinge" axis="0 {ay} 0" range="-160 160" damping="1" armature="0.01"/>
          <geom name="{p}upper_arm_l" type="capsule" fromto="0 0 0 0 0 -0.28" size="0.045" rgba="{color}"/>
          <body name="{p}forearm_l" pos="0 0 -0.28">
            <joint name="{p}elbow_l" type="hinge" axis="0 {ay} 0" range="-150 0" damping="1" armature="0.01"/>
            <geom name="{p}forearm_l" type="capsule" fromto="0 0 0 0 0 -0.25" size="0.04" rgba="{color}"/>
          </body>
        </body>
      </body>

      <body name="{p}thigh_r" pos="0 -0.09 -0.22">
        <!-- damping raised 2->3 alongside the hip/knee gear bump (60->90/40->60, see
             ACTUATOR_GEAR) -- gear alone made falls worse (68%->86% over 10 league rounds),
             stronger torque with unchanged damping likely made motion twitchier/harder to
             control rather than more stable. Scaling damping by the same 1.5x tests whether
             that's actually the cause. -->
        <joint name="{p}hip_r" type="hinge" axis="0 {ay} 0" range="-105 105" damping="3" armature="0.02"/>
        <geom name="{p}thigh_r" type="capsule" fromto="0 0 0 0 0 -0.42" size="0.07" rgba="{color}"/>
        <body name="{p}shin_r" pos="0 0 -0.42">
          <joint name="{p}knee_r" type="hinge" axis="0 {ay} 0" range="-140 0" damping="3" armature="0.02"/>
          <geom name="{p}shin_r" type="capsule" fromto="0 0 0 0 0 -0.42" size="0.055" rgba="{color}"/>
          <!-- ankle: previously the foot was a geom rigidly welded to the shin, so a real
               human's main tool for correcting fore/aft sway (shifting ground-reaction-force
               under the foot via ankle torque) didn't exist -- the policy could only react to
               an in-progress fall through the knee/hip, once it was already substantial. Added
               as an RL-actuated hinge (not passive like the toe) since active, timely correction
               is the whole point. Range kept small and SYMMETRIC (-30 to 30) rather than trying
               to guess the anatomically-correct biased range up front -- an asymmetric range
               burned real time twice already on the knee (docs: mirroring broke on asymmetric
               ranges, then the bend direction itself was backwards) and a symmetric range works
               regardless of which way {ay} ends up pointing, so there's no third direction bug
               to chase; the policy can just learn whichever sign is useful within it. -->
          <body name="{p}foot_r" pos="0 0 -0.42">
            <joint name="{p}ankle_r" type="hinge" axis="0 {ay} 0" range="-30 30" damping="1.5" armature="0.02"/>
            <!-- foot: tried a flat box sole for a genuine contact patch instead of a rolling line
                 contact, but it measurably made falls worse (68%->90% over ~2100 episodes) -- a
                 box has corners, and a corner catching the ground at even a slight foot angle
                 creates a sudden torque spike a capsule's smooth curve never would (same reason
                 robots use rounded feet, not flat-edged ones). Reverted to a capsule but fattened
                 the radius 2x (0.035->0.07) for a bigger, more forgiving contact tolerance while
                 keeping the smooth rolling contact. -->
            <!-- heel/toe x-offsets are signed by {{facing}} (+1 for a, -1 for b, see make_fighter)
                 -- a spawns facing +x (toward b) and b spawns facing -x (toward a), but this
                 template is shared verbatim between both, so an unsigned heel-behind/toe-ahead
                 layout would have the toe pointing away from the opponent for whichever fighter
                 faces -x. Caught when the toe joint made a backwards-pointing toe visually
                 obvious, but the same asymmetry was already latent in the foot capsule itself. -->
            <geom name="{p}foot_r" type="capsule" fromto="{heel_x} 0 0 {toe_x} 0 0" size="0.07" rgba="{color}"/>
            <site name="{p}foot_r" pos="0 0 0" size="0.02"/>
            <!-- toe: passive (no actuator, no RL action dim) hinge with a spring (stiffness)
                 pulling it back to neutral, like a real toe's passive compliance during push-off
                 and weight shift -- not another thing the policy has to learn to drive. Range
                 biased toward plantarflexion (curling down, -10 to 45) over dorsiflexion, like a
                 real foot. -->
            <body name="{p}toe_r" pos="{toe_x} 0 0">
              <joint name="{p}toe_r" type="hinge" axis="0 {ay} 0" range="-10 45" stiffness="8" damping="0.5" armature="0.01"/>
              <geom name="{p}toe_r" type="capsule" fromto="0 0 0 {toe_tip_x} 0 0" size="0.045" rgba="{color}"/>
            </body>
          </body>
        </body>
      </body>

      <body name="{p}thigh_l" pos="0 0.09 -0.22">
        <joint name="{p}hip_l" type="hinge" axis="0 {ay} 0" range="-105 105" damping="3" armature="0.02"/>
        <geom name="{p}thigh_l" type="capsule" fromto="0 0 0 0 0 -0.42" size="0.07" rgba="{color}"/>
        <body name="{p}shin_l" pos="0 0 -0.42">
          <joint name="{p}knee_l" type="hinge" axis="0 {ay} 0" range="-140 0" damping="3" armature="0.02"/>
          <geom name="{p}shin_l" type="capsule" fromto="0 0 0 0 0 -0.42" size="0.055" rgba="{color}"/>
          <body name="{p}foot_l" pos="0 0 -0.42">
            <joint name="{p}ankle_l" type="hinge" axis="0 {ay} 0" range="-30 30" damping="1.5" armature="0.02"/>
            <geom name="{p}foot_l" type="capsule" fromto="{heel_x} 0 0 {toe_x} 0 0" size="0.07" rgba="{color}"/>
            <site name="{p}foot_l" pos="0 0 0" size="0.02"/>
            <body name="{p}toe_l" pos="{toe_x} 0 0">
              <joint name="{p}toe_l" type="hinge" axis="0 {ay} 0" range="-10 45" stiffness="8" damping="0.5" armature="0.01"/>
              <geom name="{p}toe_l" type="capsule" fromto="0 0 0 {toe_tip_x} 0 0" size="0.045" rgba="{color}"/>
            </body>
          </body>
        </body>
      </body>
    </body>
"""

JOINTS = [
    "waist", "head", "shoulder_r", "elbow_r", "shoulder_l", "elbow_l",
    "hip_r", "knee_r", "ankle_r", "hip_l", "knee_l", "ankle_l",
]

ACTUATOR_GEAR = {
    # waist is new -- lets the policy lean/twist the upper body independently of root_ry
    # (which only ever moves the whole body at once, and isn't even RL-controlled). Gear
    # between shoulder and hip: moves more mass than an arm but less than the whole leg chain.
    "waist": 70,
    "head": 20, "shoulder_r": 45, "elbow_r": 30, "shoulder_l": 45, "elbow_l": 30,
    # hip/knee raised 60->90 / 40->60 (50%): fattening the foot capsule fixed the "sudden
    # ground-support loss" fall mode (68%->44%), but a follow-up measurement showed a second
    # mode remains -- knees now visibly buckle (bend further) during the ~20 steps before a
    # fall, instead of staying static. Testing whether more leg torque lets the policy actually
    # arrest a stumble instead of getting overpowered by torso weight/combat impacts.
    "hip_r": 90, "knee_r": 60, "hip_l": 90, "knee_l": 60,
    # ankle: small, fast corrective torque (fore/aft ground-reaction-force shifting), not gross
    # locomotion force -- gear kept modest, between an elbow and a shoulder.
    "ankle_r": 35, "ankle_l": 35,
}

ROOT_GEAR = 90
# root_ry is not part of the RL action space -- env.py drives it with a small always-on
# PD balance assist (like a person's reflexes), since a standing biped is an inverted
# pendulum that plain passive damping/stiffness can't stabilize (verified: even stiffness=200
# still toppled from a small 0.4 rad/s nudge). A strong hit can still overwhelm the assist.
BALANCE_GEAR = 250


def make_fighter(prefix, x, color, facing=1):
    # ay sets each rotation joint's axis sign. First version used ay=facing (only flipped for
    # b, so a's knee bend swung its foot toward +x -- ITS OWN forward/opponent direction).
    # That made b a true mirror of a, confirmed by direct measurement, but a real human knee
    # bends the OTHER way: standing and bending your knee swings the shin BACKWARD (behind
    # you), not forward into a front kick -- a front kick is hip flexion carrying an
    # already-bent knee forward, then the knee EXTENDING at the end, not the knee bending
    # deeper. So the original convention had both fighters' knees (and every other joint
    # sharing the same axis pattern -- waist, elbows, hips, toes) bending anatomically
    # backwards. ay=-facing flips both a and b from where they were: a's knee now swings
    # toward -x (behind it) when bending, b's swings toward +x (behind it) -- each fighter's
    # own "behind," matching how a real standing knee bends, while a and b remain mirror
    # images of each other (still true -- only the shared sign got flipped, not the
    # mirror-consistency between them).
    ay = -facing
    return FIGHTER_TEMPLATE.format(
        p=prefix, x=x, color=color, ay=ay,
        heel_x=-0.05 * facing, toe_x=0.12 * facing, toe_tip_x=0.14 * facing,
    )


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
    # a spawns on the left facing +x (toward b); b spawns on the right facing -x (toward a) --
    # facing flips the heel/toe x-asymmetry in FIGHTER_TEMPLATE so each fighter's foot/toe
    # actually points at their opponent instead of both defaulting to +x-forward.
    fighters = make_fighter("a_", -0.6, "0.85 0.2 0.2 1", facing=1) + \
        make_fighter("b_", 0.6, "0.2 0.4 0.85 1", facing=-1)
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
