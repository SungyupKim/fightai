# FightAI — Technical Summary (condensed, English)

*Full detailed history (Korean) lives in [`fightai_기술문서.md`](fightai_기술문서.md). This is a
shorter English companion covering the architecture and the key findings, not every iteration.*

## 1. What this is

A 2D ragdoll fighter simulated in MuJoCo, trained with PPO (Stable-Baselines3). The original goal
was a self-play combat league; partway through, a harder question surfaced — can this body learn
to *stand*, from an arbitrary starting posture, at all? — and the project pivoted to answer that
rigorously before returning to combat.

- Physics: MuJoCo, contact forces via `mj_contactForce`, 2D sagittal-plane ragdoll (two mirrored
  fighters, `a_`/`b_` prefixed bodies)
- RL: Gymnasium custom env + SB3 PPO, `SubprocVecEnv` with 8 parallel workers
- Combat mode: self-play with periodically-frozen opponent snapshots, mirrored observations
- Stand-only mode (`LocomotionEnv(stand_only=True)`): no opponent, learn balance/recovery alone

## 2. Methodology (the part that mattered most)

The single biggest lesson across this project: **training-log aggregates (`fall_rate`,
`ep_len_mean`, rolling-window means) are not trustworthy on their own.** Measured repeatedly that a
training-log metric can look like steady improvement while the actual behavior of interest is flat
at zero. The fix, applied consistently from here on: a **fixed-seed, deterministic-action gate** —
reset with seeds `0..N-1`, run the current policy with `deterministic=True`, and measure the thing
you actually care about directly (e.g. "does the torso stay above a height threshold for 3
continuous seconds"). Every claimed result in this project is backed by that kind of direct
measurement, not a log line.

Related habits that came out of repeated mistakes:
- **Audit every legacy reward term when entering a new mode.** Two separate bugs (a "stance
  width" reward and a "getting up" reward) kept paying out during stand-only training because
  they were designed for combat and never explicitly disabled for the new mode — same root cause
  twice.
- **Pure reward-constant changes should continue from a checkpoint (`--init-from`), not restart
  from scratch.** Body/XML changes and observation-space changes do require scratch retraining;
  reward-coefficient tweaks don't.
- **Curriculum transitions should be performance-gated, not time-gated.** A fixed-step schedule
  can't tell whether the easy phase actually finished before harder cases get mixed in.

## 3. The stand-only investigation, condensed

**Can the body physically stand?** Computed `qfrc_bias` (the torque needed to hold a pose
statically) at a hand-set fully-upright configuration: each joint needs only **1–5% of its max
torque** to hold a straight-legged stand. Actuator strength is not the bottleneck.

**Then why does it keep crouching?** A hand-coded capture-point PD controller (no RL) was used to
separate "body can't do it" from "policy hasn't learned it." Scanning actuator gear with that
controller (not the reward formula) found the real lever: tripling ankle gear and doubling hip
gear raised survival from 3% to 17%, while boosting hip alone made it *worse*. This matched a
biomechanics audit: this body's ankle:hip torque ratio was ~1:6.4 vs. a human's ~1:2 — the ankle
was under-strengthed for the fine balance corrections it's supposed to provide. Gear was retuned
(hip 90→180, ankle 14→40) to match.

**Body proportions were also off.** Measuring actual body-part masses against Winter's
anthropometric tables: head was 12.3% of body mass (human ~8%), one leg was 23.5% (human ~16%),
trunk was 33.6% (human ~50%). Fixed via density adjustments on existing geometry (head/foot/leg
density down, trunk density up) rather than resizing capsules, landing within ~1pp of the human
reference ratios at the same total body mass. The hip joint's range of motion was also found to be
symmetric (−105°..105°) when every other joint sharing the same axis convention had already been
given an anatomically asymmetric range (hips can flex to ~120° but only extend ~10–20° past
neutral in reality) — fixed to −20°..120°.

**Two reward leaks, same shape.** A "stance width" reward (pays for spreading the feet, meant to
keep a fighter from being easily swept in combat) was never disabled for stand-only and was
rewarding leg-splitting for free. A "recovery" reward (meant to pay for genuinely getting up off
the ground) was gated against a near-fully-upright reference height, so it stayed ~25% open even
in a relaxed, already-passing stance — rewarding small vertical bobbing ("tap-dancing") instead of
holding still. Both were gated off above the fall-height threshold once found.

**A reward for the thing the gate actually measures.** Training-log `fall_rate` kept improving
while the real 3-second gate stayed at 0% because the knockdown-recovery window (5 seconds) made
"fall, recover, repeat" nearly free. Added `stand_streak`: reward scales with consecutive steps
spent above the fall-height threshold (saturating at 150 steps = 3s), resetting hard to zero on any
dip — directly incentivizing the thing being measured, not a proxy for it.

**A missing observation feature.** Even after all of the above, training plateaued for 20M+ steps.
The hand-coded controller's edge was that it was handed the centre-of-mass position/velocity
relative to the feet directly; the RL policy only had raw joint angles/velocities and had to
rederive that relationship (forward kinematics) from scratch, inside episodes that often ended in
well under a second. Added the capture-point-style features (`com_rel_x`, `com_rel_z`,
`com_vel_x`) directly to the observation (66 → 70 dims).

**A curriculum that asked for two skills at once.** "Hold a stand" and "get up from a collapse"
are different skills; the existing calm/harsh episode-start mix (tuned for the combat league) had
never been re-examined for the stand-only goal. Split into a gated two-stage curriculum: phase 1
trains calm (near-standing) starts exclusively until `fall_rate ≤ 0.1` and the longest continuous
stand streak reaches `≥500` steps (10s) over 200 episodes; only then does phase 2 phase in
harsh/collapse starts.

**A knee that wouldn't straighten, twice.** Watching trained checkpoints directly: both knees sat
pinned near their flexion limit even in a genuinely stable (near-zero-velocity) equilibrium — the
existing "brace" reward for straightening knees only fired during an active fast fall, not while
already resting low. Added an always-on knee-straightness reward. First version averaged both
knees, which a sharp catch (from direct observation, not a metric) revealed was being gamed by
straightening just one leg while leaving the other fully bent — fixed to use the minimum of the two
knees, matching a fix an existing reward term (`knee_avoid`) had already made for the identical
problem.

**Discovering a stale override.** While raising the head-height reward scale, printing the value
before vs. after the intended change caught that a much older experiment had left a module-level
override in place (ambient baseline was actually 6.0, not env.py's bare default of 3.0) — the
intended "3.0→5.0" edit would have been a silent regression. Caught by verifying the actual runtime
value instead of trusting a stale comment.

**First real success.** With the height-reward scale raised and ramped as a proper scaffold
(rather than a one-off manual edit each time), a fixed-seed re-measurement with 100 seeds showed
the deterministic policy genuinely achieving the 3-second survival bar — **1/100**, the first
non-zero result across the entire stand-only effort. Still far from solved, but the central
open question — "is this even achievable?" — now has a direct, measured "yes."

## 4. Where it stands now

- Standing from an arbitrary posture, held for 3 full seconds, deterministic policy, fixed seeds:
  **~1%** and climbing, up from a flat 0% across roughly twenty prior iterations.
- Physical capability is confirmed sufficient; remaining work is about reward/curriculum shaping
  finding the easy, low-torque equilibrium reliably rather than settling for a low, stable crouch.
- Walking and the combat league are intentionally on hold until standing is solid — the project's
  own working hypothesis is that sequencing "stand" → "walk" → "fight" as separate, gated curriculum
  stages will go faster than tackling them together, for the same reason splitting "hold a stand"
  from "get up from a collapse" did.
