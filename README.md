# fightai

A 2D MuJoCo ragdoll fighter trained end-to-end with PPO — self-play combat, a from-any-posture
standing/recovery curriculum, and a diagnostic toolchain built specifically to catch the ways
reinforcement learning quietly lies to you.

설계 배경과 전체 실험 히스토리(한국어)는 [`docs/fightai_기술문서.md`](docs/fightai_기술문서.md),
간단한 영문 요약은 [`docs/TECHNICAL_SUMMARY.md`](docs/TECHNICAL_SUMMARY.md) 참고.

## What's here

- A self-play combat league: two fighters train against periodically-frozen snapshots of each
  other, with physically-grounded contact-force damage, stagger, and knockdown systems.
- A **stand-only track**: strip away everything but the question "can this body learn to stand up
  and hold it, from any starting posture?" — and answer it with the same rigor a robotics lab
  would, not with reward-curve vibes.
- A diagnostic methodology that's arguably the real deliverable: fixed-seed deterministic gates,
  direct physics measurements (`qfrc_bias` torque checks, hand-coded controllers used purely to
  separate "body can't" from "policy hasn't learned"), and a standing habit of auditing every
  legacy reward term before trusting a new mode's numbers.

## Current status

The stand-only effort went through roughly twenty iterations before the fixed-seed 3-second
survival gate moved off a flat **0%** — not because any single fix was wrong, but because getting
a biped to balance from scratch via RL is a genuinely hard control problem, and most of the
apparent progress along the way was training-log metrics that looked good while the real target
stayed at zero (see §2–3 below). The latest checkpoint clears the bar on **~1%** of fixed-seed
trials, deterministic policy, no cherry-picking — the first confirmed non-zero result in this
track. Physical capability was never the blocker: a direct torque measurement shows holding a
full standing posture costs only 1–5% of each joint's maximum output. The rest is reward and
curriculum design, which is still being actively tightened.

## Lessons (the short version)

- **Training-log numbers are not ground truth.** `fall_rate`, `ep_len_mean`, rolling-window means
  — all of them can improve steadily while the thing you actually care about is flat. Every
  reported result here comes from a fixed-seed, deterministic re-measurement, not a log line.
- **A reward term designed for one mode doesn't turn itself off in another.** Two separate bugs
  (a leg-spreading reward, a "getting up" reward) kept paying out during stand-only training
  because they were built for combat and nobody gated them off for the new mode — same shape of
  mistake, twice.
- **Reward the thing the evaluation actually measures.** `fall_rate` improving while a fixed-seed
  "stay balanced for 3 continuous seconds" gate stays at 0% is the single most repeated pattern in
  this project's history. The fix each time was making the reward pay for exactly the eval
  criterion, not a proxy for it.
- **Separate "can't" from "hasn't learned."** Hand-coded controllers (no RL) and direct inverse-
  dynamics torque checks were used repeatedly to pin down whether a limitation was physical
  (actuator strength, body proportions) or a training/reward-shaping problem — they're very
  different fixes.
- **Pure reward-constant changes warm-start; structural changes don't.** Changing a reward
  coefficient doesn't require retraining from scratch — continuing from a checkpoint
  (`--init-from`) preserves tens of millions of steps of learned skill. Body/XML changes and
  observation-space changes do need a fresh run.

## Setup

```bash
python -m venv .venv
.venv/bin/pip install mujoco gymnasium stable-baselines3 tensorboard
```

`watch.py` / `watch_locomotion.py` (viewers) need a machine with an OpenGL context. Training
itself runs headless.

## Project layout

```
models/fighter2d.xml          Generated MJCF model for both fighters (build_model.py builds it)
scripts/build_model.py        Model generator -- edit this, not the XML, to change the body
scripts/env.py                Fighter2DEnv: combat environment (reward, physics, self-play mirroring)
scripts/locomotion_env.py     LocomotionEnv: solo stand/recover environment (stand_only mode)
scripts/train_locomotion.py   PPO training for the stand-only/locomotion track
scripts/train_league.py       One combat-league round: trains one side against a frozen opponent
scripts/selfplay_league_loop.sh   Alternates league rounds between the two sides automatically
scripts/transplant_locomotion.py  Ports a locomotion policy into the combat action/observation space
scripts/gate_eval.py          Fixed-seed, deterministic combat evaluation (falls, knockdowns, wins)
scripts/stand_progress_notify.py  Periodic KakaoTalk progress pings with a real fixed-seed gate
scripts/watch.py               Combat policy viewer (health bars, hit markers)
scripts/watch_locomotion.py    Stand-only policy viewer
checkpoints/                   Trained models, autosaves, logs (gitignored)
docs/                          Technical documentation (Korean, full history + English summary)
```

## Running things

**Stand-only training (learn to stand from any posture):**
```bash
cd scripts
../.venv/bin/python train_locomotion.py --n-envs 8 --timesteps 80000000 --stand-only --out my_stand
```

**Combat league (alternating sides):**
```bash
cd scripts
./selfplay_league_loop.sh [rounds=10] [timesteps_per_round=1000000] p1_init.zip p2_init.zip
```

**Fixed-seed evaluation (compare two checkpoints under identical conditions):**
```bash
cd scripts
../.venv/bin/python gate_eval.py ../checkpoints/a.zip ../checkpoints/opponent.zip --episodes 120
```

**Watching a trained policy:**
```bash
cd scripts
DISPLAY=:0 ../.venv/bin/python watch.py ../checkpoints/my_run.zip [--opponent ../checkpoints/other.zip]
DISPLAY=:0 ../.venv/bin/python watch_locomotion.py ../checkpoints/my_stand.zip
```

Training scripts autosave periodically to `checkpoints/autosave/`, so a crash can resume from the
last checkpoint instead of from scratch.

## Diagnostic script pattern

Load a checkpoint, run a fixed number of episodes with a fixed seed range, and sample `env.data`/
`info` directly for whatever's in question (contact state, joint angles, torque, knockdown-step
fractions). Measure behavior the same way before and after any change — a change that only looks
good in the aggregate reward (a past mistake here, `EFFORT_COST`) is not a change that's actually
working.

## Roadmap

Standing (from any posture, held reliably) → walking (reintroducing velocity tracking, staged by
target speed) → combat league, in that order — each stage gated on the previous one actually
working, not on a fixed step count.
