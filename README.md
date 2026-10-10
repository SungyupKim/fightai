# fightai

A 2D MuJoCo ragdoll fighter trained end-to-end with PPO — self-play combat, and a diagnostic
toolchain built specifically to catch the ways reinforcement learning quietly lies to you.

The from-any-posture standing/recovery track that used to live here has moved to its own repo,
[`standai`](../standai), once it became clear "can this body learn to stand up at all?" was a
rigorous research question in its own right rather than a side quest. This repo keeps the shared
body definition (`build_model.py`, `env.py`) and the combat league; `standai` copies those two
files rather than depending on this repo, so a body change has to be applied by hand in both
places (see standai's README for why a copy was chosen over a shared submodule).

설계 배경과 전체 실험 히스토리(한국어)는 [`docs/fightai_기술문서.md`](docs/fightai_기술문서.md),
간단한 영문 요약은 [`docs/TECHNICAL_SUMMARY.md`](docs/TECHNICAL_SUMMARY.md) 참고 — 둘 다 격투
히스토리와 함께, stand-only 트랙이 분리되기 전까지의 기록을 담고 있음 (현재 진행 중인 stand-only
작업은 `standai` repo 참고).

## What's here

- A self-play combat league: two fighters train against periodically-frozen snapshots of each
  other, with physically-grounded contact-force damage, stagger, and knockdown systems.
- A diagnostic methodology that's arguably the real deliverable: fixed-seed deterministic gates,
  direct physics measurements (`qfrc_bias` torque checks, hand-coded controllers used purely to
  separate "body can't" from "policy hasn't learned"), and a standing habit of auditing every
  legacy reward term before trusting a new mode's numbers.
- `transplant_locomotion.py`: the planned endpoint for `standai`'s policy — ports a standing/
  walking policy into this repo's combat action/observation space once that track is solid.

## Current status

Combat league development is on hold while `standai` works through the standing problem (see that
repo for current numbers). Physical capability was never the blocker there — a direct torque
measurement shows holding a full standing posture costs only 1–5% of each joint's maximum output —
so the rest is reward and curriculum design, happening in `standai` now.

## Setup

```bash
python -m venv .venv
.venv/bin/pip install mujoco gymnasium stable-baselines3 tensorboard
```

`watch.py` (viewer) needs a machine with an OpenGL context (`DISPLAY=:1` on this box). Training
itself runs headless.

## Project layout

```
models/fighter2d.xml          Generated MJCF model for both fighters (build_model.py builds it)
scripts/build_model.py        Model generator -- edit this, not the XML, to change the body
scripts/env.py                Fighter2DEnv: combat environment (reward, physics, self-play mirroring)
scripts/train_league.py       One combat-league round: trains one side against a frozen opponent
scripts/selfplay_league_loop.sh   Alternates league rounds between the two sides automatically
scripts/transplant_locomotion.py  Ports a standai policy into the combat action/observation space
scripts/gate_eval.py          Fixed-seed, deterministic combat evaluation (falls, knockdowns, wins)
scripts/watch.py               Combat policy viewer (health bars, hit markers)
checkpoints/                   Trained models, autosaves, logs (gitignored)
docs/                          Technical documentation (Korean, full history + English summary)
```

(`build_model.py`/`env.py` are also copied into `standai`, which develops against its own copy —
see that repo for the standing track's files: `locomotion_env.py`, `train_locomotion.py`,
`watch_locomotion.py`, `stand_progress_notify.py`.)

## Running things

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
DISPLAY=:1 ../.venv/bin/python watch.py ../checkpoints/my_run.zip [--opponent ../checkpoints/other.zip]
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

Standing (from any posture, held reliably, now developed in `standai`) → walking (reintroducing
velocity tracking, staged by target speed) → transplant the resulting policy into this repo via
`transplant_locomotion.py` → combat league, in that order — each stage gated on the previous one
actually working, not on a fixed step count.
