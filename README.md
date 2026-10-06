# fightai

2D MuJoCo 랙돌 격투 에이전트를 PPO로 학습하는 실험 기록이다. 상대와 번갈아 학습하는 리그(셀프플레이),
임의의 자세에서 다시 서는 것을 배우는 locomotion 학습, 그리고 각 단계의 원인을 수치로 확인하는 진단
스크립트를 함께 담고 있다.

*A 2D MuJoCo ragdoll fighter trained with PPO: league self-play, a standing/recovery curriculum,
and diagnostic scripts that measure each failure mode instead of trusting reward curves.*

설계 배경과 실험 히스토리(성공과 실패 모두)는 [`docs/fightai_기술문서.md`](docs/fightai_기술문서.md) 참고.

## 현재 상태 (솔직하게)

결과는 아직 완성 단계가 아니다. 격투 정책은 상대와 붙어 있는 동안 넘어지는 일이 많다.

- 붙은 순간 넘어짐 비율: 약 54% → 35% (고정 시드, 120판 기준, 접촉 발 간격 보상 + 붙은 시작 리셋 적용 후)
- 서 있는 자세에서 가벼운 흔들림 후 회복 실패: 약 20%, 완전 붕괴 후 회복 실패: 약 70% (locomotion 기준선)
- 서기 전용 학습(`--stand-only`)은 처음부터 다시 돌리는 중이다.

## 배운 것 (요약)

- **리워드를 올리면 행동이 안 바뀔 때가 많다.** 머리 높이 보상을 올렸더니 리워드 로그는 좋아졌는데 실제
  머리 높이 비율은 내려갔다. 벌점을 올린 것도 쓰러진 시간을 줄이지 못했다.
- **스케일이 큰 보상이 학습을 무너뜨릴 수 있다.** 캡 없는 보상 항(걸음걸이 리워드)이 폭주해 낙상률이
  0.96까지 올라간 적이 있다. 보상 항은 모두 상한이 있는 형태로 유지한다.
- **평가는 고정 시드와 결정적 행동으로 한다.** 확률적 행동으로 재면 같은 체크포인트도 낙상률이 15%p
  가까이 흔들렸다. `scripts/gate_eval.py`가 이 방식이다.
- **리워드 곡선만으로 판단하지 않는다.** 리그처럼 상대가 함께 변하는 경우 평균 리워드는 평평하게 유지되기
  쉽다. 승패, 낙상, 자세 측정을 같이 본다.
- **원인을 먼저 측정한다.** 낙상의 대부분은 피해가 없는 접촉 중 발생했고, 넘어진 쪽은 발을 모으고 무릎을
  편 자세였다. 이 진단이 보상 설계보다 먼저 있어야 했다.

## Setup

```bash
python -m venv .venv
.venv/bin/pip install mujoco gymnasium stable-baselines3 tensorboard
```

`watch.py`(뷰어)는 OpenGL 컨텍스트가 있는 머신이 필요하다. 학습 자체는 headless로 동작한다.

## Project layout

```
models/fighter2d.xml          두 파이터 MJCF 모델 (build_model.py가 생성)
scripts/build_model.py        fighter2d.xml 생성기 -- 모델을 고칠 땐 XML이 아니라 이 파일을 수정
scripts/env.py                Fighter2DEnv: 격투 환경 (리워드, 물리, 셀프플레이 미러링, 시작 자세 변화)
scripts/locomotion_env.py     LocomotionEnv: 상대 없이 임의의 자세에서 서고 버티는 환경 (stand_only 옵션)
scripts/train_locomotion.py   locomotion PPO 학습 (--stand-only, --init-from 지원)
scripts/train_league.py       리그 한 라운드: 한쪽 정책을 고정된 상대에 맞춰 학습
scripts/selfplay_league_loop.sh  리그를 P1/P2 번갈아 여러 라운드 자동 반복
scripts/transplant_locomotion.py locomotion 정책(67차원)을 격투 환경(66차원)으로 이식
scripts/gate_eval.py          고정 시드 + 결정적 행동 평가 (낙상, 쓰러진 시간, 머리 높이, 승패)
scripts/watch.py              격투 정책 뷰어 (체력바/피격 표시)
scripts/watch_locomotion.py   locomotion 정책 뷰어
scripts/league_progress_notify.py  리그 라운드마다 카카오톡 알림 (kakao_notify.py 사용)
scripts/league_viewer_loop.sh 라운드가 바뀔 때마다 뷰어를 최신 체크포인트로 다시 띄움
scripts/train.py, train_selfplay.py, selfplay_loop.sh  초기 단일 상대 / 셀프플레이 학습
scripts/dashboard.py          로컬 웹 대시보드 (학습/뷰어 제어 + 그래프)
scripts/diag_kick_cap.py      접촉력 기반 데미지가 캡에 걸리는 비율을 측정하는 진단 스크립트
checkpoints/                  학습된 모델, autosave, 로그 (gitignored)
docs/                         기술 문서
```

## Running things

**서기 전용 학습 (임의의 자세에서 다시 서기):**
```bash
cd scripts
../.venv/bin/python train_locomotion.py --n-envs 8 --timesteps 40000000 --stand-only --out my_stand
```

**격투 리그 (P1/P2 번갈아 학습):**
```bash
cd scripts
./selfplay_league_loop.sh [rounds=10] [timesteps_per_round=1000000] p1_init.zip p2_init.zip
```

**고정 시드 평가 (두 체크포인트를 같은 조건에서 비교):**
```bash
cd scripts
../.venv/bin/python gate_eval.py ../checkpoints/a.zip ../checkpoints/opponent.zip --episodes 120
../.venv/bin/python gate_eval.py ../checkpoints/a.zip ../checkpoints/opponent.zip --episodes 120 --perturb
```

**학습된 정책 보기:**
```bash
cd scripts
DISPLAY=:0 ../.venv/bin/python watch.py ../checkpoints/my_run.zip [--opponent ../checkpoints/other.zip]
DISPLAY=:0 ../.venv/bin/python watch_locomotion.py ../checkpoints/my_stand.zip
```

**초기 단일 상대 / 셀프플레이 (이전 실험):**
```bash
cd scripts
../.venv/bin/python train.py --timesteps 5000000 --out my_run [--device cuda|cpu] [--n-envs 8]
../.venv/bin/python train_selfplay.py --timesteps 5000000 --out ppo_selfplay --init-from ../checkpoints/x.zip
```

학습 스크립트는 `checkpoints/autosave/`에 주기적으로 저장하므로(기본 25,000 스텝마다) 크래시가 나도
중간부터 재개할 수 있다.

## 진단 스크립트 작성 패턴

체크포인트를 로드하고 고정 시드로 N 에피소드를 돌리면서 `env.data`/`info`를 직접 샘플링해 수치(접촉
여부, 발 간격, 넉다운 스텝 비율 등)를 측정한다. 리워드나 물리 파라미터를 바꾸기 전후로 같은 스크립트로
행동 변화를 재는 걸 권장한다. 감으로 넘겨짚은 수정이 리워드만 왜곡시킨 사례(EFFORT_COST)가 있었다.

## TODO

- 대표 장면을 담은 GIF/영상 추가 (현재 README에는 없음)
- 기술 문서의 영문 요약
