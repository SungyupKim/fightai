# fightai

2D MuJoCo 랙돌 격투 에이전트. PPO로 학습하며, 셀프플레이(자기 자신의 과거 스냅샷과 대전)로 훈련한다.
설계 배경과 리워드 셰이핑 히스토리는 [`docs/fightai_기술문서.docx`](docs/fightai_기술문서.docx) 참고.

## Setup

```bash
python -m venv .venv
.venv/bin/pip install mujoco gymnasium stable-baselines3 tensorboard
```

`watch.py`(뷰어)는 OpenGL 컨텍스트가 있는 머신이 필요하다. 학습 자체는 headless로 동작한다.

## Project layout

```
models/fighter2d.xml     두 파이터 MJCF 모델 (build_model.py가 생성)
scripts/build_model.py   fighter2d.xml 생성기 -- 모델을 고칠 땐 XML이 아니라 이 파일을 수정
scripts/view.py          스크립트 상대(shadow-boxing) 스모크 테스트 + JOINTS/get_ctrl 정의
scripts/env.py           Fighter2DEnv: Gymnasium 환경 (리워드/물리/셀프플레이 미러링)
scripts/train.py         단일 상대(스크립트 opponent) PPO 학습
scripts/train_selfplay.py 셀프플레이 PPO 학습 (주기적 opponent 스냅샷 갱신)
scripts/selfplay_loop.sh 셀프플레이를 여러 라운드 자동으로 이어 돌리는 래퍼
scripts/watch.py         체크포인트를 뷰어로 재생 (체력바/피격 이펙트 포함)
scripts/dashboard.py     로컬 웹 대시보드 (학습/뷰어 제어 + 그래프)
scripts/diag_kick_cap.py 접촉력 기반 데미지가 캡에 걸리는 비율을 측정하는 진단 스크립트
checkpoints/              학습된 모델, autosave, 로그 (gitignored)
docs/                    기술 문서
```

## Running things

**단일 상대 학습:**
```bash
cd scripts
../.venv/bin/python train.py --timesteps 5000000 --out my_run \
  [--resume-from ../checkpoints/some_checkpoint.zip] [--device cuda|cpu] [--n-envs 8]
```

**셀프플레이 학습:**
```bash
cd scripts
../.venv/bin/python train_selfplay.py --timesteps 5000000 --out ppo_selfplay \
  --init-from ../checkpoints/some_checkpoint.zip \
  [--refresh-interval 250000] [--ent-coef 0.01] [--n-envs 8] [--device cuda|cpu]
```

**여러 라운드 자동 반복:**
```bash
cd scripts
./selfplay_loop.sh [rounds=10] [timesteps_per_round=5000000] [refresh_interval=250000]
```
직전 라운드의 최신 체크포인트(`ppo_selfplay_2*.zip`, snap 제외)를 자동으로 찾아 이어서 학습한다.

두 학습 스크립트 모두 `checkpoints/autosave/`에 주기적으로 저장하므로(`--save-freq`, 기본 25,000
스텝) 크래시가 나도 중간부터 재개할 수 있다.

**학습된 정책 보기:**
```bash
cd scripts
DISPLAY=:0 ../.venv/bin/python watch.py ../checkpoints/my_run_20260101_120000.zip \
  [--opponent ../checkpoints/same_or_other_checkpoint.zip]
```
`--opponent`를 주면 셀프플레이 미러 매치로 관전(같은 체크포인트를 주면 자기 자신과의 미러 매치).

**대시보드:**
```bash
.venv/bin/python scripts/dashboard.py
```
`http://localhost:8787` 접속. 학습/뷰어 시작·중지, 체크포인트 선택, 실시간 리워드/breakdown 그래프.

## 진단 스크립트 작성 패턴

체크포인트를 로드하고 N 스텝 돌리면서 `env.data`/`info`를 직접 샘플링해 구체적인 수치(접촉력 분포,
자세 높이, 넉다운 횟수 등)를 측정하는 방식. `scripts/diag_kick_cap.py`가 예시. 뷰어로 눈으로 보고
판단하기보다, 리워드/물리 파라미터를 바꾸기 전후로 이런 스크립트로 실제 행동 변화를 측정하는 걸
권장한다 — 감으로 넘겨짚은 수정이 아무 효과 없이 리워드만 왜곡시킨 사례(EFFORT_COST)가 있었다.
