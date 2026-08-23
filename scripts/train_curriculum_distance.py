"""Curriculum: fine-tunes a checkpoint through progressively larger spawn distances
(edits build_model.py's fighter x-offset, regenerates the model, trains, repeat) instead
of cold-starting at a large distance -- a from-scratch attempt at 1.8 got only 6.7%
engagement even with the balance-penalty fix, so this tries extending an already-good
close-range policy (scratch13, 76.7% engagement at 1.2) incrementally instead.
"""
import argparse
import pathlib
import re
import subprocess

from kakao_notify import send_message

SCRIPTS = pathlib.Path(__file__).resolve().parent
BUILD_MODEL = SCRIPTS / "build_model.py"
CKPT_DIR = SCRIPTS.parent / "checkpoints"


def set_distance(x):
    text = BUILD_MODEL.read_text()
    new_text = re.sub(
        r'make_fighter\("a_", -[\d.]+,',
        f'make_fighter("a_", -{x},',
        text,
    )
    new_text = re.sub(
        r'make_fighter\("b_", [\d.]+,',
        f'make_fighter("b_", {x},',
        new_text,
    )
    BUILD_MODEL.write_text(new_text)
    subprocess.run(["../.venv/bin/python", "build_model.py"], cwd=str(SCRIPTS), check=True)


def latest_ckpt(prefix):
    return str(max(CKPT_DIR.glob(f"{prefix}_2*.zip"), key=lambda p: p.stat().st_mtime))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-from", required=True)
    # TOTAL spawn distance (a-to-b) per stage, not the per-side x-offset -- set_distance()
    # takes half of this. 1.2 first: re-optimize at the SAME distance scratch13 already
    # knows, so it adapts to the new progress_reward term (which it's never seen) before
    # also facing a harder distance at the same time -- one new variable at a time.
    parser.add_argument("--stages", type=float, nargs="+", default=[1.2, 1.4, 1.6, 1.8])
    parser.add_argument("--timesteps", type=int, default=1_500_000)
    parser.add_argument("--out-prefix", default="ppo_curriculum")
    args = parser.parse_args()

    checkpoint = args.init_from
    send_message(f"[fightai] 거리 커리큘럼 학습 시작: {' -> '.join(map(str, args.stages))}")

    for i, dist in enumerate(args.stages, 1):
        set_distance(dist / 2)
        out_name = f"{args.out_prefix}_d{dist}"
        subprocess.run(
            ["../.venv/bin/python", "train.py", "--timesteps", str(args.timesteps),
             "--n-envs", "8", "--resume-from", checkpoint, "--out", out_name],
            cwd=str(SCRIPTS), check=True,
        )
        checkpoint = latest_ckpt(out_name)
        send_message(f"[fightai] 커리큘럼 {i}/{len(args.stages)}단계 완료 (거리 {dist}). {checkpoint}")

    send_message(f"[fightai] 거리 커리큘럼 학습 전체 완료. 최종: {checkpoint}")


if __name__ == "__main__":
    main()
