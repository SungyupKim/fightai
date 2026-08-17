"""Dynamic league self-play: after each round, measures the current P1-vs-P2 win/loss
ratio (via matchup_eval.py) and trains whichever side is currently BEHIND, instead of
a fixed alternating schedule -- so a side that's fallen behind gets consecutive
catch-up rounds instead of waiting its fixed turn, and a side that's already ahead
doesn't keep training further into a lock-in. Falls back to plain alternation when the
two are close enough (within --margin losses) so neither side goes stale.

Usage: selfplay_league_dynamic.py --p1-init CKPT --p2-init CKPT [--rounds N] [...]
"""
import argparse
import json
import pathlib
import subprocess
import time

SCRIPTS = pathlib.Path(__file__).resolve().parent
MODELS_DIR = SCRIPTS.parent / "checkpoints"
VENV_PY = SCRIPTS.parent / ".venv" / "bin" / "python"
LOG_PATH = MODELS_DIR / "league_dynamic_loop.log"


def log(msg):
    line = f"[dynamic] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def latest_ckpt(prefix):
    files = [p for p in MODELS_DIR.glob(f"{prefix}_2*.zip") if "snap" not in p.name]
    if not files:
        raise RuntimeError(f"no checkpoint found matching {prefix}_2*.zip")
    return str(max(files, key=lambda p: p.stat().st_mtime))


def run_matchup(p1, p2, episodes):
    out = subprocess.run(
        [str(VENV_PY), "matchup_eval.py", p1, p2, "--episodes", str(episodes)],
        cwd=str(SCRIPTS), capture_output=True, text=True, timeout=180,
    )
    if out.returncode != 0:
        raise RuntimeError(f"matchup_eval.py failed: {out.stderr[-1000:]}")
    return json.loads(out.stdout.strip().splitlines()[-1])


def train_side(side, own_ckpt, opp_ckpt, timesteps, out_name):
    cmd = [str(VENV_PY), "train_league.py", "--side", side, "--timesteps", str(timesteps),
           "--n-envs", "8", "--init-from", own_ckpt, "--opponent-from", opp_ckpt, "--out", out_name]
    with open(LOG_PATH, "a") as logf:
        subprocess.run(cmd, cwd=str(SCRIPTS), stdout=logf, stderr=subprocess.STDOUT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="steps per round")
    parser.add_argument("--eval-episodes", type=int, default=30, help="episodes for the between-round matchup check")
    parser.add_argument("--margin", type=int, default=3,
                         help="min loss-count gap (out of --eval-episodes) before favoring the behind side; "
                              "smaller gaps just alternate so neither side goes stale")
    parser.add_argument("--p1-init", required=True)
    parser.add_argument("--p2-init", required=True)
    args = parser.parse_args()

    p1, p2 = args.p1_init, args.p2_init
    last_trained = None

    for i in range(1, args.rounds + 1):
        result = run_matchup(p1, p2, args.eval_episodes)
        a_losses, b_losses = result["a_losses"], result["b_losses"]
        log(f"round {i}/{args.rounds}: matchup a_losses={a_losses} b_losses={b_losses} "
            f"(of {args.eval_episodes}) -- P1={p1} P2={p2}")

        if a_losses - b_losses > args.margin:
            side = "a"
        elif b_losses - a_losses > args.margin:
            side = "b"
        else:
            side = "b" if last_trained == "a" else "a"
        log(f"round {i}/{args.rounds}: training side={side}")

        t0 = time.time()
        if side == "a":
            train_side("a", p1, p2, args.timesteps, "ppo_p1")
            p1 = latest_ckpt("ppo_p1")
        else:
            train_side("b", p2, p1, args.timesteps, "ppo_p2")
            p2 = latest_ckpt("ppo_p2")
        last_trained = side
        log(f"round {i}/{args.rounds}: side={side} done in {time.time() - t0:.0f}s -- P1={p1} P2={p2}")

    log(f"all {args.rounds} rounds complete. P1={p1} P2={p2}")


if __name__ == "__main__":
    main()
