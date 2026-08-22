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
import re
import subprocess
import time

SCRIPTS = pathlib.Path(__file__).resolve().parent
MODELS_DIR = SCRIPTS.parent / "checkpoints"
VENV_PY = SCRIPTS.parent / ".venv" / "bin" / "python"
LOG_PATH = MODELS_DIR / "league_dynamic_loop.log"

try:
    from kakao_notify import send_message
except Exception:
    send_message = None


def log(msg):
    line = f"[dynamic] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def notify(msg):
    if send_message is None:
        return
    try:
        send_message(msg)
    except Exception as e:
        log(f"kakao notify failed: {e}")


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


def train_side(side, own_ckpt, opp_ckpt, timesteps, out_name, ent_coef=None):
    cmd = [str(VENV_PY), "train_league.py", "--side", side, "--timesteps", str(timesteps),
           "--n-envs", "8", "--init-from", own_ckpt, "--opponent-from", opp_ckpt, "--out", out_name]
    if ent_coef is not None:
        cmd += ["--ent-coef", str(ent_coef)]
    start_pos = LOG_PATH.stat().st_size if LOG_PATH.exists() else 0
    with open(LOG_PATH, "a") as logf:
        subprocess.run(cmd, cwd=str(SCRIPTS), stdout=logf, stderr=subprocess.STDOUT)
    # pull the policy's action std out of the training subprocess's own SB3 log lines
    # (already written to LOG_PATH above) -- a cheap, free-riding way to watch for the
    # entropy collapse/explosion failure modes hit earlier (docs 9.9) without extra compute.
    with open(LOG_PATH, "r") as f:
        f.seek(start_pos)
        chunk = f.read()
    std_matches = re.findall(r"\|\s*std\s*\|\s*([0-9.eE+-]+)\s*\|", chunk)
    return float(std_matches[-1]) if std_matches else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="steps per round")
    parser.add_argument("--eval-episodes", type=int, default=30, help="episodes for the between-round matchup check")
    parser.add_argument("--margin", type=int, default=3,
                         help="min loss-count gap (out of --eval-episodes) before favoring the behind side; "
                              "smaller gaps just alternate so neither side goes stale")
    parser.add_argument("--max-consecutive", type=int, default=2,
                         help="cap on consecutive rounds training the same side even if it's still behind by "
                              "more than --margin -- training one side against a completely static opponent "
                              "plateaus fast (measured: round 1 closed 26:4->24:6, three more rounds against "
                              "the same frozen opponent went nowhere), so force a swap to let the other side "
                              "respond instead of grinding uselessly")
    parser.add_argument("--p1-init", required=True)
    parser.add_argument("--p2-init", required=True)
    parser.add_argument("--ent-coef", type=float, default=None,
                         help="PPO entropy coefficient override (SB3 default is 0.0, i.e. no "
                              "entropy bonus at all -- std has been in monotonic freefall the "
                              "whole project because of this). A small positive value keeps some "
                              "exploration/diversity alive instead of collapsing toward a "
                              "near-deterministic policy over many cumulative training rounds.")
    args = parser.parse_args()

    p1, p2 = args.p1_init, args.p2_init
    last_trained = None
    consecutive = 0

    for i in range(1, args.rounds + 1):
        result = run_matchup(p1, p2, args.eval_episodes)
        a_losses, b_losses = result["a_losses"], result["b_losses"]
        log(f"round {i}/{args.rounds}: matchup a_losses={a_losses} b_losses={b_losses} "
            f"(of {args.eval_episodes}) -- P1={p1} P2={p2}")
        log(f"round {i}/{args.rounds}: end_reasons {json.dumps(result.get('end_reason_counts', {}))} "
            f"head_ratio_a={result.get('mean_head_ratio_a')} head_ratio_b={result.get('mean_head_ratio_b')}")
        log(f"round {i}/{args.rounds}: hits_a={result.get('hits_a')} hits_b={result.get('hits_b')}")

        if a_losses - b_losses > args.margin:
            preferred = "a"
        elif b_losses - a_losses > args.margin:
            preferred = "b"
        else:
            preferred = "b" if last_trained == "a" else "a"

        if preferred == last_trained and consecutive >= args.max_consecutive:
            side = "b" if preferred == "a" else "a"
            log(f"round {i}/{args.rounds}: {preferred} still behind but hit "
                f"--max-consecutive={args.max_consecutive} -- forcing a swap to {side}")
        else:
            side = preferred
        consecutive = consecutive + 1 if side == last_trained else 1
        log(f"round {i}/{args.rounds}: training side={side}")

        t0 = time.time()
        if side == "a":
            std = train_side("a", p1, p2, args.timesteps, "ppo_p1", ent_coef=args.ent_coef)
            p1 = latest_ckpt("ppo_p1")
        else:
            std = train_side("b", p2, p1, args.timesteps, "ppo_p2", ent_coef=args.ent_coef)
            p2 = latest_ckpt("ppo_p2")
        last_trained = side
        elapsed = time.time() - t0
        log(f"round {i}/{args.rounds}: side={side} done in {elapsed:.0f}s -- P1={p1} P2={p2} std={std}")
        reasons = result.get("end_reason_counts", {})
        std_txt = f"{std:.4f}" if std is not None else "-"
        std_warn = " ⚠비정상" if std is not None and (std > 1.0 or std < 0.005) else ""
        ep = args.eval_episodes
        fall_n = reasons.get("a_down_timeout", 0) + reasons.get("b_down_timeout", 0)
        fall_rate = 100 * fall_n / ep
        ko_n = reasons.get("a_ko", 0) + reasons.get("b_ko", 0)
        ko_rate = 100 * ko_n / ep
        notify(
            f"[fightai] round {i}/{args.rounds} done (trained {side}, {elapsed:.0f}s)\n"
            f"matchup a_losses={a_losses} b_losses={b_losses} / {ep}\n"
            f"낙상률 {fall_rate:.0f}% (a={reasons.get('a_down_timeout', 0)} b={reasons.get('b_down_timeout', 0)}) "
            f"· KO율 {ko_rate:.0f}% (a={reasons.get('a_ko', 0)} b={reasons.get('b_ko', 0)}) "
            f"· 시간초과={reasons.get('truncated', 0)}\n"
            f"타격수 a={result.get('hits_a')} b={result.get('hits_b')}\n"
            f"머리높이비율 a={result.get('mean_head_ratio_a', 0):.2f} b={result.get('mean_head_ratio_b', 0):.2f} · "
            f"std={std_txt}{std_warn}"
        )

    log(f"all {args.rounds} rounds complete. P1={p1} P2={p2}")
    notify(f"[fightai] all {args.rounds} rounds complete. P1={p1} P2={p2}")


if __name__ == "__main__":
    main()
