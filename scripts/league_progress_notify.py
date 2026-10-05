"""Watches selfplay_league_loop.sh's combined log and sends one KakaoTalk summary per finished
league round -- polls every minute, sends when the round number advances (the summary is the
last stats block of the round that just finished), plus a final message when the whole run
completes. Standalone loop, independent of the training process.

Usage: league_progress_notify.py [--poll-seconds 60] [--log PATH]
"""
import argparse
import pathlib
import re
import time

from kakao_notify import send_message

SCRIPTS = pathlib.Path(__file__).resolve().parent
DEFAULT_LOG = SCRIPTS.parent / "checkpoints" / "league_loop.log"

FIELDS = ["total_timesteps", "ep_rew_mean", "b_ep_rew_mean", "r_strike", "r_engage", "std"]
ROUND_RE = re.compile(r"\[league\] round (\d+)/(\d+): (\w+) \(\w\) vs frozen \w+")
DONE_RE = re.compile(r"\[league\] all (\d+) rounds complete")
STAT_RE = re.compile(r"\|\s*(\w+)\s*\|\s*([-+\d.eE]+)\s*\|")


def read_tail_lines(log_path):
    with open(log_path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 40_000))
        return f.read().decode(errors="ignore").splitlines()


def latest_round(lines):
    """Returns (index, round_n, total, side) of the most recent round announcement, or None."""
    for i in range(len(lines) - 1, -1, -1):
        m = ROUND_RE.match(lines[i].strip())
        if m:
            return i, int(m.group(1)), int(m.group(2)), m.group(3)
    return None


def stats_between(lines, start, end):
    """Latest value of each FIELD appearing in lines[start:end], searching backwards."""
    values = {}
    for line in reversed(lines[start:end]):
        m = STAT_RE.match(line.strip())
        if m and m.group(1) in FIELDS and m.group(1) not in values:
            values[m.group(1)] = float(m.group(2))
        if len(values) == len(FIELDS):
            break
    return values


def format_round_message(round_n, total, values):
    steps_m = values.get("total_timesteps", 0) / 1_000_000
    return (
        f"[fightai] 리그 라운드 {round_n}/{total} 완료\n"
        f"누적 스텝: {steps_m:.2f}M\n"
        f"std: {values.get('std', float('nan')):.3f}\n"
        f"ep_rew_mean: {values.get('ep_rew_mean', float('nan')):.1f}\n"
        f"b_ep_rew_mean: {values.get('b_ep_rew_mean', float('nan')):.1f}\n"
        f"r_strike: {values.get('r_strike', float('nan')):.1f}  "
        f"r_engage: {values.get('r_engage', float('nan')):.1f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--log", type=str, default=str(DEFAULT_LOG))
    args = parser.parse_args()
    log_path = pathlib.Path(args.log)

    last_round = None
    final_sent = False
    while True:
        try:
            lines = read_tail_lines(log_path)
            found = latest_round(lines)
            if found is not None:
                idx, round_n, total, _side = found
                if last_round is None:
                    last_round = round_n  # don't send for whatever was already in progress at startup
                elif round_n > last_round:
                    # the just-finished round's final stats sit right before the new round's line
                    values = stats_between(lines, 0, idx)
                    if values:
                        msg = format_round_message(last_round, total, values)
                        send_message(msg)
                        print(f"[league-progress-notify] sent round {last_round}", flush=True)
                    last_round = round_n
            if not final_sent and any(DONE_RE.match(l.strip()) for l in lines[-50:]):
                values = stats_between(lines, 0, len(lines))
                if values and found is not None:
                    send_message(format_round_message(found[1], found[2], values) + "\n전체 리그 완료")
                    print("[league-progress-notify] sent final", flush=True)
                final_sent = True
        except Exception as e:
            print(f"[league-progress-notify] failed: {e}", flush=True)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
