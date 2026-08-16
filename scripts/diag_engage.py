"""Diagnostic: is the large r_engage magnitude (avg ~-53/episode, dwarfing r_strike
~+7) actually driving approach behavior, or is it dead weight like EFFORT_COST was?

For each episode, track: length, how it ended, starting foot_dist, minimum foot_dist
reached (did they ever actually get close?), and step index of first close approach.
"""
import sys

import numpy as np
from stable_baselines3 import PPO

from env import Fighter2DEnv, FALL_HEIGHT

CHECKPOINT = sys.argv[1] if len(sys.argv) > 1 else "../checkpoints/ppo_selfplay_20260816_213302.zip"
N_EPISODES = int(sys.argv[2]) if len(sys.argv) > 2 else 60
CLOSE_THRESHOLD = 1.0  # foot distance considered "engaged"

model = PPO.load(CHECKPOINT, device="cpu")
env = Fighter2DEnv(opponent_policy_path=CHECKPOINT)

episodes = []
obs, info = env.reset()
foot_dists = []
step_idx = 0
ep_start_step = 0

while len(episodes) < N_EPISODES:
    a_foot_xs = [env.data.site_xpos[s][0] for s in env.a_feet]
    b_foot_xs = [env.data.site_xpos[s][0] for s in env.b_feet]
    fd = min(abs(bf - af) for af in a_foot_xs for bf in b_foot_xs)
    foot_dists.append(fd)

    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    step_idx += 1

    if terminated or truncated:
        fd_arr = np.array(foot_dists)
        min_fd = fd_arr.min()
        start_fd = fd_arr[0]
        first_close = np.argmax(fd_arr < CLOSE_THRESHOLD) if (fd_arr < CLOSE_THRESHOLD).any() else -1
        a_z = env.data.xpos[env.a_torso_id][2]
        b_z = env.data.xpos[env.b_torso_id][2]
        ended_by = "truncated" if truncated and not terminated else (
            "mutual_fall" if info["health_a"] <= 0 and info["health_b"] <= 0 else
            "a_ko" if info["health_a"] <= 0 else
            "b_ko" if info["health_b"] <= 0 else
            "a_down_timeout" if a_z < FALL_HEIGHT else
            "b_down_timeout" if b_z < FALL_HEIGHT else "other"
        )
        episodes.append({
            "length": len(fd_arr), "start_fd": start_fd, "min_fd": min_fd,
            "first_close_step": first_close, "ever_close": bool((fd_arr < CLOSE_THRESHOLD).any()),
            "ended_by": ended_by,
        })
        foot_dists = []
        obs, info = env.reset()

lengths = np.array([e["length"] for e in episodes])
min_fds = np.array([e["min_fd"] for e in episodes])
ever_close = np.array([e["ever_close"] for e in episodes])
first_close = np.array([e["first_close_step"] for e in episodes if e["first_close_step"] >= 0])

print(f"checkpoint: {CHECKPOINT}")
print(f"episodes: {N_EPISODES}")
print(f"episode length: mean={lengths.mean():.1f} median={np.median(lengths):.0f}")
print(f"ever got within {CLOSE_THRESHOLD}: {ever_close.sum()}/{N_EPISODES} ({100*ever_close.mean():.1f}%)")
if len(first_close):
    print(f"steps to first close approach (of those that closed): mean={first_close.mean():.1f} "
          f"median={np.median(first_close):.0f}")
print(f"min foot_dist reached: mean={min_fds.mean():.2f} median={np.median(min_fds):.2f} "
      f"max={min_fds.max():.2f} (i.e. never closed at all in the worst case)")
print()
from collections import Counter
ended = Counter(e["ended_by"] for e in episodes)
print("episode end reasons:", dict(ended))
print()
never_closed = [e for e in episodes if not e["ever_close"]]
print(f"never-closed episodes: {len(never_closed)}")
if never_closed:
    nc_len = np.array([e["length"] for e in never_closed])
    nc_end = Counter(e["ended_by"] for e in never_closed)
    print(f"  their length: mean={nc_len.mean():.1f}  end reasons: {dict(nc_end)}")
