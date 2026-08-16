"""True simultaneous self-play: instead of 'a' training against a periodically-refreshed
frozen snapshot of itself (train_selfplay.py), both sides of every match are driven by
the SAME live policy every step, and BOTH perspectives' transitions feed the same PPO
rollout buffer -- so the mirrored ('b') frame gets direct gradient exposure continuously,
not just via a frozen opponent or the observation-mirror augmentation in env.py.

Implemented as a custom VecEnv: N physical Fighter2DEnv instances (one MuJoCo sim each,
one per subprocess worker) are exposed to SB3 as 2N "slots" -- slot 2i is env i's 'a'
perspective, slot 2i+1 is the same physical match's 'b' perspective. Each physical step
advances the shared simulation once using both slots' actions (env.step's b_full_action
param), so the two slots are two views of one match, not two independent environments.
"""
import multiprocessing as mp

import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper, VecEnv


def _paired_worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.var()
    obs_a, _ = env.reset()
    env._a_mirror = False  # the b-slot already gives direct, non-synthetic mirrored-frame
    obs_b = env._obs_for_b()  # training signal every step -- no need for the a-mirror augmentation too
    ep_rew_a = ep_rew_b = 0.0  # Monitor-equivalent per-episode return/length tracking, since this
    ep_len = 0                # custom VecEnv bypasses the usual Monitor wrapper -- without an
                               # info["episode"] entry on done, SB3's ep_rew_mean/ep_len_mean never update
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                a_action = np.clip(data[0], -1.0, 1.0)
                b_action = np.clip(data[1], -1.0, 1.0)
                obs_a, reward_a, term, trunc, info = env.step(a_action, b_full_action=b_action)
                obs_b = env._obs_for_b()
                reward_b = info["reward_b"]
                ep_rew_a += reward_a
                ep_rew_b += reward_b
                ep_len += 1
                info_a = {"reward_breakdown": info["reward_breakdown"],
                          "health_a": info["health_a"], "health_b": info["health_b"]}
                info_b = {"reward_breakdown": info["reward_breakdown_b"],
                          "health_a": info["health_b"], "health_b": info["health_a"]}
                done = term or trunc
                info_a["TimeLimit.truncated"] = trunc and not term
                info_b["TimeLimit.truncated"] = trunc and not term
                if done:
                    info_a["terminal_observation"] = obs_a
                    info_b["terminal_observation"] = obs_b
                    info_a["episode"] = {"r": round(ep_rew_a, 6), "l": ep_len}
                    info_b["episode"] = {"r": round(ep_rew_b, 6), "l": ep_len}
                    ep_rew_a = ep_rew_b = 0.0
                    ep_len = 0
                    obs_a, _ = env.reset()
                    env._a_mirror = False
                    obs_b = env._obs_for_b()
                remote.send(((obs_a, obs_b), (reward_a, reward_b), done, (info_a, info_b)))
            elif cmd == "reset":
                obs_a, reset_info = env.reset()
                env._a_mirror = False
                obs_b = env._obs_for_b()
                ep_rew_a = ep_rew_b = 0.0
                ep_len = 0
                remote.send(((obs_a, obs_b), reset_info))
            elif cmd == "close":
                env.close()
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            elif cmd == "env_method":
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == "get_attr":
                remote.send(getattr(env, data))
            elif cmd == "set_attr":
                remote.send(setattr(env, data[0], data[1]))
            elif cmd == "is_wrapped":
                remote.send(False)
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the paired worker")
    except (EOFError, KeyboardInterrupt):
        pass


class PairedSelfPlayVecEnv(VecEnv):
    def __init__(self, n_physical_envs, env_fn, start_method=None):
        self.waiting = False
        self.closed = False
        self.n_physical = n_physical_envs

        if start_method is None:
            start_method = "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"
        ctx = mp.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_physical_envs)], strict=True)
        self.processes = []
        for work_remote, remote in zip(self.work_remotes, self.remotes, strict=True):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            p = ctx.Process(target=_paired_worker, args=args, daemon=True)
            p.start()
            self.processes.append(p)
            work_remote.close()

        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()
        super().__init__(2 * n_physical_envs, observation_space, action_space)

    def step_async(self, actions):
        for i, remote in enumerate(self.remotes):
            remote.send(("step", (actions[2 * i], actions[2 * i + 1])))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rewards, dones, infos = [], [], [], []
        for (obs_a, obs_b), (reward_a, reward_b), done, (info_a, info_b) in results:
            obs += [obs_a, obs_b]
            rewards += [reward_a, reward_b]
            dones += [done, done]
            infos += [info_a, info_b]
        return np.stack(obs), np.array(rewards, dtype=np.float32), np.array(dones, dtype=bool), infos

    def reset(self, seed=None, options=None):
        for remote in self.remotes:
            remote.send(("reset", None))
        results = [remote.recv() for remote in self.remotes]
        obs, infos = [], []
        for (obs_a, obs_b), reset_info in results:
            obs += [obs_a, obs_b]
            infos += [reset_info, reset_info]
        self.reset_infos = infos
        return np.stack(obs)

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.processes:
            p.join()
        self.closed = True

    def _physical_indices(self, indices):
        return sorted(set(i // 2 for i in self._get_indices(indices)))

    def get_attr(self, attr_name, indices=None):
        target = self._physical_indices(indices)
        for i in target:
            self.remotes[i].send(("get_attr", attr_name))
        return [self.remotes[i].recv() for i in target]

    def set_attr(self, attr_name, value, indices=None):
        target = self._physical_indices(indices)
        for i in target:
            self.remotes[i].send(("set_attr", (attr_name, value)))
        for i in target:
            self.remotes[i].recv()

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        target = self._physical_indices(indices)
        for i in target:
            self.remotes[i].send(("env_method", (method_name, method_args, method_kwargs)))
        return [self.remotes[i].recv() for i in target]

    def env_is_wrapped(self, wrapper_class, indices=None):
        target = self._get_indices(indices)
        return [False for _ in target]

    def get_images(self):
        raise NotImplementedError

    def render(self, mode=None):
        raise NotImplementedError
