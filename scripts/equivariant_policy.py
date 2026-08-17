"""A policy whose action-mean (and value) output is EXACTLY equivariant under the
environment's left-right mirror transform, by construction -- not just encouraged via
symmetric training data (mirror augmentation, paired self-play), which measurably
wasn't enough on its own: even after full convergence, the a/b asymmetry kept
reappearing (see docs/기술문서, section 4 and reward-history item 18).

Construction: for an ordinary, otherwise-unconstrained network g (the usual
MlpExtractor + action_net/value_net), define

    f(x) = 0.5 * (g(x) + action_mirror * g(obs_mirror * x))

This is the standard symmetrization (Reynolds operator) for a Z2 reflection group,
and is EXACT -- f(obs_mirror * x) == action_mirror * f(x) always holds -- because
obs_mirror and action_mirror are both self-inverse (diagonal +/-1 matrices, i.e.
obs_mirror * obs_mirror == I). Proof sketch: substitute x -> obs_mirror*x in f and use
(obs_mirror)^2 = I; the two terms swap and the equality falls out algebraically.

No new learnable parameters are added (obs_mirror/action_mirror are fixed buffers), so
a checkpoint trained with a plain ActorCriticPolicy can donate its weights directly via
load_state_dict(strict=False) -- see transplant_weights() below.
"""
import numpy as np
import torch as th
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy


def build_mirror_vectors():
    """Returns (obs_mirror, action_mirror) as float32 numpy arrays, matching env.py's
    _obs_view()/_action_mirror conventions exactly. Doesn't need a live MuJoCo sim --
    these are pure index/sign bookkeeping, already computed in Fighter2DEnv.__init__."""
    from env import Fighter2DEnv
    env = Fighter2DEnv()
    qpos_mirror = env._qpos_mirror  # (12,): root_x, root_z, root_ry, <9 joints>
    action_mirror = env._action_mirror.astype(np.float32)  # (10,), all -1
    # observation layout from _obs_view: [self_qpos(12), self_qvel(12), opp_qpos(12),
    # opp_qvel(12), extra(6)] -- extra = [x_diff, z_diff, self_hp, opp_hp, self_stagger,
    # opp_stagger]; only x_diff flips under the mirror (z/health/stagger don't)
    extra_mirror = np.array([-1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    obs_mirror = np.concatenate([qpos_mirror, qpos_mirror, qpos_mirror, qpos_mirror, extra_mirror]).astype(np.float32)
    env.close()
    return obs_mirror, action_mirror


class MirrorEquivariantPolicy(ActorCriticPolicy):
    def __init__(self, *args, obs_mirror, action_mirror, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("obs_mirror", th.as_tensor(np.asarray(obs_mirror), dtype=th.float32))
        self.register_buffer("action_mirror", th.as_tensor(np.asarray(action_mirror), dtype=th.float32))
        assert isinstance(self.action_dist, DiagGaussianDistribution), \
            "MirrorEquivariantPolicy assumes a continuous Box action space (DiagGaussianDistribution)"

    def _latent_pi_vf(self, obs):
        features = self.extract_features(obs)
        if self.share_features_extractor:
            return self.mlp_extractor(features)
        pi_features, vf_features = features
        return self.mlp_extractor.forward_actor(pi_features), self.mlp_extractor.forward_critic(vf_features)

    def _symmetrized(self, obs):
        """Returns (mean_actions, values), both exactly mirror-equivariant/-invariant."""
        latent_pi, latent_vf = self._latent_pi_vf(obs)
        mean_raw = self.action_net(latent_pi)
        value_raw = self.value_net(latent_vf)

        mirrored_obs = obs * self.obs_mirror
        latent_pi_m, latent_vf_m = self._latent_pi_vf(mirrored_obs)
        mean_mirrored = self.action_net(latent_pi_m) * self.action_mirror
        value_mirrored = self.value_net(latent_vf_m)  # value is a scalar quality estimate,
        # not a direction, so it should be mirror-INVARIANT (not flipped) -- same situation
        # described in either canonical frame is equally good/bad

        mean_actions = 0.5 * (mean_raw + mean_mirrored)
        values = 0.5 * (value_raw + value_mirrored)
        return mean_actions, values

    def forward(self, obs, deterministic=False):
        mean_actions, values = self._symmetrized(obs)
        distribution = self.action_dist.proba_distribution(mean_actions, self.log_std)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob

    def evaluate_actions(self, obs, actions):
        mean_actions, values = self._symmetrized(obs)
        distribution = self.action_dist.proba_distribution(mean_actions, self.log_std)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def get_distribution(self, obs):
        mean_actions, _ = self._symmetrized(obs)
        return self.action_dist.proba_distribution(mean_actions, self.log_std)

    def predict_values(self, obs):
        _, values = self._symmetrized(obs)
        return values

    def _predict(self, observation, deterministic=False):
        return self.get_distribution(observation).get_actions(deterministic=deterministic)


def transplant_weights(new_policy, old_policy_state_dict):
    """Loads mlp_extractor/action_net/value_net/log_std weights from a checkpoint
    trained with a plain ActorCriticPolicy into a freshly-constructed
    MirrorEquivariantPolicy. strict=False because the new policy has two extra
    (non-learnable) buffers -- obs_mirror/action_mirror -- that the old checkpoint
    never had; everything else must match exactly (same net_arch)."""
    missing, unexpected = new_policy.load_state_dict(old_policy_state_dict, strict=False)
    assert set(missing) <= {"obs_mirror", "action_mirror"}, f"unexpected missing keys: {missing}"
    assert not unexpected, f"unexpected extra keys in old checkpoint: {unexpected}"
