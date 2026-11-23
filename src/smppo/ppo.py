# Licence: pending owner approval (proposed MIT). The adapted sb3-contrib code is MIT; see THIRD_PARTY_NOTICES.md.
# Semi-Markov MaskablePPO.
#
# The algorithm below is copied unchanged from the validated research implementation (see docs/design.md,
# "Provenance"). SemiMarkovMaskablePPO.collect_rollouts is adapted from sb3-contrib 2.7.0
# MaskablePPO.collect_rollouts, and SemiMarkovRolloutBuffer extends sb3-contrib's MaskableRolloutBuffer.
# sb3-contrib: MIT License, Copyright (c) 2020 Stable-Baselines Team. See THIRD_PARTY_NOTICES.md.
"""MaskablePPO with per-transition discount gamma_t (semi-Markov), spec §5.

Changes vs sb3-contrib 2.7.0 MaskablePPO.collect_rollouts / RolloutBuffer.compute_returns_and_advantage:
  * buffer stores gamma_t and an episode-end flag per transition;
  * delta_t = r_t + gamma_t (1 - end_t) V(s_{t+1}) - V(s_t);  A_t = delta_t + gamma_t lambda (1 - end_t) A_{t+1};
  * on an external cutoff (truncation) r_t += gamma_t V(terminal_obs) and end_t = 1 (bootstrap once, no GAE leak);
  * SB3's scalar-gamma 'TimeLimit.truncated' correction is NOT applied.
"""
from __future__ import annotations

import numpy as np
import torch as th
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.buffers import MaskableRolloutBuffer
from sb3_contrib.common.maskable.utils import get_action_masks, is_masking_supported
from stable_baselines3.common.utils import obs_as_tensor


class SemiMarkovRolloutBuffer(MaskableRolloutBuffer):
    def reset(self) -> None:
        super().reset()
        self.gammas = np.zeros((self.buffer_size, self.n_envs), np.float32)
        self.ends = np.zeros((self.buffer_size, self.n_envs), np.float32)

    def add_sm(self, gamma_t: np.ndarray, end: np.ndarray) -> None:
        p = self.pos - 1  # called right after add()
        self.gammas[p], self.ends[p] = gamma_t, end

    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
        last_values = last_values.clone().cpu().numpy().flatten()
        adv = 0.0
        for step in reversed(range(self.buffer_size)):
            nxt = last_values if step == self.buffer_size - 1 else self.values[step + 1]
            cont = 1.0 - self.ends[step]
            g = self.gammas[step]
            delta = self.rewards[step] + g * cont * nxt - self.values[step]
            adv = delta + g * self.gae_lambda * cont * adv
            self.advantages[step] = adv
        self.returns = self.advantages + self.values


class SemiMarkovMaskablePPO(MaskablePPO):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("rollout_buffer_class", SemiMarkovRolloutBuffer)
        super().__init__(*args, **kwargs)
        self.n_bootstraps = 0

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps, use_masking=True):
        assert isinstance(rollout_buffer, SemiMarkovRolloutBuffer)
        self.policy.set_training_mode(False)
        n_steps = 0
        rollout_buffer.reset()
        if use_masking and not is_masking_supported(env):
            raise ValueError("env must implement action_masks()")
        callback.on_rollout_start()
        while n_steps < n_rollout_steps:
            with th.no_grad():
                action_masks = get_action_masks(env) if use_masking else None
                actions, values, log_probs = self.policy(obs_as_tensor(self._last_obs, self.device), action_masks=action_masks)
            actions = actions.cpu().numpy()
            new_obs, rewards, dones, infos = env.step(actions)
            self.num_timesteps += env.num_envs
            callback.update_locals(locals())
            if not callback.on_step():
                return False
            self._update_info_buffer(infos, dones)
            n_steps += 1
            actions = actions.reshape(-1, 1)
            gam = np.array([inf["gamma_t"] for inf in infos], np.float32)
            rewards = rewards.astype(np.float32).copy()
            for idx, inf in enumerate(infos):
                if dones[idx] and inf.get("truncated_cutoff", False):
                    term_obs = self.policy.obs_to_tensor(inf["terminal_observation"])[0]
                    with th.no_grad():
                        rewards[idx] += gam[idx] * float(self.policy.predict_values(term_obs)[0])
                    self.n_bootstraps += 1
            rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs,
                               action_masks=action_masks)
            rollout_buffer.add_sm(gam, dones.astype(np.float32))
            self._last_obs = new_obs
            self._last_episode_starts = dones
        with th.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        callback.on_rollout_end()
        return True
