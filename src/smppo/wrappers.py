"""Gymnasium wrappers that produce the information SemiMarkovMaskablePPO reads from ``info``.

SemiMarkovMaskablePPO needs two keys in the ``info`` dict of every step:

* ``gamma_t``: the discount for this transition, ``gamma ** duration``.
* ``truncated_cutoff``: ``truncated and not terminated``. This marks an external cutoff (time limit), which is
  bootstrapped once with ``gamma_t * V(terminal_observation)``.

Two wrappers are provided:

* :class:`MacroActionWrapper` turns a ``Discrete`` environment into an SMDP whose actions are
  (primitive action, repeat count). The duration is the number of primitive steps actually executed.
* :class:`DurationDiscountWrapper` is for environments whose transitions already take variable time. It reads a
  duration from ``info`` (or computes one from the action) and adds ``gamma_t``. Reward and dynamics are unchanged.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import gymnasium as gym


class MacroActionWrapper(gym.Wrapper):
    """Repeat one primitive action up to ``k_max`` times as a single SMDP decision.

    Action ``a`` in ``Discrete(n_primitive * len(durations))`` decodes to
    ``primitive = a // len(durations)`` and ``k_max = durations[a % len(durations)]``.

    Execution stops immediately when the wrapped environment terminates or truncates. The executed duration
    ``k`` (``1 <= k <= k_max``) is therefore exact, including when a time limit falls inside the macro-action.

    Returned reward: ``R = sum_{j<k} gamma**j * r_j`` (discounted within the macro-action).
    Added ``info`` keys: ``k``, ``gamma_t = gamma**k``, ``truncated_cutoff``, ``prim_rewards``.

    Put any time limit (``TimeLimit``) *inside* this wrapper, so that it counts primitive steps.
    The wrapper draws no random numbers and does no bootstrapping.
    """

    def __init__(self, env: gym.Env, durations: Sequence[int] = (1, 2, 4), gamma: float = 0.99):
        super().__init__(env)
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise TypeError("MacroActionWrapper needs a Discrete primitive action space")
        if not durations or any(int(d) < 1 for d in durations):
            raise ValueError("durations must be positive integers")
        self.durations = tuple(int(d) for d in durations)
        self.gamma = float(gamma)
        self.n_primitive = int(env.action_space.n)
        self.action_space = gym.spaces.Discrete(self.n_primitive * len(self.durations))

    def decode(self, action: int) -> tuple[int, int]:
        n = len(self.durations)
        return int(action) // n, self.durations[int(action) % n]

    def step(self, action):
        prim, k_max = self.decode(action)
        reward, prim_rewards = 0.0, []
        for j in range(k_max):
            obs, r, terminated, truncated, info = self.env.step(prim)
            reward += (self.gamma**j) * float(r)
            prim_rewards.append(float(r))
            if terminated or truncated:
                break
        k = len(prim_rewards)
        info = dict(info)
        info.update(
            k=k, gamma_t=self.gamma**k, truncated_cutoff=bool(truncated and not terminated), prim_rewards=prim_rewards
        )
        return obs, reward, terminated, truncated, info


class DurationDiscountWrapper(gym.Wrapper):
    """Add ``gamma_t = gamma ** (duration / dt0)`` and ``truncated_cutoff`` to ``info``. Reward is unchanged.

    The duration comes from ``info[duration_key]`` if present, else from ``duration_fn(action, info)``.
    Use this when the environment already simulates variable-duration transitions (for example, travel time),
    and its reward is already the reward of the whole transition.
    """

    def __init__(
        self,
        env: gym.Env,
        gamma: float = 0.99,
        dt0: float = 1.0,
        duration_key: str = "duration",
        duration_fn: Optional[Callable[[object, dict], float]] = None,
    ):
        super().__init__(env)
        if dt0 <= 0:
            raise ValueError("dt0 must be positive")
        self.gamma, self.dt0, self.duration_key, self.duration_fn = float(gamma), float(dt0), duration_key, duration_fn

    def step(self, action):
        obs, r, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        if self.duration_key in info:
            dt = float(info[self.duration_key])
        elif self.duration_fn is not None:
            dt = float(self.duration_fn(action, info))
        else:
            raise KeyError(f"no '{self.duration_key}' in info and no duration_fn given")
        if dt < 0:
            raise ValueError("duration must be non-negative")
        info.update(gamma_t=self.gamma ** (dt / self.dt0), truncated_cutoff=bool(truncated and not terminated))
        info.setdefault(self.duration_key, dt)
        return obs, r, terminated, truncated, info
