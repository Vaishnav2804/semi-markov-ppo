"""MountainCar-v0 benchmark arms.

A1: native MountainCar-v0, 3 primitive actions (stock MaskablePPO).
A2: MacroActionWrapper with durations (1,), so Discrete(3), identical to A1 (SemiMarkovMaskablePPO).
A3: MacroActionWrapper with durations (1, 2, 4), so Discrete(9): left/coast/right x 1, 2, 4.

gym.make("MountainCar-v0") already contains TimeLimit(200), so the limit counts primitive steps.
EpisodeStats only records per-episode summaries in info["ep"]; it changes nothing else.
"""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np

from smppo import MacroActionWrapper

ENV_ID = "MountainCar-v0"
GAMMA = 0.99
ARMS = {"A1": None, "A2": (1,), "A3": (1, 2, 4)}


class EpisodeStats(gym.Wrapper):
    """info["ep"] at episode end: prim_steps, decisions, discounted return in primitive time, success."""

    def __init__(self, env: gym.Env, gamma: float = GAMMA):
        super().__init__(env)
        self.gamma = float(gamma)

    def reset(self, **kw):
        self._ep = dict(prim_steps=0, disc_return_prim=0.0, undisc_return=0.0, decisions=0)
        return self.env.reset(**kw)

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        e = self._ep
        for pr in info.get("prim_rewards", [r]):
            e["disc_return_prim"] += (self.gamma ** e["prim_steps"]) * float(pr)
            e["undisc_return"] += float(pr)
            e["prim_steps"] += 1
        e["decisions"] += 1
        if term or trunc:
            info = dict(info)
            info["ep"] = dict(e, success=bool(term), truncated=bool(trunc and not term))
        return obs, r, term, trunc, info


def make_env(arm: str, gamma: float = GAMMA, **make_kw) -> gym.Env:
    env = gym.make(ENV_ID, **make_kw)
    if ARMS[arm] is not None:
        env = MacroActionWrapper(env, ARMS[arm], gamma)
    return EpisodeStats(env, gamma)


def simulate(state, prim):
    """Hand re-implementation of MountainCarEnv.step (gymnasium 1.2.3), used by tests."""
    position, velocity = state
    velocity += (prim - 1) * 0.001 + math.cos(3 * position) * (-0.0025)
    velocity = np.clip(velocity, -0.07, 0.07)
    position += velocity
    position = np.clip(position, -1.2, 0.6)
    if position == -1.2 and velocity < 0:
        velocity = 0
    return (position, velocity), bool(position >= 0.5 and velocity >= 0)
