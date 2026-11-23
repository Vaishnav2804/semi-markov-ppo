"""Hand-calculated macro-action rewards and per-transition discounts; wrapper unit tests."""

import gymnasium as gym
import numpy as np
import pytest
from conftest import C1, C4, L1, L2, L4, R1, R2, R4
from environment import GAMMA, make_env, simulate

from smppo import DurationDiscountWrapper, MacroActionWrapper

SEQ = [R4, R4, L2, C1, R4, L4, L4, R2, C4, R1, R4, R4, L1, R4]


def test_deterministic_replay_and_hand_discounting(report):
    """Fixed macro sequence vs a hand simulation of MountainCar physics: states, k, R and gamma_t."""
    env = make_env("A3")
    env.reset(seed=123)
    macro = env.env  # the MacroActionWrapper under EpisodeStats
    s = tuple(float(x) for x in env.unwrapped.state)
    max_r = max_g = 0.0
    for a in SEQ:
        prim, kmax = macro.decode(a)
        hand_k, done = 0, False
        for _ in range(kmax):
            s, done = simulate(s, prim)
            hand_k += 1
            if done:
                break
        obs, R, term, trunc, info = env.step(a)
        assert info["k"] == hand_k and term == done
        assert tuple(env.unwrapped.state) == s
        assert np.array_equal(obs, np.array(s, np.float32))
        hand_R = sum(GAMMA**j * -1.0 for j in range(hand_k))
        max_r = max(max_r, abs(R - hand_R))
        max_g = max(max_g, abs(info["gamma_t"] - GAMMA**hand_k))
        if term or trunc:
            break
    assert max_r < 1e-12 and max_g < 1e-12
    report["replay_hand_discount"] = dict(macros=len(SEQ), max_abs_err_R=max_r, max_abs_err_gamma=max_g)


class _Const(gym.Env):
    """Reward j+1 on the j-th primitive step; never ends unless cap reached."""

    observation_space = gym.spaces.Box(-1, 1, (1,), np.float32)
    action_space = gym.spaces.Discrete(2)

    def __init__(self, cap=100):
        self.cap = cap

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return np.zeros(1, np.float32), {}

    def step(self, a):
        self.t += 1
        return np.zeros(1, np.float32), float(self.t), False, self.t >= self.cap, {}


def test_macro_wrapper_accumulates_with_nonconstant_rewards():
    env = MacroActionWrapper(_Const(), durations=(1, 3), gamma=0.9)
    env.reset()
    assert env.action_space.n == 4 and env.decode(3) == (1, 3)
    _, R, _, _, info = env.step(3)  # primitive rewards 1, 2, 3
    assert info["k"] == 3 and info["prim_rewards"] == [1.0, 2.0, 3.0]
    assert abs(R - (1 + 0.9 * 2 + 0.81 * 3)) < 1e-12 and abs(info["gamma_t"] - 0.9**3) < 1e-12


def test_macro_wrapper_stops_at_truncation():
    env = MacroActionWrapper(_Const(cap=2), durations=(4,), gamma=0.9)
    env.reset()
    _, R, term, trunc, info = env.step(0)
    assert info["k"] == 2 and trunc and not term and info["truncated_cutoff"]
    assert abs(R - (1 + 0.9 * 2)) < 1e-12


def test_macro_wrapper_validation():
    with pytest.raises(TypeError):
        MacroActionWrapper(gym.make("MountainCarContinuous-v0"))
    with pytest.raises(ValueError):
        MacroActionWrapper(_Const(), durations=(0, 2))


def test_duration_discount_wrapper():
    env = DurationDiscountWrapper(_Const(cap=3), gamma=0.99, dt0=0.5, duration_fn=lambda a, info: 0.5 * (a + 1))
    env.reset()
    _, r, _, _, info = env.step(1)  # duration 1.0 = 2 * dt0
    assert r == 1.0 and abs(info["gamma_t"] - 0.99**2) < 1e-12 and info["duration"] == 1.0
    env.step(0)
    _, _, term, trunc, info = env.step(0)
    assert trunc and info["truncated_cutoff"]
    with pytest.raises(KeyError):
        e2 = DurationDiscountWrapper(_Const())
        e2.reset()
        e2.step(0)
