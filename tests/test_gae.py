"""Variable-duration GAE against a hand-written reference."""

import copy

import gymnasium as gym
import numpy as np
import torch as th
from conftest import collect, ref_gae, sm_model, value
from environment import GAMMA
from stable_baselines3.common.buffers import RolloutBuffer
from train import build

from smppo import MacroActionWrapper, SemiMarkovRolloutBuffer


def _buf(T, cls=SemiMarkovRolloutBuffer, gamma=0.99, lam=0.9):
    b = cls(T, gym.spaces.Box(-1, 1, (1,)), gym.spaces.Discrete(2), n_envs=1, gamma=gamma, gae_lambda=lam)
    b.reset()
    return b


def test_synthetic_gae_matches_reference():
    rng = np.random.default_rng(0)
    T = 9
    r, v, g = rng.normal(size=T), rng.normal(size=T), rng.uniform(0.9, 1.0, T)
    end = np.zeros(T)
    end[3] = 1
    b = _buf(T)
    b.rewards[:, 0], b.values[:, 0], b.gammas[:, 0], b.ends[:, 0] = r, v, g, end
    b.compute_returns_and_advantage(th.tensor([0.7]), np.array([False]))
    assert np.allclose(b.advantages[:, 0], ref_gae(r, v, 0.7, g, end, 0.9), atol=1e-5)
    assert np.isclose(b.advantages[3, 0], r[3] - v[3], atol=1e-6)  # no future term after an episode end
    assert np.allclose(b.returns, b.advantages + b.values)


def test_constant_gamma_reduces_to_sb3_buffer():
    rng = np.random.default_rng(1)
    T = 8
    r, v = rng.normal(size=T), rng.normal(size=T)
    starts = np.zeros(T)
    starts[5] = 1
    a = _buf(T)
    a.rewards[:, 0], a.values[:, 0] = r, v
    a.gammas[:] = 0.99
    a.ends[:, 0] = np.r_[starts[1:], 0]
    s = _buf(T, RolloutBuffer)
    s.rewards[:, 0], s.values[:, 0], s.episode_starts[:, 0] = r, v, starts
    for b in (a, s):
        b.compute_returns_and_advantage(th.tensor([0.3]), np.array([False]))
    assert np.allclose(a.advantages, s.advantages, atol=1e-6)


def test_variable_duration_gae_on_real_rollout(report):
    m = build("A3", 2, n_envs=4, n_steps=64)
    spy, (b,) = collect(m)
    g_info = np.array([[i["gamma_t"] for i in row] for row in spy.infos], np.float32)
    assert np.array_equal(b.gammas, g_info)
    allowed = np.array([GAMMA**k for k in (1, 2, 4)], np.float32)
    assert all(np.isclose(g, allowed, rtol=0, atol=1e-7).any() for g in np.unique(b.gammas))
    assert len(np.unique(b.gammas)) > 1
    assert np.array_equal(b.ends, np.array(spy.dones, np.float32))
    lv = [value(m, m._last_obs[e : e + 1]) for e in range(b.n_envs)]
    d = max(
        float(
            np.abs(
                b.advantages[:, e]
                - ref_gae(b.rewards[:, e], b.values[:, e], lv[e], b.gammas[:, e], b.ends[:, e], b.gae_lambda)
            ).max()
        )
        for e in range(b.n_envs)
    )
    assert d < 1e-5
    report["gae_real_rollout"] = dict(decisions=int(b.gammas.size), max_abs_diff=d)


def test_episode_end_on_last_buffer_step(report):
    import gymnasium as gym_

    m = sm_model(
        [lambda: MacroActionWrapper(gym_.make("MountainCar-v0", max_episode_steps=8), (1,), GAMMA) for _ in range(3)],
        seed=5,
        n_steps=8,
    )
    spy, (b,) = collect(m)
    assert np.all(b.ends[-1] == 1) and all(i["truncated_cutoff"] for i in spy.infos[-1])
    assert np.allclose(b.advantages[-1], b.rewards[-1] - b.values[-1], atol=1e-6)
    b2 = copy.deepcopy(b)
    b2.compute_returns_and_advantage(th.tensor([123.0] * b2.n_envs), np.ones(b2.n_envs, bool))
    assert np.allclose(b2.advantages, b.advantages, atol=1e-6)  # last_values ignored after an end
    report["end_on_last_step"] = dict(envs=int(b.ends[-1].sum()))
