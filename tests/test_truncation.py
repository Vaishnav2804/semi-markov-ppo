"""Termination vs truncation inside macro-actions; exactly-once bootstrapping from the real terminal observation."""

import gymnasium as gym
import numpy as np
from conftest import C1, L4, R4, Chain, collect, sm_model, value
from environment import ENV_ID, GAMMA, make_env, simulate
from stable_baselines3.common.vec_env import DummyVecEnv

from smppo import MacroActionWrapper, SemiMarkovMaskablePPO


def _near_goal_state():
    for p in np.arange(0.40, 0.50, 0.001):
        s = (float(p), 0.03)
        s1, t1 = simulate(s, 2)
        _, t2 = simulate(s1, 2)
        if not t1 and t2:
            return s
    raise RuntimeError("no state found")


NEAR_GOAL = _near_goal_state()


class StartNearGoal(gym.Wrapper):
    """TEST ONLY: after reset, place the car where holding 'right' reaches the goal on primitive 2."""

    def reset(self, **kw):
        _, info = self.env.reset(**kw)
        self.env.unwrapped.state = np.array(NEAR_GOAL, dtype=np.float64)
        return np.array(NEAR_GOAL, np.float32), info


def _a3(inner):
    return MacroActionWrapper(inner, (1, 2, 4), GAMMA)


def test_termination_mid_macro(report):
    env = _a3(StartNearGoal(gym.make(ENV_ID)))
    env.reset(seed=0)
    _, R, term, trunc, info = env.step(R4)
    assert info["k"] == 2 and term and not trunc and not info["truncated_cutoff"]
    assert abs(R - (-1 - GAMMA)) < 1e-12 and abs(info["gamma_t"] - GAMMA**2) < 1e-12
    m = sm_model([lambda: _a3(StartNearGoal(gym.make(ENV_ID))) for _ in range(2)], seed=3, n_steps=64)
    spy, (b,) = collect(m)
    n_term = 0
    for t, row in enumerate(spy.infos):
        for e, inf in enumerate(row):
            if spy.dones[t][e] and not inf["truncated_cutoff"]:
                n_term += 1
                assert np.isclose(b.rewards[t, e], spy.raw_r[t][e], atol=1e-6)  # no bootstrap
                assert b.ends[t, e] == 1
                assert np.isclose(b.advantages[t, e], b.rewards[t, e] - b.values[t, e], atol=1e-6)
    assert n_term > 0 and m.n_bootstraps == 0
    report["termination_mid_macro"] = dict(start_state=NEAR_GOAL, terminations=n_term)


def test_truncation_mid_macro_bootstraps_once(report):
    env = make_env("A3")
    env.reset(seed=5)
    for _ in range(198):
        _, _, te, tr, _ = env.step(C1)
        assert not (te or tr)
    _, R, term, trunc, info = env.step(L4)
    assert info["k"] == 2 and trunc and not term and info["truncated_cutoff"] and info["ep"]["prim_steps"] == 200
    assert abs(R - (-1 - GAMMA)) < 1e-12
    m = sm_model([lambda: _a3(gym.make(ENV_ID, max_episode_steps=8)) for _ in range(4)], seed=4, n_steps=64)
    spy, (b,) = collect(m)
    n_boot = n_mid = 0
    for t, row in enumerate(spy.infos):
        for e, inf in enumerate(row):
            if inf["truncated_cutoff"]:
                n_boot += 1
                n_mid += inf["k"] < (1, 2, 4)[int(spy.actions[t][e]) % 3]
                exp = spy.raw_r[t][e] + inf["gamma_t"] * value(m, inf["terminal_observation"][None])
                assert np.isclose(b.rewards[t, e], exp, atol=1e-5) and b.ends[t, e] == 1
            else:
                assert np.isclose(b.rewards[t, e], spy.raw_r[t][e], atol=1e-6)
    assert n_boot == m.n_bootstraps > 0 and n_mid > 0
    report["truncation_mid_macro"] = dict(bootstraps=n_boot, truncations_mid_macro=n_mid)


class LastRaw(gym.Wrapper):
    """TEST ONLY: remembers the raw observation of the final step of each episode."""

    def step(self, a):
        o, r, te, tr, i = self.env.step(a)
        if te or tr:
            i = dict(i)
            i["_raw_last"] = np.array(o, np.float32)
        return o, r, te, tr, i


def test_terminal_observation_is_pre_reset_state(report):
    m = sm_model([lambda: LastRaw(_a3(gym.make(ENV_ID, max_episode_steps=8))) for _ in range(4)], seed=6, n_steps=64)
    spy, _ = collect(m)
    eps, n = m.get_env().epsilon, 0
    for t, row in enumerate(spy.infos):
        for inf in row:
            if inf["truncated_cutoff"]:
                n += 1
                mean, var = spy.rms[t]
                raw = inf["terminal_observation"] * np.sqrt(var + eps) + mean
                assert np.allclose(raw, inf["_raw_last"], atol=1e-4)
                assert inf["_raw_last"][1] != 0.0  # reset velocity is exactly 0
    assert n == m.n_bootstraps > 0
    report["terminal_observation"] = dict(truncations_checked=n)


def test_chain_truncation_single_bootstrap_with_masking():
    """Masked env, alternating gamma_t: one gamma_t * V(real terminal obs) bootstrap; goal steps not bootstrapped."""
    from stable_baselines3.common.callbacks import CallbackList as CL

    env = DummyVecEnv([lambda: Chain(cap=4)])
    m = SemiMarkovMaskablePPO(
        "MlpPolicy", env, n_steps=40, batch_size=20, n_epochs=1, seed=0, gamma=0.99, gae_lambda=0.95
    )
    m._setup_learn(40, None)
    cb = CL([])
    cb.init_callback(m)
    seen, orig = [], env.step_wait

    def spy():
        o, r, d, i = orig()
        seen.append([dict(x) for x in i])
        return o, r, d, i

    env.step_wait = spy
    m.collect_rollouts(env, cb, m.rollout_buffer, 40)
    b, n_trunc = m.rollout_buffer, 0
    for t, inf in enumerate(seen):
        inf = inf[0]
        if inf.get("truncated_cutoff"):
            n_trunc += 1
            assert inf["terminal_observation"][1] == 4
            v = value(m, inf["terminal_observation"])
            assert np.isclose(b.rewards[t, 0], 1.0 + inf["gamma_t"] * v, atol=1e-5) and b.ends[t, 0] == 1
        else:
            assert np.isclose(b.rewards[t, 0], 1.0)
    assert n_trunc == m.n_bootstraps and n_trunc > 0
