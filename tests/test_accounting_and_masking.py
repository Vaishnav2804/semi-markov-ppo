"""Primitive-step accounting, primitive budget, and action-mask behaviour."""

import numpy as np
import pytest
from conftest import L4, R4, Chain, collect
from environment import make_env
from sb3_contrib.common.maskable.utils import is_masking_supported
from stable_baselines3.common.vec_env import DummyVecEnv
from train import PrimitiveBudget, build

from smppo import SemiMarkovMaskablePPO


def test_primitive_step_accounting(report):
    rng = np.random.default_rng(0)
    n_succ = n_trunc = 0
    for ep in range(50):
        env = make_env("A3")
        obs, _ = env.reset(seed=1000 + ep)
        ks, done = [], False
        while not done:
            a = (R4 if obs[1] >= 0 else L4) if ep % 2 else int(rng.integers(9))  # odd: energy pumping reaches goal
            obs, _, te, tr, info = env.step(a)
            ks.append(info["k"])
            done = te or tr
        e = info["ep"]
        assert sum(ks) == e["prim_steps"] <= 200 and e["decisions"] == len(ks)
        if te:
            n_succ += 1
            assert e["prim_steps"] < 200 and env.unwrapped.state[0] >= 0.5 and e["success"]
        else:
            n_trunc += 1
            assert e["prim_steps"] == 200 and e["truncated"]
    assert n_succ > 0 and n_trunc > 0
    report["primitive_accounting"] = dict(episodes=50, successes=n_succ, truncations=n_trunc)


@pytest.mark.parametrize("arm,kmax", [("A1", 1), ("A3", 4)])
def test_primitive_budget(arm, kmax, report):
    B = 2000
    m = build(arm, 8)
    bud = PrimitiveBudget(B, native=arm == "A1")
    m.learn(5_000_000, callback=bud, use_masking=False)
    assert bud.stopped_by_budget and B <= bud.prim < B + m.n_envs * kmax
    assert bud.prim == m.num_timesteps if arm == "A1" else bud.prim > m.num_timesteps
    report[f"primitive_budget_{arm}"] = dict(prim_final=bud.prim, decisions=m.num_timesteps, updates=m._n_updates)


@pytest.mark.parametrize("arm,n_act", [("A1", 3), ("A2", 3), ("A3", 9)])
def test_no_accidental_masking(arm, n_act, report):
    m = build(arm, 7, n_envs=4, n_steps=2500)
    assert not is_masking_supported(m.env)
    _, (b,) = collect(m)
    assert np.all(b.action_masks == 1.0)
    counts = np.bincount(b.actions.astype(int).ravel(), minlength=n_act)
    assert counts.size == n_act and counts.min() > 0
    report[f"no_masking_{arm}"] = dict(decisions=int(b.actions.size), action_counts=counts.tolist())


def test_masked_policy_never_selects_invalid_and_learns():
    env = DummyVecEnv([lambda: Chain(cap=4)])
    m = SemiMarkovMaskablePPO("MlpPolicy", env, n_steps=64, batch_size=32, n_epochs=2, seed=0)
    before = [p.detach().clone() for p in m.policy.parameters()]
    m.learn(512)  # Chain.step asserts that the chosen action is unmasked
    assert any(not (a == p).all() for a, p in zip(before, m.policy.parameters()))
    b = m.rollout_buffer  # arrays are flattened to (n_steps * n_envs, ...) after training
    at_start = b.observations.reshape(-1, 2)[:, 0] == 0
    assert at_start.any() and np.all(b.action_masks.reshape(-1, 2)[at_start, 0] == 0)  # action 0 masked at position 0
