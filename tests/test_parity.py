"""Duration-1 parity with stock sb3-contrib MaskablePPO, and seed reproducibility."""

import numpy as np
from conftest import collect, maxdiff, params
from stable_baselines3.common.utils import set_random_seed
from train import build


def test_duration1_parity_with_stock_maskableppo(report):
    m1, m2 = build("A1", 0, n_envs=4, n_steps=256), build("A2", 0, n_envs=4, n_steps=256)
    assert maxdiff(params(m1), params(m2)) == 0.0
    # Both models sample from the global torch RNG: reseed before each one so they see the same random stream.
    set_random_seed(0)
    s1, (b1,) = collect(m1)
    set_random_seed(0)
    s2, (b2,) = collect(m2)
    assert np.array_equal(b1.observations, b2.observations) and np.array_equal(b1.actions, b2.actions)
    assert np.array_equal(np.array(s1.raw_r), np.array(s2.raw_r)) and np.array_equal(
        np.array(s1.dones), np.array(s2.dones)
    )
    n_trunc = int(sum(i.get("TimeLimit.truncated", False) for row in s1.infos for i in row))
    assert n_trunc > 0
    d = dict(
        rewards=float(np.abs(b1.rewards - b2.rewards).max()),
        adv=float(np.abs(b1.advantages - b2.advantages).max()),
        ret=float(np.abs(b1.returns - b2.returns).max()),
    )
    # one update on identical batches with the same minibatch order
    m1.rollout_buffer, m2.rollout_buffer = b1, b2
    p1, p2 = params(m1), params(m2)
    np.random.seed(123)
    set_random_seed(123)
    m1.train()
    l1 = dict(m1.logger.name_to_value)
    np.random.seed(123)
    set_random_seed(123)
    m2.train()
    l2 = dict(m2.logger.name_to_value)
    keys = ["train/policy_gradient_loss", "train/value_loss", "train/entropy_loss", "train/loss"]
    d["losses"] = max(abs(l1[k] - l2[k]) for k in keys)
    d["update"] = maxdiff({k: params(m1)[k] - p1[k] for k in p1}, {k: params(m2)[k] - p2[k] for k in p2})
    # full learn() for 2 rollouts
    f1, f2 = build("A1", 1, n_envs=4), build("A2", 1, n_envs=4)
    n = 2 * f1.n_steps * f1.n_envs
    set_random_seed(1)
    f1.learn(n, use_masking=False)
    set_random_seed(1)
    f2.learn(n, use_masking=False)
    d["weights_after_learn"] = maxdiff(params(f1), params(f2))
    assert all(v < 1e-5 for v in d.values()), d
    report["parity_duration1"] = dict(truncations=n_trunc, max_abs_diff=d)


def test_seed_reproducibility(report):
    w = []
    for _ in range(2):
        m = build("A3", 11, n_envs=4)
        m.learn(2_000, use_masking=False)
        w.append(params(m))
    other = build("A3", 12, n_envs=4)
    other.learn(2_000, use_masking=False)
    same, diff = maxdiff(w[0], w[1]), maxdiff(w[0], params(other))
    assert same == 0.0 and diff > 0.0
    report["seed_reproducibility"] = dict(same_seed_max_diff=same, different_seed_max_diff=diff)
