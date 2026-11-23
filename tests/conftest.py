"""Shared helpers. Edge-case environments built in tests (short time limits, near-goal starts) are TEST ONLY."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "mountaincar"))
th.set_num_threads(1)

from train import GAMMA, PPO_KW  # noqa: E402

from smppo import SemiMarkovMaskablePPO  # noqa: E402

L1, L2, L4, C1, C2, C4, R1, R2, R4 = range(9)  # MountainCar A3 actions: primitive*3 + duration index


class Spy:
    """Records what the outermost vec env returns (raw rewards before any bootstrapping) and its obs statistics."""

    def __init__(self, venv):
        self.infos, self.raw_r, self.dones, self.actions, self.rms = [], [], [], [], []
        oa, ow = venv.step_async, venv.step_wait

        def step_async(a):
            self.actions.append(np.array(a).copy())
            return oa(a)

        def step_wait():
            o, r, d, i = ow()
            rms = getattr(venv, "obs_rms", None)
            self.rms.append(None if rms is None else (rms.mean.copy(), rms.var.copy()))
            self.infos.append([dict(x) for x in i])
            self.raw_r.append(np.array(r, np.float64).copy())
            self.dones.append(d.copy())
            return o, r, d, i

        venv.step_async, venv.step_wait = step_async, step_wait


def collect(model, n_rollouts=1):
    """Collect rollouts with masking off, without training. Returns the spy and a deep copy of each buffer."""
    model._setup_learn(10**9, None)
    cb = CallbackList([])
    cb.init_callback(model)
    spy, bufs = Spy(model.env), []
    for _ in range(n_rollouts):
        assert model.collect_rollouts(model.env, cb, model.rollout_buffer, model.n_steps, use_masking=False)
        bufs.append(copy.deepcopy(model.rollout_buffer))
    return spy, bufs


def sm_model(env_fns, seed=0, **ppo):
    venv = VecNormalize(DummyVecEnv(env_fns), norm_obs=True, norm_reward=False, gamma=GAMMA)
    return SemiMarkovMaskablePPO("MlpPolicy", venv, seed=seed, **dict(PPO_KW, **ppo))


def ref_gae(r, v, last_v, g, end, lam):
    """Hand-written semi-Markov GAE, the reference every buffer is compared against."""
    A = np.zeros(len(r), dtype=np.float64)
    adv = 0.0
    for t in reversed(range(len(r))):
        nxt = last_v if t == len(r) - 1 else v[t + 1]
        delta = r[t] + g[t] * (1 - end[t]) * nxt - v[t]
        adv = delta + g[t] * lam * (1 - end[t]) * adv
        A[t] = adv
    return A


def value(model, obs):
    with th.no_grad():
        return float(model.policy.predict_values(model.policy.obs_to_tensor(obs)[0])[0])


def params(model):
    return {k: v.detach().clone() for k, v in model.policy.state_dict().items()}


def maxdiff(a, b):
    return max(float((a[k] - b[k]).abs().max()) for k in a)


class Chain(gym.Env):
    """Tiny masked env: obs [position, t]; goal at position 3 via action 1; action 0 stays (masked at position 0).
    Truncates after `cap` steps. Reward 1. gamma_t alternates 0.95 / 0.99."""

    def __init__(self, cap=4):
        self.observation_space = gym.spaces.Box(-10, 10, (2,), np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self.cap = cap

    def action_masks(self):
        return np.array([self.pos > 0, True])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.pos, self.t = 0, 0
        return np.array([0, 0], np.float32), {}

    def step(self, a):
        assert self.action_masks()[a], "masked action selected"
        self.pos += int(a)
        self.t += 1
        term = self.pos >= 3
        trunc = (self.t >= self.cap) and not term
        g = 0.95 if self.t % 2 else 0.99
        return np.array([self.pos, self.t], np.float32), 1.0, term, trunc, {"gamma_t": g, "truncated_cutoff": trunc}


# ---------------------------------------------- measured values, written to results/test_report.json
import json  # noqa: E402

import pytest  # noqa: E402

_REPORT: dict = {}


@pytest.fixture
def report():
    return _REPORT


def pytest_sessionfinish(session, exitstatus):
    if _REPORT:
        out = Path(__file__).resolve().parents[1] / "results" / "test_report.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(dict(sorted(_REPORT.items())), indent=1, default=float))
