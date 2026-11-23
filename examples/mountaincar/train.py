"""Train one MountainCar arm x one seed with a primitive-step budget.

python examples/mountaincar/train.py --arm A3 --seed 0 --prim_steps 1000000 --out runs/A3_s0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

import pandas as pd
import torch as th
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from environment import ARMS, GAMMA, make_env  # noqa: E402

from smppo import SemiMarkovMaskablePPO  # noqa: E402

N_ENVS = 16
# RL Zoo MountainCar-v0 PPO settings (normalize: true, n_envs 16, n_steps 16, gae_lambda 0.98, gamma 0.99,
# n_epochs 4, ent_coef 0) with SB3 defaults otherwise, EXCEPT reward normalization, which is off in every arm:
# VecNormalize tracks returns with one scalar gamma, which is inconsistent with per-transition gamma_t.
PPO_KW = dict(
    n_steps=16,
    batch_size=64,
    n_epochs=4,
    learning_rate=3e-4,
    gamma=GAMMA,
    gae_lambda=0.98,
    clip_range=0.2,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    normalize_advantage=True,
    device="cpu",
    verbose=0,
    policy_kwargs=dict(net_arch=[64, 64]),
)
EVAL_SEEDS = list(range(10_000, 10_100))
CURVE_SEEDS = EVAL_SEEDS[:20]
DECISION_CAP = 5_000_000


def build(arm: str, seed: int, n_envs: int = N_ENVS, **overrides):
    venv = VecNormalize(
        DummyVecEnv([lambda: make_env(arm) for _ in range(n_envs)]), norm_obs=True, norm_reward=False, gamma=GAMMA
    )
    cls = MaskablePPO if ARMS[arm] is None else SemiMarkovMaskablePPO
    return cls("MlpPolicy", venv, seed=seed, **dict(PPO_KW, **overrides))


class PrimitiveBudget(BaseCallback):
    """Stop when executed primitive steps (info['k'], or 1 for a native env) reach the budget."""

    def __init__(self, budget: int, native: bool):
        super().__init__()
        self.budget, self.native, self.prim, self.stopped_by_budget = int(budget), bool(native), 0, False

    def _on_step(self) -> bool:
        for inf in self.locals["infos"]:
            if self.native:
                assert "k" not in inf
                self.prim += 1
            else:
                self.prim += int(inf["k"])
        if self.prim >= self.budget:
            self.stopped_by_budget = True
            return False
        return True


def evaluate(model, arm: str, seeds, vecnorm=None) -> pd.DataFrame:
    """Deterministic policy; observation-normalisation statistics frozen."""
    vn = vecnorm or model.get_env()
    rows = []
    for s in seeds:
        env = make_env(arm)
        obs, _ = env.reset(seed=s)
        done = False
        while not done:
            a, _ = model.predict(vn.normalize_obs(obs), deterministic=True)
            obs, _, te, tr, info = env.step(int(a))
            done = te or tr
        rows.append(dict(info["ep"], eval_seed=s))
    return pd.DataFrame(rows)


class Logger(BaseCallback):
    def __init__(self, budget_cb: PrimitiveBudget, arm: str, out: Path, curve_every=50_000, ckpt_every=100_000):
        super().__init__()
        self.b, self.arm, self.out = budget_cb, arm, out
        self.curve_every, self.ckpt_every = curve_every, ckpt_every
        self.next_curve, self.next_ckpt = curve_every, ckpt_every
        self.eps, self.curve = [], []

    def _on_step(self) -> bool:
        for inf in self.locals["infos"]:
            if "ep" in inf:
                self.eps.append(dict(inf["ep"], prim_total=self.b.prim, decisions_total=self.num_timesteps))
        if self.b.prim >= self.next_curve:
            ev = evaluate(self.model, self.arm, CURVE_SEEDS)
            self.curve.append(
                dict(
                    prim_total=self.b.prim,
                    updates=self.model._n_updates,
                    success=ev.success.mean(),
                    disc_return_prim=ev.disc_return_prim.mean(),
                    prim_steps=ev.prim_steps.mean(),
                )
            )
            self.next_curve += self.curve_every
        if self.b.prim >= self.next_ckpt:
            self.model.save(self.out / f"ckpt_{self.next_ckpt}.zip")
            self.next_ckpt += self.ckpt_every
        return True


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(ARMS), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--prim_steps", type=int, default=1_000_000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    th.set_num_threads(1)
    import smppo

    here = Path(__file__).resolve().parent
    cfg = dict(
        arm=a.arm,
        seed=a.seed,
        prim_steps=a.prim_steps,
        n_envs=N_ENVS,
        durations=ARMS[a.arm],
        ppo={k: str(v) for k, v in PPO_KW.items()},
        vecnormalize="norm_obs=True, norm_reward=False",
        eval_seeds=[EVAL_SEEDS[0], EVAL_SEEDS[-1]],
        sha256={
            "smppo/ppo.py": sha(Path(smppo.__file__).parent / "ppo.py"),
            "smppo/wrappers.py": sha(Path(smppo.__file__).parent / "wrappers.py"),
            "environment.py": sha(here / "environment.py"),
            "train.py": sha(here / "train.py"),
        },
    )
    (out / "config.json").write_text(json.dumps(cfg, indent=1))

    model = build(a.arm, a.seed)
    bud = PrimitiveBudget(a.prim_steps, native=ARMS[a.arm] is None)
    log = Logger(bud, a.arm, out)
    t0 = time.time()
    model.learn(DECISION_CAP, callback=[bud, log], use_masking=False)
    wall = time.time() - t0

    model.save(out / "final_model.zip")
    model.get_env().save(str(out / "vecnormalize.pkl"))
    ev = evaluate(model, a.arm, EVAL_SEEDS)
    ev.to_csv(out / "eval.csv", index=False)
    pd.DataFrame(log.eps).to_csv(out / "train_episodes.csv", index=False)
    pd.DataFrame(log.curve).to_csv(out / "curve.csv", index=False)
    rt = dict(
        wall_s=round(wall, 1),
        prim_steps_final=bud.prim,
        overshoot=bud.prim - a.prim_steps,
        stopped_by_budget=bud.stopped_by_budget,
        decisions=model.num_timesteps,
        updates=model._n_updates,
        bootstraps=getattr(model, "n_bootstraps", None),
        maxrss_MB=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
        eval_success=float(ev.success.mean()),
    )
    (out / "runtime.json").write_text(json.dumps(rt, indent=1))
    print(json.dumps(rt))


if __name__ == "__main__":
    main()
