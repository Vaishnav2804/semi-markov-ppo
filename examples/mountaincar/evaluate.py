"""Evaluate a saved run on the 100 fixed starts, or summarise many runs into tables and figures.

python examples/mountaincar/evaluate.py model --run runs/A3_s0 --arm A3
python examples/mountaincar/evaluate.py summary --runs runs --out results/mountaincar
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from environment import ARMS, make_env  # noqa: E402,F401
from train import EVAL_SEEDS, evaluate  # noqa: E402

LABEL = {
    "A1": "A1 stock MaskablePPO, primitive",
    "A2": "A2 SM-PPO, durations {1}",
    "A3": "A3 SM-PPO, durations {1,2,4}",
}
COL = {"A1": "#1b9e77", "A2": "#7570b3", "A3": "#d95f02"}
LS = {"A1": "-", "A2": "--", "A3": ":"}


def eval_model(run: Path, arm: str):
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from smppo import SemiMarkovMaskablePPO

    cls = MaskablePPO if ARMS[arm] is None else SemiMarkovMaskablePPO
    model = cls.load(run / "final_model.zip", device="cpu")
    vn = VecNormalize.load(str(run / "vecnormalize.pkl"), DummyVecEnv([lambda: make_env(arm)]))
    vn.training = False
    ev = evaluate(model, arm, EVAL_SEEDS, vecnorm=vn)
    print(ev[["success", "prim_steps", "disc_return_prim", "decisions"]].mean().to_string())


def summary(runs: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    rows, curves, evals = [], [], []
    for d in sorted(runs.glob("A?_s*")):
        if not (d / "runtime.json").exists():
            continue
        arm, seed = d.name.split("_s")
        rt = json.loads((d / "runtime.json").read_text())
        ev, cu = pd.read_csv(d / "eval.csv"), pd.read_csv(d / "curve.csv")
        succ = ev[ev.success]
        rows.append(
            dict(
                arm=arm,
                seed=int(seed),
                success=ev.success.mean(),
                steps_to_goal_median=succ.prim_steps.median() if len(succ) else np.nan,
                disc_return_prim=ev.disc_return_prim.mean(),
                undisc_return=ev.undisc_return.mean(),
                ep_prim_steps=ev.prim_steps.mean(),
                decisions_per_ep=ev.decisions.mean(),
                wall_min=rt["wall_s"] / 60,
                maxrss_MB=rt["maxrss_MB"],
                updates=rt["updates"],
                decisions=rt["decisions"],
                prim_final=rt["prim_steps_final"],
                overshoot=rt["overshoot"],
                stopped_by_budget=rt["stopped_by_budget"],
            )
        )
        curves.append(cu.assign(arm=arm, seed=int(seed)))
        evals.append(ev.assign(arm=arm, seed=int(seed)))
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(out / "summary_per_seed.csv", index=False)
    num = [
        "success",
        "steps_to_goal_median",
        "disc_return_prim",
        "ep_prim_steps",
        "decisions_per_ep",
        "wall_min",
        "updates",
    ]
    per_seed.groupby("arm")[num].agg(["mean", "std"]).round(3).to_csv(out / "summary_by_arm.csv")
    print(per_seed.round(3).to_string(index=False))

    cv, ev = pd.concat(curves), pd.concat(evals)
    n_seeds = per_seed.seed.nunique()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for arm in LABEL:
        c = cv[cv.arm == arm]
        for _, g in c.groupby("seed"):
            ax[0].plot(g.prim_total / 1e6, g.success, LS[arm], color=COL[arm], alpha=0.3, lw=1)
            ax[1].plot(g.prim_total / 1e6, g.disc_return_prim, LS[arm], color=COL[arm], alpha=0.3, lw=1)
        m = c.assign(b=(c.prim_total // 50_000) * 50_000).groupby("b")[["success", "disc_return_prim"]].mean()
        ax[0].plot(m.index / 1e6, m.success, LS[arm], color=COL[arm], lw=2.5, label=LABEL[arm])
        ax[1].plot(m.index / 1e6, m.disc_return_prim, LS[arm], color=COL[arm], lw=2.5, label=LABEL[arm])
    ax[0].set_ylabel("success rate (20 fixed eval starts)")
    ax[1].set_ylabel("discounted return, primitive time (γ=0.99)")
    for x in ax:
        x.set_xlabel("primitive environment steps (millions)")
        x.grid(alpha=0.3)
    ax[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(
        f"MountainCar-v0 learning curves. Thin: individual seeds (n={n_seeds}); thick: mean over seeds per 50k-step bin"
    )
    fig.tight_layout()
    fig.savefig(out / "learning_curves.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    rng = np.random.default_rng(0)
    for i, arm in enumerate(LABEL):
        e = ev[ev.arm == arm]
        for j, s in enumerate(sorted(e.seed.unique())):
            g = e[e.seed == s]
            x = i + (j - 1) * 0.22 + rng.uniform(-0.06, 0.06, len(g))
            ax.scatter(x, g.prim_steps, s=9, color=COL[arm], marker="o^s"[j % 3], alpha=0.6)
    ax.set_xticks(range(len(LABEL)), [LABEL[k].split(",")[0] for k in LABEL])
    ax.set_ylabel("episode length, primitive steps (200 = timeout)")
    ax.grid(alpha=0.3)
    ax.set_title(
        "Final evaluation, 100 fixed starts per run\npoints: episodes; offset and marker = seed (o 0, ^ 1, s 2)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out / "eval_distributions.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("model")
    m.add_argument("--run", type=Path, required=True)
    m.add_argument("--arm", choices=list(ARMS), required=True)
    s = sub.add_parser("summary")
    s.add_argument("--runs", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    eval_model(a.run, a.arm) if a.cmd == "model" else summary(a.runs, a.out)
