"""Side-by-side GIF: stock MaskablePPO (primitive actions) vs SemiMarkovMaskablePPO (macro-actions {1, 2, 4}).

Both agents start from the same reset seed. Frames are drawn on a shared clock of primitive environment steps, so
the two cars move at the true relative speed. The SMDP agent's current macro-action and its decision count are
overlaid. When an agent reaches the goal its last frame is held.

    python examples/mountaincar/make_gif.py --a1 runs/A1_s0 --a3 runs/A3_s0 --seed 10000 --out docs/mountaincar.gif

Needs trained runs (final_model.zip + vecnormalize.pkl from train.py) and pygame:
pip install "gymnasium[classic-control]".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gymnasium as gym
from PIL import Image, ImageDraw, ImageFont
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from environment import ENV_ID, GAMMA  # noqa: E402

from smppo import MacroActionWrapper, SemiMarkovMaskablePPO, SemiMarkovRolloutBuffer  # noqa: E402

PRIM = ("left", "coast", "right")


class FrameRecorder(gym.Wrapper):
    """Renders after every primitive step (sits below MacroActionWrapper, so macro-actions are drawn step by step)."""

    def __init__(self, env):
        super().__init__(env)
        self.frames, self.labels, self.label = [], [], ""

    def reset(self, **kw):
        out = self.env.reset(**kw)
        self.frames, self.labels = [self.env.render()], [self.label]
        return out

    def step(self, a):
        out = self.env.step(a)
        self.frames.append(self.env.render())
        self.labels.append(self.label)
        return out


def rollout(run: Path, smdp: bool, seed: int):
    base = FrameRecorder(gym.make(ENV_ID, render_mode="rgb_array"))
    env = MacroActionWrapper(base, (1, 2, 4), GAMMA) if smdp else base
    cls = SemiMarkovMaskablePPO if smdp else MaskablePPO
    # custom_objects: models saved before packaging reference the buffer class by its old module path
    extra = {"rollout_buffer_class": SemiMarkovRolloutBuffer} if smdp else {}
    model = cls.load(run / "final_model.zip", device="cpu", custom_objects=extra)
    vn = VecNormalize.load(str(run / "vecnormalize.pkl"), DummyVecEnv([lambda: gym.make(ENV_ID)]))
    vn.training = False
    obs, _ = env.reset(seed=seed)
    done, decisions = False, 0
    while not done:
        a, _ = model.predict(vn.normalize_obs(obs), deterministic=True)
        a = int(a)
        decisions += 1
        if smdp:
            prim, k = env.decode(a)
            base.label = f"decision {decisions}: {PRIM[prim]} x{k}"
        else:
            base.label = f"decision {decisions}: {PRIM[a]}"
        obs, _, te, tr, _ = env.step(a)
        done = te or tr
    return base.frames, base.labels, decisions, bool(te)


def panel(frame, title, sub, status, font, small, scale):
    img = Image.fromarray(frame).resize((int(frame.shape[1] * scale), int(frame.shape[0] * scale)), Image.LANCZOS)
    d = ImageDraw.Draw(img)
    d.text((8, 6), title, fill=(0, 0, 0), font=font)
    d.text((8, 30), sub, fill=(60, 60, 60), font=small)
    d.text((8, img.height - 24), status, fill=(0, 0, 0), font=small)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a1", type=Path, required=True, help="run dir of stock MaskablePPO (arm A1)")
    ap.add_argument("--a3", type=Path, required=True, help="run dir of SemiMarkovMaskablePPO, durations {1,2,4} (A3)")
    ap.add_argument("--seed", type=int, default=10000)
    ap.add_argument("--out", type=Path, default=Path("docs/mountaincar.gif"))
    ap.add_argument("--scale", type=float, default=0.6)
    ap.add_argument("--every", type=int, default=2, help="keep every n-th primitive step")
    ap.add_argument("--fps", type=int, default=25)
    a = ap.parse_args()

    f1, l1, d1, s1 = rollout(a.a1, False, a.seed)
    f3, l3, d3, s3 = rollout(a.a3, True, a.seed)
    n = max(len(f1), len(f3))
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 15)
        small = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        font = small = ImageFont.load_default()

    def status(t, frames, d, ok):
        if t < len(frames) - 1:
            return f"primitive step {t}"
        return f"{'GOAL' if ok else 'timeout'} at step {len(frames) - 1}, {d} decisions"

    out = []
    for t in list(range(0, n, a.every)) + [n - 1] * 12:  # hold the final frame
        i1, i3 = min(t, len(f1) - 1), min(t, len(f3) - 1)
        p1 = panel(
            f1[i1],
            "PPO (1 step per decision)",
            l1[i1] or "one primitive action per decision",
            status(t, f1, d1, s1),
            font,
            small,
            a.scale,
        )
        p3 = panel(
            f3[i3], "Semi-Markov PPO (hold 1, 2 or 4)", l3[i3] or "", status(t, f3, d3, s3), font, small, a.scale
        )
        canvas = Image.new("RGB", (p1.width + p3.width + 6, p1.height), (200, 200, 200))
        canvas.paste(p1, (0, 0))
        canvas.paste(p3, (p1.width + 6, 0))
        out.append(canvas.convert("P", palette=Image.ADAPTIVE, colors=64))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out[0].save(a.out, save_all=True, append_images=out[1:], duration=int(1000 / a.fps), loop=0, optimize=True)
    print(
        f"{a.out}: {len(out)} frames; PPO goal={s1} at {len(f1) - 1} steps ({d1} decisions); "
        f"SMDP goal={s3} at {len(f3) - 1} steps ({d3} decisions); {a.out.stat().st_size / 1e6:.2f} MB"
    )


if __name__ == "__main__":
    main()
