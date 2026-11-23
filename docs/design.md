# Design notes and provenance

## Provenance of the algorithm code

`src/smppo/ppo.py` is a byte-for-byte copy of the validated research implementation, with a 7-line licence and
attribution header added. The research file has sha256
`f3f24511a9701cb6836850f2873c4ac0ac18cd70c95c03470cdcc9803512bfd9`, and every run in `results/mountaincar/runs/*`
records that same hash in its `config.json`. The code was not refactored for release. Any change would have
invalidated the benchmark's provenance, and the class is already small (86 lines). For the same reason it is excluded
from `ruff format`, and two lint rules are ignored for that file only.

## Audit: generic versus specific components

| component (research code) | status in this repository |
|---|---|
| `SemiMarkovRolloutBuffer`, `SemiMarkovMaskablePPO` | generic; copied verbatim |
| MountainCar `MacroRepeat` wrapper | generalised to `MacroActionWrapper`: any `Discrete` env, any durations. Episode statistics moved to the example (`EpisodeStats`). On 14,609 random steps (90 episodes, all arms), it matches the validated research wrapper exactly in observation, reward, `k`, `gamma_t`, `truncated_cutoff` and episode summary. This check was run locally and is not shipped, because it imports the research code |
| duration-to-discount logic from the research environments (γ^{Δt/Δt0}) | generalised to `DurationDiscountWrapper` |
| `PrimitiveBudget` callback, evaluation, logging | MountainCar example only (`examples/mountaincar/train.py`) |
| research-domain environments, rewards, data, IRL code | not included |

## Decisions

* **Reward normalization off.** `VecNormalize` normalizes rewards by a running return computed with one scalar γ.
  That is inconsistent with per-transition γ_t, and it would give the macro-action arm a different reward scale for
  reasons unrelated to the algorithm. Observation normalization stays on in every arm. This deviates from the RL Zoo
  MountainCar setting (`normalize: true`).
* **Budget in primitive steps.** SB3's `num_timesteps` counts decisions. `PrimitiveBudget` sums the executed
  durations instead. The stop check runs once per vectorised step, so the overshoot is below n_envs × k_max (it was 0
  for A1/A2 and 4–6 for A3). The partial final rollout is not used for an update.
* **n_steps in decisions.** The PPO update keeps the same shape in every arm. Arms with longer actions therefore make
  fewer updates per primitive step, and update counts are reported.
* **Parity tolerance.** Parity is asserted at 1e-5. The observed differences were 0.0.

## Known gaps

* `SubprocVecEnv` is covered by one test (same rollout as `DummyVecEnv`, including truncations and autoreset); training runs in the benchmark used `DummyVecEnv`.
* Continuous action spaces are not supported, because MaskablePPO does not support them.
* λ and entropy are per decision, not per unit of time.
