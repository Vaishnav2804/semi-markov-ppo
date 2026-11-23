# Algorithm

## SMDP setting

A semi-Markov decision process lets each decision last a random or chosen duration. Decision t starts in state s_t,
takes action a_t, runs for duration k_t (primitive steps) or Δt_t (time), and collects primitive rewards
r_{t,0}, …, r_{t,k_t−1}. The discounted return of the underlying process, written per decision, is

$$G_t = R_t + \gamma_t\, G_{t+1}, \qquad R_t = \sum_{j=0}^{k_t-1} \gamma^{j} r_{t,j}, \qquad \gamma_t = \gamma^{k_t}.$$

When durations are continuous, γ_t = γ^{Δt_t/Δt_0}. The SMDP Bellman equation and this discounting are classical:
Bradtke & Duff (NIPS 7) for continuous-time problems, and Sutton, Precup & Singh (1999) for options.

## What changes in PPO

Only the advantage computation changes.

| step | stock MaskablePPO | SemiMarkovMaskablePPO |
|---|---|---|
| per-step reward | r_t | R_t (environment returns the accumulated discounted reward) |
| TD error | δ_t = r_t + γ (1−d_t) V(s_{t+1}) − V(s_t) | δ_t = R_t + γ_t (1−e_t) V(s_{t+1}) − V(s_t) |
| GAE recursion | A_t = δ_t + γ λ (1−d_t) A_{t+1} | A_t = δ_t + γ_t λ (1−e_t) A_{t+1} |
| time-limit cutoff | r_t += γ V(terminal obs) when `TimeLimit.truncated` | R_t += γ_t V(terminal obs) when `truncated_cutoff`, once |
| value target | A_t + V(s_t) | A_t + V(s_t) |
| policy loss, value loss, entropy, clipping, masking | unchanged | unchanged |

e_t is 1 if the episode ended at decision t, by termination or truncation. A truncated step therefore carries its
bootstrap inside R_t and contributes no further future term, so value never leaks across an automatic reset. When
the last decision of a rollout did not end an episode, the critic's value of the next observation is used as
V(s_{t+1}). When it did end an episode, that value is ignored.

## Code map (`src/smppo/ppo.py`)

* `SemiMarkovRolloutBuffer` adds two arrays, `gammas` and `ends`, filled by `add_sm()` right after each `add()`.
  `compute_returns_and_advantage` implements the GAE recursion above.
* `SemiMarkovMaskablePPO.collect_rollouts` is sb3-contrib 2.7.0's loop with three changes:
  * it reads `info["gamma_t"]`;
  * it adds the single γ_t·V bootstrap on `truncated_cutoff`;
  * it stores `gammas` and `ends`.

  `n_bootstraps` counts bootstraps for auditing.
* `train()` (the PPO update) is inherited unchanged.

## Wrappers (`src/smppo/wrappers.py`)

* `MacroActionWrapper` builds (primitive, k_max) macro-actions, stops at termination or truncation, and returns R_t,
  `k`, `gamma_t`, `truncated_cutoff` and `prim_rewards`.
* `DurationDiscountWrapper` adds `gamma_t` from a reported or computed duration. It does not change the reward.
