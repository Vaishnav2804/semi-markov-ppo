# Changelog

## 0.1.0 (unreleased)

First public version.

* `SemiMarkovMaskablePPO` and `SemiMarkovRolloutBuffer`: transition-specific discount `gamma_t` in GAE, single
  `gamma_t * V(terminal_observation)` bootstrap on truncation, action masking from sb3-contrib `MaskablePPO` 2.7.x.
* `MacroActionWrapper`: macro-actions over a `Discrete` environment, with exact executed durations.
* `DurationDiscountWrapper`: `gamma_t` from reported or computed transition durations.
* MountainCar example (training with a primitive-step budget, evaluation, GIF) and 3-seed benchmark results.
* Test suite: duration-1 parity with `MaskablePPO`, variable-duration GAE, discounting, truncation and
  termination inside macro-actions, `SubprocVecEnv` autoreset, primitive-step accounting, masking.
