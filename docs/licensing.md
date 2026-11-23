# Licensing and attribution report

## Copied or adapted code

| file | origin | licence of origin | action taken |
|---|---|---|---|
| `src/smppo/ppo.py`, `SemiMarkovMaskablePPO.collect_rollouts` | adapted from sb3-contrib 2.7.0 `MaskablePPO.collect_rollouts` | MIT, Copyright (c) 2020 Stable-Baselines Team (from the installed `sb3_contrib-2.7.0.dist-info/licenses/LICENSE`) | header comment in the file; full licence text and the list of modifications in `THIRD_PARTY_NOTICES.md` |
| `src/smppo/ppo.py`, `SemiMarkovRolloutBuffer` | subclass of sb3-contrib `MaskableRolloutBuffer`; `compute_returns_and_advantage` is a rewrite of the SB3 recursion with per-transition γ | as above; SB3 is MIT, Copyright (c) 2019 Antonin Raffin | covered by the same notice; no SB3 source text copied |
| `examples/mountaincar/environment.py`, `simulate()` | re-implementation of Gymnasium 1.2.3 `MountainCarEnv.step` equations, used in tests | Gymnasium: MIT (PyPI metadata) | written from the equations; credited in the docstring |
| everything else | written for this repository | — | — |

## Dependencies (imported, not vendored)

Licences are taken from the `METADATA` of the tested versions installed from `requirements.txt` (checked
2026-11-23):

| package | licence |
|---|---|
| stable-baselines3 | MIT (plus a NOTICE crediting OpenAI Baselines and Stable-Baselines) |
| sb3-contrib | MIT |
| gymnasium | MIT |
| torch 2.7.0 | BSD-3-Clause |
| numpy 2.2.4 | BSD (classifier "OSI Approved :: BSD License") |
| pandas 2.2.3 | BSD-3-Clause |
| matplotlib 3.11.2 | Matplotlib licence (PSF-based) |
| pytest, ruff (dev only) | MIT |

Importing these packages imposes no licence constraint on this repository's own code.

## Recommendation

Use **MIT**. The adapted code is MIT-licensed, so the whole repository can stay under one permissive licence that
matches the ecosystem it plugs into. Apache-2.0 would also be compatible and adds an explicit patent grant, at the
cost of NOTICE conventions. GPL-family licences would restrict downstream use in the SB3 ecosystem.

**Status: not approved.** A draft MIT `LICENSE` with a placeholder copyright holder exists in the working copy. It is
deliberately not committed, and `pyproject.toml` and `CITATION.cff` declare no licence. Source files carry no SPDX
licence identifier. `src/smppo/ppo.py` states that its licence is pending and points to `THIRD_PARTY_NOTICES.md`.
Until a licence is committed, the repository grants no licence for the project's own code. The adapted sb3-contrib
portions remain under MIT regardless, because that licence must be kept with them.

The owner must decide:
* the licence (MIT recommended);
* the copyright holder: an individual, or an institution if institutional policy assigns ownership.

## External approvals still required (none obtained)

* **University.** Its intellectual-property policy for software written by students or employees, and whether an
  open-source release must be disclosed to, or approved by, a technology-transfer or research office.
* **Employer, if separate from the university.** Its IP or moonlighting policy, if the work overlapped with paid
  employment.
* **Funders.** Any grant, scholarship or partner agreement covering the research: open-source or IP clauses, and
  required acknowledgements.
* **Collaborators and supervisors.** Agreement on authorship and on publishing benchmark results before a related
  paper.
