# Prior art

The search ran on 2026-11-23. It covered web search, arXiv, library documentation and direct reads of library source
files. GitHub code search and Google Scholar were not available, so the absence of a matching implementation below
is limited to what this search found. Each entry was checked against the record named in its "verified via" column.

## Conclusion

The ingredients are established:
* SMDP discounting by elapsed time or duration;
* GAE;
* PPO;
* invalid-action masking;
* temporally extended and repeated actions.

Per-step discounting inside GAE also exists in at least one major library (TorchRL). We found no public package that
combines the following as a drop-in subclass of sb3-contrib's MaskablePPO:
* per-transition γ_t in GAE;
* reward accumulation inside macro-actions;
* exactly-once truncation bootstrapping;
* tested duration-1 parity.

We describe the repository as a tested implementation. We do not claim it is a new algorithm or the first of its
kind.

## Papers

| reference | verified via | relation |
|---|---|---|
| Bradtke, S. J., & Duff, M. O. Reinforcement Learning Methods for Continuous-Time Markov Decision Problems. Advances in Neural Information Processing Systems 7 (NIPS 1994). | proceedings.neurips.cc page `paper/1994/hash/07871915a8107172b3b5dc15a6574ad3` | SMDP and continuous-time TD learning, discounting by elapsed time |
| Sutton, R. S., Precup, D., & Singh, S. Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning. Artificial Intelligence 112(1–2), 181–211, 1999. doi:10.1016/S0004-3702(99)00052-1 | Crossref record for the DOI | options and SMDP framework |
| Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. High-Dimensional Continuous Control Using Generalized Advantage Estimation. arXiv:1506.02438, 2015. | arXiv API | GAE with a scalar γ |
| Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. Proximal Policy Optimization Algorithms. arXiv:1707.06347, 2017. | arXiv API | PPO |
| Huang, S., & Ontañón, S. A Closer Look at Invalid Action Masking in Policy Gradient Algorithms. arXiv:2006.14171, 2020; FLAIRS Vol. 35 (2022). | arXiv API (journal_ref) | action masking, as used by MaskablePPO |
| Bacon, P.-L., Harb, J., & Precup, D. The Option-Critic Architecture. arXiv:1609.05140, 2016 (accepted to AAAI-17 per the arXiv comment). | arXiv API | options with actor-critic |
| Klissarov, M., Bacon, P.-L., Harb, J., & Precup, D. Learnings Options End-to-End for Continuous Action Tasks. arXiv:1712.00004, 2017. (Title spelled as on arXiv.) | arXiv API | option-critic trained with PPO (PPOC) |
| Sharma, S., Srinivas, A., & Ravindran, B. Learning to Repeat: Fine Grained Action Repetition for Deep Reinforcement Learning. arXiv:1702.06054, 2017. | arXiv API | learned action repetition (FiGAR) |
| Biedenkapp, A., Rajan, R., Hutter, F., & Lindauer, M. TempoRL: Learning When to Act. arXiv:2106.05262, 2021 (accepted at ICML'21 per the arXiv comment). | arXiv API | learned action commitment, Q-learning based |
| Kwiatkowski, A., Kalogeiton, V., Pettré, J., & Cani, M.-P. UGAE: A Novel Approach to Non-exponential Discounting. arXiv:2302.05740, 2023. | arXiv API | GAE with a general discount vector, used with PPO; the closest paper found |

## Implementations and libraries

| item | verified via | comparison with this repository |
|---|---|---|
| Stable-Baselines3 2.7.0 `RolloutBuffer.compute_returns_and_advantage` | installed source, `buffers.py:433` | GAE with scalar `self.gamma`; no per-transition discount |
| sb3-contrib 2.7.0 `MaskablePPO` | installed source | scalar γ; timeout bootstrap `rewards += self.gamma * V(terminal_obs)`; masking. This is the base class here |
| CleanRL `cleanrl/ppo.py` (repo `vwxyzjn/cleanrl`, master) | raw source, lines 49 and 229–230 | scalar `args.gamma` in the GAE loop |
| TorchRL `torchrl/objectives/value/functional.py` (repo `pytorch/rl`, main at `df1db3dc`) | raw source, line 271 | `vec_generalized_advantage_estimate(gamma: float \| torch.Tensor, ...)` accepts a tensor γ, so per-step discounting in GAE is supported. We did not verify reward accumulation inside actions or SMDP truncation handling there |
| `mklissa/PPOC` (GitHub) | `git ls-remote` (repo exists) | code for PPOC; not inspected in detail |
| `automl/TempoRL` (GitHub) | `git ls-remote` (repo exists) | code for TempoRL; DQN-style, not PPO/GAE, per the paper |

Not included because they could not be verified: two Ray/RLlib forum threads about SMDP-adapted GAE and DQN
discounting (full URLs not confirmed); RLlib's own GAE source (not retrieved); one 2026 arXiv entry seen only in search
results.
