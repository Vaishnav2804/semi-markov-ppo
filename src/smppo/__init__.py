"""PPO for semi-Markov decision processes: per-transition discount, variable-duration GAE, action masking."""

from smppo.ppo import SemiMarkovMaskablePPO, SemiMarkovRolloutBuffer
from smppo.wrappers import DurationDiscountWrapper, MacroActionWrapper

__all__ = ["SemiMarkovMaskablePPO", "SemiMarkovRolloutBuffer", "MacroActionWrapper", "DurationDiscountWrapper"]
__version__ = "0.1.0"
