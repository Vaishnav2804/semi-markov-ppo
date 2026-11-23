# Third-party notices

`src/smppo/ppo.py` contains code adapted from **sb3-contrib 2.7.0** (`sb3_contrib/ppo_mask/ppo_mask.py`, `MaskablePPO.collect_rollouts`) and subclasses sb3-contrib and Stable-Baselines3 classes. The modifications are listed in the module docstring: the buffer stores a per-transition discount and an episode-end flag; GAE uses the per-transition discount; truncations are bootstrapped once with `gamma_t * V(terminal_observation)`; the scalar-gamma `TimeLimit.truncated` correction is not applied.

sb3-contrib is distributed under the following licence (text copied from the installed sb3-contrib 2.7.0 distribution, `sb3_contrib-2.7.0.dist-info/licenses/LICENSE`):

```
MIT License

Copyright (c) 2020 Stable-Baselines Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Stable-Baselines3 2.7.0 (MIT, Copyright (c) 2019 Antonin Raffin; its NOTICE credits OpenAI Baselines and Stable-Baselines) is a runtime dependency; no Stable-Baselines3 source is copied here.
