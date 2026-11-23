# Contributing

Issues and pull requests are welcome.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
pip install -r requirements.txt ruff
pip install -e .
```

## Before opening a pull request

```bash
ruff check . && ruff format --check .
pytest -q
```

* **New behaviour needs a test.** Changes to rollout collection or GAE should be checked against a hand-written
  reference (see `tests/test_gae.py`), and must keep `tests/test_parity.py` passing.
* **`src/smppo/ppo.py` is excluded from `ruff format`.** It mirrors sb3-contrib's rollout loop; keep edits minimal
  and say in the PR why they are needed.
* **Benchmark results.** If you change results under `results/`, include the exact commands and seeds that
  produced them.

By contributing you agree that your contributions are licensed under the MIT License.
