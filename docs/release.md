# Release process (recommended)

The first release is `v0.1.0`. Publishing to PyPI is not planned for this release.

1. Resolve the owner decisions listed in `docs/licensing.md`: licence, copyright holder and external approvals.
2. Add the approved `LICENSE`. In `pyproject.toml`, add the licence field, for example `license = "MIT"` (needs
   setuptools ≥ 77) or `license = { text = "MIT" }`. Complete `CITATION.cff`: authors, ORCID, `repository-code`,
   `date-released`, `license`.
3. In a fresh environment, run:
   `pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt ruff && pip install -e . --no-deps`,
   then `ruff check . && ruff format --check . && pytest -q`.
4. Validate `CITATION.cff`, for example with `cffconvert --validate`.
5. Commit, create the GitHub repository, push, and confirm that the CI workflow passes.
6. Tag: `git tag -a v0.1.0 -m "SemiMarkovPPO 0.1.0"` and `git push origin v0.1.0`. Optionally create a GitHub
   release from the tag, with the README's benchmark table in the notes.
7. Optional, later: archive the release on Zenodo for a DOI, then add the DOI to `CITATION.cff`.
