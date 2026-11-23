# Release process (recommended)

The first release is `v0.1.0`. Publishing to PyPI is not planned for this release.

1. Licence: MIT, Copyright (c) 2026 Vaishnav (done). Confirm any external approvals listed in `docs/licensing.md`.
2. Complete `CITATION.cff`: authors, ORCID, `repository-code`, `date-released`, `license`.
3. In a fresh environment, run:
   `pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt ruff && pip install -e . --no-deps`,
   then `ruff check . && ruff format --check . && pytest -q`.
4. Validate `CITATION.cff`, for example with `cffconvert --validate`.
5. Commit, create the GitHub repository, push, and confirm that the CI workflow passes.
6. Tag: `git tag -a v0.1.0 -m "SemiMarkovPPO 0.1.0"` and `git push origin v0.1.0`. Optionally create a GitHub
   release from the tag, with the README's benchmark table in the notes.
7. Optional, later: archive the release on Zenodo for a DOI, then add the DOI to `CITATION.cff`.
