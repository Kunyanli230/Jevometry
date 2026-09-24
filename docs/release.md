# Release process

## Versioning

Three things are versioned independently:

* the Python package version (`pyproject.toml`);
* artifact schemas (`schema_version` on every document);
* the mathematical conventions (documented in `docs/mathematics.md`).

SemVer applies.  Changing the Fisher definition, default normalisation, support
semantics or the inference contract is a behaviour change and requires a
migration note.  Every fixed statistical or numerical bug needs a regression
fixture and a changelog entry.

## Pre-release checklist

1. `uv sync --locked --extra live`
2. `uv run ruff check src tests examples`
3. `uv run mypy`
4. `uv run pytest --cov=jevometry --cov-branch` (coverage gate: 80% overall;
   geometry, systems and inference cores at least 90% branch coverage)
5. `uv run python examples/analytic_geometry/run.py`
6. `uv run python examples/system_information/run.py`
7. `uv run python examples/jev_ticket_sensitivity/run.py`
8. `uv build`
9. Install the wheel in a clean environment outside the repository and run an
   example offline.
10. Manually open a generated `report.html` with networking disabled.

## Build

```bash
uv build            # dist/jevometry-<version>-py3-none-any.whl and .tar.gz
```

The wheel contains only `src/jevometry`.  The SDK is an optional `live` extra;
the offline core never imports it.

## What this repository does not do

* It does not publish to PyPI.
* It does not create remote repositories, tags or GitHub releases.
* Distribution uses the repository's MIT license; preserve `LICENSE` in
  release artifacts.
* It does not treat a model version as permanently current.  Reports always
  show the requested and resolved model.
