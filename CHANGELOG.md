# Changelog

All notable changes to Jevometry are documented here.  The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
SemVer.  Artifact schemas and mathematical conventions are versioned separately
from the package version; see `docs/release.md`.

## [0.1.0] — 2026-09-24

Initial implementation of the v0.1 specification.

### Added

* Probability core: strict support-aligned validation, raw/working simplex
  copies, entropy, expected value/variance under explicit encodings, KL, JS,
  Hellinger and categorical Fisher–Rao distances.
* Geometry: finite-difference Jacobians with h/h2 stability, one-sided
  stencils, renderer resolution detection, Fisher pullback with an independent
  square-root cross-check, matrix diagnostics, coordinate transforms.
* Systems: declared product joints, explicit joints, exact conditional tree
  enumeration with the conditional information identity, fixed-mapping
  pushforward information loss, redundancy comparison.
* Inference: sampling contracts, CRLB with nuisance handling, bounded MLE,
  profile-likelihood intervals, synthetic simulation summaries.
* Adapters: framework-agnostic protocols, analytic families, TypeSafe provider
  (live extra), replay provider, capture recorder and recording transport.
* Artifacts: run directories, immutable analysis revisions, atomic manifests,
  checksums, JSON-safe serialisation.
* CLI: `init`, `validate`, `run`, `analyze`, `compare`, `infer`, `report`,
  `doctor`, with documented exit codes.
* Reporting: self-contained offline interactive HTML plus a Markdown twin.
* Examples: analytic geometry laboratory, multi-node information and inference,
  ticket-system sensitivity with a deterministic policy and a declared
  surrogate.
* Tests: analytic unit tests, Hypothesis property tests, SDK transport mocks,
  CLI integration tests, schema contract tests and regression tests.
* CI workflow for Python 3.11–3.13 with Ruff, mypy, pytest and a build step.

### Notes

* Live TypeSafe integration is not verified without credentials; the offline
  path is complete and does not require a key.
* No license is declared yet; the owner must choose one before distribution.
