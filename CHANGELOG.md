# Changelog

All notable changes to Jevometry are documented here.  The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
SemVer.  Artifact schemas and mathematical conventions are versioned separately
from the package version; see `docs/release.md`.

## [0.2.0] — 2026-10-02

### Added

* Read-only `jevometry verify RUN [--json]` audits with exit code 2 for missing,
  malformed, incomplete or mismatched checksum evidence.
* Checksum coverage for resolved provider configuration, questions, inference
  and retained analysis revisions.

### Fixed

* Backward finite differences now use the correct near/far sample order,
  including the automatic backward stencil at an upper parameter bound.
* Custom stencil scales above one respect the effective step when selecting
  an in-bounds direction; unavailable large stencils are refused.
* Acquisition budgets count actual boundary stencil points, all repeats and
  the provider's requests per point before live execution.
* Fisher quantities inherit numerical instability and conditional derivative
  status. Capability summaries expose the actual failure reason; unreliable
  node information cannot enter a successful redundancy comparison.
* Replay requires the exact repeat and validates its declared identity and
  available request fingerprint. TypeSafe traces retain the requested stencil
  role on both success and failure paths.
* Saving a report creates a new analysis revision and refreshes its checksums,
  preserving previously saved analysis metrics.

### Migration

* Artifact schema version remains `1.0`; mathematical definitions, probability
  normalization and sampling contracts are unchanged.
* Some formerly `ok` derived results are now `unstable` or `conditional`.
  Request estimates may increase (the live ticket example requires 27
  evaluations before retries, rather than the previously displayed 11).
* Reanalyze runs that used backward stencils. The old sample ordering could
  overstate derivatives and Fisher information even when h/h2 stability passed.
* Missing repeats no longer reuse another repeat. Reacquire the required records
  or lower the declared repeat count explicitly.
* `verify_checksums` raises `ValueError` for missing or invalid indexes. The
  new CLI audit also requires coverage for managed artifacts and revisions;
  see `docs/v0.2.md` for auditing older runs.

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
