# Data handling

## What is stored where

A run directory contains:

```text
run/
  manifest.json            # identity, provider mode, budget, revisions
  experiment.resolved.yaml # resolved experiment specification
  config.resolved.yaml     # original configuration (when available)
  system.json              # declared system structure
  parameters.json          # parameter declarations
  questions.json           # question declarations and semantic hashes
  traces.jsonl             # one recorded evaluation per line
  derivatives.npz          # numeric matrices (internal diagnostics)
  metrics.json             # latest analysis revision
  inference.json           # only when inference actually ran
  report.html / report.md  # rendered report
  checksums.json           # sha256 of every artifact
  analysis/rev-NNNN/       # immutable per-revision copies
```

Raw captures (`traces.jsonl`, `requests/`, `responses/`) are written once and
never overwritten by analysis.  Every `analyze` creates a new revision and the
manifest is updated atomically.

## JSON safety

JSON artifacts never contain `NaN` or `Infinity`.  Values that are not finite
are represented with `value_kind` (`positive_infinity`, `negative_infinity`)
and a `reason_code`.  `derivatives.npz` may contain internal non-finite
diagnostics but is never used for a successful result.

## Redaction and privacy

* Only live adapters send rendered states and questions.
* A user-declared redaction is applied before hashing, so the redacted payload
  is the actual analysis input.
* Raw captures stay local.  Reports do not embed sample text; they show
  summaries, hashes and parameters.
* Report text from external sources is HTML-escaped.
* The self-contained HTML embeds Plotly.js inline, does not load a CDN and
  sends no telemetry.
* API keys are never logged, recorded or hashed.  `RecordingTransport`
  redacts `Authorization` and API-key headers.

## Failures are not outcomes

Provider failures, timeouts and authentication errors are not semantic
outcomes and are never folded into categorical probabilities.  They appear as
`ProviderStatus(ok=False)` with a reason code, and the run is marked
incomplete.  Reliability may be analysed separately only if the user declares
an explicit reliability observation model.

## Cache

The capture cache lives outside run directories and is keyed by the full
request fingerprint: provider, model, adapter version, node, question hash,
rendered input, theta, conditional history and stencil role.  Repeats bypass
the cache; live requests bypass it unless explicitly enabled.  Retries share a
semantic request hash but receive distinct capture ids.

## Integrity

`checksums.json` records a sha256 for every artifact.  `verify_checksums`
recomputes them; mismatches are reported rather than ignored.
