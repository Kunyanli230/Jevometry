# Contributing

Thanks for improving Jevometry.  This project is a statistical tool: an
incorrect number is worse than a refusal.

## Development setup

```bash
uv sync --locked --extra live
uv run pytest
uv run ruff check src tests examples
uv run mypy
```

Python 3.11–3.13 are supported; CI runs all three.

## Ground rules

1. **No fabricated results.**  A quantity that cannot be computed returns a
   structured refusal with a reason code.  Never substitute zero, a default
   independence assumption, epsilon smoothing or synthetic data.
2. **No silent semantics changes.**  Changing a Fisher definition, default
   normalisation, support semantics or inference contract requires a
   migration note in the changelog and a regression fixture.
3. **Analytic expectations only.**  Tests must derive expected values from
   closed forms or independently constructed computations, not from the
   production functions under test.
4. **Every new behaviour needs a test.**  Statistical or numerical bug fixes
   need a regression fixture.
5. **Core isolation.**  `geometry`, `systems` and `inference` must not import
   providers, the CLI or Plotly.
6. **Optional dependencies stay optional.**  The offline core must import
   without `typesafe-sdk` installed.

## Adding a provider or adapter

Implement the protocols in `jevometry.adapters.base`, add an SDK transport mock
test, and document the provider in `docs/adapters.md`.  Do not patch SDKs
globally; capture is opt-in.

## Adding a metric

Add the result to the relevant schema, emit a `MetricResult` with an
`analysis_object`, and cover both the success and refusal paths.  Update
`docs/mathematics.md` or `docs/statistical_contracts.md` when the meaning
changes.

## Reporting bugs

Include:

* the command or Python snippet;
* the run directory's `manifest.json` and `metrics.json`;
* the exact status and reason code, not just the value;
* the installed versions from `jevometry doctor`.

Never include API keys or raw customer data.
