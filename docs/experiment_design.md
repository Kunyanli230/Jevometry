# Experiment design

## Declaring an experiment

An experiment YAML declares:

```yaml
id: my-experiment
parameters:        # name, role, unit, bounds, scale, step, center
stencil:           # kind, step_scales, relative_stability_threshold
cases:             # fixed inputs (state)
theta_points:      # centre points in raw coordinates
budget:            # attempts, tokens, timeouts, repeats
renderer:          # deterministic theta -> state
questions:         # primitive + outcomes + legend
node_questions:    # node -> question
provider:          # module / replay / typesafe
```

Every theta point must define exactly the declared parameters.  Categorical
parameters are rejected because finite differences are undefined for them.

## Renderers

A renderer maps `(theta, case)` to a `RenderedInput` with a content
fingerprint.  `TextTemplateRenderer` formats a string,
`StructuredRenderer` formats JSON fields, and `CallableRenderer` wraps a
function.  Format specifiers set the effective resolution: `{amount:.2f}`
cannot resolve a step smaller than `0.005`.  When distinct theta values render
to the same input the analysis reports `resolution_limited` instead of
pretending the parameter is observable.

## Acquisition plan

Before any request, `run` builds an acquisition plan containing the cases, theta
points, nodes, stencil points, repeats and the expected number of provider
requests.  Live runs refuse to start when the plan exceeds `max_attempts`.
Each node question batch counts once when the provider batches a shared state.

The plan is persisted in the run manifest and shown by `jevometry validate`.

## Repeats

Repeats are independent provider evaluations at the same point.  They are a
repeatability diagnostic, not new evidence:

* repeats always bypass the capture cache;
* each repeat has its own capture id while sharing the semantic request hash;
* retries are not new repeats.

Live examples default to three repeats; offline analysis defaults to one.

## Stencil and stability

For each centre point the engine requires the stencil points described by
`step_scales` (default `h` and `h/2`) and the stencil kind.  `run` captures
exactly those points; `analyze` only consumes what was captured and reports
`insufficient_data` for missing points.  The h/h2 relative difference is a
diagnostic with a configurable threshold; exceeding it marks the result
`unstable`, never silently averaged.

## Budgets

| Limit | Default |
|-------|---------|
| attempts | 200 |
| known input tokens | 200 000 |
| request timeout | 45 s |
| experiment timeout | 600 s |
| concurrency | 2 |

Budget exhaustion produces structured refusals.  When a stencil is incomplete
because of budget or errors, affected metrics are marked incomplete and are not
filled from other cases, models or stale caches.

## Comparing runs

`jevometry compare` checks parameter names, units, scales, steps, stencil kind,
step scales, requested model and per-node support sets before comparing any
metric.  Incompatible runs are refused with the specific mismatch.
