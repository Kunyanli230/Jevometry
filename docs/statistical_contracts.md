# Statistical contracts

Jevometry refuses to compute rather than guessing.  This document states what
each object means and when a result may be reported.

## Three distinct objects

Every result carries `analysis_object`:

1. `reported_distribution` — the probability vector returned by Jev.
2. `declared_system_model` — a joint or conditional model declared by the user
   or adapter.
3. `empirical_observation_model` — a sampling distribution estimated from, or
   defined for, real or simulated observations.

These are not interchangeable.  For a fixed input, calling an API twice and
receiving the same vector is **not** two independent categorical draws.  The
label selected by the SDK is **not** assumed to be a random sample from the
reported vector.

## Parameter declarations

`theta` is always an external experiment coordinate (amount, wait time, prompt
threshold, business parameter), never a Jev internal weight.  Every parameter
declares a name, research role (`task_relevant`, `nuisance`, `diagnostic`),
unit, bounds, centre, finite-difference step and a positive `scale` for
dimensionless comparison.  Raw and standardized results are both stored.
Discrete prompt versions and categorical variables are compared by distance and
grouping, never differentiated.

## System information requires a joint model

Without a declared joint distribution there is no system Fisher information.
The toolkit never:

* computes mutual information from two marginals;
* treats the confidence of one answer as extra evidence;
* sums the Fisher information of correlated nodes by default;
* labels an independence-based combination as anything other than
  `assumption_based`.

`declared_product` results are conditional on the declared independence.
`explicit_joint` and `conditional_tree` results are exact for the declared
model.

## Result statuses

Every metric returns a structured `MetricResult` with one of:

`ok`, `conditional`, `unstable`, `undefined`, `not_identifiable`,
`insufficient_data`, `unsupported`, `failed`.

A refusal has `value = null`, a `reason_code` and an actionable `remedy`.
Zero is never used to mean "could not compute".  JSON artifacts never contain
NaN or Infinity; infinities are encoded with `value_kind` and reason codes.

## Routing semantics

Adapters declare how probabilities become actions:

* `sampled_outcome` — outcomes are sampled from a declared categorical law;
* `deterministic_policy` — the action is a deterministic function of the
  reported probabilities or labels;
* `externally_observed` — actions come from an external process.

For `deterministic_policy`, node geometry, action labels and flip locations are
reported, but no trajectory Fisher information or business CRLB.  A stochastic
surrogate may be declared, and is reported separately with its transformation
rule.

## CRLB eligibility

A CRLB requires a complete sampling contract:

* observable, observation unit, sampling kind, sample size;
* how observations relate to the Jev output;
* fixed support, differentiability and local identifiability, with their
  source (`analytic`, `user_asserted`, `empirically_checked`);
* the estimand.

Dependent or adaptive samples cannot be scaled by `n`.  The number of API calls
is not a sample size.  A reported-distribution bound is explicitly
model-conditional.  Ineligible requests return `unsupported` or
`not_identifiable` with the missing fields listed.

## Simulation is not validation of the real system

Synthetic MLE experiments validate the declared observation model.  Reported
bias, variance, RMSE, coverage and interval widths are accompanied by Monte
Carlo standard errors and failure counts.  Provider repeatability, synthetic
categorical sampling and real business inference are three different things and
are counted separately.
