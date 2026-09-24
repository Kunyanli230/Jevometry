# Numerical limits and failure modes

## Declared caps (v0.1)

| Resource | Limit |
|----------|-------|
| numeric parameters | 1–8 |
| outcomes per node | 32 |
| nodes per system | 64 |
| exact joint / tree paths | 4096 |

Exceeding a cap returns a structured `unsupported` status with the estimated
requirement.  Probabilities are never truncated or altered to fit.

## Tolerance defaults

| Quantity | Default |
|----------|---------|
| probability sum tolerance | `1e-6` |
| rank relative tolerance | `1e-8` |
| rank absolute tolerance | `1e-12` |
| h vs h/2 relative stability threshold | `0.10` |
| matrix relative floor | `1e-12` |
| sqrt-form mismatch tolerance | `1e-8` |
| machine clamp tolerance | `1e-12` |

All tolerances are recorded with the result.  They are diagnostics, not
guarantees.

## Failure modes and reason codes

| Situation | Status | Reason code |
|-----------|--------|-------------|
| support changes with theta | failed | `support_changed_with_theta` |
| question semantics change with theta | failed | `semantics_changed_with_theta` |
| resolved model changes across the stencil | failed | `model_identity_changed` |
| renderer cannot resolve the step | conditional | `resolution_limited` |
| h vs h/2 difference above threshold | unstable | `step_stability_exceeded` |
| stencil outside bounds | conditional | `stencil_unavailable` |
| required capture absent | insufficient_data | `missing_capture` |
| zero-probability outcome | undefined | `zero_probability_outcome` |
| declared fixed zero with a gradient | failed | `declared_zero_outcome_has_gradient` |
| singular Fisher information | not_identifiable | `not_identifiable` |
| no joint model | unsupported | `no_joint_model` |
| theta-dependent aggregation | unsupported | `policy_changed` |
| missing sampling contract | unsupported | `missing_sampling_contract` |
| dependent sampling | unsupported | `dependent_sampling` |
| budget exhausted | insufficient_data | `attempt_budget_exhausted` |
| replay fingerprint mismatch | failed | `fingerprint_mismatch` |

## Reproducibility

Simulations use explicit seeds.  Derivative results record the stencil, step
sizes and relative stability.  Manifests record the experiment hash, provider
mode, model, budget and analysis revisions.  NPZ arrays are keyed by node,
case and point so that a matrix can be traced back to its coordinates, labels
and units.
