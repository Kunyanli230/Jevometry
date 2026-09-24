# Mathematics

All logarithms are natural logarithms and all information quantities are in
nats.  This document is the software contract; the implementation and the test
suite follow it.

## 1. Probability vectors

A node reports a vector `q` over a stable support of outcome ids.  Raw values
are always preserved.  A vector is valid when every value is finite and in
`[0, 1]`, the support ids are unique, and the sum is within `1e-6` of one
(configurable).  When the sum is inside tolerance but not exactly one, a
*working* copy `q_working = q_raw / sum(q_raw)` is derived and the original
total, the correction and `derived=true` are recorded.  Outside tolerance the
vector is refused; it is never clipped, padded, smoothed or replaced by a
uniform distribution.

Primitives map to support-aligned vectors as follows:

| Primitive | Support | Vector |
|-----------|---------|--------|
| Choice | criteria keys in declared order | reported `probabilities` |
| Score | integer level ids as strings | reported `probabilities`; legend and optional numeric encoding stored |
| Noul | `false`, `true` | `(1 - noul, noul)` |

Score numeric encodings are only used when they are explicit.  Without an
explicit encoding no grade mean or variance is reported.

## 2. Distribution statistics

With `q` a validated probability vector:

* entropy `H(q) = -sum_k q_k log q_k` with `0 log 0 = 0`;
* expected value and variance only under an explicit numeric encoding `a_k`;
* `KL(p || q) = sum_k p_k log(p_k / q_k)`; if some `p_k > 0` and `q_k = 0` the
  result is a structured `positive_infinity` refusal, not a large number;
* `JS(p, q) = (KL(p||m) + KL(q||m)) / 2`, `m = (p + q) / 2`;
* Hellinger `H(p, q) = sqrt(1 - sum_k sqrt(p_k q_k))`;
* categorical Fisher–Rao `d_FR(p, q) = 2 arccos(sum_k sqrt(p_k q_k))`.

The Fisher–Rao formula is the geodesic distance on the **full** categorical
simplex.  It is not an exact geodesic distance inside a restricted parametric
family.  Floating-point clamps of `1 - BC` to zero are applied only at machine
level and are recorded.

## 3. Jacobian and Fisher pullback

For `q(theta)` with `J[k, a] = dq_k / dtheta_a`:

```
G(theta) = J^T diag(1/q) J
G(theta) = 4 (D_theta sqrt(q))^T (D_theta sqrt(q))
```

The first form is the implementation; the second is computed independently and
their relative difference is reported.  A mismatch above `1e-8` fails the
result.  Support, question semantics and model identity must be identical at
every stencil point.

Finite differences use central differences

```
J[:, a] ~ (q(theta + h_a e_a) - q(theta - h_a e_a)) / (2 h_a)
```

with `h` and `h/2`, a second-order one-sided formula when the central stencil
leaves the declared bounds, and a relative stability diagnostic

```
rel = ||J(h) - J(h/2)||_F / max(||J(h)||_F, floor)
```

with a default threshold of `0.10`.  The threshold is a diagnostic, not an
error bound.  When the renderer maps two distinct theta values to the same
input the result is flagged `resolution_limited`.

A zero-probability outcome makes the Fisher information undefined.  Only when
an adapter declares that outcome identically zero on a verified stencil is the
metric computed on the fixed active support, and the assumption is recorded.
No epsilon is ever injected.

## 4. Matrix diagnostics and coordinates

Every Fisher matrix is reported with its symmetry residual, eigenvalues,
eigenvectors, rank, condition number and null directions.  Rank uses a relative
tolerance `1e-8` and an absolute tolerance `1e-12` by default.  Negative
eigenvalues from rounding are reported; the raw matrix is never projected to
PSD.

For a linear coordinate change `theta = theta_0 + S z`:

```
G_z = S^T G_theta S
```

Standardized coordinates `z_a = theta_a / scale_a` use `S = diag(scales)`.
Eigenvalues, traces and determinants are only comparable across systems when
parameters, units, scales, experiment points and probability objects match.

## 5. System composition

| Mode | Meaning |
|------|---------|
| `node_only` | No system metric is produced |
| `declared_product` | `P(z) = prod_i q_i(z_i)` under declared conditional independence; results are assumption-based |
| `explicit_joint` | The adapter provides a full `P_theta` over a fixed support |
| `conditional_tree` | Finite conditional nodes with exact path enumeration |

Conditional trees satisfy

```
I_trajectory = sum_i E_history[I_i(theta | history)]
```

under the declared conditional distributions and regularity conditions.  The
implementation checks the identity numerically against the exact joint.  The
sum of node metrics along a single realised trajectory is never presented as
expected system information.

For a fixed aggregation `A = f(Z)` independent of theta the pushforward is
`P_theta(A = a) = sum_{z: f(z) = a} P_theta(z)` and the data-processing
inequality `I_Z - I_A >= 0` is checked as a PSD condition.  A theta-dependent
mapping receives no such guarantee and is reported as non-regular.

## 6. Inference

For `n` i.i.d. observations with non-singular `I_1`:

```
Cov(theta_hat) >= (n I_1)^-1
Cov(g_hat)     >= Dg (n I_1)^-1 Dg^T
```

in the PSD sense, for unbiased estimators under a regular, locally identifiable
model.  Independent but non-identical designs sum their information matrices:
`(sum_i I_i)^-1`.  For nuisance parameters the target sub-block of the full
inverse is used, or the Schur complement when it exists; the inverse of the
target block is never substituted.  Singular information returns
`not_identifiable`, and the pseudoinverse is only a geometric diagnostic.

A CRLB is not a confidence interval and not an accuracy guarantee.

## 7. Coordinate and support hygiene

Two probability vectors may only be compared when their supports match, or when
the caller supplies a confirmed injective `SupportMapping`.  Positional
comparison of different supports is refused.  Absent nodes are recorded as
missing, never as zero vectors.
