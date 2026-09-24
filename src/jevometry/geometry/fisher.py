"""Node Fisher pullback metric and coordinate transforms.

``G(theta) = J^T diag(1/q) J`` is the primary implementation; the equivalent
square-root form ``G = 4 (D_theta sqrt(q))^T (D_theta sqrt(q))`` is computed
independently as a consistency check.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from jevometry.geometry.diagnostics import MatrixDiagnostics, matrix_diagnostics
from jevometry.schemas.common import Diagnostic, MetricStatus

FloatArray = NDArray[np.float64]


@dataclass
class FisherResult:
    """Fisher information matrix for one node at one point."""

    values: FloatArray | None
    sqrt_form_values: FloatArray | None
    active_support: tuple[str, ...]
    status: MetricStatus = MetricStatus.OK
    reason_code: str | None = None
    assumptions: list[str] = field(default_factory=list)
    diagnostics: list[Diagnostic] = field(default_factory=list)
    symmetry_residual: float | None = None
    sqrt_form_residual: float | None = None
    matrix: MatrixDiagnostics | None = None


def fisher_pullback(
    jacobian: FloatArray,
    probabilities: FloatArray,
    *,
    support: tuple[str, ...],
    fixed_zero_outcomes: tuple[str, ...] = (),
    declared_fixed_zero: bool = False,
    zero_row_atol: float = 1e-10,
) -> FisherResult:
    """Compute the Fisher pullback metric with explicit zero-probability policy.

    By default any outcome with ``q_k = 0`` makes the metric undefined.  When
    the adapter has declared an outcome identically zero on a verified stencil,
    the metric is computed on the fixed active support and the assumption is
    recorded.
    """
    jac = np.asarray(jacobian, dtype=np.float64)
    q = np.asarray(probabilities, dtype=np.float64)
    if jac.shape[0] != q.shape[0]:
        raise ValueError("jacobian rows must match probability support")
    if len(support) != q.shape[0]:
        raise ValueError("support must match probability length")

    zero_mask = q == 0.0
    diagnostics: list[Diagnostic] = []
    assumptions: list[str] = []
    if zero_mask.any():
        zero_outcomes = tuple(
            outcome for outcome, is_zero in zip(support, zero_mask, strict=True) if is_zero
        )
        if not declared_fixed_zero:
            return FisherResult(
                values=None,
                sqrt_form_values=None,
                active_support=tuple(
                    outcome for outcome, is_zero in zip(support, zero_mask, strict=True) if not is_zero
                ),
                status=MetricStatus.UNDEFINED,
                reason_code="zero_probability_outcome",
                diagnostics=[
                    Diagnostic(
                        code="zero_probability_outcome",
                        message=(
                            "Fisher information is undefined at zero-probability outcomes: "
                            f"{', '.join(zero_outcomes)}"
                        ),
                        severity="error",
                        details={"outcomes": list(zero_outcomes)},
                    )
                ],
                assumptions=["no epsilon smoothing was applied"],
            )
        undeclared = set(zero_outcomes) - set(fixed_zero_outcomes)
        if undeclared:
            return FisherResult(
                values=None,
                sqrt_form_values=None,
                active_support=tuple(
                    outcome for outcome, is_zero in zip(support, zero_mask, strict=True) if not is_zero
                ),
                status=MetricStatus.UNSUPPORTED,
                reason_code="fixed_zero_declaration_mismatch",
                diagnostics=[
                    Diagnostic(
                        code="fixed_zero_declaration_mismatch",
                        message=(
                            "outcomes are zero but were not declared fixed zero on the stencil: "
                            f"{', '.join(sorted(undeclared))}"
                        ),
                        severity="error",
                    )
                ],
            )
        for index, is_zero in enumerate(zero_mask):
            if is_zero and float(np.max(np.abs(jac[index, :]))) > zero_row_atol:
                return FisherResult(
                    values=None,
                    sqrt_form_values=None,
                    active_support=tuple(
                        outcome
                        for outcome, zero in zip(support, zero_mask, strict=True)
                        if not zero
                    ),
                    status=MetricStatus.FAILED,
                    reason_code="declared_zero_outcome_has_gradient",
                    diagnostics=[
                        Diagnostic(
                            code="declared_zero_outcome_has_gradient",
                            message=(
                                f"declared fixed-zero outcome {support[index]!r} has non-zero "
                                "Jacobian row; declaration is inconsistent with the stencil"
                            ),
                            severity="error",
                        )
                    ],
                )
        keep = ~zero_mask
        assumptions.append(
            "fixed active support: outcomes "
            f"{', '.join(zero_outcomes)} declared and verified identically zero on the stencil"
        )
        diagnostics.append(
            Diagnostic(
                code="fixed_active_support",
                message="Fisher computed on the declared fixed active support",
                severity="info",
                details={"excluded": list(zero_outcomes)},
            )
        )
        active_support = tuple(
            outcome for outcome, is_zero in zip(support, zero_mask, strict=True) if not is_zero
        )
        jac = jac[keep, :]
        q = q[keep]
    else:
        active_support = tuple(support)

    if q.size == 0:
        return FisherResult(
            values=None,
            sqrt_form_values=None,
            active_support=(),
            status=MetricStatus.UNDEFINED,
            reason_code="empty_active_support",
        )

    weighted = jac / np.sqrt(q)[:, None]
    values = weighted.T @ weighted
    sqrt_derivative = jac / (2.0 * np.sqrt(q)[:, None])
    sqrt_form = 4.0 * (sqrt_derivative.T @ sqrt_derivative)
    scale = max(float(np.linalg.norm(values, ord="fro")), 1e-12)
    sqrt_form_residual = float(np.linalg.norm(values - sqrt_form, ord="fro") / scale)
    if sqrt_form_residual > 1e-8:
        diagnostics.append(
            Diagnostic(
                code="sqrt_form_mismatch",
                message=(
                    "primary and square-root Fisher forms differ by "
                    f"{sqrt_form_residual:.3e}; this indicates an implementation error"
                ),
                severity="error",
            )
        )
    symmetry_residual = float(np.linalg.norm(values - values.T, ord="fro") / scale)
    matrix = matrix_diagnostics(values)
    if matrix.psd_violation:
        diagnostics.append(
            Diagnostic(
                code="psd_violation",
                message=(
                    f"minimum eigenvalue {matrix.min_eigenvalue:.3e} is below tolerance; "
                    "raw matrix is reported without projection"
                ),
                severity="warning",
            )
        )
    status = MetricStatus.OK
    reason_code: str | None = None
    if sqrt_form_residual > 1e-8:
        status = MetricStatus.FAILED
        reason_code = "sqrt_form_mismatch"
    return FisherResult(
        values=values,
        sqrt_form_values=sqrt_form,
        active_support=active_support,
        status=status,
        reason_code=reason_code,
        assumptions=assumptions,
        diagnostics=diagnostics,
        symmetry_residual=symmetry_residual,
        sqrt_form_residual=sqrt_form_residual,
        matrix=matrix,
    )


def transform_coordinates(matrix: FloatArray, scales: FloatArray) -> FloatArray:
    """G_z = S^T G_theta S for theta = theta0 + S z with S = diag(scales)."""
    s = np.asarray(scales, dtype=np.float64)
    if s.ndim != 1 or s.shape[0] != matrix.shape[0]:
        raise ValueError("scales must be a vector matching the matrix dimension")
    return (matrix * s[None, :]) * s[:, None]


def transform_jacobian(jacobian: FloatArray, scales: FloatArray) -> FloatArray:
    """dq/dz = J S for theta = theta0 + S z."""
    s = np.asarray(scales, dtype=np.float64)
    if s.ndim != 1 or s.shape[0] != jacobian.shape[1]:
        raise ValueError("scales must match the Jacobian column count")
    return jacobian * s[None, :]
