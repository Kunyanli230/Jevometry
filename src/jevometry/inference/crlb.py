"""Cramer-Rao lower bounds for declared observation models.

The bound is ``Cov(theta_hat) >= (n I_1)^-1`` in the PSD sense for unbiased
estimators under a regular, locally identifiable model with i.i.d. sampling.
It is not a confidence interval and not an accuracy guarantee.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from jevometry.geometry.diagnostics import is_psd
from jevometry.schemas.common import Diagnostic, MetricStatus

FloatArray = NDArray[np.float64]

DEFAULT_IDENTIFIABILITY_RTOL = 1e-10


@dataclass
class CrlbResult:
    """A CRLB matrix with structured eligibility information."""

    crlb: FloatArray | None
    standard_errors: FloatArray | None
    parameter_names: list[str]
    status: MetricStatus = MetricStatus.OK
    reason_code: str | None = None
    remedy: str | None = None
    method: str | None = None
    assumptions: list[str] = field(default_factory=list)
    diagnostics: list[Diagnostic] = field(default_factory=list)
    smallest_eigenvalue: float | None = None
    condition_number: float | None = None


def _inverse_or_reason(
    information: FloatArray, *, rtol: float = DEFAULT_IDENTIFIABILITY_RTOL
) -> tuple[FloatArray | None, str | None, float, float | None]:
    array = np.asarray(information, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        return None, "non_square_information", float("nan"), None
    if not np.all(np.isfinite(array)):
        return None, "non_finite_information", float("nan"), None
    eigenvalues = np.linalg.eigvalsh(0.5 * (array + array.T))
    smallest = float(eigenvalues.min())
    largest = float(eigenvalues.max())
    if largest <= 0.0 or smallest <= max(rtol * largest, 0.0):
        return None, "not_identifiable", smallest, None
    condition = largest / smallest
    try:
        inverse = np.asarray(np.linalg.inv(array), dtype=np.float64)
    except np.linalg.LinAlgError:
        return None, "singular_information", smallest, condition
    return inverse, None, smallest, condition


def crlb_iid(
    information: FloatArray,
    sample_size: int,
    *,
    parameter_names: Sequence[str],
    identifiability_rtol: float = DEFAULT_IDENTIFIABILITY_RTOL,
) -> CrlbResult:
    """CRLB for n i.i.d. observations: (n I_1)^-1."""
    names = list(parameter_names)
    if sample_size <= 0:
        return CrlbResult(
            crlb=None,
            standard_errors=None,
            parameter_names=names,
            status=MetricStatus.UNSUPPORTED,
            reason_code="invalid_sample_size",
            remedy="sample_size must be a positive integer",
        )
    inverse, reason, smallest, condition = _inverse_or_reason(
        information, rtol=identifiability_rtol
    )
    if inverse is None:
        return CrlbResult(
            crlb=None,
            standard_errors=None,
            parameter_names=names,
            status=MetricStatus.NOT_IDENTIFIABLE,
            reason_code=reason,
            remedy=(
                "the Fisher information is singular; the parameter is not locally "
                "identifiable under this model. Report null directions or reparameterise."
            ),
            method="inverse",
            smallest_eigenvalue=smallest,
            condition_number=condition,
        )
    crlb = inverse / float(sample_size)
    return CrlbResult(
        crlb=crlb,
        standard_errors=np.sqrt(np.clip(np.diag(crlb), 0.0, None)),
        parameter_names=names,
        method="inverse",
        assumptions=[
            "n independent and identically distributed observations",
            "regular model with finite Fisher information",
            "locally identifiable parameter",
            "unbiased estimator",
        ],
        diagnostics=[
            Diagnostic(
                code="identifiability",
                message="information matrix inverted successfully",
                details={"smallest_eigenvalue": smallest, "condition_number": condition},
            )
        ],
        smallest_eigenvalue=smallest,
        condition_number=condition,
    )


def crlb_independent_designs(
    information_matrices: Sequence[FloatArray],
    *,
    parameter_names: Sequence[str],
) -> CrlbResult:
    """CRLB for independent (possibly non-identical) designs: (sum I_i)^-1."""
    if not information_matrices:
        return CrlbResult(
            crlb=None,
            standard_errors=None,
            parameter_names=list(parameter_names),
            status=MetricStatus.INSUFFICIENT_DATA,
            reason_code="no_design_information",
            remedy="provide at least one per-design Fisher information matrix",
        )
    total = np.sum(np.asarray(information_matrices, dtype=np.float64), axis=0)
    inverse, reason, smallest, condition = _inverse_or_reason(total)
    if inverse is None:
        return CrlbResult(
            crlb=None,
            standard_errors=None,
            parameter_names=list(parameter_names),
            status=MetricStatus.NOT_IDENTIFIABLE,
            reason_code=reason,
            remedy="combined design is not identifiable; inspect null directions",
            method="sum_of_designs",
            smallest_eigenvalue=smallest,
            condition_number=condition,
        )
    return CrlbResult(
        crlb=inverse,
        standard_errors=np.sqrt(np.clip(np.diag(inverse), 0.0, None)),
        parameter_names=list(parameter_names),
        method="sum_of_designs",
        assumptions=["independent but not necessarily identical designs"],
        smallest_eigenvalue=smallest,
        condition_number=condition,
    )


def crlb_for_function(
    information: FloatArray,
    sample_size: int,
    jacobian_g: FloatArray,
    *,
    parameter_names: Sequence[str],
    function_name: str,
) -> CrlbResult:
    """CRLB for g(theta): Dg (n I)^-1 Dg^T."""
    base = crlb_iid(information, sample_size, parameter_names=parameter_names)
    if base.crlb is None:
        return base
    gradient = np.asarray(jacobian_g, dtype=np.float64)
    if gradient.ndim == 1:
        gradient = gradient[None, :]
    if gradient.shape[1] != base.crlb.shape[0]:
        return CrlbResult(
            crlb=None,
            standard_errors=None,
            parameter_names=list(parameter_names),
            status=MetricStatus.FAILED,
            reason_code="gradient_dimension_mismatch",
        )
    transformed = gradient @ base.crlb @ gradient.T
    return CrlbResult(
        crlb=transformed,
        standard_errors=np.sqrt(np.clip(np.diag(transformed), 0.0, None)),
        parameter_names=[function_name],
        method="delta_method",
        assumptions=[
            *base.assumptions,
            f"estimand {function_name!r} with Jacobian Dg at the true parameter",
        ],
        diagnostics=base.diagnostics,
        smallest_eigenvalue=base.smallest_eigenvalue,
        condition_number=base.condition_number,
    )


def crlb_with_nuisance(
    information: FloatArray,
    sample_size: int,
    target_indices: Sequence[int],
    *,
    parameter_names: Sequence[str],
    method: str = "full_inverse",
) -> CrlbResult:
    """CRLB for a target subvector in the presence of nuisance parameters.

    ``method="full_inverse"`` uses the target sub-block of the full inverse;
    ``method="schur"`` uses the Schur complement.  Both are the same matrix when
    the full inverse exists, which is checked by the caller/tests.
    """
    array = np.asarray(information, dtype=np.float64)
    indices = list(target_indices)
    nuisance = [index for index in range(array.shape[0]) if index not in indices]
    names = list(parameter_names)
    target_names = [names[index] for index in indices]
    inverse, reason, smallest, condition = _inverse_or_reason(array)
    if method == "schur" and nuisance:
            a = array[np.ix_(indices, indices)]
            b = array[np.ix_(indices, nuisance)]
            d = array[np.ix_(nuisance, nuisance)]
            d_inverse, d_reason, _, _ = _inverse_or_reason(d)
            if d_inverse is None:
                return CrlbResult(
                    crlb=None,
                    standard_errors=None,
                    parameter_names=target_names,
                    status=MetricStatus.NOT_IDENTIFIABLE,
                    reason_code=d_reason,
                    remedy="nuisance block is not invertible; use the full inverse or reparameterise",
                    method="schur",
                )
            inverse = np.asarray(np.linalg.inv(a - b @ d_inverse @ b.T), dtype=np.float64)
    if inverse is None:
        return CrlbResult(
            crlb=None,
            standard_errors=None,
            parameter_names=target_names,
            status=MetricStatus.NOT_IDENTIFIABLE,
            reason_code=reason,
            remedy="full information matrix is singular; target CRLB is unavailable",
            method=method,
            smallest_eigenvalue=smallest,
            condition_number=condition,
        )
    target = inverse[np.ix_(indices, indices)] / float(sample_size)
    return CrlbResult(
        crlb=target,
        standard_errors=np.sqrt(np.clip(np.diag(target), 0.0, None)),
        parameter_names=target_names,
        method=method,
        assumptions=[
            "n independent and identically distributed observations",
            "target sub-block of the full inverse (never the inverse of the target block)",
            f"nuisance parameters: {[names[index] for index in nuisance]}",
        ],
        smallest_eigenvalue=smallest,
        condition_number=condition,
    )


def is_valid_crlb(matrix: FloatArray, *, atol: float = 1e-12) -> bool:
    """CRLB matrices must be symmetric PSD."""
    array = np.asarray(matrix, dtype=np.float64)
    return bool(np.max(np.abs(array - array.T)) <= atol) and is_psd(array, atol=atol)
