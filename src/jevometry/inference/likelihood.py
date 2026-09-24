"""Finite-outcome likelihood, bounded MLE and profile-likelihood intervals."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq, minimize
from scipy.stats import chi2

from jevometry.adapters.base import LikelihoodModel
from jevometry.schemas.common import MetricStatus
from jevometry.schemas.parameters import ParameterSpec

BOUND_EPSILON = 1e-9


@dataclass
class MleResult:
    """Maximum-likelihood estimate for a declared finite likelihood model."""

    theta: dict[str, float] | None
    log_likelihood: float | None
    parameter_names: list[str]
    status: MetricStatus = MetricStatus.OK
    reason_code: str | None = None
    converged: bool = True
    n_observations: int = 0
    n_iterations: int | None = None
    message: str | None = None


@dataclass
class ProfileInterval:
    """Profile-likelihood interval for one parameter."""

    parameter: str
    estimate: float
    lower: float | None
    upper: float | None
    level: float
    status: MetricStatus = MetricStatus.OK
    reason_code: str | None = None
    message: str | None = None
    width: float | None = None
    assumptions: list[str] = field(
        default_factory=lambda: [
            "asymptotic chi-square profile-likelihood interval",
            "regular model and interior maximum",
        ]
    )


def _vector(theta: Mapping[str, float], names: Sequence[str]) -> np.ndarray:
    return np.asarray([float(theta[name]) for name in names], dtype=np.float64)


def _bounds(specs: Sequence[ParameterSpec]) -> list[tuple[float, float]]:
    bounds: list[tuple[float, float]] = []
    for spec in specs:
        low, high = spec.bounds
        bounds.append((low + BOUND_EPSILON, high - BOUND_EPSILON))
    return bounds


def mle(
    model: LikelihoodModel,
    observations: Sequence[int] | NDArray[np.int64],
    *,
    parameter_specs: Sequence[ParameterSpec],
    initial: Mapping[str, float] | None = None,
) -> MleResult:
    """Bounded maximum-likelihood estimate."""
    names = [spec.name for spec in parameter_specs]
    if initial is None:
        start = np.asarray(
            [
                spec.center
                if spec.center is not None
                else 0.5 * (spec.bounds[0] + spec.bounds[1])
                for spec in parameter_specs
            ],
            dtype=np.float64,
        )
    else:
        start = _vector(initial, names)
    observations = list(observations)

    def objective(vector: np.ndarray) -> float:
        theta = dict(zip(names, vector.tolist(), strict=True))
        try:
            value = model.log_prob(observations, theta)
        except (ValueError, KeyError, IndexError):
            return 1e12
        if not np.isfinite(value):
            return 1e12
        return -float(value)

    result = minimize(
        objective,
        start,
        method="L-BFGS-B",
        bounds=_bounds(parameter_specs),
        options={"maxiter": 500},
    )
    theta = dict(zip(names, np.asarray(result.x).tolist(), strict=True))
    objective_value = float(result.fun)
    log_likelihood = -objective_value
    if not np.isfinite(log_likelihood) or objective_value >= 1e11:
        return MleResult(
            theta=None,
            log_likelihood=None,
            parameter_names=names,
            status=MetricStatus.FAILED,
            reason_code="likelihood_undefined",
            converged=False,
            n_observations=len(observations),
            message=str(result.message),
        )
    converged = bool(result.success)
    return MleResult(
        theta=theta,
        log_likelihood=log_likelihood,
        parameter_names=names,
        status=MetricStatus.OK if converged else MetricStatus.UNSTABLE,
        reason_code=None if converged else "optimiser_not_converged",
        converged=converged,
        n_observations=len(observations),
        n_iterations=int(result.nit),
        message=str(result.message),
    )


def profile_log_likelihood(
    model: LikelihoodModel,
    observations: Sequence[int] | NDArray[np.int64],
    target: str,
    value: float,
    *,
    parameter_specs: Sequence[ParameterSpec],
    initial: Mapping[str, float] | None = None,
) -> float:
    """Maximise the likelihood over all other parameters with ``target`` fixed."""
    others = [spec for spec in parameter_specs if spec.name != target]
    observations = list(observations)

    def objective(vector: np.ndarray) -> float:
        theta = {target: value}
        for spec, item in zip(others, vector.tolist(), strict=True):
            theta[spec.name] = float(item)
        try:
            likelihood = model.log_prob(observations, theta)
        except (ValueError, KeyError, IndexError):
            return 1e12
        return 1e12 if not np.isfinite(likelihood) else -float(likelihood)

    if not others:
        try:
            likelihood = model.log_prob(observations, {target: value})
        except (ValueError, KeyError, IndexError):
            return -float("inf")
        return float(likelihood)
    start = np.asarray(
        [
            (
                float(initial[spec.name])
                if initial is not None and spec.name in initial
                else (
                    spec.center
                    if spec.center is not None
                    else 0.5 * (spec.bounds[0] + spec.bounds[1])
                )
            )
            for spec in others
        ],
        dtype=np.float64,
    )
    result = minimize(
        objective,
        start,
        method="L-BFGS-B",
        bounds=_bounds(others),
        options={"maxiter": 500},
    )
    value_opt = -float(result.fun)
    return value_opt


def profile_interval(
    model: LikelihoodModel,
    observations: Sequence[int] | NDArray[np.int64],
    target: str,
    *,
    parameter_specs: Sequence[ParameterSpec],
    level: float = 0.95,
    initial: Mapping[str, float] | None = None,
) -> ProfileInterval:
    """Profile-likelihood interval based on the chi-square(1) threshold."""
    spec = next((item for item in parameter_specs if item.name == target), None)
    if spec is None:
        return ProfileInterval(
            parameter=target,
            estimate=float("nan"),
            lower=None,
            upper=None,
            level=level,
            status=MetricStatus.FAILED,
            reason_code="unknown_parameter",
        )
    fit = mle(model, observations, parameter_specs=parameter_specs, initial=initial)
    if fit.theta is None or fit.log_likelihood is None:
        return ProfileInterval(
            parameter=target,
            estimate=float("nan"),
            lower=None,
            upper=None,
            level=level,
            status=MetricStatus.FAILED,
            reason_code=fit.reason_code or "mle_failed",
        )
    estimate = fit.theta[target]
    threshold = fit.log_likelihood - 0.5 * float(chi2.ppf(level, df=1))
    low, high = spec.bounds
    low_edge = low + BOUND_EPSILON
    high_edge = high - BOUND_EPSILON

    def gap(value: float) -> float:
        return profile_log_likelihood(
            model,
            observations,
            target,
            value,
            parameter_specs=parameter_specs,
            initial=fit.theta,
        ) - threshold

    def find_root(direction: int) -> float | None:
        span = max(spec.step, 1e-3)
        edge = low_edge if direction < 0 else high_edge
        previous_value = estimate
        gap(previous_value)
        for _ in range(60):
            candidate = previous_value + direction * span
            if direction < 0 and candidate <= low_edge:
                candidate = low_edge
            if direction > 0 and candidate >= high_edge:
                candidate = high_edge
            candidate_gap = gap(candidate)
            if candidate_gap <= 0.0:
                try:
                    return float(
                        brentq(gap, min(previous_value, candidate), max(previous_value, candidate))
                    )
                except ValueError:
                    return candidate
            if candidate == edge:
                return None
            previous_value = candidate
            span *= 1.6
        return None

    lower = find_root(-1)
    upper = find_root(1)
    if lower is None or upper is None:
        return ProfileInterval(
            parameter=target,
            estimate=estimate,
            lower=lower,
            upper=upper,
            level=level,
            status=MetricStatus.CONDITIONAL,
            reason_code="interval_boundary_reached",
            message="profile interval reached the parameter bound or failed to bracket",
        )
    return ProfileInterval(
        parameter=target,
        estimate=estimate,
        lower=lower,
        upper=upper,
        level=level,
        width=upper - lower,
    )
