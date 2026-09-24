"""Closed-form distribution statistics and distances.

Every function consumes validated probability vectors in nats.  Boundary
cases (a positive mass against a zero mass) return a structured
``DistanceValue`` rather than a fabricated finite number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from jevometry.schemas.common import MetricStatus

FloatArray = NDArray[np.float64]

_CLAMP_TOLERANCE = 1e-12


@dataclass(frozen=True)
class DistanceValue:
    """A distance or divergence with its status and boundary flag."""

    value: float | None
    status: MetricStatus = MetricStatus.OK
    boundary: bool = False
    reason_code: str | None = None
    clamped: bool = False


def _as_array(values: NDArray[np.float64] | list[float]) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("probability vector must be one-dimensional")
    return array


def _clamp(value: float, *, low: float = -1.0, high: float = 1.0) -> tuple[float, bool]:
    if value < low:
        return low, True
    if value > high:
        return high, True
    return value, False


def entropy(probabilities: NDArray[np.float64] | list[float]) -> float:
    """Shannon entropy in nats, with the convention 0 log 0 = 0."""
    q = _as_array(probabilities)
    positive = q[q > 0.0]
    if positive.size == 0:
        return 0.0
    return float(-np.sum(positive * np.log(positive)))


def expected_value(
    probabilities: NDArray[np.float64] | list[float],
    encodings: NDArray[np.float64] | list[float],
) -> float:
    """Expected value under an explicit numeric encoding of the support."""
    q = _as_array(probabilities)
    a = _as_array(encodings)
    if q.shape != a.shape:
        raise ValueError("probabilities and encodings must have equal length")
    return float(np.dot(q, a))


def variance(
    probabilities: NDArray[np.float64] | list[float],
    encodings: NDArray[np.float64] | list[float],
) -> float:
    """Variance under an explicit numeric encoding of the support."""
    q = _as_array(probabilities)
    a = _as_array(encodings)
    mean = expected_value(q, a)
    return float(np.dot(q, (a - mean) ** 2))


def bhattacharyya_coefficient(
    p: NDArray[np.float64] | list[float],
    q: NDArray[np.float64] | list[float],
) -> DistanceValue:
    """Bhattacharyya coefficient BC = sum sqrt(p_k q_k)."""
    p_array = _as_array(p)
    q_array = _as_array(q)
    if p_array.shape != q_array.shape:
        raise ValueError("probability vectors must have equal length")
    coefficient = float(np.sum(np.sqrt(p_array * q_array)))
    clamped_value, clamped = _clamp(coefficient)
    return DistanceValue(
        value=clamped_value,
        status=MetricStatus.CONDITIONAL if clamped else MetricStatus.OK,
        reason_code="machine_clamp" if clamped else None,
        clamped=clamped,
    )


def hellinger_distance(
    p: NDArray[np.float64] | list[float],
    q: NDArray[np.float64] | list[float],
) -> DistanceValue:
    """Normalised Hellinger distance H = sqrt(1 - BC), in [0, 1].

    ``1 - BC`` is clamped to zero only at machine level for vectors that are
    equal up to floating-point rounding; the clamp is recorded.
    """
    coefficient = bhattacharyya_coefficient(p, q)
    assert coefficient.value is not None
    squared = 1.0 - coefficient.value
    clamped = coefficient.clamped
    if squared <= _CLAMP_TOLERANCE:
        squared = 0.0
        clamped = True
    return DistanceValue(
        value=math.sqrt(squared),
        status=coefficient.status,
        reason_code=coefficient.reason_code,
        clamped=clamped,
    )


def fisher_rao_distance(
    p: NDArray[np.float64] | list[float],
    q: NDArray[np.float64] | list[float],
) -> DistanceValue:
    """Categorical Fisher-Rao distance d = 2 arccos(BC) on the full simplex.

    This is the geodesic distance of the complete categorical simplex.  It is
    not an exact geodesic distance inside a restricted parametric family.
    """
    coefficient = bhattacharyya_coefficient(p, q)
    assert coefficient.value is not None
    if 1.0 - coefficient.value <= _CLAMP_TOLERANCE:
        return DistanceValue(
            value=0.0,
            status=coefficient.status,
            reason_code=coefficient.reason_code,
            clamped=True,
        )
    return DistanceValue(
        value=float(2.0 * math.acos(coefficient.value)),
        status=coefficient.status,
        reason_code=coefficient.reason_code,
        clamped=coefficient.clamped,
    )


def kl_divergence(
    p: NDArray[np.float64] | list[float],
    q: NDArray[np.float64] | list[float],
) -> DistanceValue:
    """KL(p || q) in nats; positive infinity when p_k > 0 and q_k = 0."""
    p_array = _as_array(p)
    q_array = _as_array(q)
    if p_array.shape != q_array.shape:
        raise ValueError("probability vectors must have equal length")
    boundary = bool(np.any((p_array > 0.0) & (q_array == 0.0)))
    if boundary:
        return DistanceValue(
            value=None,
            status=MetricStatus.CONDITIONAL,
            boundary=True,
            reason_code="boundary_zero_probability",
        )
    positive = p_array > 0.0
    terms = p_array[positive] * (np.log(p_array[positive]) - np.log(q_array[positive]))
    return DistanceValue(value=float(np.sum(terms)))


def js_divergence(
    p: NDArray[np.float64] | list[float],
    q: NDArray[np.float64] | list[float],
) -> DistanceValue:
    """Jensen-Shannon divergence in nats (always finite, in [0, log 2])."""
    p_array = _as_array(p)
    q_array = _as_array(q)
    if p_array.shape != q_array.shape:
        raise ValueError("probability vectors must have equal length")
    midpoint = 0.5 * (p_array + q_array)
    left = kl_divergence(p_array, midpoint)
    right = kl_divergence(q_array, midpoint)
    if left.value is None or right.value is None:
        return DistanceValue(
            value=None,
            status=MetricStatus.CONDITIONAL,
            boundary=True,
            reason_code="boundary_zero_probability",
        )
    return DistanceValue(value=0.5 * (left.value + right.value))


def kl_matrix(
    p: NDArray[np.float64] | list[float],
    q: NDArray[np.float64] | list[float],
) -> DistanceValue:
    """Alias kept for symmetry with the report vocabulary."""
    return kl_divergence(p, q)


def js_distance(
    p: NDArray[np.float64] | list[float],
    q: NDArray[np.float64] | list[float],
) -> DistanceValue:
    """Metric JS distance sqrt(JS divergence)."""
    divergence = js_divergence(p, q)
    if divergence.value is None:
        return divergence
    return DistanceValue(value=math.sqrt(divergence.value))
