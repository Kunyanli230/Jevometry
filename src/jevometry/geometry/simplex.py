"""Probability-vector validation, alignment and working-simplex derivation.

Raw probabilities are always preserved.  A working copy is only produced when
the raw vector is already a valid probability vector within tolerance; the
derivation records the original total and the applied correction so that API
rounding can never be silently hidden.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from jevometry.schemas.common import Primitive
from jevometry.schemas.distribution import DistributionRecord, DistributionSource

DEFAULT_SUM_TOLERANCE = 1e-6

FloatArray = NDArray[np.float64]


class ProbabilityValidationError(ValueError):
    """A probability vector failed validation; carries a stable reason code."""

    def __init__(self, reason_code: str, message: str, remedy: str | None = None) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.remedy = remedy


@dataclass(frozen=True)
class SimplexVector:
    """A validated probability vector with raw and working copies."""

    support: tuple[str, ...]
    raw: FloatArray
    working: FloatArray
    raw_total: float
    correction: float
    derived: bool
    tolerance: float = DEFAULT_SUM_TOLERANCE
    assumptions: tuple[str, ...] = field(default=())

    def as_record(
        self,
        *,
        node_id: str,
        question_id: str,
        primitive: Primitive,
        source: DistributionSource,
        semantic_hash: str,
        case_id: str,
        point_id: str,
        repeat: int = 0,
        theta: dict[str, float] | None = None,
        selected: str | None = None,
        confidence: float | None = None,
        legend: dict[str, str] | None = None,
        numeric_encoding: dict[str, float] | None = None,
        request_fingerprint: str | None = None,
    ) -> DistributionRecord:
        return DistributionRecord(
            node_id=node_id,
            question_id=question_id,
            primitive=primitive,
            support=list(self.support),
            raw=[float(value) for value in self.raw],
            working=[float(value) for value in self.working] if self.derived else None,
            raw_total=self.raw_total,
            correction=self.correction,
            derived=self.derived,
            selected=selected,
            confidence=confidence,
            source=source,
            legend=legend,
            numeric_encoding=numeric_encoding,
            semantic_hash=semantic_hash,
            case_id=case_id,
            point_id=point_id,
            repeat=repeat,
            theta=dict(theta or {}),
            request_fingerprint=request_fingerprint,
            assumptions=list(self.assumptions),
        )


def validate_probabilities(
    support: Sequence[str],
    values: Sequence[float] | NDArray[np.float64],
    *,
    tolerance: float = DEFAULT_SUM_TOLERANCE,
    label: str = "distribution",
) -> SimplexVector:
    """Validate a probability vector and derive its working copy.

    Raises ``ProbabilityValidationError`` instead of clipping, smoothing or
    filling missing outcomes.
    """
    support_tuple = tuple(str(item) for item in support)
    if not support_tuple:
        raise ProbabilityValidationError(
            "empty_support", f"{label}: support must contain at least one outcome"
        )
    if len(support_tuple) != len(set(support_tuple)):
        raise ProbabilityValidationError(
            "duplicate_support", f"{label}: support ids must be unique"
        )
    if len(values) != len(support_tuple):
        raise ProbabilityValidationError(
            "support_length_mismatch",
            f"{label}: {len(values)} probabilities for {len(support_tuple)} outcomes",
        )
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ProbabilityValidationError("shape", f"{label}: probabilities must be 1-D")
    if not np.all(np.isfinite(array)):
        raise ProbabilityValidationError(
            "non_finite",
            f"{label}: probabilities must all be finite",
            remedy="fix the provider response; NaN/Infinity cannot be treated as probabilities",
        )
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise ProbabilityValidationError(
            "out_of_range",
            f"{label}: probabilities must lie in [0, 1]",
            remedy="report the raw response and investigate the provider",
        )
    total = float(array.sum())
    if abs(total - 1.0) > tolerance:
        raise ProbabilityValidationError(
            "sum_out_of_tolerance",
            f"{label}: probabilities sum to {total!r}, outside tolerance {tolerance!r}",
            remedy="do not renormalise silently; fix the source or raise the documented tolerance",
        )
    correction = 1.0 - total
    derived = correction != 0.0
    assumptions: tuple[str, ...]
    if derived:
        working = array / total
        assumptions = (
            f"working simplex renormalised from raw total {total!r}; correction {correction!r}",
        )
    else:
        working = array.copy()
        assumptions = ()
    return SimplexVector(
        support=support_tuple,
        raw=array.copy(),
        working=working,
        raw_total=total,
        correction=correction,
        derived=derived,
        tolerance=tolerance,
        assumptions=assumptions,
    )


def noul_vector(noul: float, *, tolerance: float = DEFAULT_SUM_TOLERANCE) -> SimplexVector:
    """Convert a Noul value into a validated (false, true) probability vector."""
    if not math.isfinite(noul) or not (0.0 <= noul <= 1.0):
        raise ProbabilityValidationError(
            "noul_out_of_range",
            f"noul value {noul!r} must be finite and in [0, 1]",
        )
    return validate_probabilities(("false", "true"), (1.0 - noul, noul), tolerance=tolerance)


def align_support(
    left: SimplexVector,
    right: SimplexVector,
    *,
    mapping: dict[str, str] | None = None,
    left_name: str = "left",
    right_name: str = "right",
) -> tuple[FloatArray, FloatArray, tuple[str, ...]]:
    """Align two probability vectors onto a shared support.

    Positional comparison is only allowed when the supports are identical or an
    explicit injective mapping from left ids to right ids is supplied.
    """
    if left.support == right.support:
        return left.working.copy(), right.working.copy(), left.support
    if mapping is None:
        raise ProbabilityValidationError(
            "support_mismatch",
            f"{left_name} support {left.support!r} != {right_name} support {right.support!r}",
            remedy="supply a confirmed one-to-one SupportMapping; never align by list position",
        )
    if set(mapping) != set(left.support):
        raise ProbabilityValidationError(
            "support_mapping_incomplete",
            f"mapping must cover exactly the {left_name} support",
        )
    if len(set(mapping.values())) != len(mapping):
        raise ProbabilityValidationError("support_mapping_not_injective", "mapping must be injective")
    right_index = {outcome: index for index, outcome in enumerate(right.support)}
    for target in mapping.values():
        if target not in right_index:
            raise ProbabilityValidationError(
                "support_mapping_unknown_target", f"mapped outcome {target!r} is not in {right_name}"
            )
    left_values = left.working.copy()
    right_values = np.asarray([right.working[right_index[mapping[key]]] for key in left.support])
    return left_values, right_values, left.support
