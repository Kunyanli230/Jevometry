"""Simplex validation, raw preservation and support alignment."""

from __future__ import annotations

import math

import numpy as np
import pytest

from jevometry.geometry.simplex import (
    ProbabilityValidationError,
    align_support,
    noul_vector,
    validate_probabilities,
)


def test_valid_vector_keeps_raw_and_derives_working() -> None:
    vector = validate_probabilities(("a", "b"), (0.25, 0.75))
    assert vector.raw_total == pytest.approx(1.0)
    assert vector.derived is False
    assert vector.working == pytest.approx([0.25, 0.75])


def test_rounding_within_tolerance_derives_working_copy() -> None:
    raw = (0.1, 0.2, 0.7000004)
    vector = validate_probabilities(("a", "b", "c"), raw)
    assert vector.raw == pytest.approx(raw)
    assert vector.raw_total == pytest.approx(1.0000004)
    assert vector.derived is True
    assert vector.correction == pytest.approx(-4e-7)
    assert vector.working.sum() == pytest.approx(1.0)
    assert vector.assumptions


def test_sum_outside_tolerance_is_rejected_not_renormalised() -> None:
    with pytest.raises(ProbabilityValidationError) as error:
        validate_probabilities(("a", "b"), (0.5, 0.6))
    assert error.value.reason_code == "sum_out_of_tolerance"
    assert error.value.remedy


@pytest.mark.parametrize(
    ("values", "code"),
    [
        ((0.5,), "support_length_mismatch"),
        ((0.5, float("nan")), "non_finite"),
        ((0.5, float("inf")), "non_finite"),
        ((1.5, -0.5), "out_of_range"),
    ],
)
def test_invalid_vectors_are_rejected(values: tuple[float, float], code: str) -> None:
    with pytest.raises(ProbabilityValidationError) as error:
        validate_probabilities(("a", "b"), values)
    assert error.value.reason_code == code


def test_duplicate_support_rejected() -> None:
    with pytest.raises(ProbabilityValidationError) as error:
        validate_probabilities(("a", "a"), (0.5, 0.5))
    assert error.value.reason_code == "duplicate_support"


def test_noul_conversion() -> None:
    vector = noul_vector(0.3)
    assert vector.support == ("false", "true")
    assert vector.working == pytest.approx([0.7, 0.3])
    with pytest.raises(ProbabilityValidationError):
        noul_vector(1.2)


def test_alignment_requires_matching_or_mapping() -> None:
    left = validate_probabilities(("a", "b"), (0.4, 0.6))
    same = validate_probabilities(("a", "b"), (0.4, 0.6))
    left_values, right_values, support = align_support(left, same)
    assert support == ("a", "b")
    assert left_values == pytest.approx(right_values)

    reordered = validate_probabilities(("b", "a"), (0.6, 0.4))
    with pytest.raises(ProbabilityValidationError) as error:
        align_support(left, reordered)
    assert error.value.reason_code == "support_mismatch"

    left_values, right_values, support = align_support(
        left, reordered, mapping={"a": "a", "b": "b"}
    )
    assert support == ("a", "b")
    assert left_values == pytest.approx(right_values)


def test_alignment_mapping_must_be_injective() -> None:
    left = validate_probabilities(("a", "b"), (0.4, 0.6))
    right = validate_probabilities(("c",), (1.0,))
    with pytest.raises(ProbabilityValidationError):
        align_support(left, right, mapping={"a": "c", "b": "c"})


def test_raw_is_never_mutated_by_working_copy() -> None:
    raw = (0.5, 0.5 + 5e-7)
    vector = validate_probabilities(("a", "b"), raw)
    vector.working[0] = 0.0
    assert vector.raw[0] == 0.5
    assert not np.array_equal(vector.raw, vector.working)
    assert math.isclose(vector.raw.sum(), 1.0 + 5e-7)
