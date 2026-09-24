"""Property tests for the distance and information primitives."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from jevometry.geometry.distances import (
    entropy,
    fisher_rao_distance,
    hellinger_distance,
    js_divergence,
    kl_divergence,
)
from jevometry.schemas.common import MetricStatus

MAX_SIZE = 8


@st.composite
def probability_vectors(draw: st.DrawFn, size: int | None = None) -> np.ndarray:
    length = size or draw(st.integers(min_value=2, max_value=MAX_SIZE))
    weights = draw(
        arrays(
            dtype=np.float64,
            shape=(length,),
            elements=st.floats(
                min_value=1e-6, max_value=1.0, allow_nan=False, allow_infinity=False
            ),
        )
    )
    return weights / weights.sum()


@st.composite
def matching_pairs(draw: st.DrawFn, count: int = 2) -> list[np.ndarray]:
    size = draw(st.integers(min_value=2, max_value=MAX_SIZE))
    return [draw(probability_vectors(size)) for _ in range(count)]


@given(vectors=matching_pairs(2))
@settings(max_examples=150, deadline=None)
def test_hellinger_is_a_bounded_symmetric_metric(vectors: list[np.ndarray]) -> None:
    p, q = vectors
    distance = hellinger_distance(p, q)
    reverse = hellinger_distance(q, p)
    assert distance.value is not None and reverse.value is not None
    assert 0.0 - 1e-12 <= distance.value <= 1.0 + 1e-12
    assert distance.value == pytest.approx(reverse.value, abs=1e-9)


@given(vectors=matching_pairs(3))
@settings(max_examples=80, deadline=None)
def test_hellinger_triangle_inequality(vectors: list[np.ndarray]) -> None:
    p, q, r = vectors
    pq = hellinger_distance(p, q).value
    qr = hellinger_distance(q, r).value
    pr = hellinger_distance(p, r).value
    assert pq is not None and qr is not None and pr is not None
    assert pr <= pq + qr + 1e-9


@given(vectors=matching_pairs(2))
@settings(max_examples=150, deadline=None)
def test_fisher_rao_matches_bhattacharyya(vectors: list[np.ndarray]) -> None:
    p, q = vectors
    coefficient = float(np.sum(np.sqrt(p * q)))
    if 1.0 - coefficient <= 1e-12:
        expected = 0.0
    else:
        expected = 2.0 * math.acos(min(max(coefficient, -1.0), 1.0))
    result = fisher_rao_distance(p, q)
    assert result.value is not None
    assert abs(result.value - expected) < 1e-9
    assert result.value <= math.pi + 1e-9


@given(vectors=matching_pairs(2))
@settings(max_examples=150, deadline=None)
def test_kl_is_non_negative(vectors: list[np.ndarray]) -> None:
    p, q = vectors
    result = kl_divergence(p, q)
    if result.status is MetricStatus.OK:
        assert result.value is not None
        assert result.value >= -1e-9
    else:
        assert result.boundary is True
        assert result.value is None


@given(p=probability_vectors())
@settings(max_examples=100, deadline=None)
def test_self_distances_are_zero(p: np.ndarray) -> None:
    assert kl_divergence(p, p).value == pytest.approx(0.0, abs=1e-9)
    assert js_divergence(p, p).value == pytest.approx(0.0, abs=1e-9)
    assert hellinger_distance(p, p).value == pytest.approx(0.0, abs=1e-9)
    assert fisher_rao_distance(p, p).value == pytest.approx(0.0, abs=1e-9)


@given(vectors=matching_pairs(2))
@settings(max_examples=150, deadline=None)
def test_js_divergence_is_bounded_by_log_two(vectors: list[np.ndarray]) -> None:
    p, q = vectors
    result = js_divergence(p, q)
    assert result.value is not None
    assert -1e-9 <= result.value <= math.log(2.0) + 1e-9


@given(p=probability_vectors())
@settings(max_examples=100, deadline=None)
def test_entropy_bounds(p: np.ndarray) -> None:
    value = entropy(p)
    assert -1e-12 <= value <= math.log(len(p)) + 1e-9


@given(vectors=matching_pairs(2))
@settings(max_examples=150, deadline=None)
def test_js_symmetric(vectors: list[np.ndarray]) -> None:
    p, q = vectors
    left = js_divergence(p, q).value
    right = js_divergence(q, p).value
    assert left is not None and right is not None
    assert abs(left - right) < 1e-9
