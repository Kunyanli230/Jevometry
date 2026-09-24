"""Analytic expectations for entropy, divergences and distances."""

from __future__ import annotations

import math

import numpy as np
import pytest

from jevometry.geometry.distances import (
    bhattacharyya_coefficient,
    entropy,
    expected_value,
    fisher_rao_distance,
    hellinger_distance,
    js_divergence,
    kl_divergence,
    variance,
)
from jevometry.schemas.common import MetricStatus


def test_entropy_uniform_and_delta() -> None:
    assert entropy([0.25, 0.25, 0.25, 0.25]) == pytest.approx(math.log(4.0))
    assert entropy([1.0, 0.0]) == pytest.approx(0.0)
    assert entropy([0.5, 0.5]) == pytest.approx(math.log(2.0))


def test_entropy_zero_log_zero_convention() -> None:
    assert entropy([0.0, 0.0, 1.0]) == pytest.approx(0.0)


def test_expected_value_and_variance() -> None:
    q = [0.2, 0.3, 0.5]
    a = [1.0, 2.0, 4.0]
    mean = 0.2 * 1.0 + 0.3 * 2.0 + 0.5 * 4.0
    assert expected_value(q, a) == pytest.approx(mean)
    expected_variance = 0.2 * (1 - mean) ** 2 + 0.3 * (2 - mean) ** 2 + 0.5 * (4 - mean) ** 2
    assert variance(q, a) == pytest.approx(expected_variance)


def test_kl_analytic_value_and_asymmetry() -> None:
    p = [0.5, 0.5]
    q = [0.25, 0.75]
    expected = 0.5 * math.log(0.5 / 0.25) + 0.5 * math.log(0.5 / 0.75)
    result = kl_divergence(p, q)
    assert result.value == pytest.approx(expected)
    reverse = kl_divergence(q, p)
    assert reverse.value is not None
    assert reverse.value != pytest.approx(result.value)


def test_kl_boundary_is_positive_infinity_not_a_number() -> None:
    result = kl_divergence([0.5, 0.5], [1.0, 0.0])
    assert result.value is None
    assert result.boundary is True
    assert result.status is MetricStatus.CONDITIONAL
    assert result.reason_code == "boundary_zero_probability"


def test_js_bounds_and_identical_vectors() -> None:
    assert js_divergence([0.5, 0.5], [0.5, 0.5]).value == pytest.approx(0.0)
    orthogonal = js_divergence([1.0, 0.0], [0.0, 1.0])
    assert orthogonal.value == pytest.approx(math.log(2.0))


def test_js_is_symmetric() -> None:
    p = [0.1, 0.2, 0.7]
    q = [0.3, 0.3, 0.4]
    assert js_divergence(p, q).value == pytest.approx(js_divergence(q, p).value)


def test_hellinger_matches_analytic_formula() -> None:
    p = [0.5, 0.5]
    q = [0.25, 0.75]
    coefficient = math.sqrt(0.5 * 0.25) + math.sqrt(0.5 * 0.75)
    expected = math.sqrt(1.0 - coefficient)
    assert bhattacharyya_coefficient(p, q).value == pytest.approx(coefficient)
    assert hellinger_distance(p, q).value == pytest.approx(expected)
    assert hellinger_distance(p, p).value == pytest.approx(0.0)


def test_fisher_rao_has_factor_two() -> None:
    p = [1.0, 0.0]
    q = [0.0, 1.0]
    result = fisher_rao_distance(p, q)
    assert result.value == pytest.approx(math.pi)
    assert fisher_rao_distance(p, p).value == pytest.approx(0.0)


def test_fisher_rao_relation_to_hellinger_on_orthogonal_supports() -> None:
    p = [0.5, 0.5]
    q = [0.25, 0.75]
    coefficient = bhattacharyya_coefficient(p, q).value
    assert coefficient is not None
    fr = fisher_rao_distance(p, q).value
    hell = hellinger_distance(p, q).value
    assert fr == pytest.approx(2.0 * math.acos(coefficient))
    assert hell == pytest.approx(math.sqrt(1.0 - coefficient))


def test_distance_zero_iff_equal() -> None:
    q = np.array([0.2, 0.3, 0.5])
    assert hellinger_distance(q, q).value == pytest.approx(0.0)
    assert fisher_rao_distance(q, q).value == pytest.approx(0.0)
    assert js_divergence(q, q).value == pytest.approx(0.0)
    assert kl_divergence(q, q).value == pytest.approx(0.0)


def test_hellinger_and_fr_clamp_machine_noise() -> None:
    p = np.array([0.5, 0.5])
    q = p + np.array([1e-17, -1e-17])
    coefficient = bhattacharyya_coefficient(p, q)
    assert coefficient.clamped or coefficient.value == pytest.approx(1.0, abs=1e-9)
