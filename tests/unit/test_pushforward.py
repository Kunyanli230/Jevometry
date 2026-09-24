"""Aggregation mappings, information loss and the data-processing check."""

from __future__ import annotations

import numpy as np
import pytest

from jevometry.schemas.common import MetricStatus
from jevometry.schemas.joint import ConstructionMode, JointDistribution
from jevometry.systems.pushforward import (
    AggregationMap,
    information_loss,
    pushforward,
)


def two_outcome_joint() -> tuple[JointDistribution, np.ndarray]:
    joint = JointDistribution(
        mode=ConstructionMode.EXPLICIT,
        node_order=["Y"],
        outcomes=[("a",), ("b",)],
        probabilities=[0.25, 0.75],
    )
    jacobian = np.asarray([[-1.0], [1.0]])
    return joint, jacobian


def test_pushforward_merges_outcomes() -> None:
    joint = JointDistribution(
        mode=ConstructionMode.EXPLICIT,
        node_order=["Y"],
        outcomes=[("a",), ("b",), ("c",)],
        probabilities=[0.2, 0.3, 0.5],
    )
    aggregation = AggregationMap(
        name="coarse",
        mapping={("a",): "low", ("b",): "low", ("c",): "high"},
    )
    result = pushforward(joint, aggregation)
    masses = dict(zip(result.outcomes, result.probabilities, strict=True))
    assert masses[("low",)] == pytest.approx(0.5)
    assert masses[("high",)] == pytest.approx(0.5)


def test_fixed_mapping_does_not_increase_information() -> None:
    joint, jacobian = two_outcome_joint()
    aggregation = AggregationMap(name="all", mapping={("a",): "x", ("b",): "x"})
    result = information_loss(joint, jacobian, aggregation, parameter_names=["p"])
    assert result.status is MetricStatus.OK
    assert result.psd is True
    assert result.fisher_joint is not None
    assert result.fisher_aggregate is not None
    assert result.fisher_aggregate[0, 0] == pytest.approx(0.0, abs=1e-12)
    assert result.difference is not None
    assert result.difference[0, 0] == pytest.approx(result.fisher_joint[0, 0])


def test_identity_mapping_loses_nothing() -> None:
    joint, jacobian = two_outcome_joint()
    aggregation = AggregationMap(name="identity", mapping={("a",): "a", ("b",): "b"})
    result = information_loss(joint, jacobian, aggregation, parameter_names=["p"])
    assert result.difference is not None
    assert np.max(np.abs(result.difference)) < 1e-12


def test_theta_dependent_mapping_is_refused() -> None:
    joint, jacobian = two_outcome_joint()
    aggregation = AggregationMap(
        name="policy", mapping={("a",): "x", ("b",): "y"}, theta_dependent=True
    )
    result = information_loss(joint, jacobian, aggregation, parameter_names=["p"])
    assert result.status is MetricStatus.UNSUPPORTED
    assert result.reason_code == "policy_changed"


def test_zero_probability_is_refused() -> None:
    joint = JointDistribution(
        mode=ConstructionMode.EXPLICIT,
        node_order=["Y"],
        outcomes=[("a",), ("b",)],
        probabilities=[1.0, 0.0],
    )
    jacobian = np.asarray([[-1.0], [1.0]])
    aggregation = AggregationMap(name="identity", mapping={("a",): "a", ("b",): "b"})
    result = information_loss(joint, jacobian, aggregation, parameter_names=["p"])
    assert result.status is MetricStatus.UNDEFINED
    assert result.reason_code == "zero_probability_outcome"


def test_aggregation_must_cover_all_outcomes() -> None:
    joint, _ = two_outcome_joint()
    aggregation = AggregationMap(name="partial", mapping={("a",): "a"})
    with pytest.raises(KeyError):
        pushforward(joint, aggregation)
