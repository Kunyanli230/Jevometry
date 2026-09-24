"""Shared analytic fixtures for the test suite."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy.special import expit

from jevometry.systems.tree import ConditionalNode, ConditionalTree


def sigmoid(value: float) -> float:
    return float(expit(value))


def two_layer_tree(parameter: str = "p", delta: float = 1.0) -> ConditionalTree:
    """Y in {a,b} with P(b)=p; Z|Y has parameter theta shifted by delta when Y=b."""

    def y_distribution(theta: Mapping[str, float], history: Mapping[str, str]) -> Mapping[str, float]:
        del history
        p = float(theta[parameter])
        return {"a": 1.0 - p, "b": p}

    def y_jacobian(
        theta: Mapping[str, float], history: Mapping[str, str]
    ) -> Mapping[str, Mapping[str, float]]:
        del theta, history
        return {"a": {parameter: -1.0}, "b": {parameter: 1.0}}

    def z_distribution(theta: Mapping[str, float], history: Mapping[str, str]) -> Mapping[str, float]:
        shift = delta if history["Y"] == "b" else 0.0
        probability = sigmoid(float(theta[parameter]) + shift)
        return {"0": 1.0 - probability, "1": probability}

    def z_jacobian(
        theta: Mapping[str, float], history: Mapping[str, str]
    ) -> Mapping[str, Mapping[str, float]]:
        shift = delta if history["Y"] == "b" else 0.0
        probability = sigmoid(float(theta[parameter]) + shift)
        derivative = probability * (1.0 - probability)
        return {"0": {parameter: -derivative}, "1": {parameter: derivative}}

    return ConditionalTree(
        tree_id="two-layer",
        nodes=[
            ConditionalNode("Y", ("a", "b"), y_distribution, y_jacobian),
            ConditionalNode("Z", ("0", "1"), z_distribution, z_jacobian),
        ],
    )


def bernoulli_probabilities(p: float) -> np.ndarray:
    return np.asarray([1.0 - p, p], dtype=np.float64)


def bernoulli_jacobian() -> np.ndarray:
    return np.asarray([[-1.0], [1.0]], dtype=np.float64)
