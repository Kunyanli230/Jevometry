"""Fixed aggregation mappings and information loss.

For a mapping A = f(Z) that does not depend on theta, the pushforward is
P_theta(A=a) = sum_{z: f(z)=a} P_theta(z).  The data-processing inequality
I_Z - I_A >= 0 is only asserted for such fixed mappings under the same model
and regularity conditions.  A theta-dependent mapping is reported as
non-regular and receives no guarantee.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from jevometry.geometry.diagnostics import is_psd
from jevometry.schemas.common import MetricStatus
from jevometry.schemas.joint import ConstructionMode, JointDistribution

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class AggregationMap:
    """A fixed map from joint outcome tuples to aggregate labels."""

    name: str
    mapping: Mapping[tuple[str, ...], str]
    theta_dependent: bool = False
    description: str | None = None

    def apply(self, outcome: tuple[str, ...]) -> str:
        if outcome not in self.mapping:
            raise KeyError(f"aggregation {self.name!r} does not cover outcome {outcome!r}")
        return self.mapping[outcome]


@dataclass
class InformationLossResult:
    """Information comparison between a joint and its pushforward."""

    fisher_joint: FloatArray | None
    fisher_aggregate: FloatArray | None
    difference: FloatArray | None
    parameter_names: list[str]
    labels: list[str]
    status: MetricStatus = MetricStatus.OK
    reason_code: str | None = None
    psd: bool | None = None
    min_eigenvalue: float | None = None
    assumptions: list[str] = field(default_factory=list)


def pushforward(
    joint: JointDistribution, aggregation: AggregationMap
) -> JointDistribution:
    """Exact pushforward of a declared joint under a fixed mapping."""
    masses: dict[str, float] = {}
    for outcome, probability in zip(joint.outcomes, joint.probabilities, strict=True):
        label = aggregation.apply(outcome)
        masses[label] = masses.get(label, 0.0) + probability
    labels = sorted(masses)
    return JointDistribution(
        mode=ConstructionMode.EXPLICIT,
        node_order=[aggregation.name],
        outcomes=[(label,) for label in labels],
        probabilities=[masses[label] for label in labels],
        assumptions=[
            *joint.assumptions,
            f"pushforward under fixed aggregation {aggregation.name!r}",
        ],
    )


def pushforward_jacobian(
    joint: JointDistribution,
    joint_jacobian: FloatArray,
    aggregation: AggregationMap,
) -> tuple[list[str], FloatArray]:
    """Derivative of the pushforward masses: sum the Jacobian rows per label."""
    jacobian = np.asarray(joint_jacobian, dtype=np.float64)
    if jacobian.shape[0] != len(joint.outcomes):
        raise ValueError("joint Jacobian rows must match joint outcomes")
    labels = sorted({aggregation.apply(outcome) for outcome in joint.outcomes})
    index = {label: position for position, label in enumerate(labels)}
    aggregate = np.zeros((len(labels), jacobian.shape[1]), dtype=np.float64)
    for row, outcome in enumerate(joint.outcomes):
        aggregate[index[aggregation.apply(outcome)], :] += jacobian[row, :]
    return labels, aggregate


def information_loss(
    joint: JointDistribution,
    joint_jacobian: FloatArray,
    aggregation: AggregationMap,
    *,
    parameter_names: Sequence[str],
    zero_atol: float = 1e-15,
) -> InformationLossResult:
    """Compute I_Z, I_A and I_Z - I_A with a PSD acceptance check."""
    names = list(parameter_names)
    if aggregation.theta_dependent:
        return InformationLossResult(
            fisher_joint=None,
            fisher_aggregate=None,
            difference=None,
            parameter_names=names,
            labels=[],
            status=MetricStatus.UNSUPPORTED,
            reason_code="policy_changed",
            assumptions=["theta-dependent mappings receive no data-processing guarantee"],
        )
    values = np.asarray(joint.probabilities, dtype=np.float64)
    if np.any(values <= zero_atol):
        return InformationLossResult(
            fisher_joint=None,
            fisher_aggregate=None,
            difference=None,
            parameter_names=names,
            labels=[],
            status=MetricStatus.UNDEFINED,
            reason_code="zero_probability_outcome",
            assumptions=["no epsilon smoothing applied"],
        )
    aggregate = pushforward(joint, aggregation)
    aggregate_values = np.asarray(aggregate.probabilities, dtype=np.float64)
    if np.any(aggregate_values <= zero_atol):
        return InformationLossResult(
            fisher_joint=None,
            fisher_aggregate=None,
            difference=None,
            parameter_names=names,
            labels=[],
            status=MetricStatus.UNDEFINED,
            reason_code="zero_probability_aggregate",
        )
    jacobian = np.asarray(joint_jacobian, dtype=np.float64)
    labels, aggregate_jacobian = pushforward_jacobian(joint, jacobian, aggregation)
    weighted = jacobian / np.sqrt(values)[:, None]
    fisher_joint = weighted.T @ weighted
    aggregate_weighted = aggregate_jacobian / np.sqrt(aggregate_values)[:, None]
    fisher_aggregate = aggregate_weighted.T @ aggregate_weighted
    difference = fisher_joint - fisher_aggregate
    psd = is_psd(difference)
    status = MetricStatus.OK
    reason_code: str | None = None
    assumptions = [
        "fixed aggregation mapping independent of theta",
        "same declared joint model for both quantities",
    ]
    if not psd:
        status = MetricStatus.UNSTABLE
        reason_code = "data_processing_violation"
    return InformationLossResult(
        fisher_joint=fisher_joint,
        fisher_aggregate=fisher_aggregate,
        difference=difference,
        parameter_names=names,
        labels=labels,
        status=status,
        reason_code=reason_code,
        psd=psd,
        min_eigenvalue=float(np.linalg.eigvalsh(0.5 * (difference + difference.T)).min())
        if difference.size
        else 0.0,
        assumptions=assumptions,
    )
