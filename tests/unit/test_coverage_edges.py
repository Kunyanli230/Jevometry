"""Edge-path coverage for geometry, systems and inference cores."""

from __future__ import annotations

import math

import numpy as np
import pytest

from jevometry.geometry.diagnostics import matrix_diagnostics
from jevometry.geometry.distances import (
    bhattacharyya_coefficient,
    entropy,
    expected_value,
    variance,
)
from jevometry.geometry.fisher import fisher_pullback
from jevometry.inference.contracts import check_contract
from jevometry.inference.crlb import crlb_with_nuisance
from jevometry.inference.likelihood import mle, profile_interval
from jevometry.inference.simulation import (
    bernoulli_crlb_variance,
    crlb_standard_error_from_information,
)
from jevometry.schemas.common import AnalysisObject, MetricStatus
from jevometry.schemas.experiment import SamplingContract, SamplingKind
from jevometry.schemas.joint import ConstructionMode, JointDistribution
from jevometry.schemas.parameters import ParameterSpec
from jevometry.systems.joint import (
    conditional_mutual_information,
    joint_fisher,
    mutual_information,
    product_joint_from_nodes,
    product_joint_jacobian,
)
from jevometry.systems.tree import (
    ConditionalNode,
    ConditionalTree,
    conditional_fisher_identity,
    enumerate_tree,
    tree_joint,
    tree_path_jacobian,
)


def test_fisher_shape_validation() -> None:
    with pytest.raises(ValueError):
        fisher_pullback(np.zeros((2, 1)), np.zeros(3), support=("a", "b", "c"))
    with pytest.raises(ValueError):
        fisher_pullback(np.zeros((2, 1)), np.zeros(2), support=("a",))


def test_matrix_diagnostics_empty_matrix() -> None:
    diagnostics = matrix_diagnostics(np.zeros((0, 0)))
    assert diagnostics.rank == 0
    assert diagnostics.condition_number is None
    assert diagnostics.null_directions == []


def test_matrix_diagnostics_asymmetric_message() -> None:
    diagnostics = matrix_diagnostics(np.asarray([[1.0, 0.5], [0.0, 1.0]]))
    assert diagnostics.symmetry_residual > 1e-10
    assert diagnostics.diagnostics


def test_distance_shape_validations() -> None:
    with pytest.raises(ValueError):
        expected_value([0.5, 0.5], [1.0])
    with pytest.raises(ValueError):
        variance([0.5, 0.5], [1.0])
    with pytest.raises(ValueError):
        bhattacharyya_coefficient([0.5, 0.5], [1.0])
    with pytest.raises(ValueError):
        entropy(np.zeros((2, 2)))


def test_joint_fisher_validations() -> None:
    with pytest.raises(ValueError):
        joint_fisher(np.zeros(2), np.zeros((3, 1)), node_order=["a"], parameter_names=["p"])
    with pytest.raises(ValueError):
        joint_fisher(np.zeros(2), np.zeros((2, 2)), node_order=["a"], parameter_names=["p"])
    negative = joint_fisher(
        np.asarray([-0.1, 1.1]), np.zeros((2, 1)), node_order=["a"], parameter_names=["p"]
    )
    assert negative.status is MetricStatus.FAILED
    assert negative.reason_code == "negative_joint_probability"
    zero = joint_fisher(
        np.asarray([0.0, 1.0]), np.zeros((2, 1)), node_order=["a"], parameter_names=["p"]
    )
    assert zero.status is MetricStatus.UNDEFINED


def test_product_jacobian_with_zero_conditional() -> None:
    outcome_sets = {"Y": ("0", "1"), "Z": ("0", "1")}
    distributions = {"Y": np.asarray([0.0, 1.0]), "Z": np.asarray([0.5, 0.5])}
    jacobians = {"Y": np.asarray([[-1.0], [1.0]]), "Z": np.asarray([[-1.0], [1.0]])}
    jacobian = product_joint_jacobian(
        ["Y", "Z"], outcome_sets, distributions, jacobians, ["p"]
    )
    assert jacobian.shape == (4, 1)
    # For (Y=0, Z=0) the product is zero but the derivative is q_Z(0) * dq_Y(0).
    assert jacobian[0, 0] == pytest.approx(0.5 * -1.0)
    joint = product_joint_from_nodes(
        ["Y", "Z"],
        outcome_sets,
        distributions,
        assumptions=[],
        theta={"p": 1.0},
    )
    assert sum(joint.probabilities) == pytest.approx(1.0)


def test_mutual_information_ignores_zero_mass_states() -> None:
    joint = JointDistribution(
        mode=ConstructionMode.EXPLICIT,
        node_order=["a", "b"],
        outcomes=[("0", "0"), ("0", "1"), ("1", "0")],
        probabilities=[0.5, 0.0, 0.5],
    )
    assert mutual_information(joint, "a", "b") == pytest.approx(0.0, abs=1e-12)
    conditional = JointDistribution(
        mode=ConstructionMode.EXPLICIT,
        node_order=["a", "b", "c"],
        outcomes=[("0", "0", "0"), ("1", "1", "0"), ("0", "0", "1")],
        probabilities=[0.5, 0.5, 0.0],
    )
    assert conditional_mutual_information(conditional, "a", "b", "c") >= 0.0


def test_tree_conditional_distribution_rejects_extra_outcomes() -> None:
    tree = ConditionalTree(
        tree_id="bad",
        nodes=[ConditionalNode("Y", ("a", "b"), lambda theta, history: {"a": 0.5, "b": 0.3, "c": 0.2})],
    )
    with pytest.raises(ValueError):
        enumerate_tree(tree, {})


def test_tree_path_jacobian_missing_analytic_derivative() -> None:
    tree = ConditionalTree(
        tree_id="no-jac",
        nodes=[ConditionalNode("Y", ("a", "b"), lambda theta, history: {"a": 0.5, "b": 0.5})],
    )
    outcomes, jacobian = tree_path_jacobian(tree, {"p": 0.5})
    assert outcomes == [("a",), ("b",)]
    assert jacobian is None


def test_conditional_identity_finite_difference_fallback() -> None:
    tree = ConditionalTree(
        tree_id="fd",
        nodes=[
            ConditionalNode(
                "Y",
                ("a", "b"),
                lambda theta, history: {"a": 1.0 - theta["p"], "b": theta["p"]},
            ),
            ConditionalNode(
                "Z",
                ("0", "1"),
                lambda theta, history: {
                    "0": 1.0 - (theta["p"] if history["Y"] == "a" else 0.5),
                    "1": theta["p"] if history["Y"] == "a" else 0.5,
                },
            ),
        ],
    )
    identity = conditional_fisher_identity(
        tree,
        {"p": 0.3},
        parameter_names=["p"],
        parameter_steps={"p": 1e-6},
    )
    # The conditional contributions use finite differences, but the exact path
    # Jacobian is unavailable, so the identity is reported as conditional.
    assert identity.status is MetricStatus.CONDITIONAL
    assert identity.reason_code == "missing_path_jacobian"
    assert identity.total is not None
    assert identity.per_node["Y"][0, 0] == pytest.approx(1.0 / (0.3 * 0.7), rel=1e-5)


def test_conditional_identity_missing_jacobian_is_unsupported() -> None:
    tree = ConditionalTree(
        tree_id="missing",
        nodes=[ConditionalNode("Y", ("a", "b"), lambda theta, history: {"a": 0.5, "b": 0.5})],
    )
    identity = conditional_fisher_identity(tree, {"p": 0.5}, parameter_names=["p"])
    assert identity.status is MetricStatus.UNSUPPORTED
    assert identity.reason_code == "missing_conditional_jacobian"


def test_conditional_identity_zero_probability_is_undefined() -> None:
    tree = ConditionalTree(
        tree_id="zero",
        nodes=[
            ConditionalNode(
                "Y",
                ("a", "b"),
                lambda theta, history: {"a": 1.0, "b": 0.0},
                lambda theta, history: {"a": {"p": 0.0}, "b": {"p": 0.0}},
            )
        ],
    )
    identity = conditional_fisher_identity(tree, {"p": 0.5}, parameter_names=["p"])
    assert identity.status is MetricStatus.UNDEFINED
    assert identity.reason_code == "zero_probability_outcome"


def test_tree_joint_builder_roundtrip() -> None:
    tree = ConditionalTree(
        tree_id="simple",
        nodes=[
            ConditionalNode(
                "Y",
                ("a", "b"),
                lambda theta, history: {"a": 0.25, "b": 0.75},
                lambda theta, history: {"a": {"p": 0.0}, "b": {"p": 0.0}},
            )
        ],
    )
    joint = tree_joint(tree, {"p": 0.5})
    assert joint.mode is ConstructionMode.CONDITIONAL_TREE
    assert sum(joint.probabilities) == pytest.approx(1.0)


def test_nuisance_schur_all_targets() -> None:
    information = np.asarray([[2.0, 0.5], [0.5, 3.0]])
    result = crlb_with_nuisance(
        information, 2, [0, 1], parameter_names=["a", "b"], method="schur"
    )
    assert result.crlb is not None
    assert result.crlb.shape == (2, 2)


def test_likelihood_with_explicit_initial() -> None:
    from jevometry.adapters.analytic import bernoulli_likelihood

    model = bernoulli_likelihood("p")
    fit = mle(model, [1, 1, 0], parameter_specs=[p_spec()], initial={"p": 0.9})
    assert fit.theta is not None
    assert fit.theta["p"] == pytest.approx(2 / 3, abs=1e-3)
    interval = profile_interval(
        model, [1, 1, 0], "p", parameter_specs=[p_spec()], initial={"p": 0.9}
    )
    assert interval.lower is not None


def test_crlb_helper_functions() -> None:
    assert bernoulli_crlb_variance(0.3) == pytest.approx(0.21)
    assert crlb_standard_error_from_information(4.0) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        crlb_standard_error_from_information(0.0)


def test_contract_with_dependent_sampling_reports_remedy() -> None:
    contract = SamplingContract(
        observable="Y",
        observation_unit="unit",
        sampling=SamplingKind.DEPENDENT,
        sample_size=5,
        relation_to_jev="synthetic_simulation",
        identifiability_source="analytic",
        fixed_support=True,
        differentiable=True,
        locally_identifiable=True,
        estimand="p",
    )
    result = check_contract(
        contract,
        parameter_names=["p"],
        analysis_object=AnalysisObject.EMPIRICAL_OBSERVATION_MODEL,
    )
    assert result.reason_code == "dependent_sampling"
    assert "joint likelihood" in (result.remedy or "")
    assert math.isfinite(1.0)


def p_spec() -> ParameterSpec:
    return ParameterSpec(
        name="p",
        role="task_relevant",
        unit="probability",
        bounds=(0.0, 1.0),
        step=1e-4,
        center=0.5,
    )

class _RaisingModel:
    support = ("0", "1")

    def log_prob(self, observations, theta):
        raise ValueError("boom")

    def sample(self, theta, n, rng):
        return np.zeros(n, dtype=np.int64)


class _InfiniteModel:
    support = ("0", "1")

    def log_prob(self, observations, theta):
        return float("inf")

    def sample(self, theta, n, rng):
        return np.zeros(n, dtype=np.int64)


class _TwoParameterModel:
    """Independent Bernoulli observations with parameters p and q."""

    support = ("0", "1")

    def log_prob(self, observations, theta):
        p = float(theta["p"])
        q = float(theta["q"])
        total = 0.0
        for index, observation in enumerate(observations):
            probability = p if index % 2 == 0 else q
            if observation == 0:
                probability = 1.0 - probability
            total += math.log(probability)
        return total

    def sample(self, theta, n, rng):
        return rng.integers(0, 2, size=n).astype(np.int64)


def two_parameter_specs() -> list[ParameterSpec]:
    return [
        ParameterSpec(
            name="p", role="task_relevant", unit="probability", bounds=(0.0, 1.0), step=1e-4, center=0.5
        ),
        ParameterSpec(
            name="q", role="nuisance", unit="probability", bounds=(0.0, 1.0), step=1e-4, center=0.5
        ),
    ]


def test_mle_handles_models_that_raise() -> None:
    fit = mle(_RaisingModel(), [0, 1], parameter_specs=[p_spec()])
    assert fit.status is MetricStatus.FAILED
    assert fit.reason_code == "likelihood_undefined"


def test_mle_handles_infinite_likelihood() -> None:
    fit = mle(_InfiniteModel(), [0, 1], parameter_specs=[p_spec()])
    assert fit.status is MetricStatus.FAILED


def test_profile_log_likelihood_profiles_other_parameters() -> None:
    from jevometry.inference.likelihood import profile_log_likelihood

    observations = [1, 1, 0, 0, 1, 0]
    value = profile_log_likelihood(
        _TwoParameterModel(),
        observations,
        "p",
        0.5,
        parameter_specs=two_parameter_specs(),
    )
    assert math.isfinite(value)
    assert value <= 0.0


def test_profile_log_likelihood_with_invalid_observations() -> None:
    from jevometry.inference.likelihood import profile_log_likelihood

    value = profile_log_likelihood(
        _RaisingModel(), [0], "p", 0.5, parameter_specs=[p_spec()]
    )
    assert value == float("-inf")


def test_profile_interval_reports_failed_mle() -> None:
    interval = profile_interval(
        _RaisingModel(), [0, 1], "p", parameter_specs=[p_spec()]
    )
    assert interval.status is MetricStatus.FAILED
    assert interval.reason_code == "likelihood_undefined"


def test_profile_interval_two_parameters() -> None:
    rng = np.random.default_rng(3)
    model = _TwoParameterModel()
    observations = [1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1]
    interval = profile_interval(
        model,
        observations,
        "p",
        parameter_specs=two_parameter_specs(),
        initial={"p": 0.5, "q": 0.5},
    )
    assert interval.estimate is not None
    assert interval.status in {MetricStatus.OK, MetricStatus.CONDITIONAL}
    assert math.isfinite(float(rng.random()))
