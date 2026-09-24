"""Additional coverage for inference contracts, CRLB and simulation."""

from __future__ import annotations

import numpy as np
import pytest

from jevometry.adapters.analytic import AnalyticLikelihoodModel, bernoulli_likelihood
from jevometry.inference.contracts import check_contract
from jevometry.inference.crlb import (
    crlb_for_function,
    crlb_iid,
    crlb_independent_designs,
    crlb_with_nuisance,
)
from jevometry.inference.likelihood import mle, profile_interval, profile_log_likelihood
from jevometry.inference.simulation import SimulationConfig, run_simulation, simulation_status
from jevometry.schemas.common import AnalysisObject, MetricStatus
from jevometry.schemas.experiment import (
    IdentifiabilitySource,
    ObservationRelation,
    SamplingContract,
    SamplingKind,
)
from jevometry.schemas.parameters import ParameterSpec


def contract(**overrides: object) -> SamplingContract:
    values: dict[str, object] = {
        "observable": "Y",
        "observation_unit": "one draw",
        "sampling": SamplingKind.IID,
        "sample_size": 10,
        "relation_to_jev": ObservationRelation.SYNTHETIC_SIMULATION,
        "identifiability_source": IdentifiabilitySource.USER_ASSERTED,
        "fixed_support": True,
        "differentiable": True,
        "locally_identifiable": True,
        "estimand": "p",
    }
    values.update(overrides)
    return SamplingContract(**values)  # type: ignore[arg-type]


def p_spec() -> ParameterSpec:
    return ParameterSpec(
        name="p",
        role="task_relevant",
        unit="probability",
        bounds=(0.0, 1.0),
        step=1e-4,
        center=0.5,
    )


def test_non_identical_sampling_requires_designs() -> None:
    result = check_contract(
        contract(sampling=SamplingKind.INDEPENDENT_NON_IDENTICAL),
        parameter_names=["p"],
        analysis_object=AnalysisObject.EMPIRICAL_OBSERVATION_MODEL,
    )
    assert result.reason_code == "designs_required"
    with_designs = check_contract(
        contract(sampling=SamplingKind.INDEPENDENT_NON_IDENTICAL),
        parameter_names=["p"],
        analysis_object=AnalysisObject.EMPIRICAL_OBSERVATION_MODEL,
        independent_designs=True,
    )
    assert with_designs.eligible is True


def test_contract_without_parameters_or_relation() -> None:
    no_parameters = check_contract(
        contract(),
        parameter_names=[],
        analysis_object=AnalysisObject.EMPIRICAL_OBSERVATION_MODEL,
    )
    assert no_parameters.reason_code == "no_parameters"
    no_relation = check_contract(
        contract(relation_to_jev=ObservationRelation.NONE),
        parameter_names=["p"],
        analysis_object=AnalysisObject.EMPIRICAL_OBSERVATION_MODEL,
    )
    assert no_relation.reason_code == "incomplete_sampling_contract"


def test_contract_warns_about_api_call_counts() -> None:
    result = check_contract(
        contract(relation_to_jev=ObservationRelation.REPORTED_DISTRIBUTION_DRAW),
        parameter_names=["p"],
        analysis_object=AnalysisObject.REPORTED_DISTRIBUTION,
    )
    assert result.eligible is True
    assert any("API call count" in warning for warning in result.warnings)


def test_crlb_invalid_sample_size() -> None:
    result = crlb_iid(np.asarray([[1.0]]), 0, parameter_names=["p"])
    assert result.status is MetricStatus.UNSUPPORTED
    assert result.reason_code == "invalid_sample_size"


def test_crlb_non_square_and_non_finite() -> None:
    non_square = crlb_iid(np.zeros((2, 3)), 5, parameter_names=["a", "b"])
    assert non_square.reason_code == "non_square_information"
    non_finite = crlb_iid(np.asarray([[np.nan]]), 5, parameter_names=["a"])
    assert non_finite.reason_code == "non_finite_information"


def test_crlb_function_with_vector_gradient() -> None:
    information = np.asarray([[2.0, 0.0], [0.0, 4.0]])
    result = crlb_for_function(
        information, 10, np.asarray([1.0, 1.0]), parameter_names=["a", "b"], function_name="g"
    )
    assert result.crlb is not None
    assert result.crlb[0, 0] == pytest.approx((0.5 + 0.25) / 10.0)
    mismatch = crlb_for_function(
        information,
        10,
        np.asarray([1.0, 1.0, 1.0]),
        parameter_names=["a", "b"],
        function_name="g",
    )
    assert mismatch.reason_code == "gradient_dimension_mismatch"


def test_crlb_nuisance_schur_with_singular_nuisance_block() -> None:
    information = np.asarray([[2.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    result = crlb_with_nuisance(
        information, 1, [0], parameter_names=["a", "b", "c"], method="schur"
    )
    assert result.status is MetricStatus.NOT_IDENTIFIABLE


def test_crlb_nuisance_without_nuisance_parameters() -> None:
    information = np.asarray([[4.0]])
    result = crlb_with_nuisance(
        information, 1, [0], parameter_names=["a"], method="schur"
    )
    assert result.crlb is not None
    assert result.crlb[0, 0] == pytest.approx(0.25)


def test_crlb_independent_designs_requires_input() -> None:
    result = crlb_independent_designs([], parameter_names=["p"])
    assert result.status is MetricStatus.INSUFFICIENT_DATA


def test_profile_log_likelihood_without_other_parameters() -> None:
    model = bernoulli_likelihood("p")
    observations = [1, 0, 1]
    value = profile_log_likelihood(model, observations, "p", 0.5, parameter_specs=[p_spec()])
    assert value == pytest.approx(3 * np.log(0.5))


def test_profile_interval_unknown_parameter() -> None:
    model = bernoulli_likelihood("p")
    interval = profile_interval(model, [1, 0], "q", parameter_specs=[p_spec()])
    assert interval.status is MetricStatus.FAILED
    assert interval.reason_code == "unknown_parameter"


def test_profile_interval_near_boundary_is_conditional_or_finite() -> None:
    model = bernoulli_likelihood("p")
    interval = profile_interval(model, [1] * 100, "p", parameter_specs=[p_spec()])
    assert interval.estimate > 0.9
    if interval.upper is None:
        assert interval.status is MetricStatus.CONDITIONAL
        assert interval.reason_code == "interval_boundary_reached"
    else:
        assert interval.lower is not None
        assert interval.lower < interval.estimate < interval.upper


def test_mle_with_impossible_observations_fails_cleanly() -> None:
    model = bernoulli_likelihood("p")
    fit = mle(model, [5, 5], parameter_specs=[p_spec()])
    assert fit.status is MetricStatus.FAILED
    assert fit.reason_code == "likelihood_undefined"


class _BrokenModel(AnalyticLikelihoodModel):
    def sample(self, theta, n, rng):  # type: ignore[override]
        return np.full(n, 7, dtype=np.int64)


def test_simulation_with_all_failures_is_failed() -> None:
    model = _BrokenModel(("0", "1"), lambda theta: np.asarray([1 - theta["p"], theta["p"]]))
    config = SimulationConfig(
        parameter="p", true_value=0.5, sample_size=10, replications=3, seed=1
    )
    summary = run_simulation(model, config, parameter_specs=[p_spec()])
    assert summary.failures == 3
    assert summary.bias is None
    assert simulation_status(summary) is MetricStatus.FAILED
    assert any("no successful replications" in note for note in summary.notes)


def test_simulation_reports_monte_carlo_error() -> None:
    model = bernoulli_likelihood("p")
    config = SimulationConfig(
        parameter="p", true_value=0.4, sample_size=20, replications=30, seed=11
    )
    summary = run_simulation(model, config, parameter_specs=[p_spec()], crlb_variance=0.24 / 20)
    assert summary.monte_carlo_standard_error is not None
    assert simulation_status(summary) in {MetricStatus.OK, MetricStatus.CONDITIONAL}
    assert summary.crlb_variance == pytest.approx(0.24 / 20)
