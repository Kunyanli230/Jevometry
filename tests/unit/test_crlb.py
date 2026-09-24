"""CRLB eligibility, identifiability and nuisance handling."""

from __future__ import annotations

import numpy as np
import pytest

from jevometry.inference.contracts import check_contract
from jevometry.inference.crlb import (
    crlb_for_function,
    crlb_iid,
    crlb_independent_designs,
    crlb_with_nuisance,
    is_valid_crlb,
)
from jevometry.schemas.common import AnalysisObject, MetricStatus
from jevometry.schemas.experiment import (
    IdentifiabilitySource,
    ObservationRelation,
    SamplingContract,
    SamplingKind,
)


def complete_contract(**overrides: object) -> SamplingContract:
    values: dict[str, object] = {
        "observable": "Y",
        "observation_unit": "one independent categorical draw",
        "sampling": SamplingKind.IID,
        "sample_size": 100,
        "relation_to_jev": ObservationRelation.SYNTHETIC_SIMULATION,
        "identifiability_source": IdentifiabilitySource.ANALYTIC,
        "fixed_support": True,
        "differentiable": True,
        "locally_identifiable": True,
        "estimand": "p",
    }
    values.update(overrides)
    return SamplingContract(**values)  # type: ignore[arg-type]


def test_missing_contract_is_refused() -> None:
    result = check_contract(
        None,
        parameter_names=["p"],
        analysis_object=AnalysisObject.EMPIRICAL_OBSERVATION_MODEL,
    )
    assert result.eligible is False
    assert result.reason_code == "missing_sampling_contract"
    assert result.remedy


def test_incomplete_contract_lists_missing_fields() -> None:
    contract = SamplingContract(observable="Y")
    result = check_contract(
        contract,
        parameter_names=["p"],
        analysis_object=AnalysisObject.EMPIRICAL_OBSERVATION_MODEL,
    )
    assert result.eligible is False
    assert result.reason_code == "incomplete_sampling_contract"
    assert "sample_size" in result.missing


def test_dependent_sampling_is_refused() -> None:
    contract = complete_contract(sampling=SamplingKind.DEPENDENT)
    result = check_contract(
        contract,
        parameter_names=["p"],
        analysis_object=AnalysisObject.EMPIRICAL_OBSERVATION_MODEL,
    )
    assert result.reason_code == "dependent_sampling"


def test_reported_distribution_bound_has_sampling_assumptions() -> None:
    contract = complete_contract(
        relation_to_jev=ObservationRelation.REPORTED_DISTRIBUTION_DRAW
    )
    result = check_contract(
        contract,
        parameter_names=["p"],
        analysis_object=AnalysisObject.REPORTED_DISTRIBUTION,
    )
    assert result.eligible is True
    assert any("model-conditional" in item for item in result.assumptions)
    assert any("not treated as a random draw" in item for item in result.assumptions)


def test_scalar_crlb_value() -> None:
    information = np.asarray([[4.0]])
    result = crlb_iid(information, 100, parameter_names=["p"])
    assert result.crlb is not None
    assert result.crlb[0, 0] == pytest.approx(1.0 / 400.0)
    assert result.standard_errors is not None
    assert result.standard_errors[0] == pytest.approx(1.0 / 20.0)


def test_singular_information_is_not_identifiable() -> None:
    information = np.asarray([[1.0, 1.0], [1.0, 1.0]])
    result = crlb_iid(information, 10, parameter_names=["a", "b"])
    assert result.crlb is None
    assert result.status is MetricStatus.NOT_IDENTIFIABLE
    assert result.reason_code == "not_identifiable"


def test_function_crlb_uses_gradient() -> None:
    information = np.asarray([[2.0, 0.0], [0.0, 8.0]])
    gradient = np.asarray([1.0, 2.0])
    result = crlb_for_function(
        information, 50, gradient, parameter_names=["a", "b"], function_name="g"
    )
    expected = (1.0 / 50.0) * (1.0 / 2.0 + 4.0 / 8.0)
    assert result.crlb is not None
    assert result.crlb[0, 0] == pytest.approx(expected)


def test_nuisance_full_inverse_matches_schur_complement() -> None:
    information = np.asarray(
        [[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0]]
    )
    full = crlb_with_nuisance(
        information, 20, [0], parameter_names=["a", "b", "c"], method="full_inverse"
    )
    schur = crlb_with_nuisance(
        information, 20, [0], parameter_names=["a", "b", "c"], method="schur"
    )
    assert full.crlb is not None and schur.crlb is not None
    assert np.allclose(full.crlb, schur.crlb, rtol=1e-10)


def test_nuisance_target_block_inverse_is_not_used() -> None:
    information = np.asarray([[2.0, 1.5], [1.5, 2.0]])
    result = crlb_with_nuisance(
        information, 1, [0], parameter_names=["a", "b"], method="full_inverse"
    )
    naive = 1.0 / information[0, 0]
    assert result.crlb is not None
    assert result.crlb[0, 0] != pytest.approx(naive)


def test_independent_designs_sum_information() -> None:
    designs = [np.asarray([[1.0]]), np.asarray([[3.0]])]
    result = crlb_independent_designs(designs, parameter_names=["p"])
    assert result.crlb is not None
    assert result.crlb[0, 0] == pytest.approx(1.0 / 4.0)


def test_crlb_matrix_is_symmetric_psd() -> None:
    information = np.asarray([[4.0, 1.0], [1.0, 3.0]])
    result = crlb_iid(information, 10, parameter_names=["a", "b"])
    assert result.crlb is not None
    assert is_valid_crlb(result.crlb)
