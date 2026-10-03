"""Inference refusals for unobserved contrasts and external observations."""

from __future__ import annotations

import numpy as np
import pytest

from jevometry.inference.contracts import check_contract
from jevometry.inference.crlb import crlb_for_function, crlb_independent_designs
from jevometry.schemas.common import AnalysisObject, MetricStatus
from jevometry.schemas.experiment import (
    IdentifiabilitySource,
    ObservationRelation,
    SamplingContract,
    SamplingKind,
)


def test_external_measurements_use_their_declared_observation_contract() -> None:
    contract = SamplingContract(
        observable="binary laboratory measurement",
        observation_unit="one independently measured specimen",
        sampling=SamplingKind.IID,
        sample_size=12,
        relation_to_jev=ObservationRelation.EXTERNAL_MEASUREMENT,
        identifiability_source=IdentifiabilitySource.USER_ASSERTED,
        fixed_support=True,
        differentiable=True,
        locally_identifiable=True,
        estimand="specimen positive probability",
    )
    eligibility = check_contract(
        contract,
        parameter_names=["p"],
        analysis_object=AnalysisObject.EMPIRICAL_OBSERVATION_MODEL,
    )
    assert eligibility.eligible
    assert eligibility.reason_code is None
    assert "external measurement process" in " ".join(eligibility.assumptions)
    assert any("user_asserted" in warning for warning in eligibility.warnings)
    assert not any("API call count" in warning for warning in eligibility.warnings)


def test_repeated_designs_cannot_identify_an_unobserved_contrast() -> None:
    # Both independent designs measure only theta_1 + theta_2.  Their
    # information adds, but (1, -1) remains a null direction at any count.
    result = crlb_independent_designs(
        [np.asarray([[1.0, 1.0], [1.0, 1.0]]),
         np.asarray([[2.0, 2.0], [2.0, 2.0]])],
        parameter_names=["theta_1", "theta_2"],
    )
    assert result.status is MetricStatus.NOT_IDENTIFIABLE
    assert result.reason_code == "not_identifiable"
    assert result.method == "sum_of_designs"
    assert result.crlb is None
    assert result.standard_errors is None
    assert result.smallest_eigenvalue == pytest.approx(0.0)


def test_function_bound_refuses_an_estimand_along_a_null_direction() -> None:
    # Information depends only on the sum.  For g = theta_1 - theta_2,
    # Dg = (1, -1), so the observations carry no information about g.
    result = crlb_for_function(
        np.asarray([[2.0, 2.0], [2.0, 2.0]]),
        10,
        np.asarray([1.0, -1.0]),
        parameter_names=["theta_1", "theta_2"],
        function_name="contrast",
    )
    assert result.status is MetricStatus.NOT_IDENTIFIABLE
    assert result.reason_code == "not_identifiable"
    assert result.crlb is None
    assert result.standard_errors is None
