"""Inference eligibility: a CRLB requires a complete sampling contract.

A missing contract is a refusal, not a default.  In particular, the number of
API calls is never treated as a sample size, and repeated identical API
responses are never treated as independent draws.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from jevometry.schemas.common import AnalysisObject
from jevometry.schemas.experiment import (
    ObservationRelation,
    SamplingContract,
    SamplingKind,
)


@dataclass
class InferenceEligibility:
    """Whether a CRLB may be computed, and why not when it may not."""

    eligible: bool
    reason_code: str | None = None
    remedy: str | None = None
    missing: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def check_contract(
    contract: SamplingContract | None,
    *,
    parameter_names: Sequence[str],
    analysis_object: AnalysisObject,
    independent_designs: bool = False,
) -> InferenceEligibility:
    """Validate that inference is permitted under an explicit contract."""
    if contract is None:
        return InferenceEligibility(
            eligible=False,
            reason_code="missing_sampling_contract",
            remedy=(
                "declare SamplingContract(observable, observation_unit, sampling, "
                "sample_size, relation_to_jev, identifiability_source, fixed_support, "
                "differentiable, locally_identifiable, estimand)"
            ),
        )
    missing = contract.missing_fields()
    if missing:
        return InferenceEligibility(
            eligible=False,
            reason_code="incomplete_sampling_contract",
            remedy=f"declare the missing contract fields: {', '.join(missing)}",
            missing=missing,
        )
    if not parameter_names:
        return InferenceEligibility(
            eligible=False,
            reason_code="no_parameters",
            remedy="declare at least one unknown parameter",
        )
    if contract.sampling is SamplingKind.DEPENDENT:
        return InferenceEligibility(
            eligible=False,
            reason_code="dependent_sampling",
            remedy=(
                "dependent or adaptive samples cannot be scaled by sample_size; "
                "provide a joint likelihood over the dependent units"
            ),
        )
    if contract.sampling is SamplingKind.INDEPENDENT_NON_IDENTICAL and not independent_designs:
        return InferenceEligibility(
            eligible=False,
            reason_code="designs_required",
            remedy=(
                "independent non-identical designs require one information matrix per "
                "design; pass the per-design Fisher matrices"
            ),
        )
    if contract.relation_to_jev is ObservationRelation.REPORTED_DISTRIBUTION_DRAW:
        assumptions = [
            "model-conditional bound: observations are assumed to be independent draws "
            "from the reported categorical distribution",
            "the provider's selected label is not treated as a random draw",
        ]
    elif contract.relation_to_jev is ObservationRelation.SYNTHETIC_SIMULATION:
        assumptions = [
            "synthetic simulation of the declared model; not evidence about the real system"
        ]
    elif contract.relation_to_jev is ObservationRelation.EXTERNAL_MEASUREMENT:
        assumptions = [
            "observations come from an external measurement process declared by the user"
        ]
    if analysis_object is AnalysisObject.REPORTED_DISTRIBUTION:
        assumptions.append("analysis object is a reported distribution, not a system model")
    warnings: list[str] = []
    if contract.identifiability_source is not None:
        warnings.append(
            f"identifiability source: {contract.identifiability_source.value}"
        )
    if contract.relation_to_jev is ObservationRelation.REPORTED_DISTRIBUTION_DRAW:
        warnings.append("API call count is not a sample size")
    return InferenceEligibility(
        eligible=True,
        assumptions=assumptions,
        warnings=warnings,
    )
