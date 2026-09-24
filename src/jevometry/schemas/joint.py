"""Joint distributions over declared system outcomes."""

from __future__ import annotations

from enum import Enum

from pydantic import Field, model_validator

from jevometry.schemas.common import AnalysisObject, StrictModel

MAX_JOINT_OUTCOMES = 4096


class ConstructionMode(str, Enum):
    """How a joint distribution was obtained."""

    EXPLICIT = "explicit"
    DECLARED_PRODUCT = "declared_product"
    CONDITIONAL_TREE = "conditional_tree"
    ESTIMATED = "estimated"


class JointDistribution(StrictModel):
    """A finite joint distribution over outcome tuples or tree paths.

    ``outcomes`` holds one tuple per state, ``probabilities`` the matching
    masses.  Marginalisation, mutual information and Fisher information all
    operate on this declared object; nothing here is inferred from marginals.
    """

    schema_version: str = "1.0"
    mode: ConstructionMode
    node_order: list[str]
    outcomes: list[tuple[str, ...]]
    probabilities: list[float]
    history_provenance: list[dict[str, str]] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    analysis_object: AnalysisObject = AnalysisObject.DECLARED_SYSTEM_MODEL
    theta: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate(self) -> JointDistribution:
        if not self.node_order:
            raise ValueError("joint distribution requires a node order")
        if len(self.outcomes) != len(self.probabilities):
            raise ValueError("outcomes and probabilities must have equal length")
        if not self.outcomes:
            raise ValueError("joint distribution must not be empty")
        if len(self.outcomes) > MAX_JOINT_OUTCOMES:
            raise ValueError(
                f"{len(self.outcomes)} joint outcomes requested but the v0.1 exact "
                f"enumeration limit is {MAX_JOINT_OUTCOMES}"
            )
        width = len(self.node_order)
        for outcome in self.outcomes:
            if len(outcome) != width:
                raise ValueError("every outcome tuple must match node_order width")
        if len(set(self.outcomes)) != len(self.outcomes):
            raise ValueError("outcome tuples must be unique")
        if self.history_provenance and len(self.history_provenance) != len(self.outcomes):
            raise ValueError("history_provenance must align with outcomes")
        return self

    def total(self) -> float:
        return float(sum(self.probabilities))

    def index_of(self, outcome: tuple[str, ...]) -> int:
        return self.outcomes.index(outcome)

    def marginal(self, node: str) -> dict[str, float]:
        if node not in self.node_order:
            raise KeyError(node)
        axis = self.node_order.index(node)
        masses: dict[str, float] = {}
        for outcome, probability in zip(self.outcomes, self.probabilities, strict=True):
            masses[outcome[axis]] = masses.get(outcome[axis], 0.0) + probability
        return masses

    def marginal_support(self, node: str) -> list[str]:
        return sorted(self.marginal(node))

    def marginal_vector(self, node: str) -> tuple[list[str], list[float]]:
        masses = self.marginal(node)
        support = sorted(masses)
        return support, [masses[key] for key in support]
