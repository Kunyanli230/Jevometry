"""System composition declarations."""

from __future__ import annotations

from enum import Enum

from pydantic import Field, model_validator

from jevometry.schemas.common import RoutingSemantics, StrictModel

MAX_NODES = 64


class CompositionMode(str, Enum):
    """How much system-level structure the adapter actually declares."""

    NODE_ONLY = "node_only"
    DECLARED_PRODUCT = "declared_product"
    EXPLICIT_JOINT = "explicit_joint"
    CONDITIONAL_TREE = "conditional_tree"


class NodeSpec(StrictModel):
    """One decision node in the analysed system."""

    id: str
    question_id: str
    description: str | None = None
    parents: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate(self) -> NodeSpec:
        if not self.id:
            raise ValueError("node id must be non-empty")
        if self.id in self.parents:
            raise ValueError("a node may not be its own parent")
        return self


class EdgeSpec(StrictModel):
    """Directed dependency between two nodes."""

    source: str
    target: str
    condition: str | None = None


class Capabilities(StrictModel):
    """What the adapter is able to provide for this system."""

    analytic_jacobian: bool = False
    joint_model: bool = False
    tree_model: bool = False
    likelihood_model: bool = False
    action_distribution: bool = False
    replayable: bool = False


class CompositionAssumptions(StrictModel):
    """Assumptions that justify any system-level combination."""

    conditional_independence: bool = False
    shared_theta: bool = False
    fixed_mapping: bool = False
    notes: list[str] = Field(default_factory=list)


class SystemSpec(StrictModel):
    """Declared structure of the analysed agent system."""

    id: str
    description: str | None = None
    nodes: list[NodeSpec] = Field(default_factory=list)
    edges: list[EdgeSpec] = Field(default_factory=list)
    routing_semantics: RoutingSemantics = RoutingSemantics.NOT_APPLICABLE
    composition_mode: CompositionMode = CompositionMode.NODE_ONLY
    capabilities: Capabilities = Field(default_factory=Capabilities)
    composition_assumptions: CompositionAssumptions = Field(
        default_factory=CompositionAssumptions
    )
    model: str | None = None
    adapter: str | None = None

    @model_validator(mode="after")
    def _validate(self) -> SystemSpec:
        node_ids = [node.id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("node ids must be unique")
        known = set(node_ids)
        for edge in self.edges:
            if edge.source not in known or edge.target not in known:
                raise ValueError(f"edge references unknown node: {edge}")
        for node in self.nodes:
            unknown = set(node.parents) - known
            if unknown:
                raise ValueError(f"node {node.id} has unknown parents: {sorted(unknown)}")
        if self.composition_mode is not CompositionMode.NODE_ONLY and not self.nodes:
            raise ValueError("system-level composition requires declared nodes")
        if len(node_ids) > MAX_NODES:
            raise ValueError(
                f"{len(node_ids)} nodes requested but the v0.1 limit is {MAX_NODES}"
            )
        return self

    def node_ids(self) -> list[str]:
        return [node.id for node in self.nodes]

    def by_id(self, node_id: str) -> NodeSpec:
        for node in self.nodes:
            if node.id == node_id:
                return node
        raise KeyError(node_id)
