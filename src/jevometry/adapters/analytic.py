"""Deterministic analytic probability families.

These families are the offline backbone: they provide closed-form
probabilities, closed-form Jacobians and exact joint models so that the
geometry, system and inference layers can be tested against independent
analytic expectations.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit

from jevometry.adapters.base import ExperimentPoint, request_fingerprint
from jevometry.experiments.renderer import CallableRenderer, ParameterRenderer
from jevometry.geometry.derivatives import NodeEvaluation
from jevometry.geometry.simplex import validate_probabilities
from jevometry.schemas.common import (
    MetricStatus,
    Primitive,
    RoutingSemantics,
)
from jevometry.schemas.distribution import DistributionSource
from jevometry.schemas.experiment import CaseSpec
from jevometry.schemas.joint import ConstructionMode, JointDistribution
from jevometry.schemas.questions import OutcomeSpec, QuestionSpec
from jevometry.schemas.system import (
    Capabilities,
    CompositionAssumptions,
    CompositionMode,
    NodeSpec,
    SystemSpec,
)
from jevometry.schemas.trace import EvaluationTrace, ProviderStatus
from jevometry.systems.tree import ConditionalTree, enumerate_tree, tree_path_jacobian

FloatArray = NDArray[np.float64]


def _theta_hash(theta: Mapping[str, float]) -> str:
    payload = json.dumps({key: float(value) for key, value in sorted(theta.items())})
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def choice_question(
    question_id: str,
    outcomes: Mapping[str, str | None],
    *,
    instructions: str | None = None,
    rubric: str | None = None,
) -> QuestionSpec:
    """Build a Choice question declaration with stable outcome ids."""
    return QuestionSpec(
        id=question_id,
        primitive=Primitive.CHOICE,
        outcomes=[OutcomeSpec(id=key, description=value) for key, value in outcomes.items()],
        instructions=instructions,
        rubric=rubric,
    )


def score_question(
    question_id: str,
    levels: Sequence[str],
    *,
    numerics: Sequence[float] | None = None,
    instructions: str | None = None,
    rubric: str | None = None,
) -> QuestionSpec:
    """Build a Score question declaration; numeric encoding is optional."""
    if numerics is not None and len(numerics) != len(levels):
        raise ValueError("numerics must align with levels")
    outcomes = [
        OutcomeSpec(
            id=str(index),
            description=level,
            numeric=None if numerics is None else float(numerics[index]),
        )
        for index, level in enumerate(levels)
    ]
    legend = {str(index): level for index, level in enumerate(levels)}
    return QuestionSpec(
        id=question_id,
        primitive=Primitive.SCORE,
        outcomes=outcomes,
        legend=legend,
        instructions=instructions,
        rubric=rubric,
    )


def noul_question(
    question_id: str, *, instructions: str | None = None, rubric: str | None = None
) -> QuestionSpec:
    """Build a Noul (yes/no) question declaration."""
    return QuestionSpec(
        id=question_id,
        primitive=Primitive.NOUL,
        outcomes=[
            OutcomeSpec(id="false", description="no"),
            OutcomeSpec(id="true", description="yes"),
        ],
        instructions=instructions,
        rubric=rubric,
    )


@dataclass
class AnalyticNode:
    """A deterministic node family with optional analytic Jacobian."""

    node_id: str
    question: QuestionSpec
    model_identity: str
    probability_function: Callable[[Mapping[str, float]], FloatArray]
    jacobian_function: Callable[[Mapping[str, float]], FloatArray] | None = None
    declared_fixed_zero: tuple[str, ...] = ()
    description: str | None = None

    @property
    def support(self) -> tuple[str, ...]:
        return self.question.support

    def probabilities(self, theta: Mapping[str, float]) -> FloatArray:
        vector = np.asarray(self.probability_function(theta), dtype=np.float64)
        if vector.shape != (len(self.support),):
            raise ValueError(
                f"node {self.node_id!r} returned width {vector.shape} for support "
                f"{self.support!r}"
            )
        return vector

    def jacobian(self, theta: Mapping[str, float]) -> FloatArray | None:
        if self.jacobian_function is None:
            return None
        matrix = np.asarray(self.jacobian_function(theta), dtype=np.float64)
        if matrix.shape != (len(self.support), len(theta)):
            raise ValueError(f"node {self.node_id!r} returned an invalid analytic Jacobian")
        return matrix

    def evaluate(self, theta: Mapping[str, float]) -> NodeEvaluation:
        try:
            vector = self.probabilities(theta)
            simplex = validate_probabilities(self.support, vector)
        except Exception as error:  # noqa: BLE001 - converted to a structured status
            return NodeEvaluation(
                support=self.support,
                probabilities=np.zeros(len(self.support), dtype=np.float64),
                rendered_fingerprint=_theta_hash(theta),
                semantic_hash=self.question.semantic_hash(),
                model_identity=self.model_identity,
                status=MetricStatus.FAILED,
                reason_code=getattr(error, "reason_code", "probability_validation_failed"),
            )
        return NodeEvaluation(
            support=simplex.support,
            probabilities=simplex.working,
            rendered_fingerprint=_theta_hash(theta),
            semantic_hash=self.question.semantic_hash(),
            model_identity=self.model_identity,
            raw_total=simplex.raw_total,
            derived=simplex.derived,
        )


def logistic_node(
    node_id: str,
    *,
    parameter: str,
    coordinate: str = "logit",
    model_identity: str | None = None,
    question: QuestionSpec | None = None,
) -> AnalyticNode:
    """Bernoulli node q = (1 - p, p).

    With ``coordinate="logit"`` the parameter is theta = logit(p) and the
    Fisher information is p(1-p).  With ``coordinate="probability"`` the
    parameter is p itself and the Fisher information is 1/(p(1-p)).
    """
    if coordinate not in {"logit", "probability"}:
        raise ValueError("coordinate must be 'logit' or 'probability'")
    resolved_question = question or noul_question(node_id)

    def probability_function(theta: Mapping[str, float]) -> FloatArray:
        value = float(theta[parameter])
        p = float(expit(value)) if coordinate == "logit" else value
        return np.asarray([1.0 - p, p], dtype=np.float64)

    def jacobian_function(theta: Mapping[str, float]) -> FloatArray:
        value = float(theta[parameter])
        if coordinate == "logit":
            p = float(expit(value))
            derivative = p * (1.0 - p)
        else:
            p = value
            derivative = 1.0
        return np.asarray([[-derivative], [derivative]], dtype=np.float64)

    return AnalyticNode(
        node_id=node_id,
        question=resolved_question,
        model_identity=model_identity or f"logistic:{coordinate}",
        probability_function=probability_function,
        jacobian_function=jacobian_function,
    )


def bernoulli_node(
    node_id: str, *, parameter: str = "p", model_identity: str | None = None
) -> AnalyticNode:
    """Bernoulli node parameterised directly by its probability p."""
    return logistic_node(
        node_id,
        parameter=parameter,
        coordinate="probability",
        model_identity=model_identity or "bernoulli:probability",
    )


def softmax_node(
    node_id: str,
    *,
    outcomes: Sequence[str],
    parameters: Sequence[str],
    weights: NDArray[np.float64] | Sequence[Sequence[float]],
    bias: Sequence[float] | None = None,
    model_identity: str | None = None,
    question: QuestionSpec | None = None,
) -> AnalyticNode:
    """Categorical softmax family q(theta) = softmax(W theta + b)."""
    weight_matrix = np.asarray(weights, dtype=np.float64)
    if weight_matrix.shape != (len(outcomes), len(parameters)):
        raise ValueError("weights must have shape (len(outcomes), len(parameters))")
    bias_vector = (
        np.zeros(len(outcomes), dtype=np.float64)
        if bias is None
        else np.asarray(bias, dtype=np.float64)
    )
    if bias_vector.shape != (len(outcomes),):
        raise ValueError("bias must match the number of outcomes")

    def logits(theta: Mapping[str, float]) -> FloatArray:
        vector = np.asarray([theta[name] for name in parameters], dtype=np.float64)
        return weight_matrix @ vector + bias_vector

    def probability_function(theta: Mapping[str, float]) -> FloatArray:
        values = logits(theta)
        shifted = values - values.max()
        exponentials = np.exp(shifted)
        return np.asarray(exponentials / exponentials.sum(), dtype=np.float64)

    def jacobian_function(theta: Mapping[str, float]) -> FloatArray:
        q = probability_function(theta)
        return q[:, None] * (weight_matrix - q @ weight_matrix)

    resolved_question = question or choice_question(
        node_id, {outcome: None for outcome in outcomes}
    )
    return AnalyticNode(
        node_id=node_id,
        question=resolved_question,
        model_identity=model_identity or "softmax",
        probability_function=probability_function,
        jacobian_function=jacobian_function,
    )


def rank_deficient_softmax_node(
    node_id: str,
    *,
    outcomes: Sequence[str] = ("a", "b", "c"),
    parameters: Sequence[str] = ("theta1", "theta2"),
    model_identity: str | None = None,
) -> AnalyticNode:
    """Softmax whose weights have two identical columns: q depends on theta1+theta2.

    The Fisher information matrix is rank deficient; the difference direction
    (1, -1) is a null direction.
    """
    weights = np.asarray([[1.0, 1.0], [0.5, 0.5], [-1.5, -1.5]], dtype=np.float64)
    if len(outcomes) != weights.shape[0] or len(parameters) != weights.shape[1]:
        raise ValueError("rank-deficient family expects three outcomes and two parameters")
    return softmax_node(
        node_id,
        outcomes=outcomes,
        parameters=parameters,
        weights=weights,
        bias=[0.0, 0.0, 0.0],
        model_identity=model_identity or "softmax:rank-deficient",
    )


@dataclass
class AnalyticJointModel:
    """A declared joint probability family with optional analytic Jacobian."""

    node_order: list[str]
    parameter_names: list[str]
    probability_function: Callable[[Mapping[str, float]], tuple[list[tuple[str, ...]], FloatArray]]
    jacobian_function: Callable[[Mapping[str, float]], FloatArray | None] | None
    model_identity: str
    construction_mode: ConstructionMode = ConstructionMode.EXPLICIT
    assumptions: list[str] = field(default_factory=list)

    def probabilities(self, theta: Mapping[str, float]) -> JointDistribution:
        outcomes, masses = self.probability_function(theta)
        return JointDistribution(
            mode=self.construction_mode,
            node_order=list(self.node_order),
            outcomes=outcomes,
            probabilities=[float(value) for value in masses],
            assumptions=list(self.assumptions),
            theta=dict(theta),
        )

    def jacobian(self, theta: Mapping[str, float]) -> FloatArray | None:
        if self.jacobian_function is None:
            return None
        result = self.jacobian_function(theta)
        if result is None:
            return None
        return np.asarray(result, dtype=np.float64)


def bernoulli_pair_joint_model(
    parameter: str = "p",
    *,
    mode: str = "deterministic_copy",
    draws: int = 2,
) -> AnalyticJointModel:
    """Joint of a Bernoulli outcome with a copy of itself or independent draws.

    ``mode="deterministic_copy"`` gives support (0,0), (1,1) and information
    equal to a single observation.  ``mode="independent"`` gives the product
    law and information equal to ``draws`` times a single observation.
    """
    outcomes = ("0", "1")
    if mode not in {"deterministic_copy", "independent"}:
        raise ValueError("mode must be 'deterministic_copy' or 'independent'")

    def independent_grid() -> list[tuple[str, ...]]:
        grid: list[tuple[str, ...]] = [()]
        for _ in range(draws):
            grid = [(*prefix, outcome) for prefix in grid for outcome in outcomes]
        return grid

    def probability_function(theta: Mapping[str, float]) -> tuple[list[tuple[str, ...]], FloatArray]:
        p = float(theta[parameter])
        if mode == "deterministic_copy":
            return [("0", "0"), ("1", "1")], np.asarray([1.0 - p, p], dtype=np.float64)
        grid = independent_grid()
        masses = np.asarray(
            [
                math.prod(p if outcome == "1" else (1.0 - p) for outcome in path)
                for path in grid
            ],
            dtype=np.float64,
        )
        return grid, masses

    def analytic(theta: Mapping[str, float]) -> FloatArray:
        p = float(theta[parameter])
        if mode == "deterministic_copy":
            return np.asarray([[-1.0], [1.0]], dtype=np.float64)
        grid = independent_grid()
        jacobian = np.zeros((len(grid), 1), dtype=np.float64)
        for row, path in enumerate(grid):
            probability = math.prod(p if outcome == "1" else (1.0 - p) for outcome in path)
            score = sum(
                (1.0 / p) if outcome == "1" else (-1.0 / (1.0 - p)) for outcome in path
            )
            jacobian[row, 0] = probability * score
        return jacobian

    return AnalyticJointModel(
        node_order=["Y", "Z"],
        parameter_names=[parameter],
        probability_function=probability_function,
        jacobian_function=analytic,
        model_identity=f"bernoulli-pair:{mode}",
        construction_mode=ConstructionMode.DECLARED_PRODUCT
        if mode == "independent"
        else ConstructionMode.EXPLICIT,
        assumptions=[
            "declared deterministic copy" if mode == "deterministic_copy"
            else f"declared independent draws (n={draws})"
        ],
    )


def conditional_tree_joint_model(
    tree: ConditionalTree, *, model_identity: str = "conditional-tree"
) -> AnalyticJointModel:
    """Exact joint model of a declared finite conditional tree."""

    def probability_function(theta: Mapping[str, float]) -> tuple[list[tuple[str, ...]], FloatArray]:
        enumeration = enumerate_tree(tree, theta)
        if enumeration.status is not MetricStatus.OK:
            raise ValueError(
                f"tree enumeration refused: {enumeration.reason_code}; "
                f"{' '.join(enumeration.assumptions)}"
            )
        return enumeration.outcomes, enumeration.probabilities

    def jacobian_function(theta: Mapping[str, float]) -> FloatArray | None:
        outcomes, jacobian = tree_path_jacobian(tree, theta)
        if jacobian is None:
            return None
        return jacobian

    return AnalyticJointModel(
        node_order=tree.node_order,
        parameter_names=[],
        probability_function=probability_function,
        jacobian_function=jacobian_function,
        model_identity=model_identity,
        construction_mode=ConstructionMode.CONDITIONAL_TREE,
        assumptions=["exact enumeration of a finite conditional tree"],
    )


class AnalyticLikelihoodModel:
    """Categorical observation model over a declared finite support.

    Observations are support indices.  ``probabilities`` is a function of the
    external theta; nothing here depends on a provider call.
    """

    def __init__(
        self,
        support: Sequence[str],
        probabilities: Callable[[Mapping[str, float]], FloatArray],
        *,
        name: str = "categorical",
        parameter_names: Sequence[str] = ("p",),
    ) -> None:
        self.support = tuple(support)
        self._probabilities = probabilities
        self.name = name
        self.parameter_names = list(parameter_names)

    def probabilities(self, theta: Mapping[str, float]) -> FloatArray:
        return np.asarray(self._probabilities(theta), dtype=np.float64)

    def log_prob(self, observations: Sequence[int], theta: Mapping[str, float]) -> float:
        vector = self.probabilities(theta)
        counts = np.bincount(np.asarray(observations, dtype=np.int64), minlength=len(vector))
        if counts.shape[0] > len(vector):
            raise ValueError("observation index outside support")
        total = 0.0
        for index, count in enumerate(counts):
            if count == 0:
                continue
            if vector[index] <= 0.0:
                return -math.inf
            total += float(count) * math.log(float(vector[index]))
        return total

    def sample(
        self, theta: Mapping[str, float], n: int, rng: np.random.Generator
    ) -> NDArray[np.int64]:
        vector = self.probabilities(theta)
        return rng.choice(len(vector), size=n, p=vector).astype(np.int64)


def bernoulli_likelihood(parameter: str = "p") -> AnalyticLikelihoodModel:
    """Bernoulli observation model with parameter p directly."""

    def probabilities(theta: Mapping[str, float]) -> FloatArray:
        p = float(theta[parameter])
        return np.asarray([1.0 - p, p], dtype=np.float64)

    return AnalyticLikelihoodModel(
        ("0", "1"), probabilities, name="bernoulli", parameter_names=[parameter]
    )


class AnalyticAdapter:
    """Adapter exposing analytic node families and an optional joint model."""

    is_live = False
    provider_name = "analytic"

    def __init__(
        self,
        *,
        system_id: str,
        nodes: Mapping[str, AnalyticNode],
        joint_model: AnalyticJointModel | None = None,
        renderer: ParameterRenderer | None = None,
        routing_semantics: RoutingSemantics = RoutingSemantics.SAMPLED_OUTCOME,
        description: str | None = None,
        likelihood: AnalyticLikelihoodModel | None = None,
    ) -> None:
        if not nodes:
            raise ValueError("an analytic adapter requires at least one node")
        self.nodes = dict(nodes)
        self.joint_model = joint_model
        self._likelihood = likelihood
        self.renderer = renderer or CallableRenderer(
            lambda theta, case: dict(theta), version="analytic-identity"
        )
        self.routing_semantics = routing_semantics
        self._system = SystemSpec(
            id=system_id,
            description=description,
            nodes=[
                NodeSpec(id=node.node_id, question_id=node.question.id)
                for node in self.nodes.values()
            ],
            routing_semantics=routing_semantics,
            composition_mode=(
                CompositionMode.NODE_ONLY if joint_model is None else CompositionMode.EXPLICIT_JOINT
            ),
            capabilities=Capabilities(
                analytic_jacobian=all(node.jacobian_function is not None for node in self.nodes.values()),
                joint_model=joint_model is not None,
                likelihood_model=True,
                action_distribution=routing_semantics is RoutingSemantics.SAMPLED_OUTCOME,
                replayable=True,
            ),
            composition_assumptions=CompositionAssumptions(
                shared_theta=True,
                notes=["analytic families are deterministic functions of theta"],
            ),
        )

    def describe(self) -> SystemSpec:
        return self._system

    def questions(self) -> dict[str, QuestionSpec]:
        return {node.node_id: node.question for node in self.nodes.values()}

    def node_evaluator(self, node_id: str) -> Callable[[Mapping[str, float]], NodeEvaluation]:
        node = self.nodes[node_id]
        return node.evaluate

    def analytic_jacobian(
        self, node_id: str, theta: Mapping[str, float]
    ) -> FloatArray | None:
        node = self.nodes[node_id]
        return node.jacobian(theta)

    def declared_fixed_zero(self, node_id: str) -> tuple[str, ...]:
        return self.nodes[node_id].declared_fixed_zero

    def likelihood_model(self, node_id: str | None = None) -> AnalyticLikelihoodModel | None:
        if self._likelihood is None:
            return None
        if node_id is None:
            return self._likelihood
        node = self.nodes[node_id]
        if node.jacobian_function is None:
            return None
        return self._likelihood

    def evaluate_point(self, point: ExperimentPoint) -> list[EvaluationTrace]:
        rendered = self.renderer.render(point.theta, point.case)
        traces: list[EvaluationTrace] = []
        for node in self.nodes.values():
            evaluation = node.evaluate(point.theta)
            fingerprint = request_fingerprint(
                provider="analytic",
                model=node.model_identity,
                adapter_version="analytic-0.1.0",
                node_id=node.node_id,
                question_hash=node.question.semantic_hash(),
                rendered_fingerprint=rendered.fingerprint,
                theta=point.theta,
                history=point.history,
                stencil_role=point.stencil_role,
            )
            record = None
            if evaluation.status is MetricStatus.OK:
                simplex = validate_probabilities(evaluation.support, evaluation.probabilities)
                record = simplex.as_record(
                    node_id=node.node_id,
                    question_id=node.question.id,
                    primitive=node.question.primitive,
                    source=DistributionSource.ANALYTIC,
                    semantic_hash=node.question.semantic_hash(),
                    case_id=point.case.id,
                    point_id=point.point_id,
                    repeat=point.repeat,
                    theta=dict(point.theta),
                    legend=node.question.legend,
                    numeric_encoding=node.question.numeric_encoding(),
                    request_fingerprint=fingerprint,
                )
            traces.append(
                EvaluationTrace(
                    experiment_id=self._system.id,
                    case_id=point.case.id,
                    point_id=point.point_id,
                    repeat=point.repeat,
                    node_id=node.node_id,
                    question_id=node.question.id,
                    theta=dict(point.theta),
                    rendered_fingerprint=rendered.fingerprint,
                    request_fingerprint=fingerprint,
                    semantic_request_hash=fingerprint,
                    stencil_role=point.stencil_role,
                    distribution=record,
                    status=ProviderStatus(
                        ok=evaluation.status is MetricStatus.OK,
                        model_requested=node.model_identity,
                        model_resolved=node.model_identity,
                        reason_code=evaluation.reason_code,
                    ),
                    routing_semantics=self.routing_semantics.value,
                    provenance={"provider": "analytic", "model_identity": node.model_identity},
                )
            )
        return traces


def sample_case(case_id: str, state: object) -> CaseSpec:
    """Convenience constructor for analytic cases."""
    return CaseSpec(id=case_id, state=state)

def single_node_joint_model(node: AnalyticNode, *, model_identity: str | None = None) -> AnalyticJointModel:
    """Treat one node family as a declared single-node joint model."""

    def probability_function(
        theta: Mapping[str, float],
    ) -> tuple[list[tuple[str, ...]], FloatArray]:
        return [(outcome,) for outcome in node.support], node.probabilities(theta)

    def jacobian_function(theta: Mapping[str, float]) -> FloatArray | None:
        return node.jacobian(theta)

    return AnalyticJointModel(
        node_order=[node.node_id],
        parameter_names=[],
        probability_function=probability_function,
        jacobian_function=jacobian_function,
        model_identity=model_identity or f"{node.model_identity}:single-node-joint",
        construction_mode=ConstructionMode.EXPLICIT,
        assumptions=["single-node joint model declared by the caller"],
    )
