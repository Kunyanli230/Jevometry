"""Iris-backed analytic agents for real-data Fisher loss validation.

Maps sklearn Iris petal length/width (scaled to [0, 1]) onto the same
(evidence, ambiguity) parameter space and three-agent outcome alphabet as
``system.py``, so existing AggregationMaps / coordination rules apply unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

from system import (
    PARAMETERS,
    RULE_SPECS,
    build_aggregation_map,
    final_action,
    final_action_coarse_binary,
    final_action_near_identity,
    final_action_quarantine_review,
    final_action_quarantine_wider,
    final_action_stricter,
    final_action_wider,
)
from jevometry.adapters.analytic import AnalyticJointModel, AnalyticNode, choice_question, noul_question, score_question
from jevometry.schemas.joint import ConstructionMode
from jevometry.systems.joint import product_joint_from_nodes, product_joint_jacobian
from jevometry.systems.pushforward import AggregationMap

FloatArray = NDArray[np.float64]

# Re-export for callers that want the four core rules only.
CORE_RULE_NAMES = ("near_identity", "wider", "baseline", "stricter")


@dataclass(frozen=True)
class IrisBundle:
    """Fitted Iris twin: scaled features, centre θ, and declared agents."""

    evidence: FloatArray
    ambiguity: FloatArray
    center: FloatArray
    feature_names: tuple[str, str]
    n_samples: int
    nodes: dict[str, AnalyticNode]
    joint: AnalyticJointModel
    rule_maps: list[tuple[str, AggregationMap]]


def _scale_petal_features() -> tuple[FloatArray, FloatArray, FloatArray, list[str]]:
    iris = load_iris()
    # petal length (cm), petal width (cm)
    raw = iris.data[:, [2, 3]]
    scaler = MinMaxScaler(feature_range=(0.05, 0.95))
    scaled = scaler.fit_transform(raw)
    return (
        np.asarray(scaled[:, 0], dtype=np.float64),
        np.asarray(scaled[:, 1], dtype=np.float64),
        np.asarray(iris.target, dtype=np.int64),
        list(iris.feature_names[2:4]),
    )


def _fit_logistic_agents(
    evidence: FloatArray,
    ambiguity: FloatArray,
    species: FloatArray,
) -> dict[str, AnalyticNode]:
    x = np.column_stack([evidence, ambiguity])
    # Applicable: non-setosa are "true" (needs full routing); setosa → false.
    y_app = (species != 0).astype(np.int64)
    # Repair alphabet mirrors species as three repair modes.
    repair_labels = np.asarray(["normalize", "impute", "quarantine"])[species]
    # Risk: tertiles of 1 - max species probability from a quick classifier.
    species_clf = LogisticRegression(max_iter=2000)
    species_clf.fit(x, species)
    confidence = species_clf.predict_proba(x).max(axis=1)
    uncertainty = 1.0 - confidence
    edges = np.quantile(uncertainty, [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
    # nudge edges to avoid empty bins on flat regions
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    y_risk = np.digitize(uncertainty, edges[1:-1], right=True).astype(np.int64)  # 0,1,2

    app_clf = LogisticRegression(max_iter=2000)
    app_clf.fit(x, y_app)
    repair_clf = LogisticRegression(max_iter=2000)
    repair_clf.fit(x, repair_labels)
    risk_clf = LogisticRegression(max_iter=2000)
    risk_clf.fit(x, y_risk.astype(str))  # classes "0","1","2"

    def _theta_vec(theta: Mapping[str, float]) -> FloatArray:
        return np.asarray([float(theta["evidence"]), float(theta["ambiguity"])], dtype=np.float64)

    def applicability_node() -> AnalyticNode:
        # classes: [false, true] ordered by clf.classes_
        classes = list(app_clf.classes_)
        false_idx = classes.index(0)
        true_idx = classes.index(1)
        coef = np.asarray(app_clf.coef_[0], dtype=np.float64)  # for positive class in binary
        intercept = float(app_clf.intercept_[0])

        def probabilities(theta: Mapping[str, float]) -> FloatArray:
            point = _theta_vec(theta)
            # sklearn binary: P(class=1) = sigmoid(w·x+b) when classes_[1]==1
            logit = float(coef @ point + intercept)
            p_pos = 1.0 / (1.0 + np.exp(-logit))
            # map to [false, true] support order
            p_true = p_pos if app_clf.classes_[1] == 1 else 1.0 - p_pos
            return np.asarray([1.0 - p_true, p_true], dtype=np.float64)

        def jacobian(theta: Mapping[str, float]) -> FloatArray:
            p = probabilities(theta)
            p_true = float(p[1])
            # d p_true / d theta
            if app_clf.classes_[1] == 1:
                d_true = p_true * (1.0 - p_true) * coef
            else:
                d_true = -(p_true * (1.0 - p_true) * coef)
            return np.asarray([[-d_true[0], -d_true[1]], [d_true[0], d_true[1]]], dtype=np.float64)

        return AnalyticNode(
            node_id="applicability_agent",
            question=noul_question("applicability_agent"),
            model_identity="iris:applicability:logistic",
            probability_function=probabilities,
            jacobian_function=jacobian,
            description="Iris logistic applicability (setosa→false)",
        )

    def multinomial_node(
        node_id: str,
        clf: LogisticRegression,
        support: list[str],
        question,
        description: str,
    ) -> AnalyticNode:
        class_to_row = {str(label): index for index, label in enumerate(clf.classes_)}
        # W shape (K, 2), b shape (K,)
        weights = np.asarray(clf.coef_, dtype=np.float64)
        bias = np.asarray(clf.intercept_, dtype=np.float64)

        def probabilities(theta: Mapping[str, float]) -> FloatArray:
            point = _theta_vec(theta)
            logits = weights @ point + bias
            shifted = logits - logits.max()
            exps = np.exp(shifted)
            probs_model = exps / exps.sum()
            ordered = np.zeros(len(support), dtype=np.float64)
            for index, label in enumerate(support):
                ordered[index] = float(probs_model[class_to_row[str(label)]])
            return ordered

        def jacobian(theta: Mapping[str, float]) -> FloatArray:
            point = _theta_vec(theta)
            logits = weights @ point + bias
            shifted = logits - logits.max()
            q_model = np.exp(shifted)
            q_model = q_model / q_model.sum()
            # dq_model / dθ = q * (W - q W)
            jac_model = q_model[:, None] * (weights - q_model @ weights)
            jac = np.zeros((len(support), 2), dtype=np.float64)
            for index, label in enumerate(support):
                jac[index, :] = jac_model[class_to_row[str(label)], :]
            return jac

        return AnalyticNode(
            node_id=node_id,
            question=question,
            model_identity=f"iris:{node_id}:logistic",
            probability_function=probabilities,
            jacobian_function=jacobian,
            description=description,
        )

    repair_support = ["normalize", "impute", "quarantine"]
    risk_support = ["0", "1", "2"]
    return {
        "applicability_agent": applicability_node(),
        "repair_agent": multinomial_node(
            "repair_agent",
            repair_clf,
            repair_support,
            choice_question(
                "repair_agent",
                {
                    "normalize": "normalize path",
                    "impute": "impute path",
                    "quarantine": "quarantine path",
                },
            ),
            "Iris multinomial repair ← species",
        ),
        "risk_agent": multinomial_node(
            "risk_agent",
            risk_clf,
            risk_support,
            score_question("risk_agent", ["low", "medium", "high"]),
            "Iris multinomial risk ← species-confidence tertiles",
        ),
    }


def _build_joint(nodes: Mapping[str, AnalyticNode]) -> AnalyticJointModel:
    node_order = list(nodes)
    outcomes = {name: node.support for name, node in nodes.items()}

    def probabilities(theta: Mapping[str, float]):
        joint = product_joint_from_nodes(
            node_order,
            outcomes,
            {name: node.probabilities(theta) for name, node in nodes.items()},
            assumptions=["iris: agents conditionally independent given petal features"],
            theta=theta,
        )
        return joint.outcomes, np.asarray(joint.probabilities, dtype=np.float64)

    def jacobian(theta: Mapping[str, float]):
        return product_joint_jacobian(
            node_order,
            outcomes,
            {name: node.probabilities(theta) for name, node in nodes.items()},
            {name: node.jacobian(theta) for name, node in nodes.items()},
            PARAMETERS,
        )

    return AnalyticJointModel(
        node_order=node_order,
        parameter_names=list(PARAMETERS),
        probability_function=probabilities,
        jacobian_function=jacobian,
        model_identity="iris:declared-product:v1",
        construction_mode=ConstructionMode.DECLARED_PRODUCT,
        assumptions=["iris twin; logistic agents fitted on sklearn load_iris"],
    )


def build_iris_bundle() -> IrisBundle:
    evidence, ambiguity, species, feature_names = _scale_petal_features()
    center = np.asarray([float(evidence.mean()), float(ambiguity.mean())], dtype=np.float64)
    nodes = _fit_logistic_agents(evidence, ambiguity, species)
    joint = _build_joint(nodes)
    node_order = list(nodes)
    outcomes = {name: node.support for name, node in nodes.items()}
    rule_maps = [
        (
            name,
            build_aggregation_map(name, rule, node_order, outcomes, description=description),
        )
        for name, rule, description in RULE_SPECS
    ]
    return IrisBundle(
        evidence=evidence,
        ambiguity=ambiguity,
        center=center,
        feature_names=(feature_names[0], feature_names[1]),
        n_samples=int(len(evidence)),
        nodes=nodes,
        joint=joint,
        rule_maps=rule_maps,
    )
