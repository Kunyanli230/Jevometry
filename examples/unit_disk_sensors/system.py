"""Synthetic unit-disk 2D sensor-field twin for coord-fisher AggregationMaps.

Framing
-------
1. θ = (amplitude, length_scale): global field parameters to estimate
   (not raw mote readings).
2. Each physical mote sits in [0, 1]² with unit-disk radius R; three *role*
   AnalyticNodes (quality / exchange-mode / uncertainty) are induced by the
   network readings + neighbor geometry so the joint alphabet matches the
   cleaning RULE_SPECS (applicability × repair × risk).
3. AggregationMaps stay θ-independent: push joint judgments → who-exchanges-
   what / quarantine; compare rules with tr(ΔI) = tr(I_Z − I_A).

Semantic map (cleaning labels → sensor coordination)
----------------------------------------------------
- applicability true/false  ↔  observation quality OK to share
- repair normalize/impute/quarantine  ↔  full exchange / summary / isolate
- risk 0/1/2  ↔  low/med/high uncertainty about exchanging
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import product

import numpy as np
from numpy.typing import NDArray
from scipy.special import softmax

from jevometry.adapters.analytic import (
    AnalyticJointModel,
    AnalyticNode,
    choice_question,
    noul_question,
    score_question,
)
from jevometry.schemas.joint import ConstructionMode
from jevometry.schemas.questions import QuestionSpec
from jevometry.systems.joint import product_joint_from_nodes, product_joint_jacobian
from jevometry.systems.pushforward import AggregationMap

FloatArray = NDArray[np.float64]
PARAMETERS = ("amplitude", "length_scale")
# Centre θ used for paper centre tables (in unit cube).
CENTER = np.asarray([0.7, 0.35], dtype=np.float64)
N_SIDE = 3
N_NODES = N_SIDE * N_SIDE
RADIUS = 0.45
SOURCE = np.asarray([0.5, 0.5], dtype=np.float64)
RuleFn = Callable[[tuple[str, str, str]], str]
CORE_RULE_NAMES = ("near_identity", "wider", "baseline", "stricter")


def _grid_positions(n_side: int = N_SIDE) -> FloatArray:
    """n_side×n_side nodes in [0.15, 0.85]²."""
    axis = np.linspace(0.15, 0.85, n_side, dtype=np.float64)
    return np.asarray([(x, y) for x in axis for y in axis], dtype=np.float64)


def _unit_disk_edges(positions: FloatArray, radius: float = RADIUS) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    n = len(positions)
    r2 = radius * radius
    for i in range(n):
        for j in range(i + 1, n):
            d = positions[i] - positions[j]
            if float(d @ d) <= r2 + 1e-15:
                edges.append((i, j))
    return edges


POSITIONS = _grid_positions()
EDGES = _unit_disk_edges(POSITIONS)
MEAN_DEGREE = (2.0 * len(EDGES) / N_NODES) if N_NODES else 0.0


def _physical(amplitude: float, length_scale: float) -> tuple[float, float]:
    """Map unit-cube θ to positive field amplitude and Gaussian σ."""
    a = 0.15 + 0.85 * float(amplitude)
    sigma = 0.08 + 0.42 * float(length_scale)
    return a, sigma


def field_readings(amplitude: float, length_scale: float) -> FloatArray:
    """Smooth Gaussian bump readings at each mote (analytic, noiseless)."""
    a, sigma = _physical(amplitude, length_scale)
    d2 = np.sum((POSITIONS - SOURCE) ** 2, axis=1)
    return np.asarray(a * np.exp(-d2 / (2.0 * sigma * sigma)), dtype=np.float64)


def field_features(amplitude: float, length_scale: float) -> FloatArray:
    """Two θ-dependent network features: energy and neighbor roughness.

    energy ≈ mean reading (quality / share drive);
    roughness ≈ mean |r_i − r_j| on unit-disk edges (uncertainty / isolate drive).
    """
    readings = field_readings(amplitude, length_scale)
    energy = float(readings.mean())
    if not EDGES:
        roughness = 0.0
    else:
        diffs = [abs(float(readings[i] - readings[j])) for i, j in EDGES]
        roughness = float(np.mean(diffs))
    return np.asarray([energy, roughness], dtype=np.float64)


def field_features_jacobian(amplitude: float, length_scale: float) -> FloatArray:
    """Analytic 2×2 Jacobian of (energy, roughness) w.r.t. (amplitude, length_scale)."""
    a, sigma = _physical(amplitude, length_scale)
    da_damp = 0.85
    dsig_dls = 0.42
    d2 = np.sum((POSITIONS - SOURCE) ** 2, axis=1)
    inv_s2 = 1.0 / (sigma * sigma)
    gauss = np.exp(-d2 * 0.5 * inv_s2)
    readings = a * gauss
    # ∂r/∂a = gauss; ∂r/∂σ = a * gauss * d2 / σ³
    dr_da = gauss
    dr_dsig = a * gauss * d2 / (sigma**3)
    dr_damp = dr_da * da_damp
    dr_dls = dr_dsig * dsig_dls
    d_energy = np.asarray([dr_damp.mean(), dr_dls.mean()], dtype=np.float64)
    if not EDGES:
        d_rough = np.zeros(2, dtype=np.float64)
    else:
        acc = np.zeros(2, dtype=np.float64)
        for i, j in EDGES:
            diff = float(readings[i] - readings[j])
            sign = 1.0 if diff >= 0.0 else -1.0
            acc[0] += sign * (dr_damp[i] - dr_damp[j])
            acc[1] += sign * (dr_dls[i] - dr_dls[j])
        d_rough = acc / len(EDGES)
    return np.stack([d_energy, d_rough], axis=0)


# Feature centre at design θ (for softmax offsets).
FEATURE_CENTER = field_features(float(CENTER[0]), float(CENTER[1]))


def _role_agent(
    name: str,
    question: QuestionSpec,
    center_probabilities: list[float],
    weights: list[list[float]],
    *,
    model_tag: str,
) -> AnalyticNode:
    """Softmax family on field features with chain-ruled Jacobian to θ."""
    baseline = np.asarray(center_probabilities, dtype=np.float64)
    slopes = np.asarray(weights, dtype=np.float64)

    def probabilities(theta: Mapping[str, float]) -> FloatArray:
        feats = field_features(float(theta["amplitude"]), float(theta["length_scale"]))
        return np.asarray(
            softmax(np.log(baseline) + slopes @ (feats - FEATURE_CENTER)),
            dtype=np.float64,
        )

    def jacobian(theta: Mapping[str, float]) -> FloatArray:
        feats = field_features(float(theta["amplitude"]), float(theta["length_scale"]))
        values = np.asarray(
            softmax(np.log(baseline) + slopes @ (feats - FEATURE_CENTER)),
            dtype=np.float64,
        )
        # dp/dfeat = diag(p) (W - 1 pᵀ W) = p[:,None] * (W - p @ W)
        jac_feat = values[:, None] * (slopes - values @ slopes)
        dfeat_dtheta = field_features_jacobian(
            float(theta["amplitude"]), float(theta["length_scale"])
        )
        return np.asarray(jac_feat @ dfeat_dtheta, dtype=np.float64)

    return AnalyticNode(
        node_id=name,
        question=question,
        model_identity=f"unit-disk-sensors:{model_tag}:v1",
        probability_function=probabilities,
        jacobian_function=jacobian,
        description=f"Unit-disk role agent {name} (N={N_NODES}, R={RADIUS})",
    )


def final_action(outcome: tuple[str, str, str]) -> str:
    """Baseline: isolate / high risk → reject; full+low → auto_share; else review."""
    applicable, repair, risk = outcome
    if applicable == "false" or repair == "quarantine" or risk == "2":
        return "reject"
    if repair == "normalize" and risk == "0":
        return "auto_fix"
    return "human_review"


def final_action_stricter(outcome: tuple[str, str, str]) -> str:
    applicable, repair, risk = outcome
    if applicable == "false" or repair == "quarantine" or risk in {"1", "2"}:
        return "reject"
    if repair == "normalize" and risk == "0":
        return "auto_share"
    return "human_review"


def final_action_wider(outcome: tuple[str, str, str]) -> str:
    """Wider exchange: summary+low also auto-applies with distinct label."""
    applicable, repair, risk = outcome
    if applicable == "false" or repair == "quarantine" or risk == "2":
        return "reject"
    if repair == "normalize" and risk == "0":
        return "auto_full"
    if repair == "impute" and risk == "0":
        return "auto_summary"
    return "human_review"


def final_action_near_identity(outcome: tuple[str, str, str]) -> str:
    applicable, repair, risk = outcome
    return f"{applicable}|{repair}|{risk}"


def final_action_quarantine_review(outcome: tuple[str, str, str]) -> str:
    """Softer safety valve: isolate → review instead of hard reject."""
    applicable, repair, risk = outcome
    if applicable == "false" or risk == "2":
        return "reject"
    if repair == "quarantine":
        return "human_review"
    if repair == "normalize" and risk == "0":
        return "auto_share"
    return "human_review"


def final_action_quarantine_wider(outcome: tuple[str, str, str]) -> str:
    """Isolate→review plus wider dual auto_full / auto_summary."""
    applicable, repair, risk = outcome
    if applicable == "false" or risk == "2":
        return "reject"
    if repair == "quarantine":
        return "human_review"
    if repair == "normalize" and risk == "0":
        return "auto_full"
    if repair == "impute" and risk == "0":
        return "auto_summary"
    return "human_review"


def final_action_coarse_binary(outcome: tuple[str, str, str]) -> str:
    applicable, repair, risk = outcome
    if applicable == "false" or repair == "quarantine" or risk == "2":
        return "reject"
    return "proceed"


def build_aggregation_map(
    name: str,
    rule: RuleFn,
    node_order: Sequence[str],
    outcomes: Mapping[str, Sequence[str]],
    *,
    description: str | None = None,
) -> AggregationMap:
    mapping = {
        path: rule(path)  # type: ignore[arg-type]
        for path in product(*(outcomes[n] for n in node_order))
    }
    return AggregationMap(
        name=name,
        mapping=mapping,
        theta_dependent=False,
        description=description,
    )


def _build_nodes() -> dict[str, AnalyticNode]:
    """Three role agents; slopes tuned so energy↑ / roughness↓ favour sharing."""
    # Degree enters as a constant scale on exchange slopes (geometry fixed).
    deg_scale = 1.0 + 0.15 * MEAN_DEGREE
    return {
        "applicability_agent": _role_agent(
            "applicability_agent",
            noul_question("applicability_agent"),
            [0.25, 0.75],
            # false ← −energy + roughness; true ← +energy − roughness
            [[-2.8, 2.2], [2.8, -2.2]],
            model_tag="quality",
        ),
        "repair_agent": _role_agent(
            "repair_agent",
            choice_question(
                "repair_agent",
                {
                    "normalize": "full neighbor exchange",
                    "impute": "summary / compressed exchange",
                    "quarantine": "isolate (no exchange)",
                },
            ),
            [0.45, 0.40, 0.15],
            [
                [1.9 * deg_scale, -0.6],
                [0.4, 0.5],
                [-1.6, 1.8 * deg_scale],
            ],
            model_tag="exchange",
        ),
        "risk_agent": _role_agent(
            "risk_agent",
            score_question("risk_agent", ["low", "medium", "high"]),
            [0.55, 0.25, 0.20],
            # low ← energy; high ← roughness
            [[1.6, -1.2], [0.0, 0.3], [-1.4, 1.8]],
            model_tag="uncertainty",
        ),
    }


def _build_joint(nodes: Mapping[str, AnalyticNode]) -> AnalyticJointModel:
    node_order = list(nodes)
    outcomes = {name: node.support for name, node in nodes.items()}

    def probabilities(theta: Mapping[str, float]) -> tuple[list[tuple[str, ...]], FloatArray]:
        joint = product_joint_from_nodes(
            node_order,
            outcomes,
            {name: node.probabilities(theta) for name, node in nodes.items()},
            assumptions=[
                "unit-disk: role agents conditionally independent given field θ",
            ],
            theta=theta,
        )
        return joint.outcomes, np.asarray(joint.probabilities, dtype=np.float64)

    def jacobian(theta: Mapping[str, float]) -> FloatArray:
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
        model_identity="unit-disk-sensors:declared-product:v1",
        construction_mode=ConstructionMode.DECLARED_PRODUCT,
        assumptions=[
            f"synthetic unit-disk field; N={N_NODES}, R={RADIUS}, edges={len(EDGES)}",
        ],
    )


RULE_SPECS: list[tuple[str, RuleFn, str]] = [
    ("baseline", final_action, "Reject isolate/high-risk; full+low → auto_share"),
    ("stricter", final_action_stricter, "Mid-risk also rejects exchange"),
    ("wider", final_action_wider, "Summary+low can auto-apply (distinct labels)"),
    (
        "quarantine_review",
        final_action_quarantine_review,
        "Isolate routes to human_review instead of reject",
    ),
    (
        "quarantine_wider",
        final_action_quarantine_wider,
        "Isolate→review plus wider dual auto_full/auto_summary",
    ),
    (
        "coarse_binary",
        final_action_coarse_binary,
        "Binary reject vs proceed",
    ),
    (
        "near_identity",
        final_action_near_identity,
        "One label per joint outcome (Fisher retention upper bound)",
    ),
]


def build_rule_maps(
    nodes: Mapping[str, AnalyticNode] | None = None,
) -> list[tuple[str, AggregationMap]]:
    if nodes is None:
        nodes = _build_nodes()
    node_order = list(nodes)
    outcomes = {name: node.support for name, node in nodes.items()}
    return [
        (
            name,
            build_aggregation_map(name, rule, node_order, outcomes, description=description),
        )
        for name, rule, description in RULE_SPECS
    ]


@dataclass(frozen=True)
class UnitDiskBundle:
    """Synthetic unit-disk twin: geometry, centre θ, agents, joint, rule maps."""

    positions: FloatArray
    edges: tuple[tuple[int, int], ...]
    radius: float
    n_nodes: int
    mean_degree: float
    center: FloatArray
    feature_center: FloatArray
    nodes: dict[str, AnalyticNode]
    joint: AnalyticJointModel
    rule_maps: list[tuple[str, AggregationMap]]


def build_system() -> UnitDiskBundle:
    """Build the full unit-disk sensor twin."""
    nodes = _build_nodes()
    joint = _build_joint(nodes)
    return UnitDiskBundle(
        positions=POSITIONS.copy(),
        edges=tuple(EDGES),
        radius=RADIUS,
        n_nodes=N_NODES,
        mean_degree=MEAN_DEGREE,
        center=CENTER.copy(),
        feature_center=FEATURE_CENTER.copy(),
        nodes=nodes,
        joint=joint,
        rule_maps=build_rule_maps(nodes),
    )
