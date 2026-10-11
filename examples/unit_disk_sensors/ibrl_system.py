"""IBRL (Intel Berkeley Lab) real-data twin for coord-fisher AggregationMaps.

Loads the committed 12-mote subsample under ``data/``, maps MinMax-scaled
temperature/humidity onto θ, fits Iris-style logistic role agents, and reuses
``RULE_SPECS`` / AggregationMaps from ``system.py`` (quarantine = isolate).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

from system import (
    CORE_RULE_NAMES,
    RULE_SPECS,
    build_aggregation_map,
)
from jevometry.adapters.analytic import (
    AnalyticJointModel,
    AnalyticNode,
    choice_question,
    noul_question,
    score_question,
)
from jevometry.schemas.joint import ConstructionMode
from jevometry.systems.joint import product_joint_from_nodes, product_joint_jacobian
from jevometry.systems.pushforward import AggregationMap

FloatArray = NDArray[np.float64]
PARAMETERS = ("temperature", "humidity")
DATA_DIR = Path(__file__).resolve().parent / "data"
SUBSAMPLE_PATH = DATA_DIR / "ibrl_subsample.tsv"
# Unit-disk radius on MinMax-normalized lab coordinates (meters → [0,1]).
NORM_RADIUS = 0.35


@dataclass(frozen=True)
class IbrlBundle:
    """Fitted IBRL twin: scaled features, geometry, agents, joint, rule maps."""

    temperature: FloatArray
    humidity: FloatArray
    center: FloatArray
    mote_ids: tuple[int, ...]
    n_samples: int
    n_motes: int
    positions_norm: FloatArray
    edges: tuple[tuple[int, int], ...]
    radius_norm: float
    nodes: dict[str, AnalyticNode]
    joint: AnalyticJointModel
    rule_maps: list[tuple[str, AggregationMap]]
    feature_names: tuple[str, str]


def _load_subsample(path: Path = SUBSAMPLE_PATH) -> dict[str, FloatArray]:
    if not path.is_file():
        raise FileNotFoundError(f"missing IBRL subsample: {path}")
    rows = path.read_text(encoding="utf-8").strip().splitlines()
    header = rows[0].split("\t")
    idx = {name: i for i, name in enumerate(header)}
    mote, temp, hum, volt, xs, ys = [], [], [], [], [], []
    for line in rows[1:]:
        parts = line.split("\t")
        mote.append(int(parts[idx["mote_id"]]))
        temp.append(float(parts[idx["temperature"]]))
        hum.append(float(parts[idx["humidity"]]))
        volt.append(float(parts[idx["voltage"]]))
        xs.append(float(parts[idx["x"]]))
        ys.append(float(parts[idx["y"]]))
    return {
        "mote_id": np.asarray(mote, dtype=np.int64),
        "temperature": np.asarray(temp, dtype=np.float64),
        "humidity": np.asarray(hum, dtype=np.float64),
        "voltage": np.asarray(volt, dtype=np.float64),
        "x": np.asarray(xs, dtype=np.float64),
        "y": np.asarray(ys, dtype=np.float64),
    }


def _mote_geometry(raw: Mapping[str, FloatArray]) -> tuple[tuple[int, ...], FloatArray, list[tuple[int, int]]]:
    """Unique motes, normalized positions, unit-disk edges among motes."""
    ids = tuple(sorted({int(m) for m in raw["mote_id"].tolist()}))
    # one (x,y) per mote from first occurrence
    xy = {}
    for mid, x, y in zip(raw["mote_id"], raw["x"], raw["y"], strict=True):
        mid_i = int(mid)
        if mid_i not in xy:
            xy[mid_i] = (float(x), float(y))
    coords = np.asarray([xy[m] for m in ids], dtype=np.float64)
    # MinMax to [0,1]²
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    span = np.where(hi - lo < 1e-9, 1.0, hi - lo)
    norm = (coords - lo) / span
    edges: list[tuple[int, int]] = []
    r2 = NORM_RADIUS * NORM_RADIUS
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            d = norm[i] - norm[j]
            if float(d @ d) <= r2 + 1e-15:
                edges.append((i, j))
    return ids, norm, edges


def _label_rows(
    raw: Mapping[str, FloatArray],
    mote_ids: tuple[int, ...],
    positions_norm: FloatArray,
    edges: list[tuple[int, int]],
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Per-reading labels: applicability (0/1), repair class, risk (0/1/2)."""
    id_to_index = {m: i for i, m in enumerate(mote_ids)}
    # per-mote mean temperature for neighbor disagreement
    mote_temp: dict[int, list[float]] = {m: [] for m in mote_ids}
    for mid, t in zip(raw["mote_id"], raw["temperature"], strict=True):
        mote_temp[int(mid)].append(float(t))
    mote_mean = {m: float(np.mean(vals)) for m, vals in mote_temp.items()}
    # neighbor mean temp per mote index
    neighbor_mean = np.zeros(len(mote_ids), dtype=np.float64)
    for i, m in enumerate(mote_ids):
        neigh = []
        for a, b in edges:
            if a == i:
                neigh.append(mote_mean[mote_ids[b]])
            elif b == i:
                neigh.append(mote_mean[mote_ids[a]])
        neighbor_mean[i] = float(np.mean(neigh)) if neigh else mote_mean[m]
    disagree = np.asarray(
        [
            abs(float(t) - neighbor_mean[id_to_index[int(mid)]])
            for mid, t in zip(raw["mote_id"], raw["temperature"], strict=True)
        ],
        dtype=np.float64,
    )
    volt = raw["voltage"]
    volt_ok = volt >= float(np.median(volt))
    # applicable = healthy voltage and not extreme disagreement
    d_hi = float(np.quantile(disagree, 0.75))
    y_app = ((volt_ok) & (disagree <= d_hi)).astype(np.int64)
    # repair: low disagree → normalize (full), mid → impute (summary), high → quarantine
    q33, q66 = np.quantile(disagree, [1.0 / 3.0, 2.0 / 3.0])
    repair = np.empty(len(disagree), dtype=object)
    repair[disagree <= q33] = "normalize"
    repair[(disagree > q33) & (disagree <= q66)] = "impute"
    repair[disagree > q66] = "quarantine"
    # risk tertiles of disagreement
    edges_q = np.quantile(disagree, [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
    edges_q[0] -= 1e-9
    edges_q[-1] += 1e-9
    y_risk = np.digitize(disagree, edges_q[1:-1], right=True).astype(np.int64)
    return y_app, repair, y_risk


def _fit_logistic_agents(
    temperature: FloatArray,
    humidity: FloatArray,
    y_app: FloatArray,
    repair: FloatArray,
    y_risk: FloatArray,
) -> dict[str, AnalyticNode]:
    x = np.column_stack([temperature, humidity])
    app_clf = LogisticRegression(max_iter=2000)
    app_clf.fit(x, y_app)
    repair_clf = LogisticRegression(max_iter=2000)
    repair_clf.fit(x, repair)
    risk_clf = LogisticRegression(max_iter=2000)
    risk_clf.fit(x, y_risk.astype(str))

    def _theta_vec(theta: Mapping[str, float]) -> FloatArray:
        return np.asarray(
            [float(theta["temperature"]), float(theta["humidity"])],
            dtype=np.float64,
        )

    def applicability_node() -> AnalyticNode:
        coef = np.asarray(app_clf.coef_[0], dtype=np.float64)
        intercept = float(app_clf.intercept_[0])

        def probabilities(theta: Mapping[str, float]) -> FloatArray:
            point = _theta_vec(theta)
            logit = float(coef @ point + intercept)
            p_pos = 1.0 / (1.0 + np.exp(-logit))
            p_true = p_pos if int(app_clf.classes_[1]) == 1 else 1.0 - p_pos
            return np.asarray([1.0 - p_true, p_true], dtype=np.float64)

        def jacobian(theta: Mapping[str, float]) -> FloatArray:
            p = probabilities(theta)
            p_true = float(p[1])
            if int(app_clf.classes_[1]) == 1:
                d_true = p_true * (1.0 - p_true) * coef
            else:
                d_true = -(p_true * (1.0 - p_true) * coef)
            return np.asarray(
                [[-d_true[0], -d_true[1]], [d_true[0], d_true[1]]],
                dtype=np.float64,
            )

        return AnalyticNode(
            node_id="applicability_agent",
            question=noul_question("applicability_agent"),
            model_identity="ibrl:applicability:logistic",
            probability_function=probabilities,
            jacobian_function=jacobian,
            description="IBRL logistic quality (voltage + neighbor disagreement)",
        )

    def multinomial_node(
        node_id: str,
        clf: LogisticRegression,
        support: list[str],
        question,
        description: str,
    ) -> AnalyticNode:
        class_to_row = {str(label): index for index, label in enumerate(clf.classes_)}
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
            jac_model = q_model[:, None] * (weights - q_model @ weights)
            jac = np.zeros((len(support), 2), dtype=np.float64)
            for index, label in enumerate(support):
                jac[index, :] = jac_model[class_to_row[str(label)], :]
            return jac

        return AnalyticNode(
            node_id=node_id,
            question=question,
            model_identity=f"ibrl:{node_id}:logistic",
            probability_function=probabilities,
            jacobian_function=jacobian,
            description=description,
        )

    return {
        "applicability_agent": applicability_node(),
        "repair_agent": multinomial_node(
            "repair_agent",
            repair_clf,
            ["normalize", "impute", "quarantine"],
            choice_question(
                "repair_agent",
                {
                    "normalize": "full neighbor exchange",
                    "impute": "summary / compressed exchange",
                    "quarantine": "isolate (no exchange)",
                },
            ),
            "IBRL multinomial exchange ← neighbor temp disagreement tertiles",
        ),
        "risk_agent": multinomial_node(
            "risk_agent",
            risk_clf,
            ["0", "1", "2"],
            score_question("risk_agent", ["low", "medium", "high"]),
            "IBRL multinomial risk ← disagreement tertiles",
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
            assumptions=["ibrl: role agents conditionally independent given θ"],
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
        model_identity="ibrl:declared-product:v1",
        construction_mode=ConstructionMode.DECLARED_PRODUCT,
        assumptions=[
            "IBRL twin; logistic agents on MinMax temperature/humidity subsample",
        ],
    )


def build_ibrl_bundle(path: Path = SUBSAMPLE_PATH) -> IbrlBundle:
    raw = _load_subsample(path)
    mote_ids, positions_norm, edges = _mote_geometry(raw)
    scaler = MinMaxScaler(feature_range=(0.05, 0.95))
    scaled = scaler.fit_transform(
        np.column_stack([raw["temperature"], raw["humidity"]])
    )
    temperature = np.asarray(scaled[:, 0], dtype=np.float64)
    humidity = np.asarray(scaled[:, 1], dtype=np.float64)
    y_app, repair, y_risk = _label_rows(raw, mote_ids, positions_norm, edges)
    nodes = _fit_logistic_agents(temperature, humidity, y_app, repair, y_risk)
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
    center = np.asarray(
        [float(temperature.mean()), float(humidity.mean())],
        dtype=np.float64,
    )
    return IbrlBundle(
        temperature=temperature,
        humidity=humidity,
        center=center,
        mote_ids=mote_ids,
        n_samples=int(len(temperature)),
        n_motes=len(mote_ids),
        positions_norm=positions_norm,
        edges=tuple(edges),
        radius_norm=NORM_RADIUS,
        nodes=nodes,
        joint=joint,
        rule_maps=rule_maps,
        feature_names=("temperature", "humidity"),
    )


# re-export for tests / experiments
__all__ = [
    "CORE_RULE_NAMES",
    "PARAMETERS",
    "IbrlBundle",
    "build_ibrl_bundle",
    "SUBSAMPLE_PATH",
]
