"""Unit-disk sensor twin artifacts and centre-order sanity."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "papers" / "coord-fisher" / "data"
EXAMPLES = REPO / "examples" / "unit_disk_sensors"

sys.path.insert(0, str(EXAMPLES))


def _use_unit_disk_modules() -> None:
    """Other example twins also ship a top-level ``system`` module; pin ours."""
    for name in ("system", "ibrl_system"):
        sys.modules.pop(name, None)
    path = str(EXAMPLES)
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)


def test_unit_disk_artifacts_exist() -> None:
    csv_path = DATA / "unit_disk_center_loss.csv"
    json_path = DATA / "unit_disk_validation_summary.json"
    assert csv_path.is_file() and json_path.is_file()
    rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
    rules = {row["rule"] for row in rows}
    assert len(rules) == 7
    assert {
        "near_identity",
        "wider",
        "baseline",
        "stricter",
        "quarantine_wider",
        "quarantine_review",
        "coarse_binary",
    } <= rules
    summary = json.loads(json_path.read_text())
    assert summary["dataset"] == "synthetic_unit_disk_2d"
    assert summary["fisher_approximation_ok"] is True
    assert abs(float(summary["unit_disk_center_losses"]["near_identity"])) < 1e-8
    assert summary["geometry"]["n_nodes"] == 9


def test_unit_disk_core_order_and_live_eval() -> None:
    _use_unit_disk_modules()
    from system import CORE_RULE_NAMES, build_system
    from jevometry.systems.pushforward import information_loss

    bundle = build_system()
    assert bundle.n_nodes == 9
    assert bundle.radius == pytest.approx(0.45)
    assert len(bundle.edges) > 0
    theta = {
        "amplitude": float(bundle.center[0]),
        "length_scale": float(bundle.center[1]),
    }
    joint = bundle.joint.probabilities(theta)
    jacobian = bundle.joint.jacobian(theta)
    assert jacobian is not None
    losses: dict[str, float] = {}
    for name, aggregation in bundle.rule_maps:
        result = information_loss(
            joint,
            jacobian,
            aggregation,
            parameter_names=["amplitude", "length_scale"],
        )
        assert result.status.value == "ok"
        assert result.psd is True
        assert result.difference is not None
        losses[name] = float(np.trace(result.difference))
    assert abs(losses["near_identity"]) < 1e-8
    core = [losses[name] for name in CORE_RULE_NAMES]
    assert all(core[i] <= core[i + 1] + 1e-9 for i in range(len(core) - 1))


def test_unit_disk_feature_jacobian_fd() -> None:
    """Finite-difference check on field_features Jacobian (smoke)."""
    _use_unit_disk_modules()
    from system import field_features, field_features_jacobian

    amp, ls = 0.7, 0.35
    analytic = field_features_jacobian(amp, ls)
    eps = 1e-6
    for col, (da, dl) in enumerate([(eps, 0.0), (0.0, eps)]):
        plus = field_features(amp + da, ls + dl)
        minus = field_features(amp - da, ls - dl)
        fd = (plus - minus) / (2.0 * eps)
        assert np.allclose(analytic[:, col], fd, rtol=1e-4, atol=1e-5), (
            col,
            analytic[:, col],
            fd,
        )
