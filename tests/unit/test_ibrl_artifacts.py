"""IBRL real-data twin artifacts and centre-order sanity."""

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


def test_ibrl_artifacts_exist() -> None:
    csv_path = DATA / "ibrl_center_loss.csv"
    json_path = DATA / "ibrl_validation_summary.json"
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
    assert summary["dataset"] == "intel_berkeley_lab_subsample"
    assert summary["fisher_approximation_ok"] is True
    assert abs(float(summary["ibrl_center_losses"]["near_identity"])) < 1e-8
    assert summary["geometry"]["n_motes"] == 12
    assert len(summary["geometry"]["mote_ids"]) == 12
    assert (EXAMPLES / "data" / "ibrl_subsample.tsv").is_file()
    assert (EXAMPLES / "data" / "SOURCE.txt").is_file()


def test_ibrl_core_order_and_live_eval() -> None:
    _use_unit_disk_modules()
    from ibrl_system import CORE_RULE_NAMES, build_ibrl_bundle
    from jevometry.systems.pushforward import information_loss

    bundle = build_ibrl_bundle()
    assert bundle.n_motes == 12
    assert bundle.n_samples >= 100
    assert len(bundle.edges) > 0
    theta = {
        "temperature": float(bundle.center[0]),
        "humidity": float(bundle.center[1]),
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
            parameter_names=["temperature", "humidity"],
        )
        assert result.status.value == "ok"
        assert result.psd is True
        assert result.difference is not None
        losses[name] = float(np.trace(result.difference))
    assert abs(losses["near_identity"]) < 1e-8
    core = [losses[name] for name in CORE_RULE_NAMES]
    assert all(core[i] <= core[i + 1] + 1e-9 for i in range(len(core) - 1))


def test_ibrl_subsample_schema() -> None:
    path = EXAMPLES / "data" / "ibrl_subsample.tsv"
    rows = list(csv.DictReader(path.open(encoding="utf-8"), delimiter="\t"))
    assert len(rows) == 480
    motes = {int(r["mote_id"]) for r in rows}
    assert len(motes) == 12
    for key in ("temperature", "humidity", "voltage", "x", "y"):
        assert key in rows[0]
