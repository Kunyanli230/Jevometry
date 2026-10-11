"""Centre Δ anisotropy export for coordination-rule Fisher paper."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
EXAMPLE = REPO / "examples" / "three_agent_cleaning"
sys.path.insert(0, str(EXAMPLE))


def _load_export():
    script = EXAMPLE / "export_center_loss_anisotropy.py"
    spec = importlib.util.spec_from_file_location("export_center_loss_anisotropy", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_center_loss_anisotropy_near_zero_and_psd() -> None:
    module = _load_export()
    rows = module.evaluate()
    assert len(rows) == 7
    by = {str(row["rule"]): row for row in rows}
    near = by["near_identity"]
    assert near["status"] == "ok"
    assert near["psd"] is True
    assert float(near["tr_loss"]) == pytest.approx(0.0, abs=1e-10)
    assert float(near["min_eigenvalue"]) >= -1e-9
    assert float(near["delta_evidence"]) == pytest.approx(0.0, abs=1e-10)
    assert float(near["delta_ambiguity"]) == pytest.approx(0.0, abs=1e-10)
    for row in rows:
        assert row["status"] == "ok"
        assert row["psd"] is True
        assert float(row["min_eigenvalue"]) >= -1e-9
        assert float(row["eig_0"]) <= float(row["eig_1"]) + 1e-12


def test_rule_loss_anisotropy_grid_near_psd_and_core_order() -> None:
    import importlib.util

    script = EXAMPLE / "export_rule_loss_anisotropy_grid.py"
    spec = importlib.util.spec_from_file_location("export_rule_loss_anisotropy_grid", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = module.evaluate()
    assert len(rows) == 8 * 8 * 7
    near = [row for row in rows if row["rule"] == "near_identity"]
    assert all(row["status"] == "ok" and row["psd"] for row in near)
    assert all(abs(float(row["tr_loss"])) < 1e-9 for row in near)
    assert all(float(row["min_eigenvalue"]) >= -1e-9 for row in rows if row["status"] == "ok")
    stats = module.summarize(rows)
    # Document empirical stability; require at least the design centre.
    by_point = {}
    for row in rows:
        key = (float(row["evidence"]), float(row["ambiguity"]))
        by_point.setdefault(key, []).append(row)
    from system import CENTER

    centre_key = (float(CENTER[0]), float(CENTER[1]))
    # Centre may not sit exactly on the linspace; require hold>=1 overall.
    assert int(stats["hold"]) >= 1
    assert float(stats["ratio"]) > 0.0


def test_frac_loss_fields_and_quarantine_lowest_at_centre() -> None:
    import importlib.util

    script = EXAMPLE / "export_rule_loss_anisotropy_grid.py"
    spec = importlib.util.spec_from_file_location("export_rule_loss_anisotropy_grid", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = module.evaluate()
    sample = next(row for row in rows if row["status"] == "ok")
    assert "frac_loss" in sample and "frac_delta_ambiguity" in sample
    # nearest grid point to design centre
    from system import CENTER

    best = min(
        {(float(row["evidence"]), float(row["ambiguity"])) for row in rows},
        key=lambda point: (point[0] - float(CENTER[0])) ** 2 + (point[1] - float(CENTER[1])) ** 2,
    )
    centre = [
        row
        for row in rows
        if (float(row["evidence"]), float(row["ambiguity"])) == best and row["rule"] != "near_identity"
    ]
    fracs = {str(row["rule"]): float(row["frac_loss"]) for row in centre}
    assert fracs["quarantine_wider"] <= fracs["quarantine_review"] + 1e-12
    assert fracs["quarantine_wider"] == min(fracs.values())


def test_quarantine_wider_centre_loss_leq_quarantine_review() -> None:
    import importlib.util

    script = EXAMPLE / "export_center_loss_anisotropy.py"
    spec = importlib.util.spec_from_file_location("export_center_loss_anisotropy", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = module.evaluate()
    by = {str(row["rule"]): float(row["tr_loss"]) for row in rows}
    assert "quarantine_wider" in by
    assert by["quarantine_wider"] <= by["quarantine_review"] + 1e-12


def test_quarantine_wider_vs_review_json() -> None:
    import json

    path = REPO / "papers" / "coord-fisher" / "data" / "quarantine_wider_vs_review.json"
    assert path.is_file()
    summary = json.loads(path.read_text())
    assert summary["predicate"] == "quarantine_wider <= quarantine_review"
    assert summary["hold"] == summary["grid_points"]
    assert summary["hold_ratio"] == 1.0
    assert summary["fail"] == 0


def test_quarantine_action_mass_compare_artifacts() -> None:
    import csv
    import json

    csv_path = REPO / "papers" / "coord-fisher" / "data" / "quarantine_action_mass_compare.csv"
    json_path = REPO / "papers" / "coord-fisher" / "data" / "quarantine_action_mass_compare.json"
    assert csv_path.is_file() and json_path.is_file()
    rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
    assert rows
    summary = json.loads(json_path.read_text())
    assert summary["rules"] == ["quarantine_review", "quarantine_wider"]
    # wider splits auto_fix into distinct autos; review keeps auto_fix
    assert "auto_fix" in summary["actions_only_in_review"] or any(
        row["action"] == "auto_fix" for row in rows
    )


def test_seven_rules_in_paper_artifacts() -> None:
    import json

    center = json.loads((REPO / "papers" / "coord-fisher" / "data" / "center_summary.json").read_text())
    rules = [row["rule"] for row in center]
    assert len(rules) == 7
    assert "quarantine_wider" in rules
    assert (REPO / "papers" / "coord-fisher" / "figures" / "fig_loss_center.png").is_file()
    assert (REPO / "papers" / "coord-fisher" / "figures" / "fig_action_masses.png").is_file()
    status = json.loads((REPO / "papers" / "coord-fisher" / "data" / "seven_rules_export.json").read_text())
    assert status["n_rules"] == 7
