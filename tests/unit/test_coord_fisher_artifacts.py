"""Committed coord-fisher paper artifacts from the 10-round iteration."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PAPER = REPO / "papers" / "coord-fisher"
DATA = PAPER / "data"
FIGS = PAPER / "figures"


def test_coord_fisher_iteration_summary_and_artifacts() -> None:
    summary_path = DATA / "iteration_summary.json"
    assert summary_path.is_file()
    summary = json.loads(summary_path.read_text())
    assert summary["n_rules"] == 7
    rules = [row["rule"] for row in summary["center_tr_loss_ascending"]]
    assert "quarantine_wider" in rules
    assert rules == sorted(
        rules,
        key=lambda name: next(
            float(row["tr_loss"])
            for row in summary["center_tr_loss_ascending"]
            if row["rule"] == name
        ),
    )
    assert summary["quarantine_wider"]["hold_ratio"] == 1.0

    required = [
        DATA / "center_loss_anisotropy.csv",
        DATA / "rule_loss_anisotropy_grid.csv",
        DATA / "quarantine_action_mass_compare.csv",
        DATA / "quarantine_action_mass_compare.json",
        DATA / "quarantine_wider_vs_review.json",
        DATA / "center_summary.json",
        DATA / "rule_loss_grid.csv",
        FIGS / "fig_frac_loss_heatmap.png",
        FIGS / "fig_loss_center.png",
        FIGS / "fig_action_masses.png",
    ]
    missing = [str(path.relative_to(REPO)) for path in required if not path.is_file()]
    assert not missing, missing


def test_iris_validation_artifacts() -> None:
    import csv
    import json

    csv_path = DATA / "iris_center_loss.csv"
    json_path = DATA / "iris_validation_summary.json"
    assert csv_path.is_file() and json_path.is_file()
    rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
    rules = {row["rule"] for row in rows}
    assert {"near_identity", "wider", "baseline", "stricter"} <= rules
    summary = json.loads(json_path.read_text())
    assert summary["dataset"] == "sklearn.datasets.load_iris"
    assert summary["fisher_approximation_ok"] is True
    assert abs(float(summary["iris_center_losses"]["near_identity"])) < 1e-8
