"""Committed paper artifacts for the TypeSafe ticket surrogate study."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PAPER = REPO / "papers" / "typesafe-ticket"
DATA = PAPER / "data"
FIGS = PAPER / "figures"
EXAMPLE = REPO / "examples" / "jev_ticket_sensitivity"


def test_surrogate_paper_artifacts_exist_and_summary_fields() -> None:
    summary_path = DATA / "surrogate_rule_center_summary.json"
    grid_path = DATA / "surrogate_rule_loss_grid.csv"
    masses_path = DATA / "surrogate_action_masses.csv"
    assert summary_path.is_file()
    assert grid_path.is_file()
    assert masses_path.is_file()
    assert (FIGS / "surrogate_rule_loss_center.png").is_file()
    assert (FIGS / "surrogate_rule_loss_heatmap.png").is_file()

    summary = json.loads(summary_path.read_text())
    for key in (
        "LOGIT_SCALE",
        "PROB_FLOOR",
        "tr_Iz",
        "tr_loss",
        "action_mass_points",
        "legacy_flip_present",
        "center_order_holds",
        "grid_ok_points",
        "grid_points",
    ):
        assert key in summary

    assert summary["LOGIT_SCALE"] == pytest.approx(15.0)
    assert summary["PROB_FLOOR"] == pytest.approx(1e-12)
    assert summary["grid_ok_points"] == summary["grid_points"] == 9
    assert isinstance(summary["legacy_flip_present"], bool)
    assert set(summary["action_mass_points"]) >= {"(250,8)", "(600,2)", "(600,8)"}
    for point, payload in summary["action_mass_points"].items():
        assert "priority_queue_delta" in payload
        assert "standard_queue_delta" in payload
        assert "legacy_flip_present" in payload

    mass_rows = list(csv.DictReader(masses_path.open(newline="", encoding="utf-8")))
    assert mass_rows
    assert "LOGIT_SCALE" in mass_rows[0] and "PROB_FLOOR" in mass_rows[0]
    scales = {float(row["LOGIT_SCALE"]) for row in mass_rows}
    floors = {float(row["PROB_FLOOR"]) for row in mass_rows}
    assert scales == {float(summary["LOGIT_SCALE"])}
    assert floors == {float(summary["PROB_FLOOR"])}

    # twin module constants match committed markers
    import sys

    sys.path.insert(0, str(EXAMPLE))
    from ticket_system import LOGIT_SCALE, PROB_FLOOR

    assert float(LOGIT_SCALE) == pytest.approx(float(summary["LOGIT_SCALE"]))
    assert float(PROB_FLOOR) == pytest.approx(float(summary["PROB_FLOOR"]))
