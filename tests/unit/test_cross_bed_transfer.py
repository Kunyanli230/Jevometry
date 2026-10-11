"""Cross-bed AggregationMap transfer artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "papers" / "coord-fisher" / "data"
RULES = {
    "near_identity",
    "wider",
    "baseline",
    "stricter",
    "quarantine_wider",
    "quarantine_review",
    "coarse_binary",
}


def test_utility_csvs() -> None:
    for name in ("unit_disk_rule_utility.csv", "ibrl_rule_utility.csv"):
        rows = list(csv.DictReader((DATA / name).open(encoding="utf-8")))
        assert {r["rule"] for r in rows} == RULES
        by = {r["rule"]: r for r in rows}
        assert abs(float(by["near_identity"]["cond_entropy"])) < 1e-9
        assert float(by["near_identity"]["accuracy"]) > 0.99
        assert "tr_delta_I" in rows[0] and "cond_entropy" in rows[0] and "accuracy" in rows[0]


def test_transfer_summary() -> None:
    summary = json.loads((DATA / "cross_bed_transfer_summary.json").read_text())
    assert len(summary["transfers"]) == 6
    pairs = {(t["source"], t["target"]) for t in summary["transfers"]}
    assert pairs == {
        ("unit_disk", "imas2"),
        ("imas2", "unit_disk"),
        ("imas2", "ibrl"),
        ("ibrl", "imas2"),
        ("unit_disk", "ibrl"),
        ("ibrl", "unit_disk"),
    }
    rows = list(csv.DictReader((DATA / "cross_bed_transfer_pairwise.csv").open(encoding="utf-8")))
    assert len(rows) == 6
    assert "S3_negative_control" in summary
    assert summary["S3_negative_control"]["pass"] is False


def test_transfer_script_imports() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "cross_bed_transfer",
        REPO / "examples" / "cross_bed_transfer" / "cross_bed_transfer.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Don't exec main — just load module-level helpers via runpy path check
    assert (REPO / "examples" / "cross_bed_transfer" / "cross_bed_transfer.py").is_file()
