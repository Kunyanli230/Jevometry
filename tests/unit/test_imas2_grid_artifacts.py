"""IMAS² grid-world comparison artifacts (coord-fisher phase 3)."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "papers" / "coord-fisher" / "data"
sys.path.insert(0, str(REPO / "examples" / "imas2_grid"))

MAIN_FIELDS = ["rule", "policy_type", "dynamics", "sensor_range", "tr_delta_I", "cond_entropy",
               "accuracy", "seed", "n_trajectories", "selection_time_s", "training_time_s"]


def test_main_csv_schema_and_coverage() -> None:
    rows = list(csv.DictReader((DATA / "imas2_rule_comparison.csv").open(encoding="utf-8")))
    assert set(MAIN_FIELDS) <= set(rows[0])
    cfgs = {(r["policy_type"], r["dynamics"], r["sensor_range"]) for r in rows}
    for pt in ("fixed_orientation", "random_orientation"):
        for d in ("deterministic", "stochastic"):
            for s in ("small", "large"):
                assert (pt, d, s) in cfgs
    for cfg in cfgs:
        sub = [r for r in rows if (r["policy_type"], r["dynamics"], r["sensor_range"]) == cfg]
        assert len(sub) == 7
        by = {r["rule"]: r for r in sub}
        assert abs(float(by["near_identity"]["tr_delta_I"])) < 1e-6
        assert all(float(r["tr_delta_I"]) >= -1e-6 for r in sub)
        # near_identity keeps the full fused report: lowest entropy (MC tolerance)
        assert float(by["near_identity"]["cond_entropy"]) <= min(float(r["cond_entropy"]) for r in sub) + 0.02


def test_placements_and_summary() -> None:
    rows = list(csv.DictReader((DATA / "imas2_placements.csv").open(encoding="utf-8")))
    assert len(rows) == 126
    assert {"placement", "fisher_score", "mi_score"} <= set(rows[0])
    s = json.loads((DATA / "imas2_summary.json").read_text())
    for key in ("deterministic_small", "stochastic_small"):
        p = s["placements"][key]
        assert -1.0 <= p["spearman"] <= 1.0
        assert 0 <= p["top5_overlap"] <= 5
        assert set(p["selection_time_s"]) == {"fisher_all_126", "mi_mc_all_126"}
    assert "REPORTED" in s["reported_imas2"]["source"]
    assert (REPO / "third_party" / "imas2_grid" / "LICENSE").is_file()


def test_vectorised_emission_matches_upstream() -> None:
    """Our vectorised sensor model agrees with the vendored IMAS² POMDP emission."""
    import system as S
    from pomdp_grid import POMDP

    model = S.GridModel("stochastic", "small", tuple(range(9)))
    pomdp = POMDP(model.env)
    for k in (2, 4, 8):
        for act in S.ACTIONS:
            cov = model.coverage_all[k][act]
            for cell in model.cells[::7]:
                in_range = pomdp.in_sensor_range((cell, 0), act, k)
                assert in_range == bool(cov[model.idx[cell]])
                p_null = pomdp.emiss_list[k][(cell, 0)][act][("n", model.env.sensors[k])]
                assert np.isclose(p_null, S.OBS_NOISE if in_range else 1.0)


def test_retrained_rows_when_rollouts_present() -> None:
    if not (DATA / "imas2_retrain_rollouts.json").is_file():
        return
    rows = [r for r in csv.DictReader((DATA / "imas2_rule_comparison.csv").open(encoding="utf-8"))
            if r["policy_type"] == "retrained_imas2"]
    assert len(rows) == 7
    assert all(float(r["training_time_s"]) > 0 for r in rows)
