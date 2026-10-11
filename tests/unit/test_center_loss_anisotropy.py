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
    assert len(rows) == 6
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
