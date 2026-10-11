"""Declared-product surrogate for the offline ticket twin."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
EXAMPLE = REPO / "examples" / "jev_ticket_sensitivity"
sys.path.insert(0, str(EXAMPLE))

from ticket_system import (  # noqa: E402
    TicketSystemAdapter,
    build_declared_product_surrogate,
    build_nodes,
)
from jevometry.schemas.system import CompositionMode
from jevometry.systems.pushforward import information_loss, pushforward


CENTER = {"amount": 250.0, "wait_hours": 8.0}


def test_ticket_adapter_declares_product_composition() -> None:
    adapter = TicketSystemAdapter(system_id="ticket-test")
    spec = adapter.describe()
    assert spec.composition_mode is CompositionMode.DECLARED_PRODUCT
    assert adapter.joint_model is not None
    assert adapter.surrogate_aggregation is not None
    assert adapter.surrogate_aggregation.theta_dependent is False


def test_center_information_loss_psd_and_ok() -> None:
    nodes = build_nodes()
    joint_model, aggregation = build_declared_product_surrogate(nodes)
    joint = joint_model.probabilities(CENTER)
    jacobian = joint_model.jacobian(CENTER)
    assert jacobian is not None
    result = information_loss(
        joint,
        jacobian,
        aggregation,
        parameter_names=["amount", "wait_hours"],
    )
    assert result.status.value == "ok"
    assert result.psd is True
    assert result.difference is not None
    eigenvalues = np.linalg.eigvalsh(result.difference)
    assert float(eigenvalues.min()) >= -1e-9
    assert float(np.trace(result.difference)) >= -1e-12


def test_surrogate_masses_match_aggregation_pushforward() -> None:
    adapter = TicketSystemAdapter(system_id="ticket-test")
    soft = adapter.surrogate_action_distribution(CENTER)
    joint = adapter.joint_model.probabilities(CENTER)
    pushed = pushforward(joint, adapter.surrogate_aggregation)
    pushed_masses = {
        outcome[0]: float(mass)
        for outcome, mass in zip(pushed.outcomes, pushed.probabilities, strict=True)
    }
    assert set(soft) == set(pushed_masses)
    for action, mass in soft.items():
        assert mass == pytest.approx(pushed_masses[action], abs=1e-12)


def test_surrogate_rule_loss_ordering_at_center() -> None:
    nodes = build_nodes()
    from ticket_system import build_declared_joint, build_rule_maps

    joint_model = build_declared_joint(nodes)
    joint = joint_model.probabilities(CENTER)
    jacobian = joint_model.jacobian(CENTER)
    assert jacobian is not None
    losses: dict[str, float] = {}
    for name, aggregation in build_rule_maps(nodes):
        result = information_loss(
            joint,
            jacobian,
            aggregation,
            parameter_names=["amount", "wait_hours"],
        )
        assert result.status.value == "ok"
        assert result.psd is True
        assert result.difference is not None
        losses[name] = float(np.trace(result.difference))
    assert losses["near_identity"] == pytest.approx(0.0, abs=1e-10)
    assert losses["near_identity"] <= losses["baseline"]
    assert losses["near_identity"] <= losses["stricter"]
    assert losses["near_identity"] <= losses["coarse_binary"]


def test_surrogate_rule_loss_grid_near_identity_and_order_ratio() -> None:
    import importlib.util

    script = EXAMPLE / "compare_surrogate_rules.py"
    spec = importlib.util.spec_from_file_location("compare_surrogate_rules", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = module.evaluate_grid()
    near = [row for row in rows if row["rule"] == "near_identity"]
    assert near
    near_ok = [row for row in near if row["status"] == "ok"]
    assert near_ok
    assert all(row["psd"] for row in near_ok)
    assert all(abs(float(row["tr_loss"])) < 1e-9 for row in near_ok)
    by_point = {}
    for row in rows:
        key = (float(row["amount"]), float(row["wait_hours"]))
        by_point.setdefault(key, []).append(row)
    # Centre must remain finite and match documented CENTER_ORDER after LOGIT_SCALE.
    centre_rows = by_point[(250.0, 8.0)]
    assert all(row["status"] == "ok" for row in centre_rows)
    assert module.center_order_holds(centre_rows)


def test_surrogate_action_masses_near_identity_conserves() -> None:
    import importlib.util

    script = EXAMPLE / "export_surrogate_action_masses.py"
    spec = importlib.util.spec_from_file_location("export_surrogate_action_masses", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = module.evaluate()
    near = [row for row in rows if row["rule"] == "near_identity"]
    assert near
    for row in near:
        assert abs(float(row["mass_sum"]) - float(row["joint_sum"])) < 1e-12
    # every near_identity row is a joint atom label with positive or zero mass
    assert {row["action"] for row in near}
