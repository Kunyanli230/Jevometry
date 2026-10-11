"""Compare theta-independent surrogate AggregationMaps at the ticket centre."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from ticket_system import build_declared_joint, build_nodes, build_rule_maps
from jevometry.systems.pushforward import information_loss

CENTER = {"amount": 250.0, "wait_hours": 8.0}


def main() -> int:
    nodes = build_nodes()
    joint_model = build_declared_joint(nodes)
    joint = joint_model.probabilities(CENTER)
    jacobian = joint_model.jacobian(CENTER)
    assert jacobian is not None
    rows: list[tuple[str, float, float, float, str]] = []
    print(f"{'rule':16s}  tr(I_Z)    tr(I_A)    tr(loss)   status")
    for name, aggregation in build_rule_maps(nodes):
        result = information_loss(
            joint,
            jacobian,
            aggregation,
            parameter_names=["amount", "wait_hours"],
        )
        tr_z = float(np.trace(result.fisher_joint)) if result.fisher_joint is not None else float("nan")
        tr_a = (
            float(np.trace(result.fisher_aggregate))
            if result.fisher_aggregate is not None
            else float("nan")
        )
        tr_loss = float(np.trace(result.difference)) if result.difference is not None else float("nan")
        print(f"{name:16s}  {tr_z:8.6f}  {tr_a:8.6f}  {tr_loss:8.6f}  {result.status.value}")
        rows.append((name, tr_z, tr_a, tr_loss, result.status.value))
    by_loss = sorted(rows, key=lambda row: row[3])
    print("order (ascending loss):", " < ".join(name for name, *_ in by_loss))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
