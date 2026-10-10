"""Compare Fisher information loss under alternative coordination rules.

Keeps the three agent distributions and DECLARED_PRODUCT joint fixed; only the
AggregationMap changes. Prints tr(I_Z), tr(I_A), tr(loss) and status per rule.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Allow `uv run python examples/three_agent_cleaning/compare_rules.py` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from system import CENTER, PARAMETERS, build_rule_maps, build_system  # noqa: E402

from jevometry.systems.pushforward import information_loss  # noqa: E402


def main() -> None:
    nodes, joint_model, _baseline = build_system()
    theta = {"evidence": float(CENTER[0]), "ambiguity": float(CENTER[1])}
    joint = joint_model.probabilities(theta)
    jacobian = joint_model.jacobian(theta)
    assert jacobian is not None

    rows: list[tuple[str, float, float, float, str]] = []
    for name, aggregation in build_rule_maps(nodes):
        result = information_loss(
            joint,
            jacobian,
            aggregation,
            parameter_names=PARAMETERS,
        )
        if (
            result.fisher_joint is None
            or result.fisher_aggregate is None
            or result.difference is None
        ):
            rows.append((name, float("nan"), float("nan"), float("nan"), result.status.value))
            continue
        rows.append(
            (
                name,
                float(np.trace(result.fisher_joint)),
                float(np.trace(result.fisher_aggregate)),
                float(np.trace(result.difference)),
                result.status.value,
            )
        )

    header = f"{'rule':<16} {'tr(I_Z)':>10} {'tr(I_A)':>10} {'tr(loss)':>10} {'status':>12}"
    print(header)
    print("-" * len(header))
    for name, iz, ia, loss, status in rows:
        print(f"{name:<16} {iz:10.4f} {ia:10.4f} {loss:10.4f} {status:>12}")

    by_name = {name: loss for name, _, _, loss, _ in rows}
    baseline = by_name["baseline"]
    stricter = by_name["stricter"]
    wider = by_name["wider"]
    near = by_name["near_identity"]
    print()
    print(
        "Core ordering check: "
        f"near_identity ({near:.4f}) ≤ wider ({wider:.4f}) ≤ baseline ({baseline:.4f}) "
        f"≤ stricter ({stricter:.4f})"
    )
    assert near <= wider + 1e-9
    assert wider <= baseline + 1e-9
    assert baseline <= stricter + 1e-9
    if "coarse_binary" in by_name:
        print(
            f"Also: coarse_binary loss={by_name['coarse_binary']:.4f}; "
            f"quarantine_review loss={by_name.get('quarantine_review', float('nan')):.4f}"
        )


if __name__ == "__main__":
    main()
