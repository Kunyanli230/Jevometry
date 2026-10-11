"""Export 8×8 grid of Δ diagonals (evidence/ambiguity) for all coordination rules."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from system import CENTER, PARAMETERS, build_rule_maps, build_system
from jevometry.systems.pushforward import information_loss

EVIDENCE_GRID = np.linspace(0.2, 0.9, 8)
AMBIGUITY_GRID = np.linspace(0.1, 0.8, 8)
CORE_ORDER = ("near_identity", "wider", "baseline", "stricter")
CSV_PATH = (
    Path(__file__).resolve().parents[2]
    / "papers"
    / "coord-fisher"
    / "data"
    / "rule_loss_anisotropy_grid.csv"
)


def theta_grid() -> list[dict[str, float]]:
    return [
        {"evidence": float(evidence), "ambiguity": float(ambiguity)}
        for evidence in EVIDENCE_GRID
        for ambiguity in AMBIGUITY_GRID
    ]


def evaluate() -> list[dict[str, float | str | bool]]:
    nodes, joint_model, _baseline = build_system()
    rule_maps = build_rule_maps(nodes)
    rows: list[dict[str, float | str | bool]] = []
    for theta in theta_grid():
        joint = joint_model.probabilities(theta)
        jacobian = joint_model.jacobian(theta)
        assert jacobian is not None
        is_center = (
            abs(theta["evidence"] - float(CENTER[0])) < 1e-12
            and abs(theta["ambiguity"] - float(CENTER[1])) < 1e-12
        )
        for name, aggregation in rule_maps:
            result = information_loss(
                joint,
                jacobian,
                aggregation,
                parameter_names=PARAMETERS,
            )
            if result.difference is None:
                rows.append(
                    {
                        "evidence": theta["evidence"],
                        "ambiguity": theta["ambiguity"],
                        "rule": name,
                        "delta_evidence": float("nan"),
                        "delta_ambiguity": float("nan"),
                        "tr_loss": float("nan"),
                        "min_eigenvalue": float("nan"),
                        "psd": False,
                        "status": result.status.value,
                        "is_center": is_center,
                    }
                )
                continue
            delta = np.asarray(result.difference, dtype=np.float64)
            eigvals = np.linalg.eigvalsh(0.5 * (delta + delta.T))
            rows.append(
                {
                    "evidence": theta["evidence"],
                    "ambiguity": theta["ambiguity"],
                    "rule": name,
                    "delta_evidence": float(delta[0, 0]),
                    "delta_ambiguity": float(delta[1, 1]),
                    "tr_loss": float(np.trace(delta)),
                    "min_eigenvalue": float(eigvals.min()),
                    "psd": bool(result.psd),
                    "status": result.status.value,
                    "is_center": is_center,
                }
            )
    return rows


def core_order_holds(point_rows: list[dict[str, float | str | bool]]) -> bool:
    losses = {str(row["rule"]): float(row["tr_loss"]) for row in point_rows}
    if any(name not in losses for name in CORE_ORDER):
        return False
    sequence = [losses[name] for name in CORE_ORDER]
    return all(sequence[index] <= sequence[index + 1] + 1e-9 for index in range(len(sequence) - 1))


def summarize(rows: list[dict[str, float | str | bool]]) -> dict[str, float | int]:
    by_point: dict[tuple[float, float], list[dict[str, float | str | bool]]] = {}
    for row in rows:
        key = (float(row["evidence"]), float(row["ambiguity"]))
        by_point.setdefault(key, []).append(row)
    hold = sum(1 for point_rows in by_point.values() if core_order_holds(point_rows))
    total = len(by_point)
    return {"hold": hold, "total": total, "ratio": hold / total if total else 0.0}


def write_csv(rows: list[dict[str, float | str | bool]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "evidence",
        "ambiguity",
        "rule",
        "delta_evidence",
        "delta_ambiguity",
        "tr_loss",
        "min_eigenvalue",
        "psd",
        "status",
        "is_center",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    rows = evaluate()
    write_csv(rows, CSV_PATH)
    near = [row for row in rows if row["rule"] == "near_identity"]
    assert near
    assert all(row["status"] == "ok" and row["psd"] for row in near)
    assert all(abs(float(row["tr_loss"])) < 1e-9 for row in near)
    assert all(float(row["min_eigenvalue"]) >= -1e-9 for row in rows if row["status"] == "ok")
    stats = summarize(rows)
    print(f"wrote {CSV_PATH} ({len(rows)} rows)")
    print(
        f"core order near≤wider≤baseline≤stricter holds on "
        f"{stats['hold']}/{stats['total']} grid points ({float(stats['ratio']):.0%})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
