"""Export centre Δ = I_Z − I_A diagonals and eigenvalues for all coordination rules."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from system import CENTER, PARAMETERS, build_rule_maps, build_system
from jevometry.systems.pushforward import information_loss

CSV_PATH = (
    Path(__file__).resolve().parents[2]
    / "papers"
    / "coord-fisher"
    / "data"
    / "center_loss_anisotropy.csv"
)


def evaluate() -> list[dict[str, float | str | bool]]:
    nodes, joint_model, _baseline = build_system()
    theta = {"evidence": float(CENTER[0]), "ambiguity": float(CENTER[1])}
    joint = joint_model.probabilities(theta)
    jacobian = joint_model.jacobian(theta)
    assert jacobian is not None
    rows: list[dict[str, float | str | bool]] = []
    for name, aggregation in build_rule_maps(nodes):
        result = information_loss(
            joint,
            jacobian,
            aggregation,
            parameter_names=PARAMETERS,
        )
        if result.difference is None:
            rows.append(
                {
                    "rule": name,
                    "evidence": float(CENTER[0]),
                    "ambiguity": float(CENTER[1]),
                    "delta_evidence": float("nan"),
                    "delta_ambiguity": float("nan"),
                    "eig_0": float("nan"),
                    "eig_1": float("nan"),
                    "tr_loss": float("nan"),
                    "min_eigenvalue": float("nan"),
                    "psd": False,
                    "status": result.status.value,
                }
            )
            continue
        delta = np.asarray(result.difference, dtype=np.float64)
        # PARAMETER order is evidence, ambiguity
        eigvals = np.linalg.eigvalsh(0.5 * (delta + delta.T))
        rows.append(
            {
                "rule": name,
                "evidence": float(CENTER[0]),
                "ambiguity": float(CENTER[1]),
                "delta_evidence": float(delta[0, 0]),
                "delta_ambiguity": float(delta[1, 1]),
                "eig_0": float(eigvals[0]),
                "eig_1": float(eigvals[1]),
                "tr_loss": float(np.trace(delta)),
                "min_eigenvalue": float(eigvals.min()),
                "psd": bool(result.psd),
                "status": result.status.value,
            }
        )
    return rows


def write_csv(rows: list[dict[str, float | str | bool]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rule",
        "evidence",
        "ambiguity",
        "delta_evidence",
        "delta_ambiguity",
        "eig_0",
        "eig_1",
        "tr_loss",
        "min_eigenvalue",
        "psd",
        "status",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    rows = evaluate()
    write_csv(rows, CSV_PATH)
    near = next(row for row in rows if row["rule"] == "near_identity")
    assert near["status"] == "ok"
    assert near["psd"] is True
    assert abs(float(near["tr_loss"])) < 1e-10
    assert float(near["min_eigenvalue"]) >= -1e-9
    for row in rows:
        assert row["status"] == "ok"
        assert row["psd"] is True
        assert float(row["min_eigenvalue"]) >= -1e-9
    print(f"wrote {CSV_PATH} ({len(rows)} rules)")
    for row in rows:
        print(
            f"{row['rule']:18s} Δ_e={float(row['delta_evidence']):8.5f} "
            f"Δ_a={float(row['delta_ambiguity']):8.5f} "
            f"eig=({float(row['eig_0']):.5g}, {float(row['eig_1']):.5g}) "
            f"tr={float(row['tr_loss']):.5g} psd={row['psd']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
