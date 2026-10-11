"""Compare theta-independent surrogate AggregationMaps on the offline ticket grid."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from ticket_system import build_declared_joint, build_nodes, build_rule_maps
from jevometry.systems.pushforward import information_loss

CENTER = {"amount": 250.0, "wait_hours": 8.0}
AMOUNT_VALUES = (100.0, 250.0, 600.0)
WAIT_VALUES = (2.0, 8.0, 30.0)
CENTER_ORDER = ("near_identity", "stricter", "baseline", "coarse_binary")
CSV_PATH = (
    Path(__file__).resolve().parents[2]
    / "papers"
    / "typesafe-ticket"
    / "data"
    / "surrogate_rule_loss_grid.csv"
)


def theta_grid() -> list[dict[str, float]]:
    return [
        {"amount": amount, "wait_hours": wait}
        for amount in AMOUNT_VALUES
        for wait in WAIT_VALUES
    ]


def evaluate_grid() -> list[dict[str, float | str]]:
    nodes = build_nodes()
    joint_model = build_declared_joint(nodes)
    rule_maps = build_rule_maps(nodes)
    rows: list[dict[str, float | str]] = []
    for theta in theta_grid():
        joint = joint_model.probabilities(theta)
        jacobian = joint_model.jacobian(theta)
        assert jacobian is not None
        for name, aggregation in rule_maps:
            result = information_loss(
                joint,
                jacobian,
                aggregation,
                parameter_names=["amount", "wait_hours"],
            )
            tr_z = (
                float(np.trace(result.fisher_joint))
                if result.fisher_joint is not None
                else float("nan")
            )
            tr_a = (
                float(np.trace(result.fisher_aggregate))
                if result.fisher_aggregate is not None
                else float("nan")
            )
            tr_loss = (
                float(np.trace(result.difference))
                if result.difference is not None
                else float("nan")
            )
            rows.append(
                {
                    "amount": float(theta["amount"]),
                    "wait_hours": float(theta["wait_hours"]),
                    "rule": name,
                    "tr_Iz": tr_z,
                    "tr_Ia": tr_a,
                    "tr_loss": tr_loss,
                    "status": result.status.value,
                    "psd": bool(result.psd),
                }
            )
    return rows


def write_csv(rows: list[dict[str, float | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["amount", "wait_hours", "rule", "tr_Iz", "tr_Ia", "tr_loss", "status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def center_order_holds(point_rows: list[dict[str, float | str]]) -> bool:
    losses = {str(row["rule"]): float(row["tr_loss"]) for row in point_rows}
    sequence = [losses[name] for name in CENTER_ORDER]
    return all(sequence[index] <= sequence[index + 1] + 1e-12 for index in range(len(sequence) - 1))


def main() -> int:
    rows = evaluate_grid()
    write_csv(rows, CSV_PATH)
    # near-identity checks
    near = [row for row in rows if row["rule"] == "near_identity"]
    assert all(row["status"] == "ok" and row["psd"] for row in near)
    assert all(abs(float(row["tr_loss"])) < 1e-10 for row in near)

    # centre printout
    center_rows = [
        row
        for row in rows
        if float(row["amount"]) == CENTER["amount"]
        and float(row["wait_hours"]) == CENTER["wait_hours"]
    ]
    print(f"{'rule':16s}  tr(I_Z)    tr(I_A)    tr(loss)   status")
    for name in CENTER_ORDER:
        row = next(item for item in center_rows if item["rule"] == name)
        print(
            f"{name:16s}  {float(row['tr_Iz']):8.6f}  {float(row['tr_Ia']):8.6f}  "
            f"{float(row['tr_loss']):8.6f}  {row['status']}"
        )

    # fraction of grid points where centre ordering holds
    by_point: dict[tuple[float, float], list[dict[str, float | str]]] = {}
    for row in rows:
        key = (float(row["amount"]), float(row["wait_hours"]))
        by_point.setdefault(key, []).append(row)
    hold = sum(1 for point_rows in by_point.values() if center_order_holds(point_rows))
    total = len(by_point)
    ratio = hold / total if total else 0.0
    print(f"wrote {CSV_PATH}")
    print(f"center-order holds on {hold}/{total} grid points ({ratio:.0%})")
    anomalies = [
        point
        for point, point_rows in sorted(by_point.items())
        if not center_order_holds(point_rows)
    ]
    if anomalies:
        print("anomalies (amount, wait_hours):", anomalies)
        for amount, wait in anomalies:
            losses = {
                str(row["rule"]): float(row["tr_loss"])
                for row in by_point[(amount, wait)]
            }
            ordered = sorted(losses.items(), key=lambda item: item[1])
            print(
                f"  ({amount}, {wait}): "
                + " < ".join(f"{name}={loss:.6g}" for name, loss in ordered)
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
