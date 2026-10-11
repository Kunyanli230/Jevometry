"""Export pushforward action masses for key theta points and four rules."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ticket_system import (
    LOGIT_SCALE,
    PROB_FLOOR,
    build_declared_joint,
    build_nodes,
    build_rule_maps,
)
from jevometry.systems.pushforward import pushforward

POINTS = [
    {"amount": 250.0, "wait_hours": 8.0},
    {"amount": 600.0, "wait_hours": 2.0},
    {"amount": 600.0, "wait_hours": 8.0},
]
CSV_PATH = (
    Path(__file__).resolve().parents[2]
    / "papers"
    / "typesafe-ticket"
    / "data"
    / "surrogate_action_masses.csv"
)


def masses_for(joint, aggregation) -> dict[str, float]:
    pushed = pushforward(joint, aggregation)
    return {
        outcome[0]: float(mass)
        for outcome, mass in zip(pushed.outcomes, pushed.probabilities, strict=True)
    }


def evaluate() -> list[dict[str, float | str]]:
    nodes = build_nodes()
    joint_model = build_declared_joint(nodes)
    rule_maps = build_rule_maps(nodes)
    rows: list[dict[str, float | str]] = []
    for theta in POINTS:
        joint = joint_model.probabilities(theta)
        total_joint = float(sum(joint.probabilities))
        for name, aggregation in rule_maps:
            masses = masses_for(joint, aggregation)
            total = float(sum(masses.values()))
            for action, mass in sorted(masses.items()):
                rows.append(
                    {
                        "amount": float(theta["amount"]),
                        "wait_hours": float(theta["wait_hours"]),
                        "rule": name,
                        "action": action,
                        "mass": mass,
                        "mass_sum": total,
                        "joint_sum": total_joint,
                        "LOGIT_SCALE": float(LOGIT_SCALE),
                        "PROB_FLOOR": float(PROB_FLOOR),
                    }
                )
    return rows


def write_csv(rows: list[dict[str, float | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "amount",
        "wait_hours",
        "rule",
        "action",
        "mass",
        "mass_sum",
        "joint_sum",
        "LOGIT_SCALE",
        "PROB_FLOOR",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def describe_flips(rows: list[dict[str, float | str]]) -> None:
    def table(amount: float, wait: float, rule: str) -> dict[str, float]:
        return {
            str(row["action"]): float(row["mass"])
            for row in rows
            if float(row["amount"]) == amount
            and float(row["wait_hours"]) == wait
            and row["rule"] == rule
        }

    for amount, wait in ((600.0, 2.0), (600.0, 8.0)):
        base = table(amount, wait, "baseline")
        strict = table(amount, wait, "stricter")
        keys = sorted(set(base) | set(strict))
        deltas = {key: strict.get(key, 0.0) - base.get(key, 0.0) for key in keys}
        movers = sorted(deltas.items(), key=lambda item: abs(item[1]), reverse=True)
        print(f"flip ({amount:g},{wait:g}) stricter-baseline mass deltas:")
        for action, delta in movers:
            if abs(delta) > 1e-12:
                print(f"  {action:16s} {delta:+.6f}")


def main() -> int:
    rows = evaluate()
    write_csv(rows, CSV_PATH)
    # near_identity mass conservation
    for row in rows:
        if row["rule"] == "near_identity":
            assert abs(float(row["mass_sum"]) - float(row["joint_sum"])) < 1e-12
    print(f"wrote {CSV_PATH} ({len(rows)} rows)")
    describe_flips(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
