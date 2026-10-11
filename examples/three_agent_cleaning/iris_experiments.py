"""Run coordination-rule Fisher loss on the Iris twin and compare to synthetic centre."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from iris_system import CORE_RULE_NAMES, build_iris_bundle
from jevometry.systems.pushforward import information_loss

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "papers" / "coord-fisher" / "data"
SYN_CENTER = DATA / "center_summary.json"
OUT_CSV = DATA / "iris_center_loss.csv"
OUT_JSON = DATA / "iris_validation_summary.json"


def evaluate_iris_center() -> list[dict[str, object]]:
    bundle = build_iris_bundle()
    theta = {"evidence": float(bundle.center[0]), "ambiguity": float(bundle.center[1])}
    joint = bundle.joint.probabilities(theta)
    jacobian = bundle.joint.jacobian(theta)
    assert jacobian is not None
    rows: list[dict[str, object]] = []
    for name, aggregation in bundle.rule_maps:
        result = information_loss(
            joint,
            jacobian,
            aggregation,
            parameter_names=["evidence", "ambiguity"],
        )
        tr_z = float(np.trace(result.fisher_joint)) if result.fisher_joint is not None else float("nan")
        tr_a = float(np.trace(result.fisher_aggregate)) if result.fisher_aggregate is not None else float("nan")
        tr_loss = float(np.trace(result.difference)) if result.difference is not None else float("nan")
        rows.append(
            {
                "dataset": "iris",
                "evidence": theta["evidence"],
                "ambiguity": theta["ambiguity"],
                "rule": name,
                "tr_I_Z": tr_z,
                "tr_I_A": tr_a,
                "tr_loss": tr_loss,
                "status": result.status.value,
                "psd": bool(result.psd) if result.psd is not None else False,
                "n_samples": bundle.n_samples,
                "feature_x": bundle.feature_names[0],
                "feature_y": bundle.feature_names[1],
            }
        )
    return rows


def load_synthetic_center() -> dict[str, float]:
    if not SYN_CENTER.exists():
        return {}
    rows = json.loads(SYN_CENTER.read_text())
    return {str(row["rule"]): float(row["tr_loss"]) for row in rows}


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    by = {str(row["rule"]): row for row in rows}
    core = [name for name in CORE_RULE_NAMES if name in by]
    core_losses = [float(by[name]["tr_loss"]) for name in core]
    core_order_holds = all(
        core_losses[index] <= core_losses[index + 1] + 1e-9 for index in range(len(core_losses) - 1)
    )
    synthetic = load_synthetic_center()
    comparison = []
    for name in core:
        iris_loss = float(by[name]["tr_loss"])
        syn_loss = synthetic.get(name)
        comparison.append(
            {
                "rule": name,
                "iris_tr_loss": iris_loss,
                "synthetic_tr_loss": syn_loss,
                "abs_diff": None if syn_loss is None else abs(iris_loss - syn_loss),
            }
        )
    # Fisher standing: all core rows ok+psd and near≈0
    near = by["near_identity"]
    fisher_ok = (
        all(row["status"] == "ok" and row["psd"] for row in rows if row["rule"] in core)
        and abs(float(near["tr_loss"])) < 1e-8
    )
    return {
        "dataset": "sklearn.datasets.load_iris",
        "center": {
            "evidence": float(rows[0]["evidence"]),
            "ambiguity": float(rows[0]["ambiguity"]),
        },
        "features": [rows[0]["feature_x"], rows[0]["feature_y"]],
        "n_samples": int(rows[0]["n_samples"]),
        "core_rules": core,
        "core_order": "near_identity ≤ wider ≤ baseline ≤ stricter",
        "core_order_holds": core_order_holds,
        "fisher_approximation_ok": fisher_ok,
        "iris_center_losses": {name: float(by[name]["tr_loss"]) for name in by},
        "vs_synthetic_center": comparison,
        "artifacts": {
            "csv": str(OUT_CSV.relative_to(ROOT)),
            "json": str(OUT_JSON.relative_to(ROOT)),
        },
        "notes": [
            "θ = MinMax-scaled petal length/width; centre = feature means.",
            "Agents: logistic applicability (setosa→false), multinomial repair←species,",
            "multinomial risk←species-confidence tertiles; same AggregationMaps as synthetic twin.",
        ],
    }


def main() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    rows = evaluate_iris_center()
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    OUT_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT_CSV}")
    print(f"wrote {OUT_JSON}")
    print(
        f"core_order_holds={summary['core_order_holds']} "
        f"fisher_ok={summary['fisher_approximation_ok']}"
    )
    for name in CORE_RULE_NAMES:
        print(f"  {name:16s} tr_loss={summary['iris_center_losses'][name]:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
