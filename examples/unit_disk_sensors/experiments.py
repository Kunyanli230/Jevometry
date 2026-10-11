"""Centre-point Fisher information loss for the unit-disk sensor twin."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from system import CORE_RULE_NAMES, PARAMETERS, build_system  # noqa: E402
from jevometry.systems.pushforward import information_loss  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "papers" / "coord-fisher" / "data"
OUT_CSV = DATA / "unit_disk_center_loss.csv"
OUT_JSON = DATA / "unit_disk_validation_summary.json"
SYN_CENTER = DATA / "center_summary.json"


def evaluate_center() -> list[dict[str, object]]:
    bundle = build_system()
    theta = {
        "amplitude": float(bundle.center[0]),
        "length_scale": float(bundle.center[1]),
    }
    joint = bundle.joint.probabilities(theta)
    jacobian = bundle.joint.jacobian(theta)
    assert jacobian is not None
    rows: list[dict[str, object]] = []
    for name, aggregation in bundle.rule_maps:
        result = information_loss(
            joint,
            jacobian,
            aggregation,
            parameter_names=list(PARAMETERS),
        )
        tr_z = float(np.trace(result.fisher_joint)) if result.fisher_joint is not None else float("nan")
        tr_a = float(np.trace(result.fisher_aggregate)) if result.fisher_aggregate is not None else float("nan")
        tr_loss = float(np.trace(result.difference)) if result.difference is not None else float("nan")
        rows.append(
            {
                "dataset": "unit_disk_synthetic",
                "amplitude": theta["amplitude"],
                "length_scale": theta["length_scale"],
                "rule": name,
                "tr_I_Z": tr_z,
                "tr_I_A": tr_a,
                "tr_loss": tr_loss,
                "status": result.status.value,
                "psd": bool(result.psd) if result.psd is not None else False,
                "n_nodes": bundle.n_nodes,
                "radius": bundle.radius,
                "n_edges": len(bundle.edges),
                "mean_degree": bundle.mean_degree,
                "n_actions": len({aggregation.apply(o) for o in joint.outcomes}),
            }
        )
    return rows


def load_cleaning_center() -> dict[str, float]:
    if not SYN_CENTER.exists():
        return {}
    payload = json.loads(SYN_CENTER.read_text())
    if isinstance(payload, list):
        return {str(row["rule"]): float(row["tr_loss"]) for row in payload}
    return {}


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    by = {str(row["rule"]): row for row in rows}
    core = [name for name in CORE_RULE_NAMES if name in by]
    core_losses = [float(by[name]["tr_loss"]) for name in core]
    core_order_holds = all(
        core_losses[i] <= core_losses[i + 1] + 1e-9 for i in range(len(core_losses) - 1)
    )
    ascending = sorted(
        ({"rule": name, "tr_loss": float(by[name]["tr_loss"])} for name in by),
        key=lambda row: row["tr_loss"],
    )
    near = by["near_identity"]
    fisher_ok = (
        all(str(row["status"]) == "ok" and bool(row["psd"]) for row in rows)
        and abs(float(near["tr_loss"])) < 1e-8
    )
    cleaning = load_cleaning_center()
    vs_cleaning = []
    for name in by:
        ud = float(by[name]["tr_loss"])
        cl = cleaning.get(name)
        vs_cleaning.append(
            {
                "rule": name,
                "unit_disk_tr_loss": ud,
                "cleaning_tr_loss": cl,
                "abs_diff": None if cl is None else abs(ud - cl),
            }
        )
    return {
        "dataset": "synthetic_unit_disk_2d",
        "theta": {
            "amplitude": float(rows[0]["amplitude"]),
            "length_scale": float(rows[0]["length_scale"]),
        },
        "geometry": {
            "n_nodes": int(rows[0]["n_nodes"]),
            "radius": float(rows[0]["radius"]),
            "n_edges": int(rows[0]["n_edges"]),
            "mean_degree": float(rows[0]["mean_degree"]),
            "grid": "3x3",
        },
        "parameters": list(PARAMETERS),
        "core_rules": core,
        "core_order": "near_identity ≤ wider ≤ baseline ≤ stricter",
        "core_order_holds": core_order_holds,
        "fisher_approximation_ok": fisher_ok,
        "center_tr_loss_ascending": ascending,
        "unit_disk_center_losses": {name: float(by[name]["tr_loss"]) for name in by},
        "vs_cleaning_center": vs_cleaning,
        "artifacts": {
            "csv": str(OUT_CSV.relative_to(ROOT)),
            "json": str(OUT_JSON.relative_to(ROOT)),
        },
        "notes": [
            "θ = (amplitude, length_scale) of a Gaussian bump on [0,1]²; centre fixed.",
            "N=9 motes on a 3×3 grid; unit-disk R=0.45; role agents from energy/roughness.",
            "Same seven AggregationMap rules as cleaning (sensor semantic map in system.py).",
            "Phase-1 synthetic only; IBRL subsample is phase 2.",
        ],
    }


def main() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    rows = evaluate_center()
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
    for row in summary["center_tr_loss_ascending"]:
        print(f"  {row['rule']:20s} tr_loss={row['tr_loss']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
