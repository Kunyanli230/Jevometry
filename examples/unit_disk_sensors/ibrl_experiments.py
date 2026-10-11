"""Run coordination-rule Fisher loss on the IBRL twin; compare to unit-disk centre."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ibrl_system import CORE_RULE_NAMES, PARAMETERS, build_ibrl_bundle  # noqa: E402
from jevometry.systems.pushforward import information_loss  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "papers" / "coord-fisher" / "data"
OUT_CSV = DATA / "ibrl_center_loss.csv"
OUT_JSON = DATA / "ibrl_validation_summary.json"
UD_JSON = DATA / "unit_disk_validation_summary.json"


def evaluate_ibrl_center() -> list[dict[str, object]]:
    bundle = build_ibrl_bundle()
    theta = {
        "temperature": float(bundle.center[0]),
        "humidity": float(bundle.center[1]),
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
                "dataset": "ibrl",
                "temperature": theta["temperature"],
                "humidity": theta["humidity"],
                "rule": name,
                "tr_I_Z": tr_z,
                "tr_I_A": tr_a,
                "tr_loss": tr_loss,
                "status": result.status.value,
                "psd": bool(result.psd) if result.psd is not None else False,
                "n_samples": bundle.n_samples,
                "n_motes": bundle.n_motes,
                "mote_ids": ",".join(str(m) for m in bundle.mote_ids),
                "n_edges": len(bundle.edges),
                "radius_norm": bundle.radius_norm,
            }
        )
    return rows


def load_unit_disk_losses() -> dict[str, float]:
    if not UD_JSON.exists():
        return {}
    summary = json.loads(UD_JSON.read_text())
    return {str(k): float(v) for k, v in summary.get("unit_disk_center_losses", {}).items()}


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
    unit_disk = load_unit_disk_losses()
    comparison = []
    for name in by:
        ibrl_loss = float(by[name]["tr_loss"])
        ud = unit_disk.get(name)
        comparison.append(
            {
                "rule": name,
                "ibrl_tr_loss": ibrl_loss,
                "unit_disk_tr_loss": ud,
                "abs_diff": None if ud is None else abs(ibrl_loss - ud),
            }
        )
    # order rank correlation note
    if unit_disk:
        ibrl_rank = {row["rule"]: i for i, row in enumerate(ascending)}
        ud_sorted = sorted(unit_disk, key=unit_disk.get)
        ud_rank = {name: i for i, name in enumerate(ud_sorted)}
        common = [n for n in ibrl_rank if n in ud_rank]
        order_match = all(ibrl_rank[n] == ud_rank[n] for n in common)
    else:
        order_match = None
    return {
        "dataset": "intel_berkeley_lab_subsample",
        "source": "https://db.csail.mit.edu/labdata/labdata.html",
        "subsample": "examples/unit_disk_sensors/data/ibrl_subsample.tsv",
        "center": {
            "temperature": float(rows[0]["temperature"]),
            "humidity": float(rows[0]["humidity"]),
        },
        "geometry": {
            "n_motes": int(rows[0]["n_motes"]),
            "mote_ids": [int(x) for x in str(rows[0]["mote_ids"]).split(",") if x],
            "n_edges": int(rows[0]["n_edges"]),
            "radius_norm": float(rows[0]["radius_norm"]),
            "n_samples": int(rows[0]["n_samples"]),
        },
        "parameters": list(PARAMETERS),
        "core_rules": core,
        "core_order": "near_identity ≤ wider ≤ baseline ≤ stricter",
        "core_order_holds": core_order_holds,
        "fisher_approximation_ok": fisher_ok,
        "center_tr_loss_ascending": ascending,
        "ibrl_center_losses": {name: float(by[name]["tr_loss"]) for name in by},
        "vs_unit_disk_center": comparison,
        "unit_disk_order_exact_match": order_match,
        "artifacts": {
            "csv": str(OUT_CSV.relative_to(ROOT)),
            "json": str(OUT_JSON.relative_to(ROOT)),
        },
        "notes": [
            "θ = MinMax-scaled temperature/humidity; centre = feature means.",
            "12 farthest-point motes from IBRL; labels from voltage + unit-disk neighbor disagreement.",
            "Same seven AggregationMaps as unit_disk_sensors (quarantine = isolate safety valve).",
        ],
    }


def main() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    rows = evaluate_ibrl_center()
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
        f"fisher_ok={summary['fisher_approximation_ok']} "
        f"ud_order_match={summary['unit_disk_order_exact_match']}"
    )
    for row in summary["center_tr_loss_ascending"]:
        print(f"  {row['rule']:20s} tr_loss={row['tr_loss']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
