"""Exact fused-report utility for unit-disk / IBRL role-agent twins.

Ground-truth bit y_share = 1 iff the joint agent outcome is "safe to exchange":
  applicable == "true" AND repair != "quarantine" AND risk != "2".

Under the declared joint P(Z|θ_centre), AggregationMap A=f(Z) induces
  H = Σ_a P(a) h₂(P(y=1|a))          (bits)
  accuracy = Σ_a P(a) max(P(y=1|a), 1−P(y=1|a))   (Bayes-optimal from A)

Exact (enumerates the joint). Parallel in role to IMAS² H(τ|fused)/accuracy, but
the latent is the coordination-intent bit (Z-derived), not adversary type — beds
differ, so cross-bed ρ(srcΔI, tgtH) tests ranking transfer of utilities, not
identical semantics. Several rules may hit H=0 / acc=1 (ceiling: A determines y).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from system import PARAMETERS as UD_PARAMS  # noqa: E402
from system import build_system as build_unit_disk  # noqa: E402
from ibrl_system import PARAMETERS as IBRL_PARAMS  # noqa: E402
from ibrl_system import build_ibrl_bundle  # noqa: E402
from jevometry.schemas.joint import ConstructionMode, JointDistribution  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "papers" / "coord-fisher" / "data"
UTILITY_DEF = (
    "y_share=1 iff applicable==true & repair!=quarantine & risk!=2; "
    "H=E[h2(P(y|A))], acc=Bayes accuracy of A→y under joint at θ_centre"
)


def y_share(outcome: tuple[str, ...]) -> int:
    applicable, repair, risk = outcome
    return int(applicable == "true" and repair != "quarantine" and risk != "2")


def binary_entropy(p: float) -> float:
    p = float(np.clip(p, 0.0, 1.0))
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p)))


def _as_joint(raw, node_order) -> JointDistribution:
    if isinstance(raw, JointDistribution):
        return raw
    outcomes, probs = raw
    return JointDistribution(
        mode=ConstructionMode.DECLARED_PRODUCT,
        node_order=list(node_order),
        outcomes=list(outcomes),
        probabilities=[float(p) for p in probs],
    )


def evaluate_rule_utilities(joint: JointDistribution, rule_maps) -> list[dict[str, object]]:
    y_vec = np.asarray([y_share(o) for o in joint.outcomes], dtype=np.float64)
    p_z = np.asarray(joint.probabilities, dtype=np.float64)
    py_marg = float(np.dot(p_z, y_vec))
    rows: list[dict[str, object]] = []
    for name, aggregation in rule_maps:
        mass: dict[str, float] = {}
        y_mass: dict[str, float] = {}
        for outcome, prob, y in zip(joint.outcomes, p_z, y_vec, strict=True):
            a = aggregation.apply(outcome)
            mass[a] = mass.get(a, 0.0) + float(prob)
            y_mass[a] = y_mass.get(a, 0.0) + float(prob) * float(y)
        h = 0.0
        acc = 0.0
        for a, pa in mass.items():
            if pa <= 0.0:
                continue
            py = y_mass[a] / pa
            h += pa * binary_entropy(py)
            acc += pa * max(py, 1.0 - py)
        rows.append(
            {
                "rule": name,
                "cond_entropy": h,
                "accuracy": acc,
                "H_y": binary_entropy(py_marg),
                "I_y_A": binary_entropy(py_marg) - h,
                "n_actions": len(mass),
                "p_y_share": py_marg,
            }
        )
    return rows


def _load_tr_loss(csv_path: Path) -> dict[str, float]:
    return {r["rule"]: float(r["tr_loss"]) for r in csv.DictReader(csv_path.open(encoding="utf-8"))}


def build_utility_table(bed: str) -> tuple[list[dict[str, object]], dict[str, object]]:
    if bed == "unit_disk":
        bundle = build_unit_disk()
        theta = {"amplitude": float(bundle.center[0]), "length_scale": float(bundle.center[1])}
        params, tr_path = UD_PARAMS, DATA / "unit_disk_center_loss.csv"
        extra: dict[str, object] = {}
    elif bed == "ibrl":
        bundle = build_ibrl_bundle()
        theta = {"temperature": float(bundle.center[0]), "humidity": float(bundle.center[1])}
        params, tr_path = IBRL_PARAMS, DATA / "ibrl_center_loss.csv"
        extra = {"mote_ids": list(bundle.mote_ids)}
    else:
        raise ValueError(bed)

    joint = _as_joint(bundle.joint.probabilities(theta), bundle.nodes)
    tr = _load_tr_loss(tr_path)
    utils = evaluate_rule_utilities(joint, bundle.rule_maps)
    rows = []
    for u in utils:
        name = str(u["rule"])
        rows.append(
            {
                "bed": bed,
                "rule": name,
                "tr_delta_I": tr[name],
                "cond_entropy": u["cond_entropy"],
                "accuracy": u["accuracy"],
                "H_y": u["H_y"],
                "I_y_A": u["I_y_A"],
                "n_actions": u["n_actions"],
                "p_y_share": u["p_y_share"],
                "utility_definition": UTILITY_DEF,
                "evaluator": "exact_bayes_y_share_given_A",
            }
        )
    by = {r["rule"]: r for r in rows}
    assert float(by["near_identity"]["cond_entropy"]) <= min(float(r["cond_entropy"]) for r in rows) + 1e-9
    meta = {
        "bed": bed,
        "theta": theta,
        "parameters": list(params),
        "utility_definition": UTILITY_DEF,
        "n_joint_outcomes": len(joint.outcomes),
        **extra,
        "rows": {
            r["rule"]: {
                "tr_delta_I": r["tr_delta_I"],
                "cond_entropy": r["cond_entropy"],
                "accuracy": r["accuracy"],
            }
            for r in rows
        },
    }
    return rows, meta


def main() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    all_meta: dict[str, object] = {}
    for bed, out_name in (("unit_disk", "unit_disk_rule_utility.csv"), ("ibrl", "ibrl_rule_utility.csv")):
        rows, meta = build_utility_table(bed)
        path = DATA / out_name
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        all_meta[bed] = meta
        print(f"wrote {path}")
        for r in sorted(rows, key=lambda x: float(x["tr_delta_I"])):
            print(
                f"  {r['rule']:20s} dI={float(r['tr_delta_I']):.4f} "
                f"H={float(r['cond_entropy']):.4f} acc={float(r['accuracy']):.4f}"
            )
    (DATA / "role_agent_utility_meta.json").write_text(json.dumps(all_meta, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
