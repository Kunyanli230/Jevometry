"""Cross-scenario AggregationMap rule-audit transfer (unit-disk ↔ IBRL ↔ IMAS²).

Loads per-bed (tr_delta_I, cond_entropy, accuracy) for the seven shared rules,
computes directed Spearman transfer metrics, Heath 通用/正向提升 checks, and
pre-registered S1/S2/S3. No recalibration.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "papers" / "coord-fisher" / "data"
RULES = [
    "near_identity",
    "wider",
    "baseline",
    "stricter",
    "quarantine_wider",
    "quarantine_review",
    "coarse_binary",
]
QUARANTINE = ("quarantine_wider", "quarantine_review")
EX_NEAR = [r for r in RULES if r != "near_identity"]
N_PERM = 5000
SEED = 0
# S2: Fisher-picked rule's H may be at most ε bits worse than baseline.
# ε = max(0.05, 0.1 · H_baseline) — documented fallback (no bootstrap of H).
EPS_FRAC = 0.1
EPS_FLOOR = 0.05


def load_unit_disk() -> dict[str, dict[str, float]]:
    rows = list(csv.DictReader((DATA / "unit_disk_rule_utility.csv").open(encoding="utf-8")))
    return {
        r["rule"]: {
            "tr_delta_I": float(r["tr_delta_I"]),
            "cond_entropy": float(r["cond_entropy"]),
            "accuracy": float(r["accuracy"]),
        }
        for r in rows
    }


def load_ibrl() -> dict[str, dict[str, float]]:
    rows = list(csv.DictReader((DATA / "ibrl_rule_utility.csv").open(encoding="utf-8")))
    return {
        r["rule"]: {
            "tr_delta_I": float(r["tr_delta_I"]),
            "cond_entropy": float(r["cond_entropy"]),
            "accuracy": float(r["accuracy"]),
        }
        for r in rows
    }


def load_imas2(
    policy_type: str = "fixed_orientation",
    dynamics: str = "stochastic",
    sensor_range: str = "small",
) -> dict[str, dict[str, float]]:
    rows = [
        r
        for r in csv.DictReader((DATA / "imas2_rule_comparison.csv").open(encoding="utf-8"))
        if r["policy_type"] == policy_type
        and r["dynamics"] == dynamics
        and r["sensor_range"] == sensor_range
    ]
    if len(rows) != 7:
        raise RuntimeError(f"expected 7 IMAS² rows, got {len(rows)}")
    return {
        r["rule"]: {
            "tr_delta_I": float(r["tr_delta_I"]),
            "cond_entropy": float(r["cond_entropy"]),
            "accuracy": float(r["accuracy"]),
        }
        for r in rows
    }


def vec(bed: dict[str, dict[str, float]], key: str) -> np.ndarray:
    return np.asarray([bed[r][key] for r in RULES], dtype=np.float64)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    rho, _ = spearmanr(a, b)
    return float(rho) if np.isfinite(rho) else float("nan")


def perm_pvalue(src: np.ndarray, tgt: np.ndarray, observed: float, rng: np.random.Generator) -> float:
    """Two-sided permutation p for Spearman(src, tgt); permute tgt labels."""
    if not np.isfinite(observed):
        return float("nan")
    count = 0
    for _ in range(N_PERM):
        shuffled = rng.permutation(tgt)
        rho = spearman(src, shuffled)
        if abs(rho) >= abs(observed) - 1e-15:
            count += 1
    return (count + 1) / (N_PERM + 1)


def pick_fisher(bed: dict[str, dict[str, float]], include_near: bool) -> str:
    pool = RULES if include_near else EX_NEAR
    return min(pool, key=lambda r: bed[r]["tr_delta_I"])


def pick_entropy(bed: dict[str, dict[str, float]], include_near: bool = False) -> str:
    pool = RULES if include_near else EX_NEAR
    return min(pool, key=lambda r: bed[r]["cond_entropy"])


def quarantine_worst(bed: dict[str, dict[str, float]], key: str) -> bool:
    """True if both quarantine rules are in the bottom half by key (higher worse for H; lower worse for acc)."""
    if key == "cond_entropy":
        ranked = sorted(EX_NEAR, key=lambda r: -bed[r][key])  # worst (highest H) first
    else:
        ranked = sorted(EX_NEAR, key=lambda r: bed[r][key])  # worst (lowest acc) first
    bottom = set(ranked[: max(2, len(EX_NEAR) // 2)])
    return set(QUARANTINE) <= bottom


def directed_transfer(
    src_name: str,
    tgt_name: str,
    src: dict[str, dict[str, float]],
    tgt: dict[str, dict[str, float]],
    rng: np.random.Generator,
) -> dict[str, object]:
    src_di = vec(src, "tr_delta_I")
    tgt_di = vec(tgt, "tr_delta_I")
    tgt_h = vec(tgt, "cond_entropy")
    tgt_acc = vec(tgt, "accuracy")

    rho_di = spearman(src_di, tgt_di)
    rho_h = spearman(src_di, tgt_h)
    rho_acc = spearman(src_di, tgt_acc)
    # For accuracy, ΔI higher ⇒ worse; so ρ(srcΔI, tgtAcc) expected negative if aligned.
    # Also report ρ(srcΔI, −tgtAcc) as "order agreement" with higher=worse convention.
    rho_neg_acc = spearman(src_di, -tgt_acc)

    p_di = perm_pvalue(src_di, tgt_di, rho_di, rng)
    p_h = perm_pvalue(src_di, tgt_h, rho_h, rng)

    fisher_ex = pick_fisher(src, include_near=False)
    fisher_all = pick_fisher(src, include_near=True)
    entropy_ex = pick_entropy(tgt, include_near=False)
    baseline_acc = tgt["baseline"]["accuracy"]
    baseline_h = tgt["baseline"]["cond_entropy"]
    pick_acc = tgt[fisher_ex]["accuracy"]
    pick_h = tgt[fisher_ex]["cond_entropy"]
    q_acc = {q: tgt[q]["accuracy"] for q in QUARANTINE}
    q_h = {q: tgt[q]["cond_entropy"] for q in QUARANTINE}

    eps = max(EPS_FLOOR, EPS_FRAC * baseline_h)
    # Heath 通用: all listed cross ρ ≥ 0.5 (we flag each)
    rho_flags = {
        "rho_srcDI_tgtDI": rho_di,
        "rho_srcDI_tgtH": rho_h,
        "rho_srcDI_tgtAcc": rho_acc,
        "rho_srcDI_neg_tgtAcc": rho_neg_acc,
    }
    # For 通用 we require ρ(ΔI,ΔI)≥0.5 and ρ(ΔI,H)≥0.5; acc uses neg convention ≥0.5
    tongyong_pass = (
        (rho_di >= 0.5 if np.isfinite(rho_di) else False)
        and (rho_h >= 0.5 if np.isfinite(rho_h) else False)
        and (rho_neg_acc >= 0.5 if np.isfinite(rho_neg_acc) else False)
    )
    # 正向提升: Fisher pick (ex-near) acc > baseline AND quarantine worst by acc and H
    zhengxiang_pass = (
        pick_acc > baseline_acc + 1e-12
        and quarantine_worst(tgt, "accuracy")
        and quarantine_worst(tgt, "cond_entropy")
    )

    # S1
    s1_pass = (
        np.isfinite(rho_di)
        and rho_di >= 0.5
        and p_di < 0.05
        and np.isfinite(rho_h)
        and rho_h >= 0.5
        and p_h < 0.05
    )
    # S2: no contradiction — if src says pick better than baseline (lower ΔI),
    # target H should not be worse than baseline by more than ε; and pick acc ≥ baseline − 0
    # Also: among rules with srcΔI < src_baseline, none should have tgt H much worse... 
    # Spec: "no contradiction (ΔI says better than baseline but H/acc worse); ΔH vs target baseline ≥ -ε"
    delta_h = baseline_h - pick_h  # positive ⇒ pick better (lower H)
    s2_no_contradiction = not (
        src[fisher_ex]["tr_delta_I"] < src["baseline"]["tr_delta_I"] - 1e-12
        and (pick_h > baseline_h + eps or pick_acc < baseline_acc - 1e-12)
    )
    s2_delta_h_ok = (baseline_h - pick_h) >= -eps  # pick H ≤ baseline + ε
    s2_pass = s2_no_contradiction and s2_delta_h_ok

    return {
        "source": src_name,
        "target": tgt_name,
        "rho_srcDI_tgtDI": rho_di,
        "rho_srcDI_tgtH": rho_h,
        "rho_srcDI_tgtAcc": rho_acc,
        "rho_srcDI_neg_tgtAcc": rho_neg_acc,
        "perm_p_DI": p_di,
        "perm_p_H": p_h,
        "fisher_pick_ex_near": fisher_ex,
        "fisher_pick_with_near": fisher_all,
        "entropy_pick_ex_near_on_target": entropy_ex,
        "target_metrics": {
            "fisher_pick": {"rule": fisher_ex, "accuracy": pick_acc, "cond_entropy": pick_h,
                            "tr_delta_I": tgt[fisher_ex]["tr_delta_I"]},
            "baseline": {"accuracy": baseline_acc, "cond_entropy": baseline_h,
                         "tr_delta_I": tgt["baseline"]["tr_delta_I"]},
            "quarantine": {"accuracy": q_acc, "cond_entropy": q_h},
            "entropy_pick": {
                "rule": entropy_ex,
                "accuracy": tgt[entropy_ex]["accuracy"],
                "cond_entropy": tgt[entropy_ex]["cond_entropy"],
            },
            "near_identity": {
                "accuracy": tgt["near_identity"]["accuracy"],
                "cond_entropy": tgt["near_identity"]["cond_entropy"],
            },
        },
        "eps_H": eps,
        "delta_H_baseline_minus_pick": delta_h,
        "heath_tongyong": {
            "pass": tongyong_pass,
            "criterion": "ρ(srcΔI,tgtΔI)≥0.5 AND ρ(srcΔI,tgtH)≥0.5 AND ρ(srcΔI,−tgtAcc)≥0.5",
            "flags": {k: (float(v) >= 0.5 if np.isfinite(v) else False) for k, v in rho_flags.items()},
            "values": rho_flags,
        },
        "heath_zhengxiang": {
            "pass": zhengxiang_pass,
            "criterion": "fisher_pick_ex_near acc > baseline AND quarantine bottom-half by acc and by H",
            "pick_acc_gt_baseline": pick_acc > baseline_acc + 1e-12,
            "quarantine_worst_acc": quarantine_worst(tgt, "accuracy"),
            "quarantine_worst_H": quarantine_worst(tgt, "cond_entropy"),
        },
        "S1": {"pass": s1_pass, "rho_DI": rho_di, "p_DI": p_di, "rho_H": rho_h, "p_H": p_h},
        "S2": {
            "pass": s2_pass,
            "no_contradiction": s2_no_contradiction,
            "delta_H_ok": s2_delta_h_ok,
            "eps_H": eps,
        },
    }


def sensitivity_imas2() -> list[dict[str, object]]:
    """Cheap: Spearman of ΔI order across IMAS² configs vs representative."""
    rep = load_imas2()
    rep_di = vec(rep, "tr_delta_I")
    out = []
    for pt in ("fixed_orientation", "random_orientation"):
        for dyn in ("deterministic", "stochastic"):
            for sr in ("small", "large"):
                bed = load_imas2(pt, dyn, sr)
                out.append(
                    {
                        "policy_type": pt,
                        "dynamics": dyn,
                        "sensor_range": sr,
                        "rho_vs_fixed_sto_small_DI": spearman(rep_di, vec(bed, "tr_delta_I")),
                        "rho_vs_fixed_sto_small_H": spearman(rep_di, vec(bed, "cond_entropy")),
                    }
                )
    return out


def main() -> int:
    # Ensure utilities exist
    sys.path.insert(0, str(ROOT / "examples" / "unit_disk_sensors"))
    from rule_utility import main as util_main  # noqa: WPS433

    util_main()

    beds = {
        "unit_disk": load_unit_disk(),
        "ibrl": load_ibrl(),
        "imas2": load_imas2(),  # fixed_orientation | stochastic | small
    }
    rng = np.random.default_rng(SEED)
    pairs = [
        ("unit_disk", "imas2"),
        ("imas2", "unit_disk"),
        ("imas2", "ibrl"),
        ("ibrl", "imas2"),
        ("unit_disk", "ibrl"),
        ("ibrl", "unit_disk"),
    ]
    transfers = [directed_transfer(a, b, beds[a], beds[b], rng) for a, b in pairs]

    # Pairwise ρ CSV
    rho_rows = []
    for t in transfers:
        rho_rows.append(
            {
                "source": t["source"],
                "target": t["target"],
                "rho_srcDI_tgtDI": t["rho_srcDI_tgtDI"],
                "rho_srcDI_tgtH": t["rho_srcDI_tgtH"],
                "rho_srcDI_tgtAcc": t["rho_srcDI_tgtAcc"],
                "rho_srcDI_neg_tgtAcc": t["rho_srcDI_neg_tgtAcc"],
                "perm_p_DI": t["perm_p_DI"],
                "perm_p_H": t["perm_p_H"],
                "heath_tongyong": t["heath_tongyong"]["pass"],
                "heath_zhengxiang": t["heath_zhengxiang"]["pass"],
                "S1": t["S1"]["pass"],
                "S2": t["S2"]["pass"],
                "fisher_pick_ex_near": t["fisher_pick_ex_near"],
                "pick_acc": t["target_metrics"]["fisher_pick"]["accuracy"],
                "baseline_acc": t["target_metrics"]["baseline"]["accuracy"],
                "entropy_pick": t["target_metrics"]["entropy_pick"]["rule"],
                "entropy_pick_acc": t["target_metrics"]["entropy_pick"]["accuracy"],
            }
        )
    rho_path = DATA / "cross_bed_transfer_pairwise.csv"
    with rho_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rho_rows[0].keys()))
        w.writeheader()
        w.writerows(rho_rows)

    summary = {
        "beds": {
            "unit_disk": "papers/coord-fisher/data/unit_disk_rule_utility.csv",
            "ibrl": "papers/coord-fisher/data/ibrl_rule_utility.csv",
            "imas2": {
                "file": "papers/coord-fisher/data/imas2_rule_comparison.csv",
                "slice": "fixed_orientation|stochastic|small",
            },
        },
        "utility_notes": {
            "unit_disk_ibrl": (
                "y_share bit under declared joint at θ_centre; exact Bayes H/acc. "
                "Ceiling H=0/acc=1 possible when A determines y."
            ),
            "imas2": "adversary-type τ via HMM filter on fused categories (existing evaluator).",
        },
        "protocol": {
            "recalibration": False,
            "n_perm": N_PERM,
            "seed": SEED,
            "eps_H": f"max({EPS_FLOOR}, {EPS_FRAC}·H_baseline)",
            "tongyong_rhos": ["rho_srcDI_tgtDI", "rho_srcDI_tgtH", "rho_srcDI_neg_tgtAcc"],
        },
        "bed_tables": {name: beds[name] for name in beds},
        "transfers": transfers,
        "pairwise_csv": str(rho_path.relative_to(ROOT)),
        "imas2_config_sensitivity": sensitivity_imas2(),
        "S3_negative_control": {
            "note": (
                "Placement-transfer Fisher↔MI already failed on stochastic IMAS² "
                "(spearman≈-0.15, top5_overlap=0; see imas2_summary.json). Not rerun."
            ),
            "pass": False,
            "source": "papers/coord-fisher/data/imas2_summary.json placements.stochastic_small",
        },
        "overall": {
            "any_tongyong": any(t["heath_tongyong"]["pass"] for t in transfers),
            "any_zhengxiang": any(t["heath_zhengxiang"]["pass"] for t in transfers),
            "any_S1": any(t["S1"]["pass"] for t in transfers),
            "any_S2": any(t["S2"]["pass"] for t in transfers),
            "n_tongyong": sum(t["heath_tongyong"]["pass"] for t in transfers),
            "n_zhengxiang": sum(t["heath_zhengxiang"]["pass"] for t in transfers),
            "n_S1": sum(t["S1"]["pass"] for t in transfers),
            "n_S2": sum(t["S2"]["pass"] for t in transfers),
            "n_pairs": len(transfers),
        },
    }
    out = DATA / "cross_bed_transfer_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {rho_path}")
    print(f"wrote {out}")
    for t in transfers:
        print(
            f"{t['source']:10s}→{t['target']:10s}  "
            f"ρDI={t['rho_srcDI_tgtDI']:+.3f} ρH={t['rho_srcDI_tgtH']:+.3f} "
            f"ρ(−acc)={t['rho_srcDI_neg_tgtAcc']:+.3f}  "
            f"通={t['heath_tongyong']['pass']} 正={t['heath_zhengxiang']['pass']} "
            f"S1={t['S1']['pass']} S2={t['S2']['pass']}  "
            f"pick={t['fisher_pick_ex_near']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
