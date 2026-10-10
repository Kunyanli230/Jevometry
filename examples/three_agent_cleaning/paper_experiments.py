"""Publication experiments: Fisher information loss across coordination rules.

Fixes the three agent distributions and DECLARED_PRODUCT joint; sweeps a small
(evidence, ambiguity) grid and compares pure AggregationMaps. Writes CSV/JSON
tables and publication figures under papers/coord-fisher/.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from system import CENTER, PARAMETERS, build_rule_maps, build_system  # noqa: E402

from jevometry.systems.pushforward import information_loss, pushforward  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = REPO_ROOT / "papers" / "coord-fisher"
DATA_DIR = PAPER_DIR / "data"
FIG_DIR = PAPER_DIR / "figures"

# Modest grid in [0, 1]^2; denser than a single centre, still cheap offline.
EVIDENCE_GRID = np.linspace(0.2, 0.9, 8)
AMBIGUITY_GRID = np.linspace(0.1, 0.8, 8)
# Focus curves / heatmaps on a readable subset.
HEATMAP_RULES = ("baseline", "stricter", "wider", "quarantine_review", "coarse_binary")
STACK_RULES = ("baseline", "stricter", "wider", "quarantine_review", "coarse_binary")


def _theta_dict(evidence: float, ambiguity: float) -> dict[str, float]:
    return {"evidence": float(evidence), "ambiguity": float(ambiguity)}


def _action_masses(joint, aggregation) -> dict[str, float]:
    pushed = pushforward(joint, aggregation)
    return {
        label[0]: float(mass)
        for label, mass in zip(pushed.outcomes, pushed.probabilities, strict=True)
    }


def sweep() -> list[dict[str, object]]:
    nodes, joint_model, _ = build_system()
    rules = build_rule_maps(nodes)
    rows: list[dict[str, object]] = []
    for evidence in EVIDENCE_GRID:
        for ambiguity in AMBIGUITY_GRID:
            theta = _theta_dict(evidence, ambiguity)
            joint = joint_model.probabilities(theta)
            jacobian = joint_model.jacobian(theta)
            assert jacobian is not None
            for name, aggregation in rules:
                result = information_loss(
                    joint, jacobian, aggregation, parameter_names=PARAMETERS
                )
                masses = _action_masses(joint, aggregation)
                if (
                    result.fisher_joint is None
                    or result.fisher_aggregate is None
                    or result.difference is None
                ):
                    iz = ia = loss = float("nan")
                else:
                    iz = float(np.trace(result.fisher_joint))
                    ia = float(np.trace(result.fisher_aggregate))
                    loss = float(np.trace(result.difference))
                rows.append(
                    {
                        "evidence": float(evidence),
                        "ambiguity": float(ambiguity),
                        "rule": name,
                        "tr_I_Z": iz,
                        "tr_I_A": ia,
                        "tr_loss": loss,
                        "status": result.status.value,
                        "reason_code": result.reason_code,
                        "n_actions": len(masses),
                        "action_masses": masses,
                    }
                )
    return rows


def ensure_center_on_grid(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Append exact centre-point evaluations (paper Table 1)."""
    nodes, joint_model, _ = build_system()
    theta = _theta_dict(float(CENTER[0]), float(CENTER[1]))
    joint = joint_model.probabilities(theta)
    jacobian = joint_model.jacobian(theta)
    assert jacobian is not None
    extra: list[dict[str, object]] = []
    for name, aggregation in build_rule_maps(nodes):
        result = information_loss(joint, jacobian, aggregation, parameter_names=PARAMETERS)
        masses = _action_masses(joint, aggregation)
        assert result.fisher_joint is not None
        assert result.fisher_aggregate is not None
        assert result.difference is not None
        extra.append(
            {
                "evidence": float(CENTER[0]),
                "ambiguity": float(CENTER[1]),
                "rule": name,
                "tr_I_Z": float(np.trace(result.fisher_joint)),
                "tr_I_A": float(np.trace(result.fisher_aggregate)),
                "tr_loss": float(np.trace(result.difference)),
                "status": result.status.value,
                "reason_code": result.reason_code,
                "n_actions": len(masses),
                "action_masses": masses,
                "is_center": True,
            }
        )
    return rows + extra


def fig_loss_at_center(centre: list[dict[str, object]]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    order = [r[0] for r in __import__("system", fromlist=["RULE_SPECS"]).RULE_SPECS]
    by_rule = {str(row["rule"]): row for row in centre}
    names = [name for name in order if name in by_rule]
    losses = [float(by_rule[name]["tr_loss"]) for name in names]
    retained = [float(by_rule[name]["tr_I_A"]) for name in names]

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(len(names))
    bars = ax.bar(x, losses, color="#4C72B0", edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel(r"$\mathrm{tr}(I_Z - I_A)$")
    ax.set_xlabel("Coordination rule")
    ax.set_title(r"Fisher information loss at centre ($e{=}0.7$, $a{=}0.3$)")
    for bar, loss, _keep in zip(bars, losses, retained, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{loss:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_ylim(0, max(losses) * 1.18 + 0.1)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_loss_center.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_loss_center.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_loss_center")


def fig_loss_heatmaps(rows: list[dict[str, object]]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    grid_rows = [row for row in rows if not row.get("is_center")]
    e_vals = sorted({float(r["evidence"]) for r in grid_rows})
    a_vals = sorted({float(r["ambiguity"]) for r in grid_rows})
    rules = [r for r in HEATMAP_RULES]
    n = len(rules)
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 3.2), sharey=True, constrained_layout=True)
    if n == 1:
        axes = [axes]
    for ax, rule in zip(axes, rules, strict=True):
        mat = np.full((len(a_vals), len(e_vals)), np.nan)
        for row in grid_rows:
            if row["rule"] != rule:
                continue
            i = a_vals.index(float(row["ambiguity"]))
            j = e_vals.index(float(row["evidence"]))
            mat[i, j] = float(row["tr_loss"])  # type: ignore[index]
        im = ax.imshow(
            mat,
            origin="lower",
            aspect="auto",
            extent=[e_vals[0], e_vals[-1], a_vals[0], a_vals[-1]],
            cmap="magma",
            vmin=0.0,
            vmax=np.nanmax([float(r["tr_loss"]) for r in grid_rows if r["rule"] in rules]),
        )
        ax.set_title(rule, fontsize=9)
        ax.set_xlabel("evidence")
        if ax is axes[0]:
            ax.set_ylabel("ambiguity")
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04, label=r"$\mathrm{tr}(I_Z-I_A)$")
    fig.suptitle("Information loss over the parameter grid", fontsize=11)
    fig.savefig(FIG_DIR / "fig_loss_heatmap.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_loss_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_loss_heatmap")


def fig_loss_curves(rows: list[dict[str, object]]) -> None:
    """Loss vs evidence at fixed ambiguity=0.3 (interpolated from grid + centre)."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    target_a = 0.3
    # Prefer grid rows nearest to ambiguity=0.3
    a_vals = sorted({float(r["ambiguity"]) for r in rows if not r.get("is_center")})
    nearest_a = min(a_vals, key=lambda a: abs(a - target_a))
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for rule in ("near_identity", "wider", "baseline", "stricter", "coarse_binary"):
        pts = [
            (float(r["evidence"]), float(r["tr_loss"]))
            for r in rows
            if r["rule"] == rule
            and not r.get("is_center")
            and abs(float(r["ambiguity"]) - nearest_a) < 1e-12
        ]
        pts.sort()
        if not pts:
            continue
        xs, ys = zip(*pts, strict=True)
        ax.plot(xs, ys, marker="o", markersize=3, label=rule)
    ax.set_xlabel("evidence")
    ax.set_ylabel(r"$\mathrm{tr}(I_Z - I_A)$")
    ax.set_title(f"Loss vs evidence (ambiguity≈{nearest_a:.2f})")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_loss_curves.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_loss_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_loss_curves")


def fig_action_masses(centre: list[dict[str, object]]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    # Collect union of action labels (exclude near_identity — too many labels).
    rules = [r for r in STACK_RULES]
    label_set: list[str] = []
    for row in centre:
        if row["rule"] not in rules:
            continue
        for label in row["action_masses"]:  # type: ignore[union-attr]
            if label not in label_set:
                label_set.append(str(label))
    # Stable display order
    preferred = [
        "reject",
        "proceed",
        "auto_fix",
        "auto_normalize",
        "auto_impute",
        "human_review",
    ]
    labels = [lab for lab in preferred if lab in label_set] + [
        lab for lab in label_set if lab not in preferred
    ]
    colors = {
        "reject": "#C44E52",
        "proceed": "#55A868",
        "auto_fix": "#4C72B0",
        "auto_normalize": "#4C72B0",
        "auto_impute": "#8172B2",
        "human_review": "#CCB974",
    }
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = np.arange(len(rules))
    bottoms = np.zeros(len(rules))
    by_rule = {str(row["rule"]): row for row in centre}
    for label in labels:
        heights = []
        for rule in rules:
            masses = by_rule[rule]["action_masses"]  # type: ignore[index]
            heights.append(float(masses.get(label, 0.0)))  # type: ignore[union-attr]
        ax.bar(
            x,
            heights,
            bottom=bottoms,
            label=label,
            color=colors.get(label, "#999999"),
            edgecolor="white",
            linewidth=0.3,
        )
        bottoms = bottoms + np.asarray(heights)
    ax.set_xticks(x)
    ax.set_xticklabels(rules, rotation=20, ha="right")
    ax.set_ylabel("Action probability")
    ax.set_ylim(0, 1.05)
    ax.set_title("Action masses at centre by coordination rule")
    ax.legend(fontsize=8, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_action_masses.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_action_masses.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_action_masses")


def print_center_table(centre: list[dict[str, object]]) -> None:
    header = f"{'rule':<20} {'tr(I_Z)':>10} {'tr(I_A)':>10} {'tr(loss)':>10} {'|A|':>5} {'status':>10}"
    print(header)
    print("-" * len(header))
    for row in centre:
        print(
            f"{str(row['rule']):<20} {float(row['tr_I_Z']):10.4f} "
            f"{float(row['tr_I_A']):10.4f} {float(row['tr_loss']):10.4f} "
            f"{int(row['n_actions']):5d} {str(row['status']):>10}"
        )
    by = {str(r["rule"]): float(r["tr_loss"]) for r in centre}
    assert by["near_identity"] <= by["wider"] + 1e-9
    assert by["wider"] <= by["baseline"] + 1e-9
    assert by["baseline"] <= by["stricter"] + 1e-9
    print(
        "\nOrdering check (center): "
        f"near_identity ({by['near_identity']:.4f}) ≤ wider ({by['wider']:.4f}) ≤ "
        f"baseline ({by['baseline']:.4f}) ≤ stricter ({by['stricter']:.4f})"
    )


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    rows = sweep()
    rows = ensure_center_on_grid(rows)

    # Persist full grid (without duplicating centre flag noise in CSV — keep all).
    csv_path = DATA_DIR / "rule_loss_grid.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "evidence",
                "ambiguity",
                "rule",
                "tr_I_Z",
                "tr_I_A",
                "tr_loss",
                "status",
                "reason_code",
                "n_actions",
                "action_masses_json",
                "is_center",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "evidence": row["evidence"],
                    "ambiguity": row["ambiguity"],
                    "rule": row["rule"],
                    "tr_I_Z": row["tr_I_Z"],
                    "tr_I_A": row["tr_I_A"],
                    "tr_loss": row["tr_loss"],
                    "status": row["status"],
                    "reason_code": row.get("reason_code"),
                    "n_actions": row["n_actions"],
                    "action_masses_json": json.dumps(row["action_masses"], sort_keys=True),
                    "is_center": bool(row.get("is_center", False)),
                }
            )
    (DATA_DIR / "rule_loss_grid.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )

    centre = [row for row in rows if row.get("is_center")]
    (DATA_DIR / "center_summary.json").write_text(
        json.dumps(centre, indent=2), encoding="utf-8"
    )

    print_center_table(centre)
    fig_loss_at_center(centre)
    fig_loss_heatmaps(rows)
    fig_loss_curves(rows)
    fig_action_masses(centre)
    print(f"\nData: {DATA_DIR}")
    print(f"Figures: {FIG_DIR}")


if __name__ == "__main__":
    main()
