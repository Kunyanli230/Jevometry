"""Export 8×8 grid of Δ diagonals (evidence/ambiguity) for all coordination rules."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

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
FIG_DIR = Path(__file__).resolve().parents[2] / "papers" / "coord-fisher" / "figures"
HEATMAP_RULES = ("baseline", "stricter", "wider", "quarantine_review", "coarse_binary")


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
            if result.difference is None or result.fisher_joint is None:
                rows.append(
                    {
                        "evidence": theta["evidence"],
                        "ambiguity": theta["ambiguity"],
                        "rule": name,
                        "delta_evidence": float("nan"),
                        "delta_ambiguity": float("nan"),
                        "tr_I_Z": float("nan"),
                        "tr_loss": float("nan"),
                        "frac_loss": float("nan"),
                        "frac_delta_ambiguity": float("nan"),
                        "min_eigenvalue": float("nan"),
                        "psd": False,
                        "status": result.status.value,
                        "is_center": is_center,
                    }
                )
                continue
            delta = np.asarray(result.difference, dtype=np.float64)
            tr_z = float(np.trace(result.fisher_joint))
            tr_loss = float(np.trace(delta))
            delta_amb = float(delta[1, 1])
            eigvals = np.linalg.eigvalsh(0.5 * (delta + delta.T))
            rows.append(
                {
                    "evidence": theta["evidence"],
                    "ambiguity": theta["ambiguity"],
                    "rule": name,
                    "delta_evidence": float(delta[0, 0]),
                    "delta_ambiguity": delta_amb,
                    "tr_I_Z": tr_z,
                    "tr_loss": tr_loss,
                    "frac_loss": tr_loss / tr_z if tr_z > 0 else float("nan"),
                    "frac_delta_ambiguity": delta_amb / tr_z if tr_z > 0 else float("nan"),
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


def fig_frac_loss_heatmap(rows: list[dict[str, float | str | bool]]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    e_vals = sorted({float(row["evidence"]) for row in rows})
    a_vals = sorted({float(row["ambiguity"]) for row in rows})
    rules = list(HEATMAP_RULES)
    fig, axes = plt.subplots(1, len(rules), figsize=(12.5, 3.4), sharey=True)
    vmax = max(
        float(row["frac_loss"])
        for row in rows
        if row["rule"] in rules and row["status"] == "ok"
    )
    for ax, rule in zip(axes, rules, strict=True):
        grid = np.full((len(a_vals), len(e_vals)), np.nan)
        for row in rows:
            if row["rule"] != rule or row["status"] != "ok":
                continue
            i = a_vals.index(float(row["ambiguity"]))
            j = e_vals.index(float(row["evidence"]))
            grid[i, j] = float(row["frac_loss"])
        im = ax.imshow(grid, origin="lower", aspect="auto", cmap="magma", vmin=0.0, vmax=vmax)
        ax.set_xticks(range(len(e_vals)))
        ax.set_xticklabels([f"{value:.2f}" for value in e_vals], rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(a_vals)))
        ax.set_yticklabels([f"{value:.2f}" for value in a_vals], fontsize=7)
        ax.set_xlabel("evidence")
        if rule == rules[0]:
            ax.set_ylabel("ambiguity")
        ax.set_title(rule, fontsize=9)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04, label=r"$\mathrm{tr}(\Delta)/\mathrm{tr}(I_Z)$")
    fig.suptitle("Fractional information loss over the parameter grid", fontsize=11)
    fig.savefig(FIG_DIR / "fig_frac_loss_heatmap.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_frac_loss_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_frac_loss_heatmap")


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
        "tr_I_Z",
        "tr_loss",
        "frac_loss",
        "frac_delta_ambiguity",
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
    fig_frac_loss_heatmap(rows)
    centre = [
        row
        for row in rows
        if abs(float(row["evidence"]) - float(CENTER[0])) < 1e-12
        and abs(float(row["ambiguity"]) - float(CENTER[1])) < 1e-12
        and row["rule"] != "near_identity"
    ]
    # Centre may fall off the linspace; use nearest grid point to CENTER.
    if not centre:
        best = min(
            {
                (float(row["evidence"]), float(row["ambiguity"]))
                for row in rows
            },
            key=lambda point: (point[0] - float(CENTER[0])) ** 2
            + (point[1] - float(CENTER[1])) ** 2,
        )
        centre = [
            row
            for row in rows
            if (float(row["evidence"]), float(row["ambiguity"])) == best
            and row["rule"] != "near_identity"
        ]
    fracs = {str(row["rule"]): float(row["frac_loss"]) for row in centre}
    assert fracs["quarantine_review"] == min(fracs.values())
    print(f"wrote {CSV_PATH} ({len(rows)} rows)")
    print(
        f"core order near≤wider≤baseline≤stricter holds on "
        f"{stats['hold']}/{stats['total']} grid points ({float(stats['ratio']):.0%})"
    )
    print(
        "centre frac_loss:",
        ", ".join(f"{name}={value:.4f}" for name, value in sorted(fracs.items(), key=lambda item: item[1])),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
