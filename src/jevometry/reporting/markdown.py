"""Markdown twin of the HTML report, from the same computed results."""

from __future__ import annotations

from typing import Any

from jevometry.analysis import Analysis
from jevometry.reporting.model import build_report_model


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join("" if item is None else str(item) for item in row) + " |")
    return "\n".join(lines)


def render_markdown(
    analysis: Analysis, *, inference: dict[str, Any] | None = None
) -> str:
    model = build_report_model(analysis, inference=inference)
    lines: list[str] = []
    identity = model.identity
    lines.append(f"# Jevometry report · {identity['experiment_id']}")
    lines.append("")
    lines.append(f"- Run id: `{identity['run_id']}`")
    lines.append(f"- Analysis revision: {identity['analysis_revision']}")
    lines.append(f"- Analysis object: `{identity['analysis_object']}`")
    lines.append(f"- Provider mode: `{identity['provider_mode']}` (live: {identity['live']})")
    lines.append(f"- Requested model: `{identity['model']}`")
    lines.append(f"- Adapter: `{identity['adapter']}`")
    lines.append(f"- Seed: {identity['seed']}")
    lines.append("")
    lines.append("## Parameters")
    lines.append("")
    lines.append(
        _table(
            ["Name", "Role", "Unit", "Bounds", "Scale", "Step"],
            [
                [
                    parameter["name"],
                    parameter["role"],
                    parameter["unit"],
                    f"[{parameter['bounds'][0]}, {parameter['bounds'][1]}]",
                    parameter["scale"],
                    parameter["step"],
                ]
                for parameter in model.parameters
            ],
        )
    )
    lines.append("")
    lines.append("## Capability matrix")
    lines.append("")
    lines.append(
        _table(
            ["Capability", "Status", "Analysis object", "Reason", "Remedy"],
            [
                [
                    entry["capability"],
                    entry["status"],
                    entry["analysis_object"],
                    entry["reason_code"],
                    entry["remedy"],
                ]
                for entry in model.capability
            ],
        )
    )
    lines.append("")
    lines.append("## Data completeness")
    lines.append("")
    lines.append(f"- Total traces: {model.completeness['total_traces']}")
    lines.append(f"- Failed traces: {model.completeness['failed_traces']}")
    lines.append(f"- Incomplete: {model.completeness['incomplete']}")
    for note in model.completeness["notes"]:
        lines.append(f"- Note: {note}")
    lines.append("")
    if model.diagnostics:
        lines.append("## Diagnostics")
        lines.append("")
        for item in model.diagnostics:
            lines.append(f"- `{item['code']}` ({item['severity']}): {item['message']}")
        lines.append("")
    lines.append("## Node results")
    lines.append("")
    lines.append(
        _table(
            ["Node", "Point", "Support", "Status", "Reason"],
            [
                [
                    view["node_id"],
                    view["point_id"],
                    ", ".join(view["support"]),
                    view["status"],
                    view["reason_code"],
                ]
                for view in model.node_views
            ],
        )
    )
    lines.append("")
    lines.append("### Numerical geometry status")
    lines.append("")
    lines.append(
        "Non-ok Fisher values are retained diagnostic values, not validated Fisher "
        "geometry. Local metric unit balls are drawn only for ok matrices."
    )
    lines.append("")
    lines.append(
        _table(
            ["Node", "Case", "Point", "Quantity", "Status", "Reason"],
            [
                [
                    view["node_id"],
                    view["case_id"],
                    view["point_id"],
                    quantity,
                    result["status"],
                    result["reason_code"],
                ]
                for view in model.node_views
                for quantity in ("jacobian", "fisher")
                if (result := view[quantity]) is not None
            ],
        )
    )
    lines.append("")
    lines.append("### Node metrics")
    lines.append("")
    lines.append(
        _table(
            ["Node", "Case", "Point", "Metric", "Value", "Units", "Status", "Reason"],
            [
                [
                    view["node_id"],
                    view["case_id"],
                    view["point_id"],
                    metric["name"],
                    metric["value"],
                    metric["units"],
                    metric["status"],
                    metric["reason_code"],
                ]
                for view in model.node_views
                for metric in view["metrics"]
            ],
        )
    )
    lines.append("")
    if model.distances:
        lines.append("## Distribution distances (consecutive theta points)")
        lines.append("")
        lines.append(
            _table(
                ["Metric", "Value", "Units", "Status", "Reason"],
                [
                    [
                        distance["name"],
                        distance["value"],
                        distance["units"],
                        distance["status"],
                        distance["reason_code"],
                    ]
                    for distance in model.distances
                ],
            )
        )
        lines.append("")
    lines.append("## System composition")
    lines.append("")
    if model.system:
        lines.append(f"- Mode: `{model.system['mode']}`")
        lines.append(f"- Routing semantics: `{model.system['routing_semantics']}`")
        lines.append(f"- Status: `{model.system['status']}`")
        lines.append(f"- Reason: `{model.system['reason_code']}`")
        lines.append("")
        lines.append(
            _table(
                ["Metric", "Value", "Status", "Reason"],
                [
                    [metric["name"], metric["value"], metric["status"], metric["reason_code"]]
                    for metric in model.system["metrics"]
                ],
            )
        )
        lines.append("")
        if model.system["redundancy"]:
            lines.append("### Redundancy: independent sum versus declared joint")
            lines.append("")
            lines.append(
                _table(
                    [
                        "Case", "Independent sum trace", "Joint trace", "Difference", "Status",
                        "Reason",
                    ],
                    [
                        [
                            case["name"],
                            case["independent_sum_trace"],
                            case["joint_trace"],
                            case["difference"],
                            case["status"],
                            case["reason_code"],
                        ]
                        for case in model.system["redundancy"]
                    ],
                )
            )
            lines.append("")
    else:
        lines.append(
            "No system model is declared; no system Fisher information is reported."
        )
        lines.append("")
    lines.append("## Inference")
    lines.append("")
    if model.inference:
        lines.append("```json")
        import json

        lines.append(json.dumps(model.inference, indent=2, default=str))
        lines.append("```")
    else:
        lines.append(
            "No inference result is attached; run `jevometry infer` with a complete "
            "sampling contract."
        )
    lines.append("")
    return "\n".join(lines)
