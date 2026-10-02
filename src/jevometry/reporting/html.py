"""Self-contained interactive HTML report (no CDN, no telemetry)."""

from __future__ import annotations

import html as html_module
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from jinja2 import Environment, FileSystemLoader, select_autoescape

from jevometry.analysis import Analysis
from jevometry.reporting.model import ReportModel, build_report_model

TEMPLATE_DIR = Path(__file__).parent / "templates"


def render_html(
    analysis: Analysis, *, inference: dict[str, Any] | None = None
) -> str:
    """Render the offline HTML report for an analysis."""
    model = build_report_model(analysis, inference=inference)
    environment = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "xml", "j2"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = environment.get_template("report.html.j2")
    charts = _build_charts(model)
    plotly_js = _plotly_js()
    return template.render(
        model=model,
        model_json=json.dumps(_model_payload(model), indent=2, default=str),
        charts=charts,
        plotly_js=plotly_js,
        generated_utc=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        escape=html_module.escape,
    )


def _plotly_js() -> str:
    from plotly.offline import get_plotlyjs

    script: str = str(get_plotlyjs())
    return script


def _model_payload(model: ReportModel) -> dict[str, Any]:
    return {
        "identity": model.identity,
        "capability": model.capability,
        "completeness": model.completeness,
        "nodes": [
            {
                key: view[key]
                for key in (
                    "node_id", "case_id", "point_id", "status", "reason_code", "metrics",
                    "fisher", "jacobian",
                )
            }
            for view in model.node_views
        ],
        "distributions": model.distributions,
        "distances": model.distances,
        "system": model.system,
        "inference": model.inference,
        "diagnostics": model.diagnostics,
    }


def _figure(fig: go.Figure) -> str:
    payload: str = str(fig.to_json())
    return payload


def _fisher_presentation(fisher: dict[str, Any]) -> dict[str, str]:
    status = str(fisher.get("status", "unknown"))
    reason = str(fisher.get("reason_code") or "")
    label = status if not reason else f"{status} · {reason}"
    return {
        "status": status,
        "reason_code": reason,
        "label": label,
        "note": (
            "Retained diagnostic values; not validated Fisher geometry. "
            f"Status: {label}."
            if status != "ok"
            else ""
        ),
    }


def _build_charts(model: ReportModel) -> list[dict[str, str]]:
    charts: list[dict[str, str]] = []
    charts.extend(_distribution_charts(model))
    charts.extend(_fisher_charts(model))
    charts.extend(_spectrum_charts(model))
    charts.extend(_stability_charts(model))
    charts.extend(_sweep_charts(model))
    charts.extend(_local_metric_charts(model))
    charts.extend(_system_charts(model))
    return charts


def _distribution_charts(model: ReportModel) -> list[dict[str, str]]:
    charts: list[dict[str, str]] = []
    for entry in model.distributions:
        figure = go.Figure(
            data=[
                go.Bar(
                    x=entry["support"],
                    y=entry["probabilities"],
                    marker_color="#3b6ea5",
                )
            ]
        )
        figure.update_layout(
            title=(
                f"node {entry['node_id']} · case {entry['case_id']} · point "
                f"{entry['point_id']}"
            ),
            xaxis_title="outcome",
            yaxis_title="probability",
            yaxis_range=[0, 1],
            margin={"l": 60, "r": 20, "t": 60, "b": 50},
        )
        charts.append(
            {
                "group": "distributions",
                "id": f"dist__{entry['node_id']}__{entry['case_id']}__{entry['point_id']}",
                "title": f"{entry['node_id']} / {entry['case_id']} / {entry['point_id']}",
                "json": _figure(figure),
            }
        )
    return charts


def _fisher_charts(model: ReportModel) -> list[dict[str, str]]:
    charts: list[dict[str, str]] = []
    for view in model.node_views:
        fisher = view["fisher"]
        if not fisher or fisher.get("values") is None:
            continue
        names = fisher["parameter_names"]
        presentation = _fisher_presentation(fisher)
        figure = go.Figure(
            data=go.Heatmap(
                z=fisher["values"],
                x=names,
                y=names,
                colorscale="Viridis",
                colorbar={"title": "nats"},
            )
        )
        figure.update_layout(
            title=(
                f"Fisher information · {view['node_id']} · {view['point_id']} · "
                f"{html_module.escape(presentation['label'])}"
            ),
            yaxis={"autorange": "reversed"},
            margin={"l": 80, "r": 20, "t": 60, "b": 60},
        )
        charts.append(
            {
                "group": "fisher",
                "id": f"fisher__{view['node_id']}__{view['case_id']}__{view['point_id']}",
                "title": (
                    f"{view['node_id']} / {view['point_id']} · {presentation['label']}"
                ),
                **presentation,
                "json": _figure(figure),
            }
        )
    return charts


def _spectrum_charts(model: ReportModel) -> list[dict[str, str]]:
    charts: list[dict[str, str]] = []
    for view in model.node_views:
        fisher = view["fisher"]
        if not fisher or not fisher.get("eigenvalues"):
            continue
        presentation = _fisher_presentation(fisher)
        figure = go.Figure(
            data=[
                go.Bar(
                    x=[f"λ{index + 1}" for index in range(len(fisher["eigenvalues"]))],
                    y=fisher["eigenvalues"],
                    marker_color="#7a5195",
                )
            ]
        )
        figure.update_layout(
            title=(
                f"Fisher spectrum · {view['node_id']} · rank "
                f"{fisher['rank']} · cond {fisher['condition_number']} · "
                f"{html_module.escape(presentation['label'])}"
            ),
            yaxis_title="eigenvalue",
            margin={"l": 60, "r": 20, "t": 60, "b": 50},
        )
        charts.append(
            {
                "group": "spectrum",
                "id": f"spectrum__{view['node_id']}__{view['case_id']}__{view['point_id']}",
                "title": (
                    f"{view['node_id']} / {view['point_id']} · {presentation['label']}"
                ),
                **presentation,
                "json": _figure(figure),
            }
        )
    return charts


def _stability_charts(model: ReportModel) -> list[dict[str, str]]:
    labels: list[str] = []
    values: list[float] = []
    for view in model.node_views:
        jacobian = view["jacobian"]
        if not jacobian:
            continue
        stability = jacobian.get("stability_relative", {})
        if "matrix" not in stability:
            continue
        labels.append(f"{view['node_id']} / {view['point_id']}")
        values.append(stability["matrix"])
    if not labels:
        return []
    figure = go.Figure(
        data=[go.Bar(x=labels, y=values, marker_color="#d17b0f")]
    )
    figure.update_layout(
        title="Finite-difference stability: ||J(h) - J(h/2)|| / ||J(h)||",
        yaxis_title="relative difference",
        margin={"l": 60, "r": 20, "t": 60, "b": 80},
    )
    return [
        {
            "group": "stability",
            "id": "stability__matrix",
            "title": "h versus h/2",
            "json": _figure(figure),
        }
    ]


def _sweep_charts(model: ReportModel) -> list[dict[str, str]]:
    series: dict[tuple[str, str], list[tuple[str, float]]] = {}
    for view in model.node_views:
        for metric in view["metrics"]:
            if metric["name"].startswith("entropy:") and metric["value"] is not None:
                series.setdefault((view["node_id"], view["case_id"]), []).append(
                    (view["point_id"], float(metric["value"]))
                )
    charts: list[dict[str, str]] = []
    for (node_id, case_id), points in series.items():
        points = sorted(points)
        figure = go.Figure(
            data=[
                go.Scatter(
                    x=[item[0] for item in points],
                    y=[item[1] for item in points],
                    mode="lines+markers",
                    line={"color": "#3b6ea5"},
                )
            ]
        )
        figure.update_layout(
            title=f"Entropy sweep · {node_id} · {case_id}",
            xaxis_title="theta point",
            yaxis_title="entropy (nats)",
            margin={"l": 60, "r": 20, "t": 60, "b": 80},
        )
        charts.append(
            {
                "group": "sweep",
                "id": f"sweep__{node_id}__{case_id}",
                "title": f"{node_id} / {case_id}",
                "json": _figure(figure),
            }
        )
    return charts


def _local_metric_charts(model: ReportModel) -> list[dict[str, str]]:
    charts: list[dict[str, str]] = []
    for view in model.node_views:
        fisher = view["fisher"]
        if not fisher or fisher.get("values") is None or fisher.get("status") != "ok":
            continue
        names = fisher["parameter_names"]
        if len(names) < 2:
            continue
        matrix = np.asarray(fisher["values"], dtype=np.float64)[:2, :2]
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(matrix)
            if np.any(eigenvalues <= 0):
                continue
            inverse_sqrt = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
        except np.linalg.LinAlgError:
            continue
        angles = np.linspace(0.0, 2.0 * np.pi, 200)
        unit = np.vstack([np.cos(angles), np.sin(angles)])
        ellipse = inverse_sqrt @ unit
        figure = go.Figure(
            data=[
                go.Scatter(
                    x=ellipse[0, :],
                    y=ellipse[1, :],
                    mode="lines",
                    name="unit ball under the Fisher metric",
                    line={"color": "#7a5195"},
                ),
                go.Scatter(
                    x=[0.0],
                    y=[0.0],
                    mode="markers",
                    name="center",
                    marker={"color": "#333333"},
                ),
            ]
        )
        figure.update_layout(
            title=(
                "Local metric unit ball (geometry, not a confidence ellipse) · "
                f"{view['node_id']} · {names[0]} vs {names[1]} · ok"
            ),
            xaxis_title=f"delta {names[0]}",
            yaxis_title=f"delta {names[1]}",
            yaxis={"scaleanchor": "x", "scaleratio": 1},
            margin={"l": 60, "r": 20, "t": 60, "b": 60},
        )
        charts.append(
            {
                "group": "local_metric",
                "id": (
                    f"local_metric__{view['node_id']}__{view['case_id']}__{view['point_id']}"
                ),
                "title": f"{view['node_id']} · {names[0]}/{names[1]} · ok",
                **_fisher_presentation(fisher),
                "json": _figure(figure),
            }
        )
        break
    return charts


def _system_charts(model: ReportModel) -> list[dict[str, str]]:
    system = model.system
    if not system:
        return []
    node_order = system.get("node_order", [])
    if not node_order:
        return []
    charts: list[dict[str, str]] = []
    angles = np.linspace(0.0, 2.0 * np.pi, len(node_order), endpoint=False)
    positions = {
        node: (float(np.cos(angle)), float(np.sin(angle)))
        for node, angle in zip(node_order, angles, strict=True)
    }
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for edge in model.identity.get("edges", []):
        source, target = edge
        if source in positions and target in positions:
            edge_x.extend([positions[source][0], positions[target][0], None])
            edge_y.extend([positions[source][1], positions[target][1], None])
    figure = go.Figure()
    if edge_x:
        figure.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line={"color": "#bbbbbb"},
                hoverinfo="skip",
                showlegend=False,
            )
        )
    figure.add_trace(
        go.Scatter(
            x=[positions[node][0] for node in node_order],
            y=[positions[node][1] for node in node_order],
            mode="markers+text",
            text=node_order,
            textposition="top center",
            marker={"size": 18, "color": "#3b6ea5"},
            showlegend=False,
        )
    )
    figure.update_layout(
        title="Declared system graph",
        xaxis={"visible": False},
        yaxis={"visible": False},
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
    )
    charts.append(
        {
            "group": "system",
            "id": "system__graph",
            "title": "system graph",
            "json": _figure(figure),
        }
    )
    return charts
