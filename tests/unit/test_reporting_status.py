"""Reports must preserve qualifications on retained numerical results."""

from __future__ import annotations

import json
from typing import Any

import pytest

from jevometry import Experiment, analyze
from jevometry.adapters.analytic import AnalyticAdapter, bernoulli_node
from jevometry.analysis import Analysis
from jevometry.reporting import render_html, render_markdown
from jevometry.reporting.html import (
    _fisher_charts,
    _local_metric_charts,
    _spectrum_charts,
)
from jevometry.reporting.model import ReportModel
from jevometry.schemas.common import MetricStatus
from jevometry.schemas.experiment import CaseSpec, ExperimentSpec
from jevometry.schemas.parameters import ParameterSet, ParameterSpec
from jevometry.schemas.results import RedundancyCase, SystemAnalysisDocument


@pytest.fixture
def analysis() -> Analysis:
    experiment = Experiment.from_spec(
        ExperimentSpec(
            id="report-status",
            parameter_set=ParameterSet(
                parameters=[
                    ParameterSpec(
                        name="p",
                        role="task_relevant",
                        unit="probability",
                        bounds=(0.0, 1.0),
                        center=0.3,
                        step=1e-4,
                    )
                ]
            ),
            cases=[CaseSpec(id="case", state="PRIVATE-CASE-TEXT")],
            theta_points=[{"p": 0.3}],
        )
    )
    adapter = AnalyticAdapter(system_id="report-status", nodes={"Y": bernoulli_node("Y")})
    return analyze(experiment.run(adapter), adapter=adapter)


@pytest.mark.parametrize(
    ("status", "reason"),
    [
        (MetricStatus.UNSTABLE, "step_stability_exceeded"),
        (MetricStatus.CONDITIONAL, "resolution_limited"),
    ],
)
def test_reports_show_retained_fisher_metric_status(
    analysis: Analysis, status: MetricStatus, reason: str
) -> None:
    node = analysis.document.nodes[0]
    assert node.geometry is not None
    assert node.geometry.fisher is not None
    assert node.geometry.jacobian is not None
    node.status = status
    node.reason_code = reason
    node.geometry.fisher.status = status
    node.geometry.fisher.reason_code = reason
    node.geometry.jacobian.status = status
    node.geometry.jacobian.reason_code = reason
    fisher_metrics = [metric for metric in node.metrics if metric.name.startswith("fisher_")]
    assert fisher_metrics
    for metric in fisher_metrics:
        metric.status = status
        metric.reason_code = reason

    html = render_html(analysis)
    markdown = render_markdown(analysis)
    for report in (html, markdown):
        assert "Numerical geometry status" in report
        assert "Node metrics" in report
        assert status.value in report
        assert reason in report
        assert "retained diagnostic values" in report
        assert "PRIVATE-CASE-TEXT" not in report
        for metric in fisher_metrics:
            assert metric.name in report
            assert str(metric.value) in report
    for metric in fisher_metrics:
        assert (
            f"| {metric.name} | {metric.value} | {metric.units} | {status.value} | {reason} |"
            in markdown
        )


def test_reports_show_redundancy_refusal_reason(analysis: Analysis) -> None:
    analysis.document.system = SystemAnalysisDocument(
        mode="declared_joint",
        node_order=["Y"],
        redundancy=[
            RedundancyCase(
                name="copy",
                description="Copy comparison requires valid node geometry.",
                status=MetricStatus.UNSTABLE,
                reason_code="independent_geometry_unstable",
            )
        ],
    )
    markdown = render_markdown(analysis)
    html = render_html(analysis)
    assert "| copy |  |  |  | unstable | independent_geometry_unstable |" in markdown
    assert "<td><code>independent_geometry_unstable</code></td>" in html


def geometry_view(status: str, reason: str | None, point_id: str) -> dict[str, Any]:
    # A declared diagonal metric: its exact unit-ball axes are 1/2 and 1/3.
    return {
        "node_id": "Y",
        "case_id": "case",
        "point_id": point_id,
        "fisher": {
            "parameter_names": ["a", "b"],
            "values": [[4.0, 0.0], [0.0, 9.0]],
            "eigenvalues": [4.0, 9.0],
            "rank": 2,
            "condition_number": 2.25,
            "status": status,
            "reason_code": reason,
        },
    }


@pytest.mark.parametrize(
    ("status", "reason"),
    [("unstable", "step_stability_exceeded"), ("conditional", "resolution_limited")],
)
def test_fisher_charts_label_diagnostic_values_and_omit_unit_ball(status: str, reason: str) -> None:
    model = ReportModel(
        identity={},
        capability=[],
        completeness={},
        node_views=[geometry_view(status, reason, "questionable")],
    )
    for charts in (_fisher_charts(model), _spectrum_charts(model)):
        assert len(charts) == 1
        chart = charts[0]
        assert status in chart["title"]
        assert reason in chart["title"]
        assert status in json.loads(chart["json"])["layout"]["title"]["text"]
        assert reason in json.loads(chart["json"])["layout"]["title"]["text"]
        assert "Retained diagnostic values" in chart["note"]
        assert "not validated Fisher geometry" in chart["note"]
    assert _local_metric_charts(model) == []


def test_local_metric_can_select_stable_geometry_after_qualified_geometry() -> None:
    model = ReportModel(
        identity={},
        capability=[],
        completeness={},
        node_views=[
            geometry_view("unstable", "step_stability_exceeded", "questionable"),
            geometry_view("ok", None, "stable"),
        ],
    )
    charts = _local_metric_charts(model)
    assert len(charts) == 1
    chart = charts[0]
    assert chart["status"] == "ok"
    assert chart["note"] == ""
    assert chart["id"].endswith("__stable")
    assert "ok" in chart["title"]


def test_stable_reports_keep_valid_scalar_results(analysis: Analysis) -> None:
    node = analysis.document.nodes[0]
    assert node.geometry is not None
    assert node.geometry.fisher is not None
    assert node.geometry.fisher.status is MetricStatus.OK
    trace = next(metric for metric in node.metrics if metric.name.startswith("fisher_trace:"))
    # Bernoulli Fisher information at p=0.3 equals 1/(p*(1-p)).
    assert trace.value == pytest.approx(1.0 / (0.3 * 0.7), rel=1e-8)
    for report in (render_html(analysis), render_markdown(analysis)):
        assert trace.name in report
        assert str(trace.value) in report
        assert "Node metrics" in report
