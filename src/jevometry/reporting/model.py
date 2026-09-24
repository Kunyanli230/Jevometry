"""Report data model built from an analysis document.

The report layer never recomputes statistics; it only reshapes already-computed
results into presentation structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from jevometry.analysis import Analysis
from jevometry.schemas.common import MetricStatus


@dataclass
class ReportModel:
    """Everything the templates need, already computed."""

    identity: dict[str, Any]
    capability: list[dict[str, Any]]
    completeness: dict[str, Any]
    node_views: list[dict[str, Any]] = field(default_factory=list)
    distributions: list[dict[str, Any]] = field(default_factory=list)
    distances: list[dict[str, Any]] = field(default_factory=list)
    system: dict[str, Any] | None = None
    inference: dict[str, Any] | None = None
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    parameters: list[dict[str, Any]] = field(default_factory=list)


def build_report_model(
    analysis: Analysis, *, inference: dict[str, Any] | None = None
) -> ReportModel:
    document = analysis.document
    captures = analysis.captures
    parameters: list[dict[str, Any]] = [
        {
            "name": parameter.name,
            "role": parameter.role.value,
            "unit": parameter.unit,
            "bounds": list(parameter.bounds),
            "scale": parameter.scale,
            "step": parameter.step,
        }
        for parameter in captures.experiment.parameters()
    ]
    identity: dict[str, Any] = {
        "run_id": document.run_id,
        "experiment_id": document.experiment_id,
        "analysis_revision": document.analysis_revision,
        "created_utc": document.created_utc,
        "analysis_object": document.analysis_object.value
        if document.analysis_object is not None
        else None,
        "provider_mode": captures.provider_mode,
        "live": captures.live,
        "model": captures.experiment.model,
        "adapter": captures.experiment.adapter,
        "seed": captures.experiment.seed,
        "budget": captures.plan_summary,
        "started_utc": captures.started_utc,
        "finished_utc": captures.finished_utc,
        "nodes": [node["id"] for node in captures.system.get("nodes", [])],
        "edges": [
            [edge["source"], edge["target"]] for edge in captures.system.get("edges", [])
        ],
        "parameters": parameters,
        "sampling_contract": captures.experiment.sampling_contract.model_dump(mode="json")
        if captures.experiment.sampling_contract is not None
        else None,
    }
    completeness = {
        "total_traces": len(captures.traces),
        "failed_traces": len(captures.failures()),
        "incomplete": captures.incomplete,
        "notes": list(captures.notes),
    }
    capability = [
        {
            "capability": entry.capability,
            "status": entry.status.value,
            "analysis_object": entry.analysis_object.value
            if entry.analysis_object is not None
            else None,
            "reason_code": entry.reason_code,
            "remedy": entry.remedy,
            "requires": entry.requires,
        }
        for entry in document.capability_matrix.entries
    ]
    node_views: list[dict[str, Any]] = []
    distributions: list[dict[str, Any]] = []
    for node in document.nodes:
        fisher = node.geometry.fisher if node.geometry is not None else None
        jacobian = node.geometry.jacobian if node.geometry is not None else None
        view = {
            "node_id": node.node_id,
            "question_id": node.question_id,
            "case_id": node.case_id,
            "point_id": node.point_id,
            "theta": node.theta,
            "support": node.support,
            "status": node.status.value,
            "reason_code": node.reason_code,
            "distribution": node.distribution.model_dump(mode="json")
            if node.distribution is not None
            else None,
            "metrics": [metric.model_dump(mode="json") for metric in node.metrics],
            "fisher": fisher.model_dump(mode="json") if fisher is not None else None,
            "jacobian": jacobian.model_dump(mode="json") if jacobian is not None else None,
        }
        node_views.append(view)
        if node.distribution is not None:
            distributions.append(
                {
                    "node_id": node.node_id,
                    "case_id": node.case_id,
                    "point_id": node.point_id,
                    "theta": node.theta,
                    "support": node.distribution.support,
                    "probabilities": node.distribution.probabilities(),
                    "raw_total": node.distribution.raw_total,
                    "derived": node.distribution.derived,
                    "selected": node.distribution.selected,
                    "confidence": node.distribution.confidence,
                    "source": node.distribution.source.value,
                }
            )
    distances = [
        {
            "name": metric.name,
            "value": metric.value,
            "units": metric.units,
            "status": metric.status.value,
            "reason_code": metric.reason_code,
        }
        for metric in document.metrics
        if metric.name.startswith("distance:")
    ]
    system = None
    if document.system is not None:
        system = document.system.model_dump(mode="json")
    inference_view = inference
    if inference_view is None and document.system is not None:
        inference_view = None
    diagnostics = [
        {
            "code": item.code,
            "message": item.message,
            "severity": item.severity,
            "details": item.details,
        }
        for item in document.diagnostics
    ]
    return ReportModel(
        identity=identity,
        capability=capability,
        completeness=completeness,
        node_views=node_views,
        distributions=distributions,
        distances=distances,
        system=system,
        inference=inference_view,
        diagnostics=diagnostics,
        parameters=parameters,
    )


def status_is_ok(status: str) -> bool:
    return status == MetricStatus.OK.value
