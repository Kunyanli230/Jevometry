"""Compatibility checks and metric comparison across two runs.

Comparing Fisher eigenvalues, traces or entropies requires matching parameters,
units, scales, experiment points, support semantics and probability objects.
Incompatible runs are refused rather than compared.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jevometry.pipeline import load_captures
from jevometry.schemas.common import MetricStatus


@dataclass
class ComparisonResult:
    """Outcome of a run comparison."""

    compatible: bool
    reasons: list[str] = field(default_factory=list)
    checks: list[dict[str, Any]] = field(default_factory=list)
    metrics: list[dict[str, Any]] = field(default_factory=list)
    status: MetricStatus = MetricStatus.OK

    def to_json(self) -> dict[str, Any]:
        return {
            "compatible": self.compatible,
            "status": self.status.value,
            "reasons": self.reasons,
            "checks": self.checks,
            "metrics": self.metrics,
        }


def _parameter_signature(experiment: Any) -> dict[str, tuple[str, float, float]]:
    return {
        parameter.name: (parameter.unit, parameter.scale, parameter.step)
        for parameter in experiment.parameters()
    }


def _support_signature(captures: Any) -> dict[str, tuple[str, ...]]:
    supports: dict[str, tuple[str, ...]] = {}
    for trace in captures.center_traces():
        if trace.distribution is not None:
            supports[trace.node_id] = tuple(trace.distribution.support)
    return supports


def compare_runs(run_a: Path | str, run_b: Path | str) -> ComparisonResult:
    """Check compatibility then compare shared metrics."""
    captures_a = load_captures(run_a)
    captures_b = load_captures(run_b)
    result = ComparisonResult(compatible=True)
    experiment_a = captures_a.experiment
    experiment_b = captures_b.experiment

    signature_a = _parameter_signature(experiment_a)
    signature_b = _parameter_signature(experiment_b)
    if signature_a != signature_b:
        result.compatible = False
        result.reasons.append(
            "parameter declarations differ (names, units, scales or steps); "
            "dimensionless comparison is not valid"
        )
    else:
        result.checks.append({"check": "parameters", "ok": True})

    if experiment_a.model != experiment_b.model:
        result.compatible = False
        result.reasons.append(
            f"requested models differ: {experiment_a.model!r} vs {experiment_b.model!r}"
        )
    else:
        result.checks.append({"check": "model", "ok": True})

    if experiment_a.stencil.kind != experiment_b.stencil.kind:
        result.compatible = False
        result.reasons.append("finite-difference stencil kinds differ")
    if experiment_a.stencil.step_scales != experiment_b.stencil.step_scales:
        result.compatible = False
        result.reasons.append("finite-difference step scales differ")

    supports_a = _support_signature(captures_a)
    supports_b = _support_signature(captures_b)
    for node in sorted(set(supports_a) | set(supports_b)):
        left = supports_a.get(node)
        right = supports_b.get(node)
        if left != right:
            result.compatible = False
            result.reasons.append(
                f"node {node!r} support differs: {left!r} vs {right!r}; an explicit "
                "confirmed mapping is required"
            )
    if all(supports_a.get(node) == supports_b.get(node) for node in supports_a):
        result.checks.append({"check": "supports", "ok": True})

    if not result.compatible:
        result.status = MetricStatus.UNSUPPORTED
        return result

    metrics_a = _metric_index(captures_a)
    metrics_b = _metric_index(captures_b)
    for name in sorted(set(metrics_a) & set(metrics_b)):
        value_a = metrics_a[name]
        value_b = metrics_b[name]
        if value_a is None or value_b is None:
            continue
        result.metrics.append(
            {
                "metric": name,
                "run_a": value_a,
                "run_b": value_b,
                "difference": value_a - value_b,
            }
        )
    result.status = MetricStatus.OK
    return result


def _metric_index(captures: Any) -> dict[str, float | None]:
    from jevometry.analysis import analyze

    analysis = analyze(captures)
    index: dict[str, float | None] = {}
    for metric in analysis.document.metrics:
        if isinstance(metric.value, (int, float)):
            index[metric.name] = float(metric.value)
    for node in analysis.document.nodes:
        for metric in node.metrics:
            if isinstance(metric.value, (int, float)):
                index[f"{node.node_id}:{metric.name}"] = float(metric.value)
    return index


def write_comparison(result: ComparisonResult, output: Path | str) -> Path:
    import json

    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)
    path = target / "comparison.json"
    path.write_text(json.dumps(result.to_json(), indent=2), encoding="utf-8")
    return path
