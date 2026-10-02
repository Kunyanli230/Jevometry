"""Replay provider: match existing captures by full request fingerprint.

Replay never falls back to a live provider.  A missing or mismatched capture
produces a structured ``missing_capture`` failure so that incomplete runs stay
visibly incomplete.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from jevometry.adapters.base import ExperimentPoint, request_fingerprint
from jevometry.schemas.common import MetricStatus
from jevometry.schemas.system import SystemSpec
from jevometry.schemas.trace import EvaluationTrace, ProviderStatus

theta_key = tuple[tuple[str, float], ...]


class ReplayError(RuntimeError):
    """Replay could not be satisfied from the recorded artifacts."""

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


def _theta_key(theta: Mapping[str, float]) -> theta_key:
    return tuple(sorted((name, round(float(value), 12)) for name, value in theta.items()))


@dataclass
class ReplayIndex:
    """Index of recorded traces for exact lookup."""

    traces: list[EvaluationTrace]

    @classmethod
    def load(cls, run_directory: Path) -> ReplayIndex:
        path = Path(run_directory) / "traces.jsonl"
        if not path.exists():
            raise ReplayError(
                "missing_traces_file",
                f"replay directory {run_directory} has no traces.jsonl",
            )
        traces: list[EvaluationTrace] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            traces.append(EvaluationTrace.model_validate(json.loads(line)))
        return cls(traces=traces)

    def candidates(
        self,
        *,
        case_id: str,
        point_id: str,
        node_id: str,
        theta: Mapping[str, float],
        stencil_role: str,
    ) -> list[EvaluationTrace]:
        key = _theta_key(theta)
        return [
            trace
            for trace in self.traces
            if trace.case_id == case_id
            and trace.point_id == point_id
            and trace.node_id == node_id
            and _theta_key(trace.theta) == key
            and trace.stencil_role == stencil_role
        ]

    def node_ids(self) -> list[str]:
        return sorted({trace.node_id for trace in self.traces})


class ReplayAdapter:
    """An adapter that only replays previously recorded traces."""

    is_live = False
    provider_name = "replay"

    def __init__(
        self,
        run_directory: Path,
        *,
        system_id: str,
        model: str | None = None,
        adapter_version: str | None = None,
        question_hashes: Mapping[str, str] | None = None,
        provider: str | None = None,
        node_ids: list[str] | None = None,
    ) -> None:
        self.run_directory = Path(run_directory)
        self.system_id = system_id
        self.model = model
        self.adapter_version = adapter_version
        self.question_hashes = dict(question_hashes or {})
        self.index = ReplayIndex.load(self.run_directory)
        self._declared_provider = provider
        self.provider = provider or self._infer_provider()
        self._node_ids = node_ids or self.index.node_ids()

    def _infer_provider(self) -> str:
        """Infer the source provider from the recorded trace provenance."""
        for trace in self.index.traces:
            provider = self._source_provider(trace)
            if isinstance(provider, str) and provider:
                return provider
        return "typesafe"

    @staticmethod
    def _source_provider(trace: EvaluationTrace) -> str | None:
        value = trace.provenance.get("source_provider", trace.provenance.get("provider"))
        return value if isinstance(value, str) and value else None

    def describe_node_ids(self) -> list[str]:
        return list(self._node_ids)

    def describe(self) -> SystemSpec:
        from jevometry.schemas.system import CompositionMode, NodeSpec

        return SystemSpec(
            id=self.system_id,
            nodes=[
                NodeSpec(id=node_id, question_id=node_id) for node_id in self._node_ids
            ],
            composition_mode=CompositionMode.NODE_ONLY,
            model=self.model,
            adapter="replay-0.1.0",
        )

    def evaluate_point(self, point: ExperimentPoint) -> list[EvaluationTrace]:
        if not self._node_ids:
            raise ReplayError(
                "missing_capture",
                f"no recorded traces in {self.run_directory}",
            )
        results: list[EvaluationTrace] = []
        for node_id in self._node_ids:
            candidates = self.index.candidates(
                case_id=point.case.id,
                point_id=point.point_id,
                node_id=node_id,
                theta=point.theta,
                stencil_role=point.stencil_role,
            )
            if not candidates:
                raise ReplayError(
                    "missing_capture",
                    (
                        f"no capture for node {node_id!r} case {point.case.id!r} "
                        f"point {point.point_id!r} role {point.stencil_role!r}; replay "
                        "never falls back to a live provider"
                    ),
                )
            match = next(
                (trace for trace in candidates if trace.repeat == point.repeat), None
            )
            if match is None:
                raise ReplayError(
                    "missing_capture",
                    (
                        f"no capture for node {node_id!r} case {point.case.id!r} "
                        f"point {point.point_id!r} role {point.stencil_role!r} "
                        f"repeat {point.repeat}; replay never reuses another repeat"
                    ),
                )
            self._validate_capture(match, point)
            provenance = {**match.provenance, "provider": "replay"}
            source_provider = self._source_provider(match)
            if source_provider is not None:
                provenance["source_provider"] = source_provider
            results.append(
                match.model_copy(update={"provenance": provenance})
            )
        return results

    def _validate_capture(self, trace: EvaluationTrace, point: ExperimentPoint) -> None:
        """Validate declarations and the fingerprint of the selected repeat.

        Older captures may omit source metadata.  Unspecified declarations do
        not invent that metadata; explicit declarations must be verifiable.
        """
        provider = self._source_provider(trace)
        adapter = trace.provenance.get("adapter_version")
        adapter_version = adapter if isinstance(adapter, str) else None
        question_hash = (
            trace.distribution.semantic_hash if trace.distribution is not None else None
        )
        declared_question = self.question_hashes.get(trace.node_id)
        declarations = (
            (self._declared_provider, provider),
            (self.model, trace.status.model_requested),
            (self.adapter_version, adapter_version),
            (declared_question, question_hash),
        )
        mismatch = trace.history != dict(point.history) or any(
            declared is not None and recorded is not None and declared != recorded
            for declared, recorded in declarations
        )
        if self.model is not None and self.model != trace.status.model_requested:
            mismatch = True
        expected_provider = self._declared_provider or provider
        expected_adapter = self.adapter_version or adapter_version
        expected_question = declared_question or question_hash
        can_verify_fingerprint = (
            expected_provider is not None
            and expected_adapter is not None
            and expected_question is not None
        )
        if can_verify_fingerprint:
            assert expected_provider is not None
            assert expected_question is not None
            expected = request_fingerprint(
                provider=expected_provider,
                model=self.model if self.model is not None else trace.status.model_requested,
                adapter_version=expected_adapter,
                node_id=trace.node_id,
                question_hash=expected_question,
                rendered_fingerprint=trace.rendered_fingerprint,
                theta=point.theta,
                history=point.history,
                stencil_role=point.stencil_role,
            )
            mismatch = mismatch or trace.request_fingerprint != expected
        elif any(
            declared is not None and recorded is None
            for declared, recorded in declarations
        ):
            mismatch = True
        if mismatch:
            raise ReplayError(
                "fingerprint_mismatch",
                (
                    f"capture for node {trace.node_id!r} repeat {point.repeat} does not "
                    "match the current provider, model, adapter, question or history; "
                    "refusing to mix requests"
                ),
            )

    def describe_status(self) -> ProviderStatus:
        return ProviderStatus(
            ok=True,
            model_requested=self.model,
            model_resolved=self.model,
            reason_code="replay",
        )

    def available_fingerprints(self) -> list[str]:
        return sorted({trace.request_fingerprint for trace in self.index.traces})

    def missing_reason(self, node_id: str, point: ExperimentPoint) -> MetricStatus:
        del node_id, point
        return MetricStatus.INSUFFICIENT_DATA
