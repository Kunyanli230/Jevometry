"""Experiment acquisition: evaluate the planned grid and record every trace."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from jevometry.adapters.base import ExperimentPoint, SystemAdapter
from jevometry.artifacts.store import RunStore
from jevometry.experiments.cache import CaptureCache
from jevometry.experiments.design import build_acquisition_plan, point_id
from jevometry.geometry.derivatives import stencil_points
from jevometry.schemas.experiment import ExperimentSpec
from jevometry.schemas.trace import EvaluationTrace


class LiveModeError(RuntimeError):
    """The requested live/offline mode does not match the provider."""

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class Captures:
    """All recorded traces for one acquisition run."""

    experiment: ExperimentSpec
    traces: list[EvaluationTrace]
    provider_mode: str
    live: bool
    run_id: str
    started_utc: str
    finished_utc: str
    incomplete: bool = False
    notes: list[str] = field(default_factory=list)
    plan_summary: dict[str, Any] = field(default_factory=dict)
    system: dict[str, Any] = field(default_factory=dict)

    def node_ids(self) -> list[str]:
        return sorted({trace.node_id for trace in self.traces})

    def failures(self) -> list[EvaluationTrace]:
        return [trace for trace in self.traces if not trace.status.ok]

    def center_traces(self) -> list[EvaluationTrace]:
        return [trace for trace in self.traces if trace.stencil_role == "center"]


def adapter_is_live(adapter: object) -> bool:
    return bool(getattr(adapter, "is_live", False))


def run_experiment(
    experiment: ExperimentSpec,
    adapter: SystemAdapter,
    *,
    live: bool = False,
    repeats: int | None = None,
    cache: CaptureCache | None = None,
    include_stencil: bool = True,
    store: RunStore | None = None,
    provider_mode: str | None = None,
) -> Captures:
    """Evaluate every planned point and return immutable captures.

    ``live=False`` refuses a provider that would make real network requests;
    ``live=True`` refuses a provider that cannot.  Nothing is silently
    downgraded or upgraded.
    """
    is_live = adapter_is_live(adapter)
    if live and not is_live:
        raise LiveModeError(
            "live_requested_for_offline_provider",
            "live=True requires a provider that makes real requests; refusing to "
            "label offline results as live",
        )
    if not live and is_live:
        raise LiveModeError(
            "live_provider_without_live_flag",
            "this provider makes real requests; pass live=True explicitly",
        )
    system_spec = adapter.describe()
    node_ids = system_spec.node_ids()
    effective_repeats = repeats if repeats is not None else experiment.budget.repeats
    plan = build_acquisition_plan(
        experiment,
        node_ids,
        live=live,
        repeats=effective_repeats,
        include_stencil=include_stencil,
        requests_per_point=getattr(adapter, "requests_per_point", lambda: 1)(),
    )
    if live and not plan.within_budget:
        raise LiveModeError(
            "plan_exceeds_budget",
            f"acquisition plan requires {plan.expected_requests} attempts, exceeding "
            f"max_attempts={experiment.budget.max_attempts}",
        )
    started = utc_now()
    traces: list[EvaluationTrace] = []
    notes: list[str] = []
    if is_live and cache is not None:
        notes.append(
            "live requests bypass the cache; captures are written for later replay"
        )
    for case in experiment.cases:
        for theta in experiment.theta_points:
            pid = point_id(theta)
            for repeat in range(effective_repeats):
                traces.extend(
                    _evaluate(
                        adapter,
                        case=case,
                        theta=theta,
                        point_id=pid,
                        repeat=repeat,
                        stencil_role="center",
                        cache=cache,
                        read_cache=not is_live,
                        write_cache=cache is not None,
                    )
                )
                if include_stencil:
                    for stencil_point in stencil_points(
                        theta, experiment.parameters(), experiment.stencil
                    ):
                        traces.extend(
                            _evaluate(
                                adapter,
                                case=case,
                                theta=stencil_point.theta,
                                point_id=pid,
                                repeat=repeat,
                                stencil_role=stencil_point.role,
                                cache=cache,
                                read_cache=not is_live,
                                write_cache=cache is not None,
                            )
                        )
        if store is not None:
            store.write_traces(traces)
    failures = [trace for trace in traces if not trace.status.ok]
    incomplete = bool(failures)
    if incomplete:
        notes.append(
            f"{len(failures)} of {len(traces)} traces are incomplete; affected metrics "
            "are marked incomplete rather than filled from other sources"
        )
    finished = utc_now()
    resolved_mode = provider_mode or ("live" if is_live else "offline")
    return Captures(
        experiment=experiment,
        traces=traces,
        provider_mode=resolved_mode,
        live=is_live,
        run_id=store.manifest.run_id if store is not None and store.manifest else "unsaved",
        started_utc=started,
        finished_utc=finished,
        incomplete=incomplete,
        notes=notes,
        plan_summary=plan.summary(),
        system=system_spec.model_dump(mode="json"),
    )


def _evaluate(
    adapter: SystemAdapter,
    *,
    case: Any,
    theta: Mapping[str, float],
    point_id: str,
    repeat: int,
    stencil_role: str,
    cache: CaptureCache | None,
    read_cache: bool,
    write_cache: bool,
) -> list[EvaluationTrace]:
    point = ExperimentPoint(
        case=case,
        theta=dict(theta),
        point_id=point_id,
        repeat=repeat,
        stencil_role=stencil_role,
    )
    if read_cache and cache is not None:
        from jevometry.adapters.base import request_fingerprint

        system = adapter.describe()
        cached: list[EvaluationTrace] = []
        for node in system.nodes:
            fingerprint = request_fingerprint(
                provider=getattr(adapter, "provider_name", "unknown"),
                model=system.model,
                adapter_version=system.adapter,
                node_id=node.id,
                question_hash=node.question_id,
                rendered_fingerprint=f"{case.id}:{point_id}:{stencil_role}",
                theta=theta,
                stencil_role=stencil_role,
            )
            entry = cache.get(fingerprint, repeat=repeat)
            if entry is None:
                cached = []
                break
            cached.append(entry)
        if cached:
            return cached
    traces = adapter.evaluate_point(point)
    if write_cache and cache is not None:
        for trace in traces:
            cache.put(trace)
    return traces
