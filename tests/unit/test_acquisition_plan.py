"""Request budgets match independently counted acquisition calls."""

from __future__ import annotations

from pathlib import Path

import pytest

from jevometry.adapters.base import ExperimentPoint
from jevometry.artifacts.store import RunStore
from jevometry.experiments.acquisition import LiveModeError, run_experiment
from jevometry.experiments.design import build_acquisition_plan
from jevometry.schemas.experiment import Budget, CaseSpec, ExperimentSpec
from jevometry.schemas.parameters import ParameterSet, ParameterSpec, StencilKind, StencilSpec
from jevometry.schemas.system import NodeSpec, SystemSpec
from jevometry.schemas.trace import EvaluationTrace


def spec(
    *,
    repeats: int = 3,
    max_attempts: int = 100,
    bounds: tuple[float, float] = (-1.0, 1.0),
    step: float = 0.1,
    theta: float = 0.0,
    kind: StencilKind = StencilKind.CENTRAL,
    step_scales: list[float] | None = None,
) -> ExperimentSpec:
    return ExperimentSpec(
        id="plan-test",
        parameter_set=ParameterSet(
            parameters=[
                ParameterSpec(
                    name=name,
                    role="task_relevant",
                    unit="dimensionless",
                    bounds=bounds,
                    step=step,
                )
                for name in ("x", "y")
            ]
        ),
        stencil=StencilSpec(kind=kind, step_scales=step_scales or [1.0, 0.5]),
        cases=[CaseSpec(id="case", state="state")],
        theta_points=[{"x": theta, "y": theta}],
        budget=Budget(repeats=repeats, max_attempts=max_attempts),
    )


class CountingAdapter:
    """Fake live transport that records calls without accessing the network."""

    is_live = True

    def __init__(self, *, batched: bool = True) -> None:
        self.batched = batched
        self.points: list[ExperimentPoint] = []
        self.calls = 0

    def describe(self) -> SystemSpec:
        return SystemSpec(
            id="counting",
            nodes=[NodeSpec(id=name, question_id=name) for name in ("a", "b", "c")],
        )

    def requests_per_point(self) -> int:
        return 1 if self.batched else 3

    def evaluate_point(self, point: ExperimentPoint) -> list[EvaluationTrace]:
        self.points.append(point)
        self.calls += self.requests_per_point()
        return []


@pytest.mark.parametrize("batched,expected", [(True, 27), (False, 81)])
def test_plan_counts_every_repeat_and_node_request(batched: bool, expected: int) -> None:
    # Per repeat: one center, four offsets for x, four for y.  Three repeats.
    adapter = CountingAdapter(batched=batched)
    captures = run_experiment(spec(), adapter, live=True)
    assert adapter.calls == expected
    assert captures.plan_summary["expected_requests"] == expected
    assert captures.plan_summary["center_requests"] == (3 if batched else 9)
    assert captures.plan_summary["stencil_requests"] == (24 if batched else 72)
    assert len(adapter.points) == 27
    assert {point.repeat for point in adapter.points} == {0, 1, 2}


def test_plan_refuses_excess_budget_before_first_transport_call() -> None:
    adapter = CountingAdapter()
    with pytest.raises(LiveModeError) as error:
        run_experiment(spec(max_attempts=26), adapter, live=True)
    assert error.value.reason_code == "plan_exceeds_budget"
    assert adapter.calls == 0
    assert adapter.points == []


def test_plan_accepts_exact_budget() -> None:
    adapter = CountingAdapter()
    captures = run_experiment(spec(max_attempts=27), adapter, live=True)
    assert adapter.calls == 27
    assert captures.plan_summary["within_budget"] is True


def test_plan_omits_stencil_when_requested() -> None:
    adapter = CountingAdapter()
    captures = run_experiment(spec(max_attempts=3), adapter, live=True, include_stencil=False)
    assert adapter.calls == 3
    assert captures.plan_summary["center_requests"] == 3
    assert captures.plan_summary["stencil_requests"] == 0
    assert captures.plan_summary["expected_requests"] == 3
    assert {point.stencil_role for point in adapter.points} == {"center"}


def test_plan_counts_only_available_stencil_at_narrow_bounds() -> None:
    # Neither parameter admits +-h or a one-sided 2h step in [0, 1].
    experiment = spec(bounds=(0.0, 1.0), step=1.0, theta=0.5, max_attempts=3)
    adapter = CountingAdapter()
    captures = run_experiment(experiment, adapter, live=True)
    assert adapter.calls == 3
    assert captures.plan_summary["stencil_requests"] == 0
    assert captures.plan_summary["expected_requests"] == 3


def test_plan_counts_boundary_roles_even_when_step_scales_overlap() -> None:
    # At the lower bound, +2*(h/2) and +1*h visit the same theta, with
    # distinct stencil roles; both are captured for each parameter/repeat.
    adapter = CountingAdapter()
    captures = run_experiment(
        spec(bounds=(0.0, 1.0), theta=0.0), adapter, live=True
    )
    assert adapter.calls == 27
    assert captures.plan_summary["stencil_requests"] == 24
    first_repeat = [point for point in adapter.points if point.repeat == 0]
    assert {point.stencil_role for point in first_repeat} == {
        "center",
        "x:1:+1", "x:1:+2", "x:0.5:+1", "x:0.5:+2",
        "y:1:+1", "y:1:+2", "y:0.5:+1", "y:0.5:+2",
    }
    assert len({tuple(sorted(point.theta.items())) for point in first_repeat}) == 7


@pytest.mark.parametrize("kind", [StencilKind.FORWARD, StencilKind.BACKWARD])
def test_plan_counts_unavailable_direction(kind: StencilKind) -> None:
    theta = 1.0 if kind is StencilKind.FORWARD else 0.0
    adapter = CountingAdapter()
    captures = run_experiment(
        spec(bounds=(0.0, 1.0), theta=theta, kind=kind, max_attempts=3),
        adapter,
        live=True,
    )
    assert adapter.calls == 3
    assert captures.plan_summary["stencil_requests"] == 0


@pytest.mark.parametrize("repeats", [0, -1])
def test_plan_rejects_nonpositive_repeat_override(repeats: int) -> None:
    with pytest.raises(ValueError, match="repeats must be positive"):
        build_acquisition_plan(spec(), ["a"], repeats=repeats)
    adapter = CountingAdapter()
    with pytest.raises(ValueError, match="repeats must be positive"):
        run_experiment(spec(), adapter, live=True, repeats=repeats)
    assert adapter.calls == 0


@pytest.mark.parametrize("requests_per_point", [0, -1])
def test_plan_rejects_nonpositive_request_factor(requests_per_point: int) -> None:
    with pytest.raises(ValueError, match="requests_per_point must be positive"):
        build_acquisition_plan(spec(), ["a"], requests_per_point=requests_per_point)


def test_plan_multiplies_cases_and_unequal_stencils_per_theta() -> None:
    experiment = spec(bounds=(0.0, 1.0), step=0.4, theta=0.0)
    experiment.cases.append(CaseSpec(id="second", state="state"))
    experiment.theta_points.append({"x": 0.7, "y": 0.7})
    # At zero, each coordinate admits four forward stencil roles; at 0.7
    # neither admits a central or a one-sided nominal stencil.
    plan = build_acquisition_plan(experiment, ["a"])
    assert plan.center_requests == 12
    assert plan.stencil_requests == 48
    assert plan.expected_requests == 60


def test_manifest_preserves_limits_and_records_actual_plan(tmp_path: Path) -> None:
    experiment = spec(max_attempts=27)
    store = RunStore.create(
        tmp_path,
        experiment=experiment,
        provider_mode="live",
        live=True,
        budget_summary=experiment.budget.model_dump(mode="json"),
    )
    adapter = CountingAdapter()
    captures = run_experiment(experiment, adapter, live=True, store=store)
    manifest = RunStore.load(tmp_path).manifest
    assert manifest is not None
    assert manifest.budget["max_attempts"] == 27
    assert manifest.budget["max_input_tokens"] == 200_000
    assert manifest.budget["repeats"] == 3
    assert manifest.budget["center_requests"] == 3
    assert manifest.budget["stencil_requests"] == 24
    assert manifest.budget["expected_requests"] == adapter.calls == 27
    assert manifest.budget["within_budget"] is True
    for name, value in captures.plan_summary.items():
        assert manifest.budget[name] == value
