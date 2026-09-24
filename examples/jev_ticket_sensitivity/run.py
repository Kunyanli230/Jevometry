"""Example C: ticket-system sensitivity (offline synthetic, optional live).

Offline (default): analyses the side-effect-free synthetic ticket system with a
deterministic policy.  The policy's action labels and flip locations are
reported, but no trajectory Fisher information or business CRLB is claimed,
because the production routing is deterministic.  A declared product surrogate
is reported separately.

Live (``--live``, requires TYPESAFE_API_KEY): runs the same three questions
against the real TypeSafe API on a minimal grid, with the budget printed before
any request.  Without credentials the live path is refused, never faked.

Run from the repository root:

    uv run python examples/jev_ticket_sensitivity/run.py
    uv run python examples/jev_ticket_sensitivity/run.py --live --model MODEL_ID
    uv run python examples/jev_ticket_sensitivity/run.py --replay output/live
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

from jevometry import Experiment, analyze, render_report
from jevometry.adapters.typesafe import (
    TypeSafeAdapter,
    typesafe_choice_question,
    typesafe_noul_question,
    typesafe_score_question,
)
from jevometry.experiments.renderer import StructuredRenderer
from jevometry.schemas.common import (
    AnalysisObject,
    MetricResult,
    MetricStatus,
    RoutingSemantics,
    ValueKind,
)
from jevometry.schemas.experiment import CaseSpec, ExperimentSpec
from jevometry.schemas.parameters import ParameterSet, ParameterSpec, StencilSpec
from jevometry.systems.policy import analyse_policy

sys.path.insert(0, str(Path(__file__).parent))

from ticket_system import (  # noqa: E402
    STATE_TEMPLATE,
    STATIC_STATE,
    TicketSystemAdapter,
    node_probability_map,
)

OUTPUT = Path(__file__).parent / "output"
LIVE_BUDGET_ATTEMPTS = 60
LIVE_REPEATS = 3


def parameters() -> list[ParameterSpec]:
    return [
        ParameterSpec(
            name="amount",
            role="task_relevant",
            unit="USD",
            bounds=(10.0, 1000.0),
            scale=100.0,
            step=1.0,
            center=250.0,
        ),
        ParameterSpec(
            name="wait_hours",
            role="task_relevant",
            unit="hours",
            bounds=(0.0, 48.0),
            scale=12.0,
            step=0.25,
            center=8.0,
        ),
    ]


def theta_grid() -> list[dict[str, float]]:
    points: list[dict[str, float]] = []
    for amount in (100.0, 250.0, 600.0):
        for wait in (2.0, 8.0, 30.0):
            points.append({"amount": amount, "wait_hours": wait})
    return points


def build_spec(*, experiment_id: str = "ticket-sensitivity") -> ExperimentSpec:
    return ExperimentSpec(
        id=experiment_id,
        description="Synthetic ticket triage sensitivity over amount and wait time",
        model="synthetic-ticket",
        adapter="module",
        parameter_set=ParameterSet(parameters=parameters()),
        stencil=StencilSpec(step_scales=[1.0, 0.5], relative_stability_threshold=0.10),
        cases=[
            CaseSpec(
                id="ticket-1",
                state={"ticket": "Synthetic ticket fixture", "channel": "email"},
                description="fixed synthetic case",
            )
        ],
        theta_points=theta_grid(),
    )


def add_policy_metrics(analysis, adapter: TicketSystemAdapter) -> None:
    points = analysis.captures.experiment.theta_points
    probabilities = [
        node_probability_map(adapter.nodes, theta) for theta in points
    ]
    policy = analyse_policy(
        adapter.policy,
        policy_name="ticket_triage",
        points=points,
        probabilities=probabilities,
    )
    analysis.document.metrics.append(
        MetricResult.ok(
            "policy_flips",
            float(len(policy.flips)),
            analysis_object=AnalysisObject.DECLARED_SYSTEM_MODEL,
            value_kind=ValueKind.SCALAR,
            units="count",
            assumptions=list(policy.assumptions),
            diagnostics={"action_counts": policy.action_counts, "flips": policy.flips},
        )
    )
    for action, count in sorted(policy.action_counts.items()):
        analysis.document.metrics.append(
            MetricResult.ok(
                f"policy_action_count:{action}",
                float(count),
                analysis_object=AnalysisObject.DECLARED_SYSTEM_MODEL,
                units="design points",
                assumptions=["design-point counts, not observed routing frequencies"],
            )
        )
    analysis.document.provenance["policy"] = {
        "name": policy.policy_name,
        "actions": policy.actions,
        "flips": policy.flips,
        "assumptions": policy.assumptions,
    }
    # Declared product surrogate, reported separately from production routing.
    center = analysis.captures.experiment.center()
    surrogate = adapter.surrogate_action_distribution(center)
    for action, probability in sorted(surrogate.items()):
        analysis.document.metrics.append(
            MetricResult.ok(
                f"surrogate_action_probability:{action}",
                float(probability),
                analysis_object=AnalysisObject.EMPIRICAL_OBSERVATION_MODEL,
                units="probability",
                assumptions=[
                    "declared product surrogate: conditional independence across nodes",
                    "not the production routing law",
                ],
            )
        )
    analysis.document.provenance["surrogate"] = surrogate
    analysis.document.warnings.append(
        "deterministic policy: no trajectory Fisher information or business CRLB is claimed"
    )


def run_offline() -> int:
    adapter = TicketSystemAdapter(system_id="ticket-sensitivity")
    experiment = Experiment.from_spec(build_spec())
    captures = experiment.run(adapter, output=OUTPUT / "synthetic")
    analysis = analyze(captures, adapter=adapter)
    add_policy_metrics(analysis, adapter)
    render_report(
        analysis,
        output=OUTPUT / "synthetic" / "report.html",
        run_directory=OUTPUT / "synthetic",
    )

    print("== C1. node geometry ==")
    for node in analysis.document.nodes:
        if node.geometry and node.geometry.fisher:
            stability = node.geometry.jacobian.stability_relative if node.geometry.jacobian else {}
            print(
                f"  {node.node_id:18s} point={node.point_id} "
                f"trace={np.trace(node.geometry.fisher.values):.4f} "
                f"stability={stability.get('matrix')!r}"
            )
    system = analysis.document.system
    assert system is not None
    print("== C2. system composition ==")
    print(f"  mode={system.mode} status={system.status.value} reason={system.reason_code}")
    assert system.status is MetricStatus.UNSUPPORTED
    print("== C3. deterministic policy ==")
    flips = next(
        metric for metric in analysis.document.metrics if metric.name == "policy_flips"
    )
    print(f"  policy flips across the grid: {flips.value}")
    print(f"  action counts: {flips.diagnostics['action_counts']}")
    print("== C4. declared product surrogate (separate from production routing) ==")
    surrogate = analysis.document.provenance["surrogate"]
    for action, probability in sorted(surrogate.items()):
        print(f"  {action:16s} {probability:.4f}")
    print("report written to", OUTPUT / "synthetic" / "report.html")
    return 0


def live_questions():
    return {
        "request_type": typesafe_choice_question(
            "request_type",
            {"billing": "Billing question", "technical": "Technical issue", "refund": "Refund request"},
            instructions="What kind of request is this?",
        ),
        "needs_escalation": typesafe_noul_question(
            "needs_escalation", instructions="Does this ticket need escalation?"
        ),
        "impact": typesafe_score_question(
            "impact", ["low", "medium", "high"], instructions="How severe is the impact?"
        ),
    }


def run_live(model: str) -> int:
    if not os.environ.get("TYPESAFE_API_KEY"):
        print(
            "TYPESAFE_API_KEY is not set; the live path is not exercised. "
            "Offline capabilities are unaffected.",
            file=sys.stderr,
        )
        return 0
    adapter = TypeSafeAdapter.from_env(
        model=model,
        nodes=live_questions(),
        renderer=StructuredRenderer(STATE_TEMPLATE, static=STATIC_STATE, version="ticket-v1"),
        routing_semantics=RoutingSemantics.DETERMINISTIC_POLICY,
        system_id="ticket-sensitivity-live",
    )
    spec = build_spec(experiment_id="ticket-sensitivity-live")
    spec.model = model
    spec.adapter = "typesafe"
    spec.theta_points = [{"amount": 250.0, "wait_hours": 8.0}]
    spec.budget = spec.budget.model_copy(
        update={"max_attempts": LIVE_BUDGET_ATTEMPTS, "repeats": LIVE_REPEATS}
    )
    experiment = Experiment.from_spec(spec)
    plan = experiment.spec
    expected = (
        len(plan.cases)
        * len(plan.theta_points)
        * (LIVE_REPEATS + 4 * len(plan.parameter_names()))
    )
    print(f"live budget: expected {expected} attempts, cap {LIVE_BUDGET_ATTEMPTS}")
    captures = experiment.run(adapter, live=True, output=OUTPUT / "live")
    print(f"live traces: {len(captures.traces)}; incomplete: {captures.incomplete}")
    if captures.incomplete:
        return 3
    analysis = analyze(captures, adapter=adapter)
    render_report(
        analysis,
        output=OUTPUT / "live" / "report.html",
        run_directory=OUTPUT / "live",
    )
    print("live report written to", OUTPUT / "live" / "report.html")
    return 0


def run_replay(run_directory: Path) -> int:
    from jevometry.adapters.replay import ReplayAdapter
    from jevometry.pipeline import load_run_experiment

    if not run_directory.exists():
        print(f"{run_directory} does not exist", file=sys.stderr)
        return 2
    adapter = TicketSystemAdapter(system_id="ticket-sensitivity")
    experiment = load_run_experiment(run_directory)
    replay = ReplayAdapter(
        run_directory,
        system_id=experiment.spec.id,
        model=experiment.spec.model,
    )
    captures = experiment.run(replay, output=OUTPUT / "replayed")
    analysis = analyze(captures, adapter=adapter)
    add_policy_metrics(analysis, adapter)
    render_report(
        analysis,
        output=OUTPUT / "replayed" / "report.html",
        run_directory=OUTPUT / "replayed",
    )
    print(f"replayed provider: {replay.provider}")
    print("replayed traces:", len(captures.traces))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="run the minimal live grid")
    parser.add_argument("--model", default=os.environ.get("TYPESAFE_MODEL"), help="model id")
    parser.add_argument("--replay", type=Path, help="replay a recorded run directory")
    args = parser.parse_args(argv)
    if args.replay is not None:
        return run_replay(args.replay)
    if args.live:
        if not args.model:
            print("--model or TYPESAFE_MODEL is required for --live", file=sys.stderr)
            return 2
        return run_live(args.model)
    return run_offline()


if __name__ == "__main__":
    sys.exit(main())
