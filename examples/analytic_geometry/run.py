"""Example A: analytic geometry laboratory (fully offline).

Demonstrates:

* logistic Bernoulli q(theta) = sigmoid(theta) with analytic G = q(1-q);
* a two-parameter softmax with analytic Jacobian and Fisher information;
* a rank-deficient family q(theta1 + theta2) with an explicit null direction;
* a fixed aggregation mapping and the resulting information loss;
* distribution distances, coordinate transforms and step stability.

Run from the repository root:

    uv run python examples/analytic_geometry/run.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from jevometry import Experiment, analyze, render_report
from jevometry.adapters.analytic import (
    AnalyticAdapter,
    logistic_node,
    rank_deficient_softmax_node,
    single_node_joint_model,
    softmax_node,
)
from jevometry.schemas.common import RoutingSemantics
from jevometry.schemas.experiment import CaseSpec, ExperimentSpec
from jevometry.schemas.parameters import ParameterSet, ParameterSpec, StencilSpec
from jevometry.systems.pushforward import AggregationMap

OUTPUT = Path(__file__).parent / "output"


def logistic_experiment() -> tuple[Experiment, AnalyticAdapter]:
    node = logistic_node("risk", parameter="theta", coordinate="logit")
    adapter = AnalyticAdapter(
        system_id="analytic-logistic",
        nodes={"risk": node},
        routing_semantics=RoutingSemantics.SAMPLED_OUTCOME,
        description="single logistic node",
    )
    spec = ExperimentSpec(
        id="analytic-logistic",
        description="Logistic Bernoulli q(theta)=sigmoid(theta)",
        model="analytic",
        adapter="analytic",
        parameter_set=ParameterSet(
            parameters=[
                ParameterSpec(
                    name="theta",
                    role="task_relevant",
                    unit="logit",
                    bounds=(-8.0, 8.0),
                    scale=1.0,
                    step=0.01,
                    center=0.0,
                )
            ]
        ),
        stencil=StencilSpec(step_scales=[1.0, 0.5]),
        cases=[CaseSpec(id="case", state="analytic case")],
        theta_points=[{"theta": -1.0}, {"theta": 0.0}, {"theta": 1.0}],
    )
    return Experiment.from_spec(spec), adapter


def softmax_experiment() -> tuple[Experiment, AnalyticAdapter]:
    softmax = softmax_node(
        "topic",
        outcomes=("billing", "technical", "other"),
        parameters=("theta1", "theta2"),
        weights=[[1.0, 0.2], [0.1, 1.0], [-1.1, -1.2]],
        question=None,
    )
    rank_deficient = rank_deficient_softmax_node("degenerate")
    adapter = AnalyticAdapter(
        system_id="analytic-softmax",
        nodes={"topic": softmax, "degenerate": rank_deficient},
        routing_semantics=RoutingSemantics.SAMPLED_OUTCOME,
        description="two-parameter softmax plus a rank-deficient family",
    )
    spec = ExperimentSpec(
        id="analytic-softmax",
        description="Softmax with an analytic Jacobian and a rank-deficient sibling",
        model="analytic",
        adapter="analytic",
        parameter_set=ParameterSet(
            parameters=[
                ParameterSpec(
                    name="theta1",
                    role="task_relevant",
                    unit="dimensionless",
                    bounds=(-5.0, 5.0),
                    scale=1.0,
                    step=0.005,
                    center=0.0,
                ),
                ParameterSpec(
                    name="theta2",
                    role="task_relevant",
                    unit="dimensionless",
                    bounds=(-5.0, 5.0),
                    scale=1.0,
                    step=0.005,
                    center=0.0,
                ),
            ]
        ),
        stencil=StencilSpec(step_scales=[1.0, 0.5]),
        cases=[CaseSpec(id="case", state="analytic case")],
        theta_points=[{"theta1": 0.2, "theta2": -0.1}],
    )
    return Experiment.from_spec(spec), adapter


def aggregation_experiment() -> tuple[Experiment, AnalyticAdapter, AggregationMap]:
    softmax = softmax_node(
        "verdict",
        outcomes=("approve", "approve_with_review", "reject"),
        parameters=("theta1", "theta2"),
        weights=[[1.0, 0.0], [0.4, 0.4], [-0.8, -0.6]],
        question=None,
    )
    joint = single_node_joint_model(softmax)
    adapter = AnalyticAdapter(
        system_id="analytic-aggregation",
        nodes={"verdict": softmax},
        joint_model=joint,
        routing_semantics=RoutingSemantics.SAMPLED_OUTCOME,
        description="single node with a declared joint for pushforward analysis",
    )
    spec = ExperimentSpec(
        id="analytic-aggregation",
        description="Fixed aggregation of a categorical output",
        model="analytic",
        adapter="analytic",
        parameter_set=ParameterSet(
            parameters=[
                ParameterSpec(
                    name="theta1",
                    role="task_relevant",
                    unit="dimensionless",
                    bounds=(-5.0, 5.0),
                    scale=1.0,
                    step=0.005,
                    center=0.0,
                ),
                ParameterSpec(
                    name="theta2",
                    role="task_relevant",
                    unit="dimensionless",
                    bounds=(-5.0, 5.0),
                    scale=1.0,
                    step=0.005,
                    center=0.0,
                ),
            ]
        ),
        stencil=StencilSpec(step_scales=[1.0, 0.5]),
        cases=[CaseSpec(id="case", state="analytic case")],
        theta_points=[{"theta1": 0.1, "theta2": 0.2}],
    )
    aggregation = AggregationMap(
        name="binary_decision",
        mapping={
            ("approve",): "approve",
            ("approve_with_review",): "approve",
            ("reject",): "reject",
        },
        description="collapse the two approval outcomes into one",
    )
    return Experiment.from_spec(spec), adapter, aggregation


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    print("== A1. logistic Bernoulli ==")
    experiment, adapter = logistic_experiment()
    captures = experiment.run(adapter, output=OUTPUT / "logistic")
    analysis = analyze(captures, adapter=adapter)
    for node in analysis.document.nodes:
        if node.geometry and node.geometry.fisher:
            print(
                f"  theta={node.theta['theta']:+.1f}  "
                f"G={node.geometry.fisher.values[0][0]:.6f}  "
                f"rank={node.geometry.fisher.rank}"
            )
    render_report(
        analysis,
        output=OUTPUT / "logistic" / "report.html",
        run_directory=OUTPUT / "logistic",
    )

    print("== A2. softmax and rank deficiency ==")
    experiment, adapter = softmax_experiment()
    captures = experiment.run(adapter, output=OUTPUT / "softmax")
    analysis = analyze(captures, adapter=adapter)
    for node in analysis.document.nodes:
        if node.geometry and node.geometry.fisher:
            fisher = node.geometry.fisher
            print(
                f"  {node.node_id}: rank={fisher.rank} "
                f"cond={fisher.condition_number!r} "
                f"null_directions={len(fisher.null_directions)}"
            )
            if fisher.null_directions:
                print(f"    null direction: {np.round(fisher.null_directions[0], 6)}")
    render_report(
        analysis,
        output=OUTPUT / "softmax" / "report.html",
        run_directory=OUTPUT / "softmax",
    )

    print("== A3. fixed aggregation and information loss ==")
    experiment, adapter, aggregation = aggregation_experiment()
    captures = experiment.run(adapter, output=OUTPUT / "aggregation")
    analysis = analyze(captures, adapter=adapter, aggregation=aggregation)
    system = analysis.document.system
    assert system is not None
    for metric in system.information_loss:
        print(f"  {metric.name}: {metric.value!r} status={metric.status.value}")
    render_report(
        analysis,
        output=OUTPUT / "aggregation" / "report.html",
        run_directory=OUTPUT / "aggregation",
    )

    print("reports written under", OUTPUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
