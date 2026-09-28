"""Run the three-agent data-cleaning council completely offline."""

from pathlib import Path

import numpy as np
from system import build_system

from jevometry import Experiment, analyze, render_report
from jevometry.adapters import AnalyticAdapter
from jevometry.schemas.common import RoutingSemantics
from jevometry.schemas.experiment import CaseSpec, ExperimentSpec
from jevometry.schemas.parameters import ParameterSet, ParameterSpec, StencilSpec

OUTPUT = Path(__file__).parent / "output"
CASE = {"column": "monthly_income_usd", "before": " 1,200 ", "candidate": 1200}


def build_experiment() -> ExperimentSpec:
    return ExperimentSpec(
        id="three-agent-cleaning",
        model="analytic-three-agent-cleaning-v1",
        adapter="analytic",
        parameter_set=ParameterSet(
            parameters=[
                ParameterSpec(
                    name="evidence",
                    role="task_relevant",
                    unit="strength",
                    bounds=(0.0, 1.0),
                    step=0.01,
                    center=0.7,
                ),
                ParameterSpec(
                    name="ambiguity",
                    role="task_relevant",
                    unit="degree",
                    bounds=(0.0, 1.0),
                    step=0.01,
                    center=0.3,
                ),
            ]
        ),
        stencil=StencilSpec(),
        cases=[CaseSpec(id="income-cleaning", state=CASE)],
        theta_points=[{"evidence": 0.7, "ambiguity": 0.3}],
    )


def main() -> None:
    nodes, joint_model, final_policy = build_system()
    adapter = AnalyticAdapter(
        system_id="three-agent-cleaning",
        nodes=nodes,
        joint_model=joint_model,
        routing_semantics=RoutingSemantics.SAMPLED_OUTCOME,
    )
    experiment = Experiment.from_spec(build_experiment())
    captures = experiment.run(adapter, output=OUTPUT)
    analysis = analyze(captures, adapter=adapter, aggregation=final_policy)
    render_report(analysis, output=OUTPUT / "report.html", run_directory=OUTPUT)

    for node in analysis.document.nodes:
        fisher = node.geometry.fisher if node.geometry is not None else None
        assert fisher is not None and fisher.values is not None
        print(f"{node.node_id}: Fisher trace {np.trace(fisher.values):.3f}, rank {fisher.rank}")
    system = analysis.document.system
    assert system is not None
    system_trace = next(
        metric.value for metric in system.metrics if metric.name == "system_fisher_trace"
    )
    print(f"System Fisher trace: {system_trace:.3f}")
    print(f"Information loss trace: {system.information_loss[0].value:.3f}")
    assert analysis.joint is not None
    actions: dict[str, float] = {}
    for outcome, mass in zip(analysis.joint.outcomes, analysis.joint.probabilities, strict=True):
        action = final_policy.apply(outcome)
        actions[action] = actions.get(action, 0.0) + mass
    for action, mass in sorted(actions.items()):
        print(f"{action}: {mass:.3f}")
    print(f"Report: {OUTPUT / 'report.html'}")


if __name__ == "__main__":
    main()
