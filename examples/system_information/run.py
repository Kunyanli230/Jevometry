"""Example B: multi-node information, redundancy and inference (fully offline).

Demonstrates:

* one Bernoulli outcome Y with analytic Fisher information 1/[p(1-p)];
* a deterministic copy Z=Y whose declared joint has *the same* information;
* two independent draws whose declared joint has exactly twice the information;
* a two-layer conditional tree checked against the conditional information
  identity ``I_trajectory = sum_i E_history[I_i(theta | history)]``;
* a synthetic MLE experiment compared with the analytic CRLB p(1-p)/n.

Run from the repository root:

    uv run python examples/system_information/run.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from jevometry import Experiment, analyze, render_report, run_inference
from jevometry.adapters.analytic import (
    AnalyticAdapter,
    bernoulli_likelihood,
    bernoulli_node,
    bernoulli_pair_joint_model,
    conditional_tree_joint_model,
)
from jevometry.schemas.common import (
    AnalysisObject,
    Diagnostic,
    MetricResult,
    RoutingSemantics,
    ValueKind,
)
from jevometry.schemas.experiment import (
    CaseSpec,
    ExperimentSpec,
    IdentifiabilitySource,
    ObservationRelation,
    SamplingContract,
    SamplingKind,
)
from jevometry.schemas.parameters import ParameterSet, ParameterSpec, StencilSpec
from jevometry.systems.tree import ConditionalNode, ConditionalTree, conditional_fisher_identity

OUTPUT = Path(__file__).parent / "output"


def two_layer_tree() -> ConditionalTree:
    """Y in {a,b} with P(b)=p; Z depends on theta with a shift after Y=b."""

    def y_distribution(theta, history):
        return {"a": 1.0 - theta["p"], "b": theta["p"]}

    def y_jacobian(theta, history):
        return {"a": {"p": -1.0}, "b": {"p": 1.0}}

    def z_distribution(theta, history):
        from scipy.special import expit

        shift = 1.0 if history["Y"] == "b" else 0.0
        probability = float(expit(theta["p"] + shift))
        return {"0": 1.0 - probability, "1": probability}

    def z_jacobian(theta, history):
        from scipy.special import expit

        shift = 1.0 if history["Y"] == "b" else 0.0
        probability = float(expit(theta["p"] + shift))
        derivative = probability * (1.0 - probability)
        return {"0": {"p": -derivative}, "1": {"p": derivative}}

    return ConditionalTree(
        tree_id="two-layer",
        nodes=[
            ConditionalNode("Y", ("a", "b"), y_distribution, y_jacobian),
            ConditionalNode("Z", ("0", "1"), z_distribution, z_jacobian),
        ],
    )


def p_parameter() -> ParameterSpec:
    return ParameterSpec(
        name="p",
        role="task_relevant",
        unit="probability",
        bounds=(0.0, 1.0),
        scale=1.0,
        step=1e-4,
        center=0.3,
    )


def base_experiment(experiment_id: str) -> ExperimentSpec:
    return ExperimentSpec(
        id=experiment_id,
        model="analytic-bernoulli",
        adapter="analytic",
        parameter_set=ParameterSet(parameters=[p_parameter()]),
        stencil=StencilSpec(step_scales=[1.0, 0.5]),
        cases=[CaseSpec(id="case", state="declared Bernoulli system")],
        theta_points=[{"p": 0.3}],
    )


def build_adapter(
    system_id: str, *, with_likelihood: bool = True, two_nodes: bool = False
) -> AnalyticAdapter:
    nodes = {"Y": bernoulli_node("Y", parameter="p")}
    if two_nodes:
        nodes["Z"] = bernoulli_node("Z", parameter="p")
    return AnalyticAdapter(
        system_id=system_id,
        nodes=nodes,
        routing_semantics=RoutingSemantics.SAMPLED_OUTCOME,
        likelihood=bernoulli_likelihood("p") if with_likelihood else None,
        description="Bernoulli node(s) with a declared joint model",
    )


def single_information(p: float) -> float:
    return 1.0 / (p * (1.0 - p))


def run_redundancy_case(name: str, mode: str) -> dict[str, float]:
    experiment = Experiment.from_spec(base_experiment(f"system-{name}"))
    adapter = build_adapter(f"system-{name}", two_nodes=True)
    joint_model = bernoulli_pair_joint_model("p", mode=mode, draws=2)
    captures = experiment.run(adapter, output=OUTPUT / name)
    analysis = analyze(captures, adapter=adapter, joint_model=joint_model)
    system = analysis.document.system
    assert system is not None and system.redundancy
    case = system.redundancy[0]
    render_report(
        analysis,
        output=OUTPUT / name / "report.html",
        run_directory=OUTPUT / name,
    )
    return {
        "independent_sum_trace": float(case.independent_sum_trace or 0.0),
        "joint_trace": float(case.joint_trace or 0.0),
        "difference": float(case.difference or 0.0),
    }


def run_tree_case() -> tuple[float, float, float]:
    tree = two_layer_tree()
    experiment = Experiment.from_spec(base_experiment("system-tree"))
    adapter = build_adapter("system-tree", with_likelihood=False)
    joint_model = conditional_tree_joint_model(tree)
    captures = experiment.run(adapter, output=OUTPUT / "tree")
    analysis = analyze(captures, adapter=adapter, joint_model=joint_model)
    identity = conditional_fisher_identity(tree, {"p": 0.3}, parameter_names=["p"])
    assert identity.total is not None and identity.joint_fisher is not None
    analysis.document.metrics.append(
        MetricResult.ok(
            "conditional_identity_residual",
            float(identity.residual or 0.0),
            analysis_object=AnalysisObject.DECLARED_SYSTEM_MODEL,
            value_kind=ValueKind.SCALAR,
            units="relative",
            assumptions=list(identity.assumptions),
            diagnostics={"joint_fisher_trace": float(np.trace(identity.joint_fisher))},
        )
    )
    analysis.document.diagnostics.append(
        Diagnostic(
            code="conditional_information_identity",
            message=(
                "exact joint Fisher equals the history-weighted sum of conditional "
                f"Fisher information; residual {identity.residual:.3e}"
            ),
            severity="info",
            details={
                "per_node_trace": {
                    node: float(np.trace(matrix)) for node, matrix in identity.per_node.items()
                }
            },
        )
    )
    render_report(
        analysis,
        output=OUTPUT / "tree" / "report.html",
        run_directory=OUTPUT / "tree",
    )
    return (
        float(identity.total[0, 0]),
        float(identity.joint_fisher[0, 0]),
        float(identity.residual or 0.0),
    )


def run_inference_case() -> dict[str, float | None]:
    adapter = build_adapter("system-inference")
    contract = SamplingContract(
        observable="Bernoulli outcome Y",
        observation_unit="one independent Bernoulli draw",
        sampling=SamplingKind.IID,
        sample_size=200,
        relation_to_jev=ObservationRelation.SYNTHETIC_SIMULATION,
        identifiability_source=IdentifiabilitySource.ANALYTIC,
        fixed_support=True,
        differentiable=True,
        locally_identifiable=True,
        estimand="p",
        notes=["synthetic validation of the declared observation model"],
    )
    spec = base_experiment("system-inference")
    spec.sampling_contract = contract
    experiment = Experiment.from_spec(spec)
    captures = experiment.run(adapter, output=OUTPUT / "inference")
    result = run_inference(
        captures,
        adapter=adapter,
        contract=contract,
        simulate={"sample_size": 200, "replications": 200, "seed": 20260924},
    )
    import json

    (OUTPUT / "inference").mkdir(parents=True, exist_ok=True)
    (OUTPUT / "inference" / "inference.json").write_text(
        json.dumps(result.model_dump(mode="json"), indent=2), encoding="utf-8"
    )
    analysis = analyze(captures, adapter=adapter)
    render_report(
        analysis,
        output=OUTPUT / "inference" / "report.html",
        markdown_output=OUTPUT / "inference" / "report.md",
        run_directory=OUTPUT / "inference",
        inference=result.model_dump(mode="json"),
    )
    assert result.crlb is not None
    assert result.simulations
    summary = result.simulations[0]
    return {
        "crlb_variance": float(result.crlb[0][0]),
        "empirical_variance": summary.variance,
        "coverage": summary.coverage,
        "rmse": summary.rmse,
        "failures": float(summary.failures),
    }


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    p = 0.3
    analytic_information = single_information(p)

    print("== B1. one Bernoulli outcome ==")
    experiment = Experiment.from_spec(base_experiment("system-single"))
    adapter = build_adapter("system-single")
    captures = experiment.run(adapter, output=OUTPUT / "single")
    analysis = analyze(captures, adapter=adapter)
    fisher = analysis.document.nodes[0].geometry.fisher
    assert fisher is not None and fisher.values is not None
    print(f"  analytic I(p) = {analytic_information:.6f}")
    print(f"  computed  I(p) = {fisher.values[0][0]:.6f}")

    print("== B2. deterministic copy Z = Y ==")
    copy = run_redundancy_case("copy", "deterministic_copy")
    print(f"  independent sum trace = {copy['independent_sum_trace']:.6f}")
    print(f"  declared joint trace  = {copy['joint_trace']:.6f}")
    assert abs(copy["joint_trace"] - analytic_information) < 1e-6
    assert copy["difference"] > 0.0

    print("== B3. two independent draws ==")
    independent = run_redundancy_case("independent", "independent")
    print(f"  independent sum trace = {independent['independent_sum_trace']:.6f}")
    print(f"  declared joint trace  = {independent['joint_trace']:.6f}")
    assert abs(independent["joint_trace"] - 2.0 * analytic_information) < 1e-6
    assert abs(independent["difference"]) < 1e-9

    print("== B4. conditional two-layer tree ==")
    total, joint, residual = run_tree_case()
    print(f"  conditional sum trace = {total:.6f}")
    print(f"  exact joint trace     = {joint:.6f}")
    print(f"  identity residual     = {residual:.3e}")
    assert residual < 1e-9

    print("== B5. synthetic MLE versus the analytic CRLB ==")
    inference = run_inference_case()
    print(f"  CRLB variance         = {inference['crlb_variance']:.8f}")
    print(f"  empirical variance    = {inference['empirical_variance']:.8f}")
    print(f"  coverage              = {inference['coverage']}")
    print(f"  failures              = {inference['failures']}")
    assert inference["crlb_variance"] is not None
    assert abs(float(inference["crlb_variance"]) - p * (1 - p) / 200) < 1e-12

    print("reports written under", OUTPUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
