"""Report rendering from computed results only."""

from __future__ import annotations

import json

from jevometry import Experiment, analyze, run_inference
from jevometry.adapters.analytic import (
    AnalyticAdapter,
    bernoulli_likelihood,
    bernoulli_node,
    bernoulli_pair_joint_model,
)
from jevometry.reporting import build_report_model, render_html, render_markdown
from jevometry.schemas.common import RoutingSemantics
from jevometry.schemas.experiment import (
    CaseSpec,
    ExperimentSpec,
    IdentifiabilitySource,
    ObservationRelation,
    SamplingContract,
    SamplingKind,
)
from jevometry.schemas.parameters import ParameterSet, ParameterSpec, StencilSpec


def spec() -> ExperimentSpec:
    return ExperimentSpec(
        id="reporting",
        model="analytic-bernoulli",
        adapter="analytic",
        parameter_set=ParameterSet(
            parameters=[
                ParameterSpec(
                    name="p",
                    role="task_relevant",
                    unit="probability",
                    bounds=(0.0, 1.0),
                    step=1e-4,
                    center=0.3,
                )
            ]
        ),
        stencil=StencilSpec(step_scales=[1.0, 0.5]),
        cases=[CaseSpec(id="case", state="SENSITIVE-RAW-TEXT")],
        theta_points=[{"p": 0.3}, {"p": 0.5}],
    )


def build() -> tuple:
    adapter = AnalyticAdapter(
        system_id="reporting",
        nodes={"Y": bernoulli_node("Y", parameter="p")},
        joint_model=bernoulli_pair_joint_model("p", mode="deterministic_copy"),
        routing_semantics=RoutingSemantics.SAMPLED_OUTCOME,
        likelihood=bernoulli_likelihood("p"),
    )
    experiment = Experiment.from_spec(spec())
    captures = experiment.run(adapter)
    analysis = analyze(captures, adapter=adapter)
    contract = SamplingContract(
        observable="Y",
        observation_unit="one draw",
        sampling=SamplingKind.IID,
        sample_size=100,
        relation_to_jev=ObservationRelation.SYNTHETIC_SIMULATION,
        identifiability_source=IdentifiabilitySource.ANALYTIC,
        fixed_support=True,
        differentiable=True,
        locally_identifiable=True,
        estimand="p",
    )
    inference = run_inference(captures, adapter=adapter, contract=contract)
    return analysis, inference


def test_report_model_exposes_capabilities_and_distributions() -> None:
    analysis, _ = build()
    model = build_report_model(analysis)
    capabilities = {entry["capability"] for entry in model.capability}
    assert {"distribution_comparison", "node_geometry", "system_information", "inference"} <= (
        capabilities
    )
    assert model.distributions
    assert model.identity["provider_mode"] == "offline"
    assert model.identity["nodes"] == ["Y"]
    assert model.completeness["total_traces"] > 0


def test_markdown_report_contains_all_sections() -> None:
    analysis, inference = build()
    markdown = render_markdown(analysis, inference=inference.model_dump(mode="json"))
    for heading in (
        "# Jevometry report",
        "## Capability matrix",
        "## Node results",
        "## Distribution distances",
        "## System composition",
        "## Inference",
    ):
        assert heading in markdown
    assert "SENSITIVE-RAW-TEXT" not in markdown
    assert "crlb" in markdown


def test_html_report_is_self_contained_and_hides_raw_text() -> None:
    analysis, inference = build()
    html = render_html(analysis, inference=inference.model_dump(mode="json"))
    import re

    assert "Plotly.newPlot" in html
    assert "plotly" in html.lower()
    assert not re.search(r'src="https?://', html)
    assert not re.search(r'href="https?://', html)
    assert "SENSITIVE-RAW-TEXT" not in html
    assert "capability matrix" in html.lower()
    assert "not a confidence ellipse" in html


def test_report_without_system_or_inference() -> None:
    adapter = AnalyticAdapter(system_id="node-only", nodes={"Y": bernoulli_node("Y")})
    experiment = Experiment.from_spec(spec())
    captures = experiment.run(adapter)
    analysis = analyze(captures, adapter=adapter)
    html = render_html(analysis)
    markdown = render_markdown(analysis)
    assert "no_joint_model" in html
    assert "no_joint_model" in markdown
    assert "No inference result is attached" in markdown


def test_report_json_payload_is_serialisable() -> None:
    analysis, _ = build()
    model = build_report_model(analysis)
    payload = {
        "identity": model.identity,
        "capability": model.capability,
        "completeness": model.completeness,
        "system": model.system,
    }
    assert "NaN" not in json.dumps(payload, default=str)
