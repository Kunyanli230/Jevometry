"""Public orchestration: Experiment, analyze, inference and reporting entry points."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from jevometry.adapters.base import SystemAdapter
from jevometry.analysis import Analysis, analyze
from jevometry.artifacts.store import RunStore
from jevometry.config import ExperimentConfig, build_adapter, load_config
from jevometry.experiments.acquisition import Captures, run_experiment
from jevometry.experiments.cache import CaptureCache
from jevometry.geometry.fisher import fisher_pullback
from jevometry.inference.contracts import check_contract
from jevometry.inference.crlb import CrlbResult, crlb_iid
from jevometry.inference.simulation import SimulationConfig, run_simulation
from jevometry.schemas.common import AnalysisObject, Diagnostic, MetricStatus
from jevometry.schemas.experiment import ExperimentSpec, SamplingContract
from jevometry.schemas.results import InferenceResult


@dataclass
class Experiment:
    """A resolved experiment specification plus its optional on-disk config."""

    spec: ExperimentSpec
    config: ExperimentConfig | None = None
    source_path: Path | None = None

    @classmethod
    def from_spec(cls, spec: ExperimentSpec) -> Experiment:
        return cls(spec=spec)

    @classmethod
    def from_yaml(cls, path: Path | str) -> Experiment:
        source = Path(path)
        config = load_config(source)
        return cls(spec=config.to_spec(), config=config, source_path=source)

    def build_adapter(self) -> SystemAdapter:
        if self.config is None:
            raise ValueError("no configuration file is attached to this experiment")
        return build_adapter(self.config)

    def run(
        self,
        adapter: SystemAdapter | None = None,
        *,
        live: bool = False,
        output: Path | str | None = None,
        repeats: int | None = None,
        cache_dir: Path | str | None = None,
        use_cache: bool = False,
        include_stencil: bool = True,
    ) -> Captures:
        """Execute the acquisition plan and optionally persist captures."""
        resolved_adapter = adapter or self.build_adapter()
        store: RunStore | None = None
        if output is not None:
            store = RunStore.create(
                Path(output),
                experiment=self.spec,
                provider_mode="live" if live else "offline",
                live=live,
                budget_summary=self.spec.budget.model_dump(mode="json"),
                adapter=self.spec.adapter or getattr(resolved_adapter, "provider_name", None),
                system=resolved_adapter.describe().model_dump(mode="json"),
                questions=_questions_payload(resolved_adapter),
                config_payload=self.config.model_dump(mode="json", exclude_none=True)
                if self.config is not None
                else None,
            )
        cache = CaptureCache(Path(cache_dir)) if cache_dir is not None else None
        captures = run_experiment(
            self.spec,
            resolved_adapter,
            live=live,
            repeats=repeats,
            cache=cache if use_cache else None,
            include_stencil=include_stencil,
            store=store,
        )
        if store is not None:
            store.write_traces(captures.traces)
            store.finalize_checksums()
        return captures


def _questions_payload(adapter: SystemAdapter) -> dict[str, Any] | None:
    questions_fn = getattr(adapter, "questions", None)
    if not callable(questions_fn):
        return None
    questions = questions_fn()
    if not questions:
        return None
    return {
        "questions": [
            {
                "id": question.id,
                "primitive": question.primitive.value,
                "semantic_hash": question.semantic_hash(),
            }
            for question in questions.values()
        ]
    }


def load_captures(run_directory: Path | str) -> Captures:
    """Reconstruct the capture bundle stored in a run directory."""
    import yaml

    store = RunStore.load(Path(run_directory))
    manifest = store.manifest
    assert manifest is not None
    spec = ExperimentSpec.model_validate(
        yaml.safe_load(store.read_text("experiment.resolved.yaml"))
    )
    traces = store.read_traces()
    system: dict[str, Any] = {}
    if store.path("system.json").exists():
        system = store.read_json("system.json")
    return Captures(
        experiment=spec,
        traces=traces,
        provider_mode=manifest.provider_mode,
        live=manifest.live,
        run_id=manifest.run_id,
        started_utc=manifest.created_utc,
        finished_utc=manifest.updated_utc or manifest.created_utc,
        incomplete=any(not trace.status.ok for trace in traces),
        notes=[],
        plan_summary=manifest.budget,
        system=system,
    )


def load_run_experiment(run_directory: Path | str) -> Experiment:
    """Load the resolved experiment (and config when available) from a run."""
    store = RunStore.load(Path(run_directory))
    config_path = store.path("config.resolved.yaml")
    if config_path.exists():
        config = load_config(config_path)
        return Experiment(spec=config.to_spec(), config=config, source_path=config_path)
    return Experiment.from_yaml(store.path("experiment.resolved.yaml"))


def run_inference(
    captures: Captures,
    *,
    adapter: SystemAdapter | None = None,
    contract: SamplingContract | None = None,
    node_id: str | None = None,
    simulate: Mapping[str, Any] | None = None,
) -> InferenceResult:
    """Compute a CRLB under an explicit sampling contract, or refuse clearly."""
    experiment = captures.experiment
    parameter_names = experiment.parameter_names()
    resolved_contract = contract or experiment.sampling_contract
    analysis_object = AnalysisObject.EMPIRICAL_OBSERVATION_MODEL
    if (
        resolved_contract is not None
        and resolved_contract.relation_to_jev is not None
        and resolved_contract.relation_to_jev.value == "reported_distribution_draw"
    ):
        analysis_object = AnalysisObject.REPORTED_DISTRIBUTION
    eligibility = check_contract(
        resolved_contract,
        parameter_names=parameter_names,
        analysis_object=analysis_object,
    )
    resolved_node = node_id or (captures.node_ids()[0] if captures.node_ids() else None)
    if resolved_contract is None:
        return _inference_refusal(
            captures,
            contract=None,
            parameter_names=parameter_names,
            reason_code=eligibility.reason_code or "missing_sampling_contract",
            remedy=eligibility.remedy,
        )
    if not eligibility.eligible:
        return _inference_refusal(
            captures,
            contract=resolved_contract,
            parameter_names=parameter_names,
            reason_code=eligibility.reason_code or "ineligible",
            remedy=eligibility.remedy,
            missing=eligibility.missing,
        )
    if resolved_node is None:
        return _inference_refusal(
            captures,
            contract=resolved_contract,
            parameter_names=parameter_names,
            reason_code="no_nodes",
            remedy="capture at least one node before inference",
        )
    information = _observation_information(captures, adapter, resolved_node)
    if information is None:
        return _inference_refusal(
            captures,
            contract=resolved_contract,
            parameter_names=parameter_names,
            reason_code="missing_information_matrix",
            remedy=(
                "inference needs a declared observation model with an analytic Jacobian or a "
                "provided Fisher information matrix"
            ),
        )
    sample_size = resolved_contract.sample_size or 1
    crlb: CrlbResult = crlb_iid(
        information, sample_size, parameter_names=parameter_names
    )
    result = InferenceResult(
        run_id=captures.run_id,
        parameter_names=parameter_names,
        estimand=resolved_contract.estimand or parameter_names[0],
        contract=resolved_contract,
        analysis_object=analysis_object,
        crlb=crlb.crlb.tolist() if crlb.crlb is not None else None,
        crlb_units=f"1/({', '.join(parameter_names)})^2",
        standard_errors=crlb.standard_errors.tolist()
        if crlb.standard_errors is not None
        else None,
        status=crlb.status,
        reason_code=crlb.reason_code,
        remedy=crlb.remedy,
        method=crlb.method,
        assumptions=[*eligibility.assumptions, *crlb.assumptions],
        diagnostics=[
            *crlb.diagnostics,
            Diagnostic(
                code="identifiability_source",
                message=(
                    "identifiability source: "
                    f"{resolved_contract.identifiability_source.value if resolved_contract.identifiability_source else 'unknown'}"
                ),
                severity="info",
            ),
        ],
    )
    if simulate is not None and crlb.crlb is not None:
        likelihood = getattr(adapter, "likelihood_model", None) if adapter is not None else None
        if callable(likelihood):
            likelihood = likelihood(resolved_node)
        if likelihood is None:
            result.diagnostics.append(
                Diagnostic(
                    code="simulation_skipped",
                    message="no likelihood model is available; simulation was not run",
                    severity="warning",
                )
            )
        else:
            estimand = result.estimand
            index = (
                parameter_names.index(estimand) if estimand in parameter_names else 0
            )
            config = SimulationConfig(
                parameter=estimand,
                true_value=float(experiment.center()[estimand]),
                sample_size=int(simulate.get("sample_size", sample_size)),
                replications=int(simulate.get("replications", 100)),
                seed=int(simulate.get("seed", experiment.seed)),
                confidence_level=float(simulate.get("confidence_level", 0.95)),
                other_parameters={
                    name: float(value)
                    for name, value in experiment.center().items()
                    if name != estimand
                },
            )
            result.simulations.append(
                run_simulation(
                    likelihood,
                    config,
                    parameter_specs=experiment.parameters(),
                    crlb_variance=float(crlb.crlb[index, index]),
                )
            )
    return result


def _observation_information(
    captures: Captures, adapter: SystemAdapter | None, node_id: str
) -> np.ndarray[Any, np.dtype[np.float64]] | None:
    if adapter is None:
        return None
    theta = captures.experiment.center()
    analytic = getattr(adapter, "analytic_jacobian", None)
    if not callable(analytic):
        return None
    jacobian = analytic(node_id, theta)
    if jacobian is None:
        return None
    evaluator = getattr(adapter, "node_evaluator", None)
    if not callable(evaluator):
        return None
    evaluation = evaluator(node_id)(theta)
    if evaluation.status is not MetricStatus.OK:
        return None
    fisher = fisher_pullback(
        np.asarray(jacobian, dtype=np.float64),
        np.asarray(evaluation.probabilities, dtype=np.float64),
        support=evaluation.support,
    )
    if fisher.values is None:
        return None
    return fisher.values


def _inference_refusal(
    captures: Captures,
    *,
    contract: SamplingContract | None,
    parameter_names: Sequence[str],
    reason_code: str,
    remedy: str | None,
    missing: Sequence[str] | None = None,
) -> InferenceResult:
    return InferenceResult(
        run_id=captures.run_id,
        parameter_names=list(parameter_names),
        estimand=(contract.estimand if contract is not None and contract.estimand else "unknown"),
        contract=contract or SamplingContract(),
        status=MetricStatus.UNSUPPORTED
        if reason_code != "not_identifiable"
        else MetricStatus.NOT_IDENTIFIABLE,
        reason_code=reason_code,
        remedy=remedy,
        diagnostics=[
            Diagnostic(
                code=reason_code,
                message=remedy or reason_code,
                severity="warning",
                details={"missing": list(missing or [])},
            )
        ],
    )


def render_report(
    analysis: Analysis,
    *,
    output: Path | str,
    markdown_output: Path | str | None = None,
    run_directory: Path | str | None = None,
    inference: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Render the self-contained offline HTML report and its Markdown twin."""
    from jevometry.reporting import render_html, render_markdown

    store: RunStore | None = None
    if run_directory is not None:
        store = RunStore.load(Path(run_directory))
        if analysis.document.analysis_revision == 0:
            analysis.document = analysis.document.model_copy(
                update={"analysis_revision": store.next_analysis_revision()}
            )
    html_path = Path(output)
    html_payload = dict(inference) if inference is not None else None
    html = render_html(analysis, inference=html_payload)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html, encoding="utf-8")
    markdown_path = (
        Path(markdown_output)
        if markdown_output is not None
        else html_path.with_suffix(".md")
    )
    markdown_payload = dict(inference) if inference is not None else None
    markdown = render_markdown(analysis, inference=markdown_payload)
    markdown_path.write_text(markdown, encoding="utf-8")
    if store is not None:
        store.write_analysis(
            analysis.document,
            arrays=analysis.arrays,
            report_html=html,
            report_markdown=markdown,
        )
    return html_path, markdown_path


__all__ = [
    "Analysis",
    "Experiment",
    "analyze",
    "render_report",
    "run_inference",
]
