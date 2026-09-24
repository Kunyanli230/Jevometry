"""Experiment configuration files and adapter construction.

A configuration file is declarative: it names parameters, cases, theta points,
the renderer, the questions and the provider.  Building an adapter never makes
a request; ``run`` decides whether requests are permitted.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, model_validator

from jevometry.adapters.base import SystemAdapter
from jevometry.experiments.renderer import (
    ParameterRenderer,
    StructuredRenderer,
    TextTemplateRenderer,
)
from jevometry.schemas.common import RoutingSemantics, StrictModel
from jevometry.schemas.experiment import (
    Budget,
    CaseSpec,
    ExperimentSpec,
    SamplingContract,
)
from jevometry.schemas.parameters import ParameterSet, ParameterSpec, StencilSpec
from jevometry.schemas.questions import QuestionSpec


class RendererConfig(StrictModel):
    """Declarative renderer selection."""

    kind: Literal["text", "structured"]
    template: str | dict[str, str]
    static: dict[str, Any] | None = None
    version: str = "1"

    @model_validator(mode="after")
    def _validate_template(self) -> RendererConfig:
        if self.kind == "text" and not isinstance(self.template, str):
            raise ValueError("text renderers require a string template")
        if self.kind == "structured" and not isinstance(self.template, dict):
            raise ValueError("structured renderers require a mapping template")
        return self


class ProviderConfig(StrictModel):
    """Declarative provider selection."""

    kind: Literal["typesafe", "replay", "module", "analytic", "none"] = "none"
    model: str | None = None
    entrypoint: str | None = None
    run_directory: str | None = None
    system_id: str | None = None
    routing_semantics: RoutingSemantics = RoutingSemantics.NOT_APPLICABLE
    batch_questions: bool = True
    endpoint: str | None = None
    replay_source_provider: str = "typesafe"
    replay_source_model: str | None = None
    replay_source_adapter_version: str | None = None

    @model_validator(mode="after")
    def _validate(self) -> ProviderConfig:
        if self.kind == "module" and not self.entrypoint:
            raise ValueError("module providers require an entrypoint 'module:function'")
        if self.kind == "replay" and not self.run_directory:
            raise ValueError("replay providers require run_directory")
        if self.kind == "typesafe" and not self.model:
            raise ValueError("typesafe providers require a concrete model id")
        return self


class ExperimentConfig(StrictModel):
    """The on-disk experiment file."""

    schema_version: str = "1.0"
    id: str
    description: str | None = None
    model: str | None = None
    adapter: str | None = None
    parameters: list[ParameterSpec]
    stencil: StencilSpec = Field(default_factory=StencilSpec)
    cases: list[CaseSpec]
    theta_points: list[dict[str, float]]
    budget: Budget = Field(default_factory=Budget)
    seed: int = 0
    sampling_contract: SamplingContract | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    renderer: RendererConfig | None = None
    questions: list[QuestionSpec] = Field(default_factory=list)
    node_questions: dict[str, str] = Field(default_factory=dict)
    provider: ProviderConfig = Field(default_factory=ProviderConfig)

    def to_spec(self) -> ExperimentSpec:
        return ExperimentSpec(
            schema_version=self.schema_version,
            id=self.id,
            description=self.description,
            model=self.model,
            adapter=self.adapter,
            parameter_set=ParameterSet(parameters=self.parameters),
            stencil=self.stencil,
            cases=self.cases,
            theta_points=self.theta_points,
            budget=self.budget,
            seed=self.seed,
            sampling_contract=self.sampling_contract,
            metadata=self.metadata,
        )

    @model_validator(mode="after")
    def _validate_questions(self) -> ExperimentConfig:
        ids = {question.id for question in self.questions}
        for node_id, question_id in self.node_questions.items():
            if question_id not in ids:
                raise ValueError(
                    f"node {node_id!r} references unknown question {question_id!r}"
                )
        return self


def load_config(path: Path) -> ExperimentConfig:
    """Load and validate an experiment YAML file."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return ExperimentConfig.model_validate(raw)


def dump_config(config: ExperimentConfig) -> str:
    return yaml.safe_dump(
        config.model_dump(mode="json", exclude_none=True), sort_keys=False, allow_unicode=True
    )


def build_renderer(config: ExperimentConfig) -> ParameterRenderer | None:
    """Build the declared renderer, or return None when no renderer is declared."""
    if config.renderer is None:
        return None
    if config.renderer.kind == "text":
        assert isinstance(config.renderer.template, str)
        return TextTemplateRenderer(
            config.renderer.template, version=config.renderer.version
        )
    assert isinstance(config.renderer.template, dict)
    return StructuredRenderer(
        config.renderer.template,
        static=config.renderer.static,
        version=config.renderer.version,
    )


def question_hashes(config: ExperimentConfig) -> dict[str, str]:
    by_id = {question.id: question for question in config.questions}
    return {
        node_id: by_id[question_id].semantic_hash()
        for node_id, question_id in config.node_questions.items()
        if question_id in by_id
    }


def build_adapter(config: ExperimentConfig) -> SystemAdapter:
    """Build the declared provider without making any request."""
    provider = config.provider
    renderer = build_renderer(config)
    if provider.kind == "typesafe":
        from jevometry.adapters.typesafe import TypeSafeAdapter, question_spec_to_typesafe

        if not config.questions:
            raise ValueError("typesafe providers require question declarations")
        by_id = {question.id: question for question in config.questions}
        nodes = {}
        for node_id, question_id in config.node_questions.items():
            spec = by_id[question_id]
            nodes[node_id] = question_spec_to_typesafe(spec, node_id=node_id)
        if not nodes:
            raise ValueError("typesafe providers require node_questions")
        return TypeSafeAdapter(
            model=provider.model or "",
            nodes=nodes,
            budget=config.budget,
            renderer=renderer,
            routing_semantics=provider.routing_semantics,
            batch_questions=provider.batch_questions,
            system_id=provider.system_id or config.id,
            endpoint=provider.endpoint,
        )
    if provider.kind == "replay":
        from jevometry.adapters.replay import ReplayAdapter

        return ReplayAdapter(
            Path(provider.run_directory or ""),
            system_id=provider.system_id or config.id,
            model=provider.replay_source_model or provider.model or config.model,
            adapter_version=provider.replay_source_adapter_version,
            provider=provider.replay_source_provider,
            question_hashes=question_hashes(config),
        )
    if provider.kind == "module":
        return _load_entrypoint(provider.entrypoint or "", config)
    if provider.kind == "analytic":
        raise ValueError(
            "analytic adapters are constructed in Python; use a module entrypoint "
            "that returns an AnalyticAdapter"
        )
    raise ValueError("no provider is declared in the experiment configuration")


def _load_entrypoint(entrypoint: str, config: ExperimentConfig) -> SystemAdapter:
    if ":" not in entrypoint:
        raise ValueError("entrypoint must look like 'package.module:function'")
    module_name, function_name = entrypoint.split(":", 1)
    working_directory = str(Path.cwd())
    if working_directory not in sys.path:
        sys.path.insert(0, working_directory)
    module = importlib.import_module(module_name)
    function = getattr(module, function_name)
    adapter = function(config)
    if not isinstance(adapter, SystemAdapter):
        raise TypeError(f"{entrypoint} did not return a SystemAdapter")
    return adapter


def init_config(directory: Path, *, experiment_id: str = "example") -> list[Path]:
    """Create a minimal analytic experiment directory."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": "1.0",
        "id": experiment_id,
        "description": "Minimal logistic experiment (offline analytic provider).",
        "model": "analytic-logistic",
        "adapter": "module",
        "parameters": [
            {
                "name": "theta",
                "role": "task_relevant",
                "unit": "logit",
                "bounds": [-8.0, 8.0],
                "scale": 1.0,
                "step": 0.01,
                "center": 0.0,
            }
        ],
        "stencil": {"kind": "central", "step_scales": [1.0, 0.5]},
        "cases": [{"id": "case-1", "state": "example state"}],
        "theta_points": [{"theta": 0.0}],
        "budget": {"repeats": 1, "max_attempts": 50},
        "renderer": {"kind": "text", "template": "theta={theta:.4f}"},
        "questions": [
            {
                "id": "q",
                "primitive": "noul",
                "outcomes": [
                    {"id": "false", "description": "no"},
                    {"id": "true", "description": "yes"},
                ],
            }
        ],
        "node_questions": {"node": "q"},
        "provider": {"kind": "module", "entrypoint": "example_adapter:build_adapter"},
    }
    config_path = directory / "experiment.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    adapter_path = directory / "example_adapter.py"
    adapter_path.write_text(_EXAMPLE_ADAPTER, encoding="utf-8")
    readme_path = directory / "README.md"
    readme_path.write_text(_INIT_README, encoding="utf-8")
    return [config_path, adapter_path, readme_path]


_EXAMPLE_ADAPTER = '''"""Example analytic adapter used by `jevometry init`."""

from jevometry.adapters.analytic import AnalyticAdapter, logistic_node


def build_adapter(config):
    node = logistic_node("node", parameter="theta", coordinate="logit")
    return AnalyticAdapter(system_id=config.id, nodes={"node": node})
'''

_INIT_README = """# Jevometry experiment

`experiment.yaml` declares the parameters, cases, theta points, renderer,
questions and provider.  `example_adapter.py` builds an offline analytic
adapter so that the whole pipeline can be exercised without credentials.

```bash
jevometry validate experiment.yaml
jevometry run experiment.yaml --output runs/example
jevometry analyze runs/example --output runs/example
jevometry report runs/example --output runs/example/report.html
```
"""


def summarize_questions(config: ExperimentConfig) -> Sequence[Mapping[str, Any]]:
    return [
        {
            "id": question.id,
            "primitive": question.primitive.value,
            "outcomes": [outcome.id for outcome in question.outcomes],
            "semantic_hash": question.semantic_hash(),
        }
        for question in config.questions
    ]
