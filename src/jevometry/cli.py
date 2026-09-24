"""Command line interface.

Exit codes:
  0  success (may include explicitly unsupported metrics)
  2  invalid configuration or data
  3  acquisition incomplete
  4  provider or authentication failure
  5  a core analysis the user requested cannot be computed
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import typer
import yaml

from jevometry import __version__
from jevometry.analysis import analyze as analyze_captures
from jevometry.artifacts.store import RunStore
from jevometry.compare import compare_runs, write_comparison
from jevometry.config import ExperimentConfig, build_adapter, load_config
from jevometry.experiments.acquisition import LiveModeError
from jevometry.pipeline import (
    Experiment,
    load_captures,
    load_run_experiment,
    render_report,
    run_inference,
)
from jevometry.schemas.common import MetricStatus
from jevometry.schemas.experiment import SamplingContract

app = typer.Typer(
    name="jevometry",
    help=(
        "Information-geometric analysis of Jev agent systems.\n\n"
        "Metrics may be partial: an unsupported metric is reported with a reason code "
        "and does not by itself make the command fail. Exit code 5 means a metric you "
        "explicitly requested could not be computed; exit code 0 means the command "
        "succeeded even if some optional metrics are unsupported."
    ),
    no_args_is_help=True,
    add_completion=False,
)


def _fail(message: str, code: int) -> None:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(code=code)


@app.command()
def init(
    directory: Path = typer.Argument(..., help="Directory to initialise"),
    experiment_id: str = typer.Option("example", help="Experiment id"),
) -> None:
    """Create an experiment configuration, an example adapter and instructions."""
    from jevometry.config import init_config

    if directory.exists() and any(directory.iterdir()):
        _fail(f"{directory} is not empty; refusing to overwrite", 2)
    paths = init_config(directory, experiment_id=experiment_id)
    for path in paths:
        typer.echo(str(path))
    typer.echo("Next: jevometry validate experiment.yaml")


@app.command()
def validate(config: Path = typer.Argument(..., exists=True, readable=True)) -> None:
    """Validate configuration, capability and request budget without calling any API."""
    try:
        experiment_config = load_config(config)
    except Exception as error:  # noqa: BLE001
        _fail(f"invalid configuration: {error}", 2)
        return
    experiment = Experiment.from_yaml(config)
    provider = experiment_config.provider
    typer.echo(f"schema_version: {experiment_config.schema_version}")
    typer.echo(f"experiment: {experiment_config.id}")
    typer.echo(f"parameters: {', '.join(experiment.spec.parameter_names())}")
    typer.echo(f"cases: {len(experiment_config.cases)}")
    typer.echo(f"theta_points: {len(experiment_config.theta_points)}")
    typer.echo(f"provider: {provider.kind}")
    if provider.kind == "typesafe" and not os.environ.get("TYPESAFE_API_KEY"):
        typer.echo("TYPESAFE_API_KEY: not set (live runs will be refused)")
    adapter = None
    if provider.kind != "none":
        try:
            adapter = build_adapter(experiment_config)
        except Exception as error:  # noqa: BLE001
            _fail(f"cannot build provider: {error}", 2)
            return
    node_ids = adapter.describe().node_ids() if adapter is not None else ["node"]
    from jevometry.experiments.design import build_acquisition_plan

    plan = build_acquisition_plan(
        experiment.spec,
        node_ids,
        live=provider.kind == "typesafe",
        include_stencil=True,
        requests_per_point=getattr(adapter, "requests_per_point", lambda: 1)()
        if adapter is not None
        else 1,
    )
    typer.echo(yaml.safe_dump(plan.summary(), sort_keys=False))
    if not plan.within_budget:
        _fail(
            "the acquisition plan exceeds max_attempts; raise the budget or reduce the grid",
            2,
        )


@app.command()
def run(
    config: Path = typer.Argument(..., exists=True, readable=True),
    output: Path = typer.Option(..., "--output", help="Run directory"),
    live: bool = typer.Option(False, "--live", help="Permit real provider requests"),
    repeats: int = typer.Option(0, help="Repeats (0 uses the configured budget)"),
    cache_dir: Path | None = typer.Option(None, help="Capture cache directory"),
    use_cache: bool = typer.Option(False, help="Use the capture cache for offline providers"),
    no_stencil: bool = typer.Option(False, help="Skip finite-difference stencil captures"),
) -> None:
    """Execute the acquisition plan (analytic/replay offline; TypeSafe with --live)."""
    try:
        experiment_config = load_config(config)
    except Exception as error:  # noqa: BLE001
        _fail(f"invalid configuration: {error}", 2)
        return
    provider = experiment_config.provider
    if provider.kind == "typesafe" and not live:
        _fail(
            "the configured provider makes real API requests; pass --live to confirm, "
            "or use a replay provider",
            4,
        )
    if live and provider.kind != "typesafe":
        _fail("--live is only valid for the typesafe provider", 2)
    experiment = Experiment.from_yaml(config)
    try:
        adapter = experiment.build_adapter()
    except Exception as error:  # noqa: BLE001
        _fail(f"cannot build provider: {error}", 4)
        return
    try:
        captures = experiment.run(
            adapter,
            live=live,
            output=output,
            repeats=repeats if repeats > 0 else None,
            cache_dir=cache_dir,
            use_cache=use_cache,
            include_stencil=not no_stencil,
        )
    except LiveModeError as error:
        _fail(str(error), 4)
        return
    except Exception as error:  # noqa: BLE001
        _fail(f"acquisition failed: {type(error).__name__}: {error}", 4)
        return
    typer.echo(f"run directory: {output}")
    typer.echo(f"traces: {len(captures.traces)}")
    if captures.incomplete:
        _fail(
            f"{len(captures.failures())} traces are incomplete; see traces.jsonl",
            3,
        )


@app.command()
def analyze(
    run_dir: Path = typer.Argument(..., exists=True, readable=True),
    output: Path | None = typer.Option(None, "--output", help="Optional export directory"),
    config: Path | None = typer.Option(
        None, "--config", help="Original config (enables analytic cross-checks and joints)"
    ),
) -> None:
    """Compute every metric the captured data supports (offline)."""
    try:
        captures = load_captures(run_dir)
    except Exception as error:  # noqa: BLE001
        _fail(f"cannot read run: {error}", 2)
        return
    adapter = None
    experiment_config: ExperimentConfig | None = None
    source = config
    if source is None:
        candidate = Path(run_dir) / "config.resolved.yaml"
        if candidate.exists():
            source = candidate
    if source is not None:
        try:
            experiment_config = load_config(source)
            adapter = build_adapter(experiment_config)
        except Exception as error:  # noqa: BLE001
            typer.echo(f"warning: adapter unavailable ({error}); analysing traces only", err=True)
            adapter = None
    try:
        analysis = analyze_captures(captures, adapter=adapter)
    except Exception as error:  # noqa: BLE001
        _fail(f"analysis failed: {type(error).__name__}: {error}", 2)
        return
    store = RunStore.load(run_dir)
    revision = store.next_analysis_revision()
    analysis.document.analysis_revision = revision
    store.write_analysis(analysis.document, arrays=analysis.arrays)
    store.finalize_checksums()
    typer.echo(f"analysis revision: {revision}")
    typer.echo(f"metrics: {run_dir / 'metrics.json'}")
    if output is not None and Path(output) != Path(run_dir):
        export = Path(output)
        export.mkdir(parents=True, exist_ok=True)
        (export / "metrics.json").write_text(
            json.dumps(analysis.to_json(), indent=2), encoding="utf-8"
        )
        typer.echo(f"exported: {export / 'metrics.json'}")
    if not any(
        entry.status is MetricStatus.OK for entry in analysis.document.capability_matrix.entries
    ):
        _fail("no analysis capability was available for this run", 5)


@app.command()
def compare(
    run_a: Path = typer.Argument(..., exists=True, readable=True),
    run_b: Path = typer.Argument(..., exists=True, readable=True),
    output: Path = typer.Option(..., "--output", help="Output directory"),
) -> None:
    """Compare two runs after checking coordinate, support and model compatibility."""
    try:
        result = compare_runs(run_a, run_b)
    except Exception as error:  # noqa: BLE001
        _fail(f"comparison failed: {type(error).__name__}: {error}", 2)
        return
    path = write_comparison(result, output)
    typer.echo(str(path))
    if not result.compatible:
        for reason in result.reasons:
            typer.echo(f"incompatible: {reason}", err=True)
        raise typer.Exit(code=2)
    typer.echo(f"comparable metrics: {len(result.metrics)}")


@app.command()
def infer(
    run_dir: Path = typer.Argument(..., exists=True, readable=True),
    contract: Path = typer.Option(..., "--contract", exists=True, readable=True),
    output: Path | None = typer.Option(None, "--output", help="Optional export directory"),
    simulate: bool = typer.Option(False, "--simulate", help="Run synthetic MLE validation"),
    replications: int = typer.Option(100, help="Synthetic replications"),
    node: str | None = typer.Option(None, help="Node providing the observation model"),
) -> None:
    """Compute a CRLB under an explicit contract, or refuse with reasons."""
    raw = yaml.safe_load(Path(contract).read_text(encoding="utf-8"))
    try:
        sampling_contract = SamplingContract.model_validate(raw)
    except Exception as error:  # noqa: BLE001
        _fail(f"invalid sampling contract: {error}", 2)
        return
    try:
        captures = load_captures(run_dir)
    except Exception as error:  # noqa: BLE001
        _fail(f"cannot read run: {error}", 2)
        return
    try:
        experiment = load_run_experiment(run_dir)
        adapter = experiment.build_adapter()
    except Exception as error:  # noqa: BLE001
        _fail(
            f"inference requires a rebuildable adapter: {error}",
            5,
        )
        return
    simulation = (
        {"replications": replications, "sample_size": sampling_contract.sample_size or 1}
        if simulate
        else None
    )
    result = run_inference(
        captures,
        adapter=adapter,
        contract=sampling_contract,
        node_id=node,
        simulate=simulation,
    )
    store = RunStore.load(run_dir)
    store.write_json("inference.json", result.model_dump(mode="json"))
    store.finalize_checksums()
    typer.echo(f"inference: {run_dir / 'inference.json'}")
    if output is not None and Path(output) != Path(run_dir):
        export = Path(output)
        export.mkdir(parents=True, exist_ok=True)
        (export / "inference.json").write_text(
            json.dumps(result.model_dump(mode="json"), indent=2), encoding="utf-8"
        )
    if result.status is not MetricStatus.OK:
        typer.echo(f"reason: {result.reason_code}: {result.remedy}", err=True)
        raise typer.Exit(code=5)


@app.command()
def report(
    run_dir: Path = typer.Argument(..., exists=True, readable=True),
    output: Path = typer.Option(..., "--output", help="HTML report path"),
    config: Path | None = typer.Option(None, "--config", help="Original config"),
) -> None:
    """Render the self-contained offline HTML report and its Markdown twin."""
    try:
        captures = load_captures(run_dir)
    except Exception as error:  # noqa: BLE001
        _fail(f"cannot read run: {error}", 2)
        return
    adapter = None
    source = config or (Path(run_dir) / "config.resolved.yaml")
    if Path(source).exists():
        try:
            adapter = build_adapter(load_config(source))
        except Exception as error:  # noqa: BLE001
            typer.echo(f"warning: adapter unavailable ({error})", err=True)
    analysis = analyze_captures(captures, adapter=adapter)
    metrics_path = Path(run_dir) / "metrics.json"
    if metrics_path.exists():
        from jevometry.schemas.results import AnalysisDocument

        stored = AnalysisDocument.model_validate(
            json.loads(metrics_path.read_text(encoding="utf-8"))
        )
        analysis.document = stored
    inference_payload = None
    inference_path = Path(run_dir) / "inference.json"
    if inference_path.exists():
        inference_payload = json.loads(inference_path.read_text(encoding="utf-8"))
    html_path, markdown_path = render_report(
        analysis,
        output=output,
        markdown_output=Path(output).with_suffix(".md"),
        run_directory=run_dir,
        inference=inference_payload,
    )
    typer.echo(str(html_path))
    typer.echo(str(markdown_path))
    if inference_payload is not None:
        typer.echo("inference.json included in the report")


@app.command()
def doctor() -> None:
    """Report environment and provider availability (key presence only)."""
    import importlib.metadata

    typer.echo(f"jevometry: {__version__}")
    typer.echo(f"python: {sys.version.split()[0]}")
    for package in ("numpy", "scipy", "pydantic", "plotly", "jinja2", "typer", "typesafe-sdk"):
        try:
            typer.echo(f"{package}: {importlib.metadata.version(package)}")
        except importlib.metadata.PackageNotFoundError:
            typer.echo(f"{package}: not installed")
    typer.echo(
        "TYPESAFE_API_KEY: "
        + ("set" if os.environ.get("TYPESAFE_API_KEY") else "not set")
    )
    typer.echo("providers: analytic (offline), replay (offline), typesafe (live, opt-in)")
    if not os.environ.get("TYPESAFE_API_KEY"):
        typer.echo("live integration: credentials unavailable; live path not exercised")


def main() -> Any:
    return app()


if __name__ == "__main__":
    app()
