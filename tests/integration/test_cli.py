"""End-to-end offline CLI flows and exit codes."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from jevometry.cli import app

FIXTURES = Path(__file__).parent.parent / "fixtures"
REPO_ROOT = Path(__file__).parent.parent.parent


def invoke(*args: str) -> tuple[int, str]:
    runner = CliRunner()
    result = runner.invoke(app, list(args))
    output = result.output + (result.exception.__str__() if result.exception else "")
    return result.exit_code, output


def test_init_validate_run_analyze_report_flow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    code, output = invoke("init", str(project))
    assert code == 0, output
    config = project / "experiment.yaml"
    assert config.exists()
    monkeypatch.chdir(project)

    code, output = invoke("validate", str(config))
    assert code == 0, output
    assert "within_budget" in output or "budget_max_attempts" in output

    run_dir = tmp_path / "run"
    code, output = invoke("run", str(config), "--output", str(run_dir))
    assert code == 0, output
    assert (run_dir / "traces.jsonl").exists()
    assert (run_dir / "manifest.json").exists()

    code, output = invoke("analyze", str(run_dir))
    assert code == 0, output
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["nodes"]

    report = tmp_path / "report.html"
    code, output = invoke("report", str(run_dir), "--output", str(report))
    assert code == 0, output
    assert report.exists()
    assert report.with_suffix(".md").exists()
    html = report.read_text(encoding="utf-8")
    assert "Plotly.newPlot" in html
    assert not re.search(r'src="https?://', html)
    assert "no telemetry" in html


def test_invalid_config_exits_2(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("id: only-an-id\n", encoding="utf-8")
    code, output = invoke("validate", str(bad))
    assert code == 2, output


def test_missing_run_directory_exits_2(tmp_path: Path) -> None:
    code, output = invoke("analyze", str(tmp_path / "missing"))
    assert code != 0


def test_typesafe_without_live_flag_exits_4(tmp_path: Path) -> None:
    config = FIXTURES / "bernoulli_experiment.yaml"
    text = config.read_text(encoding="utf-8").replace(
        "kind: module", "kind: typesafe"
    ).replace("entrypoint: tests.fixtures.bernoulli_adapter:build_adapter", "model: some-model")
    target = tmp_path / "live.yaml"
    target.write_text(text, encoding="utf-8")
    code, output = invoke("run", str(target), "--output", str(tmp_path / "run"))
    assert code == 4, output
    assert "--live" in output


def test_fixture_run_and_infer_flow(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    code, output = invoke(
        "run",
        str(FIXTURES / "bernoulli_experiment.yaml"),
        "--output",
        str(run_dir),
    )
    assert code == 0, output
    code, output = invoke("analyze", str(run_dir))
    assert code == 0, output

    code, output = invoke(
        "infer",
        str(run_dir),
        "--contract",
        str(FIXTURES / "bernoulli_contract.yaml"),
        "--simulate",
        "--replications",
        "20",
    )
    assert code == 0, output
    inference = json.loads((run_dir / "inference.json").read_text(encoding="utf-8"))
    assert inference["crlb"] is not None
    assert inference["status"] == "ok"
    assert inference["simulations"]
    expected = 0.3 * 0.7 / 100
    assert inference["crlb"][0][0] == pytest.approx(expected, rel=1e-6)

    code, output = invoke(
        "infer",
        str(run_dir),
        "--contract",
        str(FIXTURES / "incomplete_contract.yaml"),
    )
    assert code == 5, output
    assert "missing" in output.lower() or "incomplete" in output.lower()


def test_compare_compatible_and_incompatible(tmp_path: Path) -> None:
    run_a = tmp_path / "a"
    run_b = tmp_path / "b"
    for target in (run_a, run_b):
        code, output = invoke(
            "run",
            str(FIXTURES / "bernoulli_experiment.yaml"),
            "--output",
            str(target),
        )
        assert code == 0, output
        code, output = invoke("analyze", str(target))
        assert code == 0, output
    code, output = invoke("compare", str(run_a), str(run_b), "--output", str(tmp_path / "cmp"))
    assert code == 0, output
    comparison = json.loads((tmp_path / "cmp" / "comparison.json").read_text(encoding="utf-8"))
    assert comparison["compatible"] is True
    assert comparison["metrics"]

    altered = tmp_path / "altered.yaml"
    text = (FIXTURES / "bernoulli_experiment.yaml").read_text(encoding="utf-8")
    altered.write_text(text.replace("unit: probability", "unit: logit"), encoding="utf-8")
    run_c = tmp_path / "c"
    code, output = invoke("run", str(altered), "--output", str(run_c))
    assert code == 0, output
    code, output = invoke("compare", str(run_a), str(run_c), "--output", str(tmp_path / "cmp2"))
    assert code == 2, output
    assert "differ" in output


def test_doctor_reports_key_presence_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    code, output = invoke("doctor")
    assert code == 0
    assert "TYPESAFE_API_KEY: not set" in output
    assert "not exercised" in output


def test_replay_via_cli(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    code, output = invoke(
        "run",
        str(FIXTURES / "bernoulli_experiment.yaml"),
        "--output",
        str(run_dir),
    )
    assert code == 0, output
    replay_config = tmp_path / "replay.yaml"
    text = (FIXTURES / "bernoulli_experiment.yaml").read_text(encoding="utf-8")
    text = text.replace("kind: module", "kind: replay").replace(
        "entrypoint: tests.fixtures.bernoulli_adapter:build_adapter",
        "run_directory: " + str(run_dir),
    )
    text = text.replace(
        "  system_id: bernoulli-fixture",
        "  system_id: bernoulli-fixture\n"
        "  replay_source_provider: analytic\n"
        "  replay_source_model: bernoulli:probability\n"
        "  replay_source_adapter_version: analytic-0.1.0",
    )
    replay_config.write_text(text, encoding="utf-8")
    code, output = invoke("run", str(replay_config), "--output", str(tmp_path / "replayed"))
    assert code == 0, output
    assert (tmp_path / "replayed" / "traces.jsonl").exists()


def test_example_ticket_config_runs_offline(tmp_path: Path) -> None:
    config = REPO_ROOT / "examples" / "jev_ticket_sensitivity" / "experiment.yaml"
    if not config.exists():  # pragma: no cover - repository layout guarantee
        pytest.skip("example config is unavailable")
    code, output = invoke("validate", str(config))
    assert code == 0, output
    run_dir = tmp_path / "ticket"
    code, output = invoke("run", str(config), "--output", str(run_dir))
    assert code == 0, output
    code, output = invoke("analyze", str(run_dir))
    assert code == 0, output
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    capability = {entry["capability"]: entry["status"] for entry in metrics["capability_matrix"]["entries"]}
    assert capability["system_information"] == "unsupported"
    assert capability["action_information"] == "unsupported"
    assert capability["node_geometry"] == "ok"


def test_examples_do_not_require_network() -> None:
    assert (REPO_ROOT / "examples" / "analytic_geometry" / "run.py").exists()
    assert (REPO_ROOT / "examples" / "system_information" / "run.py").exists()
    assert (REPO_ROOT / "examples" / "jev_ticket_sensitivity" / "run.py").exists()

def test_incomplete_acquisition_exits_3(tmp_path: Path) -> None:
    run_dir = tmp_path / "failing"
    code, output = invoke(
        "run",
        str(FIXTURES / "failing_experiment.yaml"),
        "--output",
        str(run_dir),
    )
    assert code == 3, output
    assert "incomplete" in output
    assert (run_dir / "traces.jsonl").exists()
