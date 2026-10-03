"""Artifact audits refuse incomplete evidence and preserve analysis history."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from jevometry.artifacts.integrity import hash_file, verify_checksums, verify_run
from jevometry.cli import app

FIXTURE = Path(__file__).parents[1] / "fixtures" / "bernoulli_experiment.yaml"


def make_run(root: Path) -> Path:
    run = root / "run"
    result = CliRunner().invoke(app, ["run", str(FIXTURE), "--output", str(run)])
    assert result.exit_code == 0, result.output
    return run


def test_verify_checks_complete_run_without_writing(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    before = {p.name: hash_file(p) for p in run.iterdir() if p.is_file()}
    result = CliRunner().invoke(app, ["verify", str(run), "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["valid"] is True
    assert {p.name: hash_file(p) for p in run.iterdir() if p.is_file()} == before
    recorded = json.loads((run / "checksums.json").read_text())
    assert "config.resolved.yaml" in recorded
    assert "system.json" in recorded


@pytest.mark.parametrize("name", ["traces.jsonl", "config.resolved.yaml", "parameters.json"])
def test_verify_rejects_tampered_or_deleted_artifacts(tmp_path: Path, name: str) -> None:
    run = make_run(tmp_path)
    target = run / name
    target.write_text(target.read_text() + "\n")
    result = CliRunner().invoke(app, ["verify", str(run)])
    assert result.exit_code == 2
    assert f"checksum mismatch: {name}" in result.output
    target.unlink()
    assert not verify_run(run).valid


def test_verify_rejects_missing_index_and_missing_coverage(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    index = run / "checksums.json"
    recorded = json.loads(index.read_text())
    recorded.pop("traces.jsonl")
    index.write_text(json.dumps(recorded))
    assert "missing checksum: traces.jsonl" in verify_run(run).issues
    index.unlink()
    with pytest.raises(ValueError, match="missing checksums"):
        verify_checksums(run)
    assert CliRunner().invoke(app, ["verify", str(run)]).exit_code == 2


@pytest.mark.parametrize("payload", ["{", "[]", "{}", '{"traces.jsonl":"abc"}'])
def test_verify_rejects_malformed_index(tmp_path: Path, payload: str) -> None:
    run = make_run(tmp_path)
    (run / "checksums.json").write_text(payload)
    assert not verify_run(run).valid


def test_index_cannot_read_outside_run(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("external data")
    index = run / "checksums.json"
    index.write_text(json.dumps({"../outside.txt": hash_file(outside)}))
    with pytest.raises(ValueError, match="invalid artifact path"):
        verify_checksums(run)
    assert not verify_run(run).valid
    (run / "linked.txt").symlink_to(outside)
    index.write_text(json.dumps({"linked.txt": hash_file(outside)}))
    with pytest.raises(ValueError, match="escapes run directory"):
        verify_checksums(run)


def test_symlink_index_and_loop_fail_with_structured_audit(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    index = run / "checksums.json"
    outside = tmp_path / "outside-index.json"
    index.rename(outside)
    index.symlink_to(outside)
    result = CliRunner().invoke(app, ["verify", str(run), "--json"])
    assert result.exit_code == 2
    assert json.loads(result.output)["valid"] is False
    index.unlink()
    (run / "loop.txt").symlink_to("loop.txt")
    index.write_text(json.dumps({"loop.txt": "sha256:" + "0" * 64}))
    result = CliRunner().invoke(app, ["verify", str(run), "--json"])
    assert result.exit_code == 2
    assert json.loads(result.output)["valid"] is False


def test_reports_create_covered_revisions_and_preserve_old_metrics(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    runner = CliRunner()
    assert runner.invoke(app, ["analyze", str(run)]).exit_code == 0
    first = run / "analysis" / "rev-0001" / "metrics.json"
    original = hash_file(first)
    for expected_revision in (2, 3):
        result = runner.invoke(app, ["report", str(run), "--output", str(run / "report.html")])
        assert result.exit_code == 0, result.output
        assert hash_file(first) == original
        manifest = json.loads((run / "manifest.json").read_text())
        assert manifest["latest_analysis_revision"] == expected_revision
        assert verify_run(run).valid, verify_run(run).issues
    index = json.loads((run / "checksums.json").read_text())
    assert "analysis/rev-0001/metrics.json" in index
    first.write_text(first.read_text() + "\n")
    assert "analysis/rev-0001/metrics.json" in verify_checksums(run)


@pytest.mark.parametrize("relative", ["traces.jsonl", "metrics.json", "analysis/rev-0001/metrics.json"])
def test_report_cannot_overwrite_managed_artifacts(tmp_path: Path, relative: str) -> None:
    run = make_run(tmp_path)
    runner = CliRunner()
    result = runner.invoke(app, ["analyze", str(run)])
    assert result.exit_code == 0, result.output
    before = {p.relative_to(run): hash_file(p) for p in run.rglob("*") if p.is_file()}
    result = runner.invoke(app, ["report", str(run), "--output", str(run / relative)])
    assert result.exit_code == 2
    assert "overwrite a managed artifact" in result.output
    assert {p.relative_to(run): hash_file(p) for p in run.rglob("*") if p.is_file()} == before
    assert verify_run(run).valid
