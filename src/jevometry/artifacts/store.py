"""Run directory layout, immutable captures and analysis revisions.

Raw captures are written once and never overwritten.  Every ``analyze`` call
creates a new numbered revision under ``analysis/rev-NNNN`` and mirrors the
latest revision at the run root; the manifest is updated atomically.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from jevometry.artifacts.integrity import hash_object, run_artifact_paths, write_checksums
from jevometry.schemas.experiment import ExperimentSpec
from jevometry.schemas.results import AnalysisDocument, RunManifest
from jevometry.schemas.trace import EvaluationTrace

RUN_FILES = (
    "manifest.json",
    "experiment.resolved.yaml",
    "config.resolved.yaml",
    "system.json",
    "questions.json",
    "parameters.json",
    "traces.jsonl",
    "derivatives.npz",
    "metrics.json",
    "diagnostics.json",
    "inference.json",
    "report.html",
    "report.md",
    "checksums.json",
)


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_run_id(experiment_id: str, seed: int) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    return f"{experiment_id}-{stamp}-{seed}"


@dataclass
class RunStore:
    """Filesystem store for one run."""

    root: Path
    manifest: RunManifest | None = None
    written_files: set[str] = field(default_factory=set)

    @classmethod
    def create(
        cls,
        root: Path,
        *,
        experiment: ExperimentSpec,
        provider_mode: str,
        live: bool,
        budget_summary: dict[str, Any] | None = None,
        run_id: str | None = None,
        adapter: str | None = None,
        system: dict[str, Any] | None = None,
        questions: dict[str, Any] | None = None,
        config_payload: dict[str, Any] | None = None,
    ) -> RunStore:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        resolved_id = run_id or new_run_id(experiment.id, experiment.seed)
        created = utc_now()
        manifest = RunManifest(
            run_id=resolved_id,
            created_utc=created,
            updated_utc=created,
            experiment_id=experiment.id,
            experiment_hash=hash_object(experiment.model_dump(mode="json")),
            model=experiment.model,
            adapter=adapter,
            provider_mode=provider_mode,
            live=live,
            budget=budget_summary or {},
        )
        store = cls(root=root, manifest=manifest)
        store.write_manifest()
        store.write_text(
            "experiment.resolved.yaml", _dump_yaml(experiment.model_dump(mode="json"))
        )
        store.write_json("parameters.json", [p.model_dump(mode="json") for p in experiment.parameters()])
        if system is not None:
            store.write_json("system.json", system)
        if questions is not None:
            store.write_json("questions.json", questions)
        if config_payload is not None:
            store.write_text("config.resolved.yaml", _dump_yaml(config_payload))
        return store

    @classmethod
    def load(cls, root: Path) -> RunStore:
        root = Path(root)
        manifest_path = root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"{root} is not a Jevometry run directory")
        manifest = RunManifest.model_validate(json.loads(manifest_path.read_text(encoding="utf-8")))
        return cls(root=root, manifest=manifest)

    def path(self, name: str) -> Path:
        return self.root / name

    def write_json(self, name: str, payload: Any) -> Path:
        target = self.root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        temporary.replace(target)
        self.written_files.add(name)
        return target

    def read_json(self, name: str) -> Any:
        return json.loads((self.root / name).read_text(encoding="utf-8"))

    def write_text(self, name: str, content: str) -> Path:
        target = self.root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        self.written_files.add(name)
        return target

    def read_text(self, name: str) -> str:
        return (self.root / name).read_text(encoding="utf-8")

    def write_traces(self, traces: list[EvaluationTrace]) -> Path:
        target = self.root / "traces.jsonl"
        temporary = target.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for trace in traces:
                handle.write(
                    json.dumps(trace.model_dump(mode="json"), ensure_ascii=False, allow_nan=False)
                )
                handle.write("\n")
        temporary.replace(target)
        self.written_files.add("traces.jsonl")
        return target

    def read_traces(self) -> list[EvaluationTrace]:
        target = self.root / "traces.jsonl"
        if not target.exists():
            return []
        traces: list[EvaluationTrace] = []
        for line in target.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                traces.append(EvaluationTrace.model_validate(json.loads(line)))
        return traces

    def write_npz(self, name: str, arrays: dict[str, np.ndarray]) -> Path:
        target = self.root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez(target, **arrays)  # type: ignore[arg-type]
        self.written_files.add(name)
        return target

    def read_npz(self, name: str) -> dict[str, np.ndarray]:
        with np.load(self.root / name) as data:
            return {key: data[key] for key in data.files}

    def analysis_revision_dir(self, revision: int) -> Path:
        return self.root / "analysis" / f"rev-{revision:04d}"

    def next_analysis_revision(self) -> int:
        assert self.manifest is not None
        return (self.manifest.latest_analysis_revision or 0) + 1

    def write_analysis(
        self,
        document: AnalysisDocument,
        *,
        arrays: dict[str, np.ndarray] | None = None,
        report_html: str | None = None,
        report_markdown: str | None = None,
    ) -> Path:
        revision_dir = self.analysis_revision_dir(document.analysis_revision)
        revision_dir.mkdir(parents=True, exist_ok=True)
        payloads: dict[str, Any] = {
            "metrics.json": document.model_dump(mode="json"),
        }
        for name, payload in payloads.items():
            (revision_dir / name).write_text(
                json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
                encoding="utf-8",
            )
        if arrays:
            np.savez(revision_dir / "derivatives.npz", **arrays)  # type: ignore[arg-type]
        if report_html is not None:
            (revision_dir / "report.html").write_text(report_html, encoding="utf-8")
        if report_markdown is not None:
            (revision_dir / "report.md").write_text(report_markdown, encoding="utf-8")
        for name in ("metrics.json", "derivatives.npz", "report.html", "report.md"):
            source = revision_dir / name
            if source.exists():
                shutil.copyfile(source, self.root / name)
                self.written_files.add(name)
        assert self.manifest is not None
        self.manifest.latest_analysis_revision = document.analysis_revision
        if document.analysis_revision not in self.manifest.analysis_revisions:
            self.manifest.analysis_revisions.append(document.analysis_revision)
        self.manifest.updated_utc = utc_now()
        self.write_manifest()
        return revision_dir

    def write_manifest(self) -> None:
        if self.manifest is None:
            raise ValueError("no manifest to write")
        target = self.root / "manifest.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self.manifest.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary.replace(target)

    def finalize_checksums(self) -> dict[str, str]:
        self.write_manifest()
        candidates = {name: name for name in run_artifact_paths(self.root)}
        return write_checksums(self.root, candidates)


def _dump_yaml(payload: Any) -> str:
    import yaml

    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
