"""Replay integrity, version mixing and cache repeat handling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx2
import pytest
from typesafe_sdk import RetryPolicy, TypeSafeClient

from jevometry.adapters.base import ExperimentPoint
from jevometry.adapters.replay import ReplayAdapter, ReplayError
from jevometry.adapters.typesafe import ADAPTER_VERSION, TypeSafeAdapter, typesafe_noul_question
from jevometry.artifacts.store import RunStore
from jevometry.experiments.acquisition import run_experiment
from jevometry.experiments.cache import CaptureCache
from jevometry.experiments.design import point_id
from jevometry.pipeline import Experiment, load_captures
from jevometry.schemas.experiment import Budget, CaseSpec, ExperimentSpec
from jevometry.schemas.parameters import ParameterSet, ParameterSpec, StencilSpec


def build_spec() -> ExperimentSpec:
    return ExperimentSpec(
        id="replay-test",
        model="test-model",
        adapter="typesafe",
        parameter_set=ParameterSet(
            parameters=[
                ParameterSpec(
                    name="theta",
                    role="task_relevant",
                    unit="logit",
                    bounds=(-5.0, 5.0),
                    step=0.1,
                    center=0.0,
                )
            ]
        ),
        stencil=StencilSpec(step_scales=[1.0]),
        cases=[CaseSpec(id="c1", state="state")],
        theta_points=[{"theta": 0.0}, {"theta": 0.5}],
        budget=Budget(repeats=1),
    )


def build_live_adapter(counter: dict[str, int]) -> TypeSafeAdapter:
    def handler(request: httpx2.Request) -> httpx2.Response:
        counter["calls"] += 1
        body = json.loads(request.content.decode("utf-8"))
        theta = float(body["state"]["theta"])
        p = 0.5 + 0.1 * theta
        return httpx2.Response(
            200,
            json={
                "model": "resolved-model",
                "usage": {"input_tokens": 5, "output_tokens": 1},
                "answers": {"applicable": {"type": "noul", "noul": p}},
            },
        )

    client = TypeSafeClient(
        api_key="test-key",
        transport=httpx2.MockTransport(handler),
        retry=RetryPolicy(max_retries=0),
    )
    from jevometry.experiments.renderer import StructuredRenderer

    return TypeSafeAdapter(
        model="test-model",
        nodes={"applicable": typesafe_noul_question("applicable")},
        client=client,
        renderer=StructuredRenderer({"theta": "{theta:.4f}"}),
        system_id="replay-test",
    )


def test_replay_roundtrip_and_no_live_fallback(tmp_path: Path) -> None:
    counter = {"calls": 0}
    spec = build_spec()
    experiment = Experiment.from_spec(spec)
    adapter = build_live_adapter(counter)
    captures = experiment.run(adapter, live=True, output=tmp_path / "run")
    assert counter["calls"] > 0
    assert not captures.incomplete

    replay = ReplayAdapter(
        tmp_path / "run",
        system_id="replay-test",
        model="test-model",
        adapter_version=ADAPTER_VERSION,
        question_hashes={"applicable": adapter.nodes["applicable"].spec.semantic_hash()},
    )
    point = ExperimentPoint(
        case=CaseSpec(id="c1", state="state"),
        theta={"theta": 0.0},
        point_id=point_id({"theta": 0.0}),
    )
    traces = replay.evaluate_point(point)
    assert traces[0].status.ok
    assert traces[0].distribution is not None
    assert traces[0].distribution.source.value == "reported"

    missing = ExperimentPoint(
        case=CaseSpec(id="c1", state="state"),
        theta={"theta": 9.0},
        point_id=point_id({"theta": 9.0}),
    )
    with pytest.raises(ReplayError) as error:
        replay.evaluate_point(missing)
    assert error.value.reason_code == "missing_capture"


def test_replay_refuses_version_mixing(tmp_path: Path) -> None:
    counter = {"calls": 0}
    experiment = Experiment.from_spec(build_spec())
    experiment.run(build_live_adapter(counter), live=True, output=tmp_path / "run")
    replay = ReplayAdapter(
        tmp_path / "run",
        system_id="replay-test",
        model="different-model",
        adapter_version="other-version",
        question_hashes={"applicable": "sha256:deadbeef"},
    )
    point = ExperimentPoint(
        case=CaseSpec(id="c1", state="state"),
        theta={"theta": 0.0},
        point_id=point_id({"theta": 0.0}),
    )
    with pytest.raises(ReplayError) as error:
        replay.evaluate_point(point)
    assert error.value.reason_code == "fingerprint_mismatch"


@pytest.fixture
def recorded_run(tmp_path: Path) -> tuple[Path, dict[str, int]]:
    counter = {"calls": 0}
    directory = tmp_path / "recorded"
    Experiment.from_spec(build_spec()).run(
        build_live_adapter(counter), live=True, repeats=2, output=directory
    )
    return directory, counter


def center_point(*, repeat: int = 0) -> ExperimentPoint:
    return ExperimentPoint(
        case=CaseSpec(id="c1", state="state"),
        theta={"theta": 0.0},
        point_id=point_id({"theta": 0.0}),
        repeat=repeat,
    )


def read_recorded_traces(directory: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (directory / "traces.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_recorded_traces(directory: Path, traces: list[dict[str, Any]]) -> None:
    (directory / "traces.jsonl").write_text(
        "".join(json.dumps(trace) + "\n" for trace in traces), encoding="utf-8"
    )


def test_replay_requires_the_requested_repeat(
    recorded_run: tuple[Path, dict[str, int]],
) -> None:
    directory, counter = recorded_run
    calls = counter["calls"]
    replay = ReplayAdapter(directory, system_id="replay-test")
    assert replay.evaluate_point(center_point(repeat=1))[0].repeat == 1
    with pytest.raises(ReplayError) as error:
        replay.evaluate_point(center_point(repeat=2))
    assert error.value.reason_code == "missing_capture"
    assert "repeat 2" in str(error.value)
    assert counter["calls"] == calls


@pytest.mark.parametrize(
    "declaration",
    [
        {"model": "different-model"},
        {"adapter_version": "other-version"},
        {"provider": "other-provider"},
        {"question_hashes": {"applicable": "sha256:deadbeef"}},
    ],
)
def test_replay_checks_each_independent_declaration(
    recorded_run: tuple[Path, dict[str, int]], declaration: dict[str, Any]
) -> None:
    directory, counter = recorded_run
    calls = counter["calls"]
    replay = ReplayAdapter(directory, system_id="replay-test", **declaration)
    with pytest.raises(ReplayError) as error:
        replay.evaluate_point(center_point())
    assert error.value.reason_code == "fingerprint_mismatch"
    assert counter["calls"] == calls


def test_replay_validates_selected_repeat_fingerprint(
    recorded_run: tuple[Path, dict[str, int]],
) -> None:
    directory, _ = recorded_run
    traces = read_recorded_traces(directory)
    for trace in traces:
        if trace["stencil_role"] == "center" and trace["repeat"] == 1:
            trace["request_fingerprint"] = "sha256:corrupted-repeat-one"
    write_recorded_traces(directory, traces)
    replay = ReplayAdapter(
        directory,
        system_id="replay-test",
        model="test-model",
        adapter_version=ADAPTER_VERSION,
        question_hashes={
            "applicable": traces[0]["distribution"]["semantic_hash"],
        },
    )
    assert replay.evaluate_point(center_point())[0].status.ok
    with pytest.raises(ReplayError) as error:
        replay.evaluate_point(center_point(repeat=1))
    assert error.value.reason_code == "fingerprint_mismatch"


def test_replay_rejects_corrupted_fingerprint_without_declarations(
    recorded_run: tuple[Path, dict[str, int]],
) -> None:
    directory, _ = recorded_run
    traces = read_recorded_traces(directory)
    traces[0]["request_fingerprint"] = "sha256:corrupted"
    write_recorded_traces(directory, traces)
    replay = ReplayAdapter(directory, system_id="replay-test")
    with pytest.raises(ReplayError) as error:
        replay.evaluate_point(center_point())
    assert error.value.reason_code == "fingerprint_mismatch"


def test_replay_without_declarations_replays_all_stencil_roles(
    recorded_run: tuple[Path, dict[str, int]],
) -> None:
    directory, counter = recorded_run
    calls = counter["calls"]
    replay = ReplayAdapter(directory, system_id="replay-test")
    captures = run_experiment(build_spec(), replay, repeats=2)
    # Two centers, two off-center roles per center, two repeats.
    assert len(captures.traces) == 12
    assert {trace.repeat for trace in captures.traces} == {0, 1}
    assert {trace.stencil_role for trace in captures.traces} == {
        "center", "theta:1:-1", "theta:1:+1",
    }
    assert all(trace.provenance["source_provider"] == "typesafe" for trace in captures.traces)
    assert counter["calls"] == calls


def test_replayed_captures_retain_original_provider_for_further_replay(
    recorded_run: tuple[Path, dict[str, int]], tmp_path: Path
) -> None:
    directory, _ = recorded_run
    replay = ReplayAdapter(directory, system_id="replay-test")
    trace = replay.evaluate_point(center_point())[0]
    replayed_directory = tmp_path / "replayed"
    replayed_directory.mkdir()
    write_recorded_traces(replayed_directory, [trace.model_dump(mode="json")])
    replay_again = ReplayAdapter(
        replayed_directory, system_id="replay-test", provider="typesafe"
    )
    assert replay_again.provider == "typesafe"
    rerecorded = replay_again.evaluate_point(center_point())[0]
    assert rerecorded.request_fingerprint == trace.request_fingerprint
    assert rerecorded.provenance["source_provider"] == "typesafe"


def test_replay_checks_history(
    recorded_run: tuple[Path, dict[str, int]],
) -> None:
    directory, _ = recorded_run
    replay = ReplayAdapter(directory, system_id="replay-test")
    point = ExperimentPoint(
        case=CaseSpec(id="c1", state="state"),
        theta={"theta": 0.0},
        point_id=point_id({"theta": 0.0}),
        history={"upstream": "changed"},
    )
    with pytest.raises(ReplayError) as error:
        replay.evaluate_point(point)
    assert error.value.reason_code == "fingerprint_mismatch"


def test_replay_allows_unspecified_source_metadata_in_existing_capture(
    recorded_run: tuple[Path, dict[str, int]],
) -> None:
    directory, _ = recorded_run
    traces = read_recorded_traces(directory)
    # Existing analytic captures omit adapter_version; replay must avoid
    # replacing that missing metadata with a guessed version or question id.
    traces[0]["provenance"].pop("adapter_version")
    write_recorded_traces(directory, traces)
    replay = ReplayAdapter(directory, system_id="replay-test")
    assert replay.evaluate_point(center_point())[0].status.ok


def test_replay_reports_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(ReplayError) as error:
        ReplayAdapter(tmp_path, system_id="x")
    assert error.value.reason_code == "missing_traces_file"


def test_cache_keeps_repeats_separate(tmp_path: Path) -> None:
    cache = CaptureCache(tmp_path / "cache")
    counter = {"calls": 0}
    experiment = Experiment.from_spec(build_spec())
    adapter = build_live_adapter(counter)
    captures = run_experiment(
        experiment.spec, adapter, live=True, cache=cache, repeats=2
    )
    assert not captures.incomplete
    fingerprints = {trace.request_fingerprint for trace in captures.traces}
    assert len(fingerprints) >= 1
    for fingerprint in fingerprints:
        assert cache.contains(fingerprint, repeat=0)
        assert cache.contains(fingerprint, repeat=1)
        assert not cache.contains(fingerprint, repeat=2)
    cache.invalidate(fingerprints.pop(), repeat=0)


def test_store_preserves_raw_captures_across_revisions(tmp_path: Path) -> None:
    from jevometry.analysis import analyze
    from jevometry.artifacts.integrity import hash_file
    from jevometry.schemas.results import AnalysisDocument

    counter = {"calls": 0}
    experiment = Experiment.from_spec(build_spec())
    captures = experiment.run(build_live_adapter(counter), live=True, output=tmp_path / "run")
    store = RunStore.load(tmp_path / "run")
    traces_hash = hash_file(tmp_path / "run" / "traces.jsonl")
    analysis = analyze(captures)
    for revision in (1, 2):
        analysis.document.analysis_revision = revision
        analysis.document = AnalysisDocument.model_validate(
            analysis.document.model_dump(mode="json")
        )
        store.write_analysis(analysis.document, arrays=analysis.arrays)
        assert hash_file(tmp_path / "run" / "traces.jsonl") == traces_hash
    reloaded = RunStore.load(tmp_path / "run")
    assert reloaded.manifest is not None
    assert reloaded.manifest.analysis_revisions == [1, 2]
    assert reloaded.manifest.latest_analysis_revision == 2
    payload = json.loads((tmp_path / "run" / "metrics.json").read_text(encoding="utf-8"))
    assert "NaN" not in json.dumps(payload)
    assert (tmp_path / "run" / "analysis" / "rev-0001" / "metrics.json").exists()


def test_load_captures_roundtrip(tmp_path: Path) -> None:
    counter = {"calls": 0}
    experiment = Experiment.from_spec(build_spec())
    original = experiment.run(
        build_live_adapter(counter), live=True, output=tmp_path / "run"
    )
    loaded = load_captures(tmp_path / "run")
    assert len(loaded.traces) == len(original.traces)
    assert loaded.live is True
    assert loaded.experiment.id == original.experiment.id
