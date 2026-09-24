# Adapters

Jevometry is framework-agnostic.  The geometry, system and inference cores
depend only on the protocols in `jevometry.adapters.base`; they never import a
provider, the CLI or Plotly.  You can connect LangGraph, AutoGen, CrewAI or a
bespoke system by exporting traces or implementing an adapter.

## Protocols

```python
class SystemAdapter(Protocol):
    def describe(self) -> SystemSpec: ...
    def evaluate_point(self, point: ExperimentPoint) -> list[EvaluationTrace]: ...

class JointModel(Protocol):
    node_order: list[str]
    def probabilities(self, theta) -> JointDistribution: ...
    def jacobian(self, theta) -> FloatArray | None: ...

class LikelihoodModel(Protocol):
    support: tuple[str, ...]
    def log_prob(self, observations, theta) -> float: ...
    def sample(self, theta, n, rng) -> NDArray[np.int64]: ...

class ParameterRenderer(Protocol):
    version: str
    def render(self, theta, case) -> RenderedInput: ...
```

An adapter optionally exposes:

* `analytic_jacobian(node_id, theta)` for finite-difference cross-checks;
* `declared_fixed_zero(node_id)` for verified fixed active supports;
* `likelihood_model(node_id)` for inference;
* `joint_model` for system composition;
* `questions()` for the question declarations;
* `is_live` and `provider_name`.

## Providers

### Analytic (offline)

Deterministic families with closed-form probabilities and Jacobians:
`logistic_node`, `bernoulli_node`, `softmax_node`,
`rank_deficient_softmax_node`, `single_node_joint_model`,
`bernoulli_pair_joint_model`, `conditional_tree_joint_model`,
`bernoulli_likelihood`.  These are the mathematical test backbone.

### Replay (offline)

`ReplayAdapter(run_directory, ...)` matches recorded traces by request
fingerprint, repeat index, theta and stencil role.  A missing or mismatched
capture raises `missing_capture` / `fingerprint_mismatch`; replay never falls
back to a live provider.  The provider, model and adapter version of the
original run must be declared in the replay configuration:

```yaml
provider:
  kind: replay
  run_directory: runs/live
  replay_source_provider: typesafe
  replay_source_model: MODEL_ID
  replay_source_adapter_version: typesafe-0.1.0
```

### TypeSafe (live, opt-in)

```python
from jevometry.adapters.typesafe import (
    TypeSafeAdapter, typesafe_choice_question, typesafe_noul_question,
    typesafe_score_question,
)

adapter = TypeSafeAdapter.from_env(
    model="MODEL_ID",
    nodes={
        "kind": typesafe_choice_question("kind", {"a": "A", "b": "B"}),
        "applicable": typesafe_noul_question("applicable"),
        "risk": typesafe_score_question("risk", ["low", "high"]),
    },
)
```

* Credentials come from `TYPESAFE_API_KEY`; the key value is never logged,
  recorded or hashed.  `doctor` reports presence only.
* A concrete model id is required; `latest` is not accepted.
* SDK version, endpoint, requested/resolved model, request id and usage are
  stored per trace.
* Retry is single-layer: the SDK's `RetryPolicy` retries 408/429/5xx,
  connection and timeout errors, honours `Retry-After`, and does not retry auth,
  bad-request, unprocessable-entity or response-validation errors.  The
  application does not add a second retry loop.
* Budgets: 200 attempts, 200 000 known input tokens, 45 s per request, 600 s
  per experiment, concurrency cap 2.  Attempts count real HTTP attempts,
  including SDK retries.  Unknown usage is counted separately and never
  fabricated.
* Questions for a shared state are batched into one `system_one` call; the
  batch shape is recorded in the acquisition plan.

## Capture

```python
from jevometry.adapters.capture import CaptureRecorder, RecordingTransport, capture

recorder = CaptureRecorder(redact=my_redactor)
with capture(recorder):
    ...  # nothing is patched; the recorder is opt-in
```

`RecordingTransport` wraps any httpx-style transport and records request and
response bodies.  Sensitive headers are redacted automatically; user redaction
is applied before hashing, so the redacted payload is the analysis input.
Capture never replays business tools and never patches an SDK globally.

## Custom adapter skeleton

```python
from jevometry.adapters.base import ExperimentPoint
from jevometry.schemas.system import SystemSpec, NodeSpec

class MyAdapter:
    is_live = False
    provider_name = "my-system"

    def describe(self) -> SystemSpec:
        return SystemSpec(id="my-system", nodes=[NodeSpec(id="n", question_id="q")])

    def evaluate_point(self, point: ExperimentPoint) -> list[EvaluationTrace]:
        ...  # render, call, convert to DistributionRecord, return traces
```

Return `EvaluationTrace` objects whose `distribution` is a validated
`DistributionRecord`.  If a call fails, set `status.ok = False` and a
`reason_code`; API failures are never mapped onto categories.
