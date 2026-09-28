# Jevometry

**Measure the geometry, sensitivity, and information structure of Jev-powered
decisions.**

Jevometry is an information-geometric analysis toolkit for Jev agent systems.
It analyses output-distribution geometry, parameter sensitivity, Fisher
information, system dependencies and information loss, and computes
Cramér–Rao lower bounds and parameter inference under explicit statistical
conditions.

Version 0.1.0 (alpha). Python 3.11–3.13. MIT licensed. The offline analysis
core requires no network access or API key. Jevometry is an independent
project, not an official TypeSafe product.

## What it does

* Validates Choice / Score / Noul probability vectors with strict support
  alignment and traceable working-simplex derivation.
* Computes entropy, KL, Jensen–Shannon, Hellinger and categorical
  Fisher–Rao distances.
* Computes node Fisher pullback metrics with finite-difference stability
  diagnostics and an independent square-root cross-check.
* Combines nodes only through declared joints, product models or conditional
  trees, and checks the conditional information identity by exact enumeration.
* Measures information loss under fixed aggregation mappings.
* Computes CRLBs and synthetic MLE validation under an explicit sampling
  contract.
* Produces structured results with statuses, analysis objects, assumptions,
  provenance and diagnostic reason codes. See the known limitations below
  before interpreting live results.

## What it deliberately does not do

* It does not treat an API response as an independent categorical draw.
* It does not sum correlated nodes' Fisher information as "system information".
* It does not produce a CRLB without a complete sampling contract.
* It does not use zero, epsilon smoothing, uniform fallbacks or synthetic data
  in place of a result it cannot compute.
* It does not modify an analysed system, call its tools, or run a service.

## Install

```bash
git clone https://github.com/Kunyanli230/Jevometry.git
cd Jevometry
uv sync --locked                # offline core
uv sync --locked --extra live   # adds the official TypeSafe SDK
```

Install [uv](https://docs.astral.sh/uv/) first. To build and install a wheel:

```bash
uv build
python -m pip install dist/jevometry-0.1.0-py3-none-any.whl
```

## Quickstart

Start with the offline
[Three-Agent Data Cleaning Council](examples/three_agent_cleaning/README.md):

```bash
uv run python examples/three_agent_cleaning/run.py
```

Open `examples/three_agent_cleaning/output/report.html` to compare each
agent's sensitivity with the declared joint and final cleaning decision.
For a standalone Python experiment:

```python
from jevometry import Experiment, analyze, render_report
from jevometry.adapters import AnalyticAdapter
from jevometry.adapters.analytic import logistic_node
from jevometry.schemas.experiment import CaseSpec, ExperimentSpec
from jevometry.schemas.parameters import ParameterSet, ParameterSpec, StencilSpec

node = logistic_node("risk", parameter="theta", coordinate="logit")
adapter = AnalyticAdapter(system_id="logistic", nodes={"risk": node})
spec = ExperimentSpec(
    id="logistic",
    parameter_set=ParameterSet(parameters=[ParameterSpec(
        name="theta", role="task_relevant", unit="logit",
        bounds=(-8.0, 8.0), step=0.01, center=0.0,
    )]),
    stencil=StencilSpec(),
    cases=[CaseSpec(id="case", state="example")],
    theta_points=[{"theta": 0.0}],
)
experiment = Experiment.from_spec(spec)
captures = experiment.run(adapter)
analysis = analyze(captures, adapter=adapter)
render_report(analysis, output="runs/logistic/report.html")
```

```bash
uv run jevometry init project/
uv run jevometry validate project/experiment.yaml
uv run jevometry run project/experiment.yaml --output runs/example
uv run jevometry analyze runs/example
uv run jevometry report runs/example --output runs/example/report.html
```

See [docs/quickstart.md](docs/quickstart.md) for the full walkthrough and
[docs/adapters.md](docs/adapters.md) for connecting your own system.

## Examples

Four complete, runnable examples live under `examples/`. Start with the
[Three-Agent Data Cleaning Council](examples/three_agent_cleaning/README.md)
for a compact multi-agent walkthrough:

| Example | Command | Shows |
|---------|---------|-------|
| Three-Agent Data Cleaning Council | `uv run python examples/three_agent_cleaning/run.py` | Three analytic agent distributions, a declared product joint, fixed action coordinator and information loss |
| Analytic geometry laboratory | `uv run python examples/analytic_geometry/run.py` | Logistic and softmax Fisher information, rank deficiency, aggregation loss |
| Multi-node information and inference | `uv run python examples/system_information/run.py` | Deterministic copies add no information, independent draws double it, conditional trees, MLE vs CRLB |
| Ticket-system sensitivity | `uv run python examples/jev_ticket_sensitivity/run.py` | A synthetic ticket system with a deterministic policy, action flips and a declared surrogate; `--live` runs a minimal real grid |

Each example writes a run directory and a self-contained interactive HTML
report under `examples/*/output/`.

## Capability and honesty

For a categorical node, the Fisher pullback describes the local geometry of
the probability family with respect to declared parameters:

```text
I(θ) = Σ_y [∇θ p(y | θ) ∇θ p(y | θ)ᵀ] / p(y | θ)
```

This does not establish empirical calibration or an observation likelihood.
Boundary probabilities, rank deficiency and unstable derivatives require
separate diagnostics. Marginal probabilities alone cannot determine a joint
system distribution, and deterministic routing is not categorical sampling.

* Every metric is a structured result: `ok`, `conditional`, `unstable`,
  `undefined`, `not_identifiable`, `insufficient_data`, `unsupported` or
  `failed`, with a reason code and a remedy.
* Missing structure produces a refusal, never a default number.
* Synthetic simulation validates the declared model, not the real system.
* Capture traces record the requested and resolved model. No model version is
  presented as permanently current.

Read [docs/statistical_contracts.md](docs/statistical_contracts.md) and
[docs/mathematics.md](docs/mathematics.md) before interpreting results.

## Live TypeSafe provider

The following Bash commands work in WSL. Live calls consume provider quota
and may incur charges. `.env.example` is documentation only; environment
files are not loaded automatically.

```bash
uv sync --locked --extra live
read -rsp 'TypeSafe API key: ' TYPESAFE_API_KEY
echo
export TYPESAFE_API_KEY
uv run python examples/jev_ticket_sensitivity/run.py --live --model jev-1.13.0
unset TYPESAFE_API_KEY
```

`jev-1.13.0` is the model used for the initial smoke test, not a claim about
the latest model. Use a concrete model ID available to your account.

The example evaluates one centre and eight perturbations, repeated three
times: **27 batch evaluations and 81 node records**, before any retries.
Output goes to `examples/jev_ticket_sensitivity/output/live/`; preserve the
directory elsewhere before rerunning if you need the previous artifacts.

* A concrete model id is required; `latest` is rejected.
* Defaults: 45 s per request, 600 s per experiment, 200 attempts,
  200 000 known input tokens, concurrency 2.
* The SDK performs the only retry layer; auth and schema errors are not
  retried.
* Supply credentials through the environment and never commit them. Review
  captured inputs and outputs for sensitive data before sharing artifacts.

Validated provider version: `typesafe-sdk 0.7.1`.  The exact SDK contract is
re-verified by the transport-mock tests whenever the SDK is upgraded.

## Known v0.1 limitations

The initial live smoke test returned 81 successful node records. All three
nodes failed the finite-difference step-stability check. This verified live
capture and report generation, **not reliable live Fisher estimation**.
The example declares neither a joint model nor an observation likelihood,
so it does not produce system Fisher information or a CRLB.

Three implementation issues remain unresolved:

1. The request estimator reports 11 rather than 27 evaluations for the live
   example because it undercounts repeated perturbations. Do not rely on
   that estimate alone when planning provider spending.
2. The capability summary can report `missing_stencil_captures` despite
   complete captures when the actual problem is derivative instability.
3. A node marked `unstable` can contain Fisher sub-results marked `ok`.
   Treat their values as unvalidated whenever the parent geometry or
   derivative diagnostics are unstable.

A successful process exit does not imply valid statistical estimates.
The API and artifact formats may evolve before 1.0. Read
[numerical limits](docs/numerical_limits.md) and
[data handling](docs/data_handling.md) before running your own experiments.

## Development

```bash
uv sync --locked --extra live
uv run ruff check src tests examples
uv run mypy
uv run pytest --cov=jevometry --cov-branch
uv build
```

See [CONTRIBUTING.md](CONTRIBUTING.md) and
[docs/release.md](docs/release.md).

The live extra enables SDK contract tests using mocked transport; tests do
not require an API key. CI is configured for Python 3.11, 3.12 and 3.13.
Report bugs with versions, commands and relevant diagnostics, never API keys
or private customer data. See [SECURITY.md](SECURITY.md) for security reporting.

## Repository layout

```text
src/jevometry/   package: schemas, geometry, systems, inference, adapters,
                 experiments, artifacts, reporting, CLI
examples/        the three complete examples
tests/           unit, property, integration, contract and fixtures
docs/            mathematics, contracts, adapters, design, data, limits, release
```

## License

Jevometry is distributed under the [MIT License](LICENSE).
