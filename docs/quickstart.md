# Quickstart

Jevometry measures the geometry, sensitivity and information structure of
Jev-powered decisions.  Everything in this quickstart runs offline; the live
TypeSafe path is opt-in and never required.

## Install

```bash
git clone <repository>
cd jevometry
uv sync --locked            # offline core, no SDK
uv sync --locked --extra live   # adds the official TypeSafe SDK
```

Python 3.11–3.13 are supported.  `uv.lock` records the resolved versions.

## Python API

```python
from jevometry import Experiment, analyze, render_report
from jevometry.adapters import TypeSafeAdapter

experiment = Experiment.from_yaml("experiment.yaml")
captures = experiment.run(adapter=TypeSafeAdapter.from_env(...), live=True)
analysis = analyze(captures, adapter=adapter)
render_report(analysis, output="analysis/report.html")
```

For a fully offline experiment, build an analytic adapter in Python:

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
captures = Experiment.from_spec(spec).run(adapter)
analysis = analyze(captures, adapter=adapter)
render_report(analysis, output="report.html")
```

## CLI

```bash
jevometry init project/                 # config + example adapter + README
jevometry validate project/experiment.yaml
jevometry run project/experiment.yaml --output runs/example
jevometry analyze runs/example
jevometry report runs/example --output runs/example/report.html
jevometry infer runs/example --contract contract.yaml --simulate
jevometry compare runs/a runs/b --output comparison/
jevometry doctor
```

`run` only accepts offline providers (analytic, replay, module) unless
`--live` is passed for the TypeSafe provider.

### Exit codes

| Code | Meaning |
|------|---------|
| 0 | Command succeeded; some optional metrics may be explicitly unsupported |
| 2 | Invalid configuration or data |
| 3 | Acquisition incomplete |
| 4 | Provider or authentication failure |
| 5 | A core analysis the user requested cannot be computed |

## Three complete examples

```bash
uv run python examples/analytic_geometry/run.py
uv run python examples/system_information/run.py
uv run python examples/jev_ticket_sensitivity/run.py
```

Each writes runs and self-contained HTML reports under `examples/*/output/`.
The ticket example also supports `--live` (requires `TYPESAFE_API_KEY`) and
`--replay RUN_DIRECTORY`.

## Next steps

* [mathematics.md](mathematics.md) — the exact formulas and conventions.
* [statistical_contracts.md](statistical_contracts.md) — what each result means.
* [adapters.md](adapters.md) — connecting your own system.
* [numerical_limits.md](numerical_limits.md) — caps and failure modes.
