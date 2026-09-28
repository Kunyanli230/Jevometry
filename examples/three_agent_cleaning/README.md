# Three-Agent Data Cleaning Council

This offline tutorial connects a small decision system to Jevometry. Three
specialized agents propose probability distributions for one candidate repair:

```json
{"column": "monthly_income_usd", "before": " 1,200 ", "candidate": 1200}
```

| Agent | Jev primitive | Outcomes |
| --- | --- | --- |
| `applicability_agent` | Noul | `false`, `true` |
| `repair_agent` | Choice | `normalize`, `impute`, `quarantine` |
| `risk_agent` | Score | `0` low, `1` medium, `2` high |

The example varies `evidence` (strength of support for the candidate) and
`ambiguity` (uncertainty about the field's meaning), both in `[0, 1]`. The
three agent models are transparent analytic softmax families defined in
[`system.py`](system.py). They are illustrative probability models: running
this example does not query Jev, execute a repair or measure cleaning quality.

The coordinator is a fixed function over one outcome from each agent:

```python
if applicable == "false" or repair == "quarantine" or risk == "high":
    final_action = "reject"
elif repair == "normalize" and risk == "low":
    final_action = "auto_fix"
else:
    final_action = "human_review"
```

The coordinator is not a fourth agent. In code, Score outcomes use `0` for
low and `2` for high. `SAMPLED_OUTCOME` describes the three node outputs;
the mapping from their joint outcome to the final action is deterministic.

## Run

From the repository root:

```bash
uv sync --locked
uv run python examples/three_agent_cleaning/run.py
```

Open `examples/three_agent_cleaning/output/report.html`. The output directory
also contains the declared system, experiment configuration, captured node
distributions and numerical results. The five steps in [`run.py`](run.py) are:
build the system, create an analytic adapter, capture an experiment, analyze
the declared joint and its action aggregation, and render the report.

At the centre `evidence=0.7`, `ambiguity=0.3`, the model produces:

| Quantity | Value |
| --- | ---: |
| Applicability Fisher trace | 3.074 |
| Repair Fisher trace | 1.773 |
| Risk Fisher trace | 1.995 |
| System Fisher trace | 6.842 |
| Information loss trace after action aggregation | 3.034 |
| `auto_fix` / `human_review` / `reject` probability | 0.236 / 0.294 / 0.470 |

The applicability node has Fisher rank 1 at the centre: a binary output
cannot locally distinguish both parameter directions by itself. The joint
distribution retains more information than the final three-way action.

The model **declares conditional independence of the three agent outcomes
given the parameters**. The system Fisher and action probabilities depend on
that assumption. Information loss is computed for the fixed action mapping
under this same declared joint. These values are local properties of this
illustrative model, not empirical accuracy or calibration estimates. No CRLB
is claimed because this example does not declare an observation likelihood
and sampling contract.
