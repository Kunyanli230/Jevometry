# TypeSafe live ticket sensitivity (Paper 2)

arXiv / JAAMAS-oriented draft measuring Fisher geometry of a live TypeSafe
decision-model API on the Jevometry ticket-triage example.

Paper 1 (coordination rules × Fisher loss) lives in `papers/coord-fisher/` and
is out of scope here; cite it only as a companion.

## Contents

| Path | Role |
| --- | --- |
| `main.tex` / `references.bib` | English article source |
| `main.pdf` | Built PDF (pdfLaTeX) |
| `data/` | CSV/JSON summaries (no secrets, no huge traces) |
| `figures/` | Publication figures (PDF/PNG) |
| `data/surrogate_*` | Surrogate coordination-rule loss / action-mass exports |

Companion code:

- `examples/jev_ticket_sensitivity/run.py` — offline synthetic + optional `--live`
- `examples/jev_ticket_sensitivity/ticket_system.py` — analytic twin
- `examples/jev_ticket_sensitivity/paper_export.py` — tables/figures for this paper
- `examples/jev_ticket_sensitivity/compare_surrogate_rules.py` — 3×3 surrogate loss grid
- `examples/jev_ticket_sensitivity/export_surrogate_action_masses.py` — action-mass CSV
- Stencil shrink (amount `0.25`, wait `0.05`, live single-scale) from PR #3

## Reproduce

```bash
# from repository root
uv sync --locked

# offline synthetic (no credentials)
uv run python examples/jev_ticket_sensitivity/run.py

# optional live (requires TYPESAFE_API_KEY + TYPESAFE_MODEL; never commit values)
uv run python examples/jev_ticket_sensitivity/run.py --live

# export CSV/JSON/PNG into papers/typesafe-ticket/{data,figures}
uv run python examples/jev_ticket_sensitivity/paper_export.py

# build PDF
cd papers/typesafe-ticket && pdflatex -interaction=nonstopmode main.tex \
  && bibtex main && pdflatex -interaction=nonstopmode main.tex \
  && pdflatex -interaction=nonstopmode main.tex
```

Live run directories under `examples/jev_ticket_sensitivity/output/` are
gitignored. Paper tables use the exported summaries in `data/`.


## Surrogate coordination rules × Fisher loss

Offline twin (`DECLARED_PRODUCT` + theta-independent `AggregationMap`) compares
four pushforward rules: `near_identity`, `baseline`, `stricter`, `coarse_binary`.

Defaults (documented in `ticket_system.py`):

- `LOGIT_SCALE = 15` — smallest scale in `{5,8,10,12,15,20}` with centre order
  `baseline ≤ stricter ≤ coarse_binary`, `tr(I_Z) ≥ 0.02`, and max finite grid cells
- `PROB_FLOOR = 1e-12` — twin-only clip at node softmax / joint exits so the 3×3
  grid (including `amount=600`) stays Fisher-defined without changing library ε

Reproduce:

```bash
uv run python examples/jev_ticket_sensitivity/compare_surrogate_rules.py
uv run python examples/jev_ticket_sensitivity/export_surrogate_action_masses.py
uv run python examples/jev_ticket_sensitivity/paper_export.py
```

Artifacts:

| Path | Role |
| --- | --- |
| `data/surrogate_rule_loss_grid.csv` | 3×3 × 4 rules: `tr_Iz`, `tr_Ia`, `tr_loss`, status |
| `data/surrogate_action_masses.csv` | pushforward action masses at `(250,8)`, `(600,2)`, `(600,8)` |
| `data/surrogate_rule_center_summary.json` | centre losses, `LOGIT_SCALE`/`PROB_FLOOR`, `action_mass_points`, `legacy_flip_present` |
| `figures/surrogate_rule_loss_center.{png,pdf}` | centre four-rule loss bar |
| `figures/surrogate_rule_loss_heatmap.{png,pdf}` | amount×wait heatmaps (non-`near_identity`) |

Under the current defaults the centre still shows a legacy mass flip
(`standard_queue → priority_queue`); high-amount cells are finite but nearly
saturated (`escalate ≈ 1`), so rule losses collapse — a limitation for the draft.

## Centre-point snapshot (live re-run after stencil shrink)

At `(amount, wait_hours)=(250, 8)`, Fisher status **ok** on all nodes:

| Node | H (nats) | tr(I) | I_amount | I_wait | Identifiability |
| --- | ---: | ---: | ---: | ---: | --- |
| request_type | 1.040 | 0.858 | 0.138 | 0.720 | — |
| needs_escalation | 0.500 | 0.085 | 0.0225 | 0.0625 | rank-deficient |
| impact | 0.309 | 2.613 | 0.154 | 2.460 | rank-deficient |

System / action Fisher remain **unsupported** (`no_joint_model`,
`deterministic_policy`).

## Before → after stencil

| | amount step | wait step | live stencil | node_geometry |
| --- | ---: | ---: | --- | --- |
| before | 1.0 | 0.25 | two-scale | unstable (`step_stability_exceeded`) |
| after | 0.25 | 0.05 | single-scale | ok |
