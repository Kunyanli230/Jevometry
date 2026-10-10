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

Companion code:

- `examples/jev_ticket_sensitivity/run.py` — offline synthetic + optional `--live`
- `examples/jev_ticket_sensitivity/ticket_system.py` — analytic twin
- `examples/jev_ticket_sensitivity/paper_export.py` — tables/figures for this paper
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
