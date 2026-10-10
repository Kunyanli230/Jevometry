# Coordination rules × Fisher information loss

Paper draft (arXiv / JAAMAS-oriented) and reproducible experiments for ranking
deterministic multi-agent coordination rules by Fisher information loss after
the action layer.

## Contents

| Path | Role |
| --- | --- |
| `main.tex` / `references.bib` | English article source |
| `main.pdf` | Built PDF (Tectonic) |
| `data/` | CSV/JSON from the grid sweep |
| `figures/` | Publication figures (PDF/PNG) |

Companion code lives in `examples/three_agent_cleaning/`:

- `system.py` — analytic agents + six pure AggregationMaps
- `compare_rules.py` — centre-point table
- `paper_experiments.py` — grid sweep, tables, figures

## Reproduce

```bash
# from repository root
uv sync --locked
uv pip install matplotlib
uv run python examples/three_agent_cleaning/compare_rules.py
uv run python examples/three_agent_cleaning/paper_experiments.py
```

Build the PDF (optional; needs [Tectonic](https://tectonic-typesetting.github.io/)):

```bash
cd papers/coord-fisher && tectonic main.tex
```

## Centre-point snapshot

At `evidence=0.7`, `ambiguity=0.3`, $\mathrm{tr}(I_Z)\approx 6.84$:

| Rule | tr(loss) | \|A\| |
| --- | ---: | ---: |
| near_identity | 0.00 | 18 |
| quarantine_review | 2.27 | 3 |
| wider | 3.01 | 4 |
| baseline | 3.03 | 3 |
| stricter | 3.42 | 3 |
| coarse_binary | 3.71 | 2 |

Core ordering: `near_identity ≤ wider ≤ baseline ≤ stricter`.

## Scope note

Paper 2 (TypeSafe live ticket sensitivity) is out of scope here; see
`papers/NOTE.md` if present.
