# Coordination rules × Fisher information loss

Paper draft (arXiv / JAAMAS-oriented) and reproducible experiments for ranking
deterministic multi-agent coordination rules by Fisher information loss after
the action layer.

## Contents

| Path | Role |
| --- | --- |
| `main.tex` / `references.bib` | English article source |
| `main.pdf` | Built PDF (Tectonic) |
| `data/` | CSV/JSON from the grid sweep and iteration exports |
| `figures/` | Publication figures (PDF/PNG) |
| `data/iteration_summary.json` | Machine-readable wrap-up of the 10-round iteration |

Companion code lives in `examples/three_agent_cleaning/`:

- `system.py` — analytic agents + **seven** pure AggregationMaps
- `compare_rules.py` — centre-point table
- `paper_experiments.py` — 8×8 grid sweep, tables, figures
- `export_center_loss_anisotropy.py` — centre Δ diagonals / eigenvalues
- `export_rule_loss_anisotropy_grid.py` — grid anisotropy + `frac_loss` heatmap

## Seven coordination rules

| Rule | Idea |
| --- | --- |
| `near_identity` | One label per joint outcome (upper bound on retention) |
| `quarantine_wider` | Quarantine→`human_review` **and** wider dual `auto_normalize` / `auto_impute` |
| `quarantine_review` | Quarantine→`human_review` instead of reject |
| `wider` | Split low-risk autos; quarantine still rejects |
| `baseline` | Fixed reject / auto_fix / human_review |
| `stricter` | Mid-risk also rejects |
| `coarse_binary` | Binary reject vs proceed |

`quarantine_wider` mechanism: reject only on inapplicable or high risk; quarantine
routes to review; normalize/impute at low risk keep distinct auto labels. On the
design grid, `tr_loss(quarantine_wider) ≤ tr_loss(quarantine_review)` holds on
**65/65** points (`hold_ratio=1.0`, see `data/quarantine_wider_vs_review.json`).

## Anisotropy / fractional-loss artifacts

| Path | Role |
| --- | --- |
| `data/center_loss_anisotropy.csv` | Centre Δ evidence/ambiguity diagonals + eigenvalues |
| `data/rule_loss_anisotropy_grid.csv` | 8×8 grid Δ diagonals + `frac_loss` / `frac_delta_ambiguity` |
| `figures/fig_frac_loss_heatmap.png` | Fractional loss heatmaps |
| `data/quarantine_action_mass_compare.csv` | Centre action masses: review vs wider |
| `data/quarantine_wider_vs_review.json` | Grid dominance stats |
| `data/center_summary.json` / `rule_loss_grid.csv` | Main paper_experiments exports |
| `figures/fig_loss_center.png` / `fig_action_masses.png` | Centre bar / stacked masses (tr_loss ascending) |

## Reproduce

```bash
# from repository root
uv sync --locked
uv pip install matplotlib
uv run python examples/three_agent_cleaning/compare_rules.py
uv run python examples/three_agent_cleaning/paper_experiments.py
uv run python examples/three_agent_cleaning/export_center_loss_anisotropy.py
uv run python examples/three_agent_cleaning/export_rule_loss_anisotropy_grid.py
uv run pytest tests/unit/test_coord_fisher_artifacts.py tests/unit/test_center_loss_anisotropy.py -q
```

Build the PDF (optional; needs [Tectonic](https://tectonic-typesetting.github.io/)):

```bash
cd papers/coord-fisher && tectonic main.tex
```

## Centre-point snapshot

At `evidence=0.7`, `ambiguity=0.3`, $\mathrm{tr}(I_Z)\approx 6.84$ (ascending loss):

| Rule | tr(loss) | \|A\| |
| --- | ---: | ---: |
| near_identity | 0.00 | 18 |
| quarantine_wider | 2.16 | 4 |
| quarantine_review | 2.27 | 3 |
| wider | 3.01 | 4 |
| baseline | 3.03 | 3 |
| stricter | 3.42 | 3 |
| coarse_binary | 3.71 | 2 |

Core ordering: `near_identity ≤ wider ≤ baseline ≤ stricter` (64/64 on the 8×8 grid).

## Scope note

Paper 2 (TypeSafe live ticket sensitivity) is out of scope here; see
`papers/NOTE.md` if present.
