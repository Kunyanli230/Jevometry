# Coordination rules × Fisher information loss

Paper draft (arXiv / JAAMAS-oriented) and reproducible experiments for ranking
deterministic multi-agent coordination rules by Fisher information loss after
the action layer.

**Primary framing:** decentralized sensor-agent networks (AggregationMap as
exchange + quarantine; rank by `tr(Δ)=tr(I_Z−I_A)`). Main synthetic twin:
unit-disk 2D field (`examples/unit_disk_sensors/`). Real-data对照: IBRL 12-mote
subsample (same package). Cleaning-council tables/figures remain the analytic
illustration / proof of concept in `main.tex` until prose is updated.

> **NOTE:** `main-zh.tex` / `main-zh.pdf` is the Simplified Chinese 1:1 counterpart of `main.tex` (same sections, equations, tables, figures, citations; prose translated). Build with `lualatex` + `bibtex` (Noto Serif CJK SC / Noto Sans Mono CJK SC via fontspec).

## Contents

| Path | Role |
| --- | --- |
| `main.tex` / `references.bib` | English article source |
| `main-zh.tex` / `main-zh.pdf` | Simplified Chinese 1:1 counterpart (lualatex) |
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

Chinese PDF (`main-zh.pdf`):

```bash
cd papers/coord-fisher
lualatex -interaction=nonstopmode main-zh.tex
bibtex main-zh
lualatex -interaction=nonstopmode main-zh.tex
lualatex -interaction=nonstopmode main-zh.tex
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

## Unit-disk sensor twin (phase 1)

Synthetic decentralized sensing frame: θ = `(amplitude, length_scale)` of a
Gaussian bump on `[0,1]²`; **N=9** motes on a 3×3 grid with unit-disk radius
`R=0.45`. Three role agents (quality / exchange-mode / uncertainty) induce the
same joint alphabet as cleaning, so the seven AggregationMaps apply; quarantine
rules act as the uncertain→isolate / less-exchange safety valve.

```bash
uv run python examples/unit_disk_sensors/experiments.py
uv run pytest tests/unit/test_unit_disk_artifacts.py -q
```

| Path | Role |
| --- | --- |
| `examples/unit_disk_sensors/system.py` | Geometry, field, role agents, RULE_SPECS |
| `examples/unit_disk_sensors/experiments.py` | Centre `information_loss` export |
| `data/unit_disk_center_loss.csv` | Per-rule centre tr(I_Z), tr(I_A), tr_loss |
| `data/unit_disk_validation_summary.json` | Order / Fisher-ok / vs cleaning centre |


## IBRL real-data对照 (phase 2)

12-mote farthest-point subsample of Intel Berkeley Lab
([source](https://db.csail.mit.edu/labdata/labdata.html); files under
`examples/unit_disk_sensors/data/` with `SOURCE.txt` attribution).
θ = MinMax temperature/humidity; Iris-style logistic role agents; same seven
AggregationMaps as the unit-disk twin.

```bash
uv run python examples/unit_disk_sensors/ibrl_experiments.py
uv run pytest tests/unit/test_ibrl_artifacts.py -q
```

| Path | Role |
| --- | --- |
| `examples/unit_disk_sensors/ibrl_system.py` | Load subsample, fit agents, RULE_SPECS |
| `examples/unit_disk_sensors/ibrl_experiments.py` | Centre `information_loss` export |
| `data/ibrl_center_loss.csv` | Per-rule centre tr loss |
| `data/ibrl_validation_summary.json` | fisher_ok, mote ids, vs unit-disk |

## IMAS² grid-world comparison (phase 3)

Upstream: https://github.com/AronYoung414/multi-agent-active-perception-grid-world
(MIT, commit `0eae156`; vendored files + LICENSE in `third_party/imas2_grid/`).
θ = (adversary prior logit, slip) — a local-sensitivity surrogate, **not** MI.

```bash
uv run python examples/imas2_grid/experiments.py
uv run pytest tests/unit/test_imas2_grid_artifacts.py -q
```

| Path | Role |
| --- | --- |
| `data/imas2_rule_comparison.csv` | rule × policy_type × dynamics × sensor_range: tr_delta_I, cond_entropy, accuracy, seed, n_trajectories, selection_time_s, training_time_s |
| `data/imas2_placements.csv` | 126 placements (det., small): fisher_score, mi_score, entropy, accuracy |
| `data/imas2_placements_stochastic.csv` | same, stochastic |
| `data/imas2_summary.json` | Spearman / top-5 overlap, timing, retrain metadata, paper-REPORTED numbers (kept separate) |

Retrained rows (`policy_type=retrained_imas2`) come from a reduced local rerun of
upstream `model_training.py` in an isolated torch venv (`/tmp/imas2-run`, not a
project dependency): reported selection order 4,2,8,6,3, fewer iterations than
the paper's N=1000. Full upstream pipeline (35 sensor trainings × 1000 iters) was
not rerun on CPU.

## Iris real-data check

```bash
uv run python examples/three_agent_cleaning/iris_experiments.py
```

Uses `sklearn.datasets.load_iris` only (petal length/width → θ). Writes
`data/iris_center_loss.csv` and `data/iris_validation_summary.json`.
