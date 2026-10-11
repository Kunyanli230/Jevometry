# IMAS² grid-world comparison (coord-fisher phase 3)

Reuses the IMAS² grid world (vendored, MIT: `third_party/imas2_grid/`, upstream
commit `0eae156`) — 10×10 grid, 12 obstacles, 9 candidate sensors, value-iteration
nominal/adversary robots, NW/NE/SW/SE sensing footprints (small 2×2 / large 3×3),
obs_noise 0.5. Main placement = IMAS²'s reported pick (2,3,4,6,8).

- `system.py` — vectorised dynamics/emissions, per-sensor categories (adv/nom/unc/none),
  seven fusion rules, exact per-step mixture law + FD Jacobian for θ=(prior_logit, slip).
- `experiments.py` — tr(ΔI) vs fused posterior entropy / accuracy (same HMM evaluator,
  same seeds) for fixed / random / retrained orientation × det/stoch × small/large;
  Fisher-vs-MI scoring of all 126 placements.

```bash
uv run python examples/imas2_grid/experiments.py
uv run pytest tests/unit/test_imas2_grid_artifacts.py -q
```

Retrained IMAS² rows need a separate torch venv (not a project dependency); see
`papers/coord-fisher/README.md`. Fisher here is a local-sensitivity surrogate, not MI.
