# Unit-disk sensor twin (coord-fisher phase 1)

Synthetic 3×3 mote field with unit-disk radius `R=0.45`. Global θ =
`(amplitude, length_scale)` of a Gaussian bump; three role agents
(quality / exchange / uncertainty) reuse the cleaning joint alphabet so the
seven AggregationMaps apply unchanged.

```bash
# from repository root
uv run python examples/unit_disk_sensors/experiments.py
uv run pytest tests/unit/test_unit_disk_artifacts.py -q
```

Artifacts: `papers/coord-fisher/data/unit_disk_center_loss.csv`,
`unit_disk_validation_summary.json`.


## IBRL real-data对照 (phase 2)

12 farthest-point motes from Intel Berkeley Lab
([labdata](https://db.csail.mit.edu/labdata/labdata.html)); committed subsample
in `data/ibrl_subsample.tsv` (see `data/SOURCE.txt`). θ = MinMax temperature /
humidity; logistic role agents; same seven AggregationMaps.

```bash
uv run python examples/unit_disk_sensors/ibrl_experiments.py
uv run pytest tests/unit/test_ibrl_artifacts.py -q
```

Artifacts: `papers/coord-fisher/data/ibrl_center_loss.csv`,
`ibrl_validation_summary.json`.

