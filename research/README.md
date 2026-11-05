# Optional research workspace

Experiments, labeling and papers live here; production does not import this package.
All tests remain in the repository's top-level `tests/` directory.

Install alongside the production/common distribution:

```bash
uv pip install --python .venv/bin/python -e '.[train,dev]' -e ./research
.venv/bin/rift-research --help
```

Use `compare` for frozen serving bundles, `evaluate` for CQL/BC/hybrid ablations,
`memory` for causal-history experiments, `cohort` for explicitly configured collection
studies, and `audit` / `review` for independent labeling. Labeling is currently deferred.
These commands write new experiment directories; they do not publish an active model.

```bash
.venv/bin/rift-research compare \
  --dataset /path/to/development/transitions.npz \
  --bundles /path/to/reference /path/to/candidate \
  --output /path/to/new/comparison
```

Comparisons require identical training label counts and evaluation matches. Reports
include per-action metrics, majority baseline, coverage/collapse checks, paired
match-bootstrap intervals, model hashes and local CPU serving latency. `--unseen`
excludes every model's saved training/development matches and requires equal evaluation
masks across models. It does not make a previously consulted cohort an untouched test.

The maintained trainers belong to `production/`; reusable features, model components,
metrics, splits and artifact handling belong to `common/`. Add experimental algorithms
here until their behavior and runtime contract are ready for deliberate production use.

Historical records: [synthesis](docs/RESEARCH_SYNTHESIS.md), [objective rules and label review](docs/OBJECTIVE_RULES.md),
[data plan](docs/DATA_PLAN.md), [model directions](docs/MODEL_IMPROVEMENTS.md),
[memory study](docs/MEMORY_STUDY.md), [original results](docs/RESULTS.md), and
[V2 results](docs/RESULTS_V2.md). Local datasets/weights remain outside Git under
`data/research/` and `models/`.
