# Classifier iteration and serving separation — 2026-11-05

The selected candidate is `models/candidates/20261105-imitation-mlp-seed43-v1`.
It is an experimental formation classifier with deferred independent labeling.
The active model reference remains `models/faker/20261104T163045Z-2e7c3564`.
No authenticated live-game integration was exercised.

## Fixed development comparison

All candidates use the corrected `sr-2026-v3` features from the frozen 443-match
development corpus: 355 training matches / 109,056 transitions and 88 validation
matches / 26,772 transitions. Both team perspectives stay together. The 846 later-cohort
matches were excluded from all fitting and development model selection in this iteration.
The 88-match split was previously consulted; it is a development comparison.

| Model | Seed | Updates | Batch | Macro F1 | Label agreement | Balanced accuracy | Predicted categories |
|---|---:|---:|---:|---:|---:|---:|---:|
| Corrected CQL + team-set hybrid | 42 | 10,000 CQL + 10,000 BC | 64 | 22.22% | 37.81% | 27.21% | 9/12 |
| Residual imitation | 42 | 20,000 | 256 | 20.99% | 31.48% | 22.14% | 12/12 |
| MLP imitation | 42 | 10,000 | 128 | 23.51% | 36.56% | 31.20% | 10/12 |
| MLP imitation, selected | 43 | 10,000 | 128 | 23.67% | 35.03% | 30.57% | 11/12 |

The residual candidate was rejected despite broader coverage. The selected MLP has
177 inputs, two 128-unit ReLU layers and a twelve-logit head. It uses AdamW at 0.0003,
weight decay 0.0001, gradient clipping at 5 and heuristic confidence weighting.
Inverse-square-root-frequency sampling replaces inverse-square-root class loss weights;
there is no extra double weighting. There is no early stopping or fitted probability
calibration. Two seeds do not establish broad optimizer robustness.

The majority label is dispersed lane coverage, with 27.89% development agreement.
All twelve training labels are present, but label counts remain substantially imbalanced.
The selected candidate never predicts unclassified on the development split; broader
coverage is not calibrated abstention or proof of valid advice. Its higher macro F1
comes with lower overall label agreement than the corrected hybrid.

Artifacts: `data/research/20261105-{corrected-dev-exploratory,imitation-residual-v1,imitation-mlp-v1,imitation-mlp-seed43-v1}/`.
The final vectorized-serving comparison is
`data/research/20261105-serving-comparison-vectorized-v2/report.json`.
Earlier `serving-comparison-v1` and `serving-comparison-seeds-v1` reports retain an
intermediate runtime's inefficient Q-call timings and are superseded for latency claims.

## Frozen evaluation on the later cohort

After selecting seed 43 on development data, its frozen weights and the corrected
hybrid were evaluated once on 846 other matches / 253,480 transitions. Saved train and
development IDs were excluded. This cohort had been consulted in historical studies;
it is not a fresh untouched confirmatory test. No further fitting followed these results.

| Model | Macro F1 | Label agreement | Balanced accuracy | Predicted categories |
|---|---:|---:|---:|---:|
| Corrected hybrid | 21.98% | 37.92% | 27.60% | 9/12 |
| Selected MLP | 23.32% | 35.70% | 29.38% | 11/12 |
| Majority baseline | 3.68% | 28.30% | 8.33% | 1/12 |

The MLP-minus-hybrid macro F1 difference is +1.34 percentage points, with a paired
1,000-draw match-bootstrap 95% interval of [+1.05, +1.64] points. This conditions on
these fitted models and this cohort; it excludes seed uncertainty, annotation validity,
selection effects and strategic usefulness. Neither learned model collapses to one label.

Artifact: `data/research/20261105-candidate-later-cohort-v1/report.json`.

## Runtime and packaging checks

The classifier contains 166,453 weight bytes versus 1,012,435 for the corrected hybrid
(about 84% fewer). Local CPU serving, one thread, five warmups and fifty measured calls:
MLP median 0.294 ms / p95 0.340 ms; vectorized hybrid median 0.692 ms / p95 0.825 ms.
These are repeated single-state microbenchmarks, excluding live transport and feature
encoding. Training keeps the corpus on CPU and transfers batches; NPZ loading is still
in memory and is not an out-of-core/distributed solution.

The production wheel includes application and common code, objective rules and UI assets,
and excludes research. An installed-wheel check loaded and served the classifier while
imports of research, d3rlpy and training orchestration were blocked. Classifier weights
load with `weights_only=True`, and bundle hashes are checked before deserialization.
Legacy CQL bundles still load through the optional d3rlpy factory. Checksums detect
corruption; they do not authenticate publishers.

Preview the candidate explicitly without changing the active reference:

```bash
.venv/bin/rift-ui --model models/candidates/20261105-imitation-mlp-seed43-v1
```

This is a usable inference artifact, not a validated strategy engine. The Live Client
still lacks the geometry that defines most targets. Scores are weak-label evidence,
not action-success or win probabilities. Independent labeling and reviewed promotion
remain separate work.

Validation: 93 tests passed, including real CPU optimizer/save/reload checks,
serving parity, dependency boundaries and localhost dashboard integration. Ruff lint,
format checks, CLI help, local documentation links and `git diff --check` passed.
All tests remain in the single top-level `tests/` folder.
