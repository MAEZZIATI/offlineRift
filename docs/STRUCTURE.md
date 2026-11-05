# Workspace structure and dependencies

The code is physically separated by responsibility within this repository:

```text
offlineRift/
  production/src/offline_rift/
    workflow.py                 # Stable collection/build/train/publication workflow
    training/                   # Maintained CQL and classifier trainers
    inference/                  # Policy loading, live adaptation and transport
    ui/                         # Dashboard service, HTTP server and browser assets
  common/src/rift_common/
    pipeline/                   # Causal timeline, shared features, labels and validation
    models/                     # Torch encoders, scoring, metrics, match splits, optional CQL factory
    artifacts.py                # Atomic persistence, digests and bundle integrity
    review.py                   # Portable review contracts and promotion checks
  research/
    pyproject.toml              # Separately installed optional package
    src/rift_research/          # Compare, evaluation, memory, cohort, audit and review
    docs/                       # Historical findings, plans and papers
  tests/                        # Every test and the shared conftest.py
  docs/                         # Operational guides and production interfaces
  configs/                      # Tracked examples; local workflow.json is ignored
  scripts/                      # Thin compatibility launchers, no implementation
  pyproject.toml                # Builds production + common; excludes research
  workflow.py                   # Compatibility launcher
  data/                         # Local cached games, datasets and research artifacts
  models/                       # Local historical production bundles
```

```mermaid
flowchart LR
    Production --> Common
    Research --> Common
    Research --> Production
```

Common code imports neither application nor research packages. The inference package
imports no training orchestration. These boundaries are enforced in `tests/test_architecture.py`.
Research can invoke the production collector/trainers and load production bundles;
production commands remain usable when research is not installed.

## Packaging and commands

The root wheel includes `offline_rift` from `production/src` and `rift_common` from
`common/src`. Keeping common in the same distribution pins serving and feature contracts
together. `research/pyproject.toml` builds `offline-rift-research` and depends on the
production distribution, without copying shared implementation.

```bash
uv pip install --python .venv/bin/python -e '.[train,dev]' -e ./research
rift --help
rift-train --help
rift-research --help
```

The former `rift research-evaluate`, `research-memory`, `research-cohort`, `research-audit`
and `research-review` commands now live under `rift-research evaluate`, `memory`, `cohort`,
`audit` and `review`. This is an intentional CLI migration. For frozen serving comparisons,
use `rift-research compare`; the older `evaluate` command retains CQL-specific ablations.

Shared Python imports now use `rift_common`:

```python
from rift_common.pipeline.features import FeatureConfig, FeatureEncoder
from rift_common.pipeline.action_labeler import MDPDatasetBuilder, load_dataset
from offline_rift.inference.policy import MacroPolicy
from offline_rift.training.imitation import ImitationTrainer
```

Compatibility scripts still delegate to these implementations. Existing artifact paths
and local workflow configuration are preserved. Source fingerprints can create new
versioned datasets after migration; they never replace historical artifacts in place.

## Model bundles

Classifier bundles contain `metadata.json`, `macro_predictor.pt`, `bundle.json` and
`run.json`. The loader verifies metadata/weight hashes before deserialization and uses
Torch's weights-only loading. Checksums detect corruption; they do not authenticate a
publisher. CQL/hybrid legacy bundles continue to load with their original `policy.d3`
and optional predictor. No filename/schema migration of stored bundles is required.

`MacroPolicy.predict_batch` and `recommend` use the same scoring implementation. Serving
uses bounded batches. Classifier bundles do not import d3rlpy, research or training
orchestration, and return `null` Q values rather than inventing returns.
