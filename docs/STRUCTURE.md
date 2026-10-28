# Project structure

```text
offlineRift/
  src/offline_rift/              # Backend implementation
    __init__.py
    __main__.py                 # python -m offline_rift
    workflow.py                 # Collection/build/train orchestration
    pipeline/
      data_pipeline.py          # Riot requests and timeline reconstruction
      features.py               # Shared state schema and objective/item features
      action_labeler.py         # Weak labels and MDP dataset validation
    training/
      rl_agent.py               # CQL training and policy persistence
      train_macro_model.py      # Direct training CLI
    inference/
      live_client.py            # Local live-game adaptation and polling
  tests/
    conftest.py                 # Shared deterministic match fixtures
    test_workflow.py            # Workflow/cache/persistence integration tests
    pipeline/test_pipeline.py
    training/test_training.py
    inference/test_live.py
  configs/
    classic_timers.json         # Illustrative timer configuration
  docs/                         # Workflow, interfaces, recorded results, structure
  AGENTS.md                     # Repository conventions and validation instructions
  README.md                     # Setup and everyday commands
  pyproject.toml                # Dependencies, packaging, and command entry points
  workflow.json                 # Local settings, ignored by Git
  data/                         # Local raw games, catalogs, datasets, manifests
  models/                       # Local learned weights and training reports
```

The root `workflow.py`, `data_pipeline.py`, `action_labeler.py`, `features.py`,
`rl_agent.py`, `train_macro_model.py`, and `live_client.py` are small compatibility
modules. They re-export the existing public interfaces; script entry points delegate
to the package. They do not duplicate implementation.

## Commands and imports

Existing commands remain supported:

```bash
rift status
rift run --refresh
python workflow.py status
python train_macro_model.py --help
```

You can also run the workflow as a package:

```bash
python -m offline_rift status
```

New Python integrations should import the implementation package:

```python
from offline_rift.pipeline.data_pipeline import RiotDataIngestor, TimelineProcessor
from offline_rift.pipeline.action_labeler import MDPDatasetBuilder
from offline_rift.pipeline.features import FeatureConfig, FeatureEncoder
from offline_rift.training.rl_agent import OfflineMacroTrainer, MacroPolicy
from offline_rift.inference.live_client import LiveClientInference
from offline_rift.workflow import ResearchWorkflow
```

After updating a checkout, refresh the editable installation so the package and console
entry points resolve correctly:

```bash
uv pip install -e '.[train,dev]'
```

The configuration schema, collection behavior, features, labels, rewards, training
algorithm, and inference behavior are unchanged by this reorganization. Existing
`workflow.json`, raw matches, datasets, models, and latest-artifact references keep their
original locations. Nothing is retrained or recollected as part of the move.

Source hashes participate in dataset/model fingerprints. The next explicit build/run
can therefore create a new artifact after this source-layout change; previous artifacts
are retained. This is the existing source-change detection behavior.
