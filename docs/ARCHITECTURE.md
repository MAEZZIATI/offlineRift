# Architecture

```text
production/src/offline_rift/
  collection.py          Riot regional client, retries and raw collection
  catalogs.py            Exact-patch item catalogs
  workflow.py            One collect/build/train/use orchestration path
  training/trainer.py    Shared training path for CQL, IQL and imitation
  training/iql.py        Discrete expectile/twin-critic/actor updates
  training/augmentation.py  Sampled observation omissions with feasibility preservation
  inference/policy.py    Weights-only runtime and supported action ranking
  inference/live_client.py  Live adaptation, history and subject selection
  inference/transport.py    Bounded local / Windows loopback transport
  inference/availability.py  Explicit opt-in aggregate availability counts
  ui/                    Snapshot cache, localhost server, browser assets
common/src/rift_common/
  actions.py             Action taxonomy, detailed instructions, feasibility
  state.py               Shared causal actor/role/public-history feature contract
  timeline.py            Real-frame reconstruction and timestamped events
  behavior.py            Interval-based executed-option inference
  dataset.py             Complete actor episodes and transition validation
  objectives.py + objective_rules.json  Pinned explicit objective estimates
  items.py               Patch catalog valuation and unknown masks
  network.py             Shared value/support network and ranking functions
  observation_masks.py   Named omissions shared by augmentation and research diagnostics
  metrics.py             Behavior/support diagnostics
  artifacts.py           Atomic persistence, hashes and bundle integrity
research/src/rift_research/
  evaluation.py          Dataset audits and frozen-policy diagnostics
  cli.py                 Optional research entry point
  experiments.py         Matched-split studies using the maintained builder/trainer
  sensitivity.py         Optimizer-seed, value-range and omission diagnostics
  combat.py              Raw-provenance-checked combat target evidence audit
  calibration.py         Match-separated coverage calibration and threshold transfer
  decision_regret.py     Experimental minimax supported action selection across seeds/views
research/README.md        New-objective study protocol
research/docs/            Specific observed runs
tests/                  All tests in one folder
```

Common imports no application package. Production imports no research package. The
research package depends on production/common and is excluded from the production wheel.
Runtime loading never imports training orchestration. Tests enforce those boundaries.
No old formation code, paper drafts, audit viewer, encoders, compatibility script tree
or old direct-training orchestration is maintained alongside the new objective.

Version 2 configuration retains verified source PUUIDs and primary account. Data and model
namespaces are separate from historical artifacts. Builds fingerprint raw bytes, catalogs
and shared reconstruction source; completed outputs are immutable. Failed/interrupted
reports remain stored, and latest pointers change only after successful validation.

Bundles contain `policy.pt`, `metadata.json`, `bundle.json` and `run.json`. The loader
checks next-action schema, exact feature/action names and hashes before weights-only
Torch loading. Checksums detect corruption, not publisher authenticity. Legacy formation
bundles are rejected rather than silently reinterpreted.

IQL bundles explicitly identify the `independent_actor_v1` architecture; existing CQL/BC
bundles retain `shared_v1`. Runtime loads only the categorical actor and first critic.
The second critic and expectile value network are training-only. Policy loading remains
independent of the training package.

Experimental bounded CQL bundles declare `bounded_shared_v1`; only their value head
applies tanh. Ordinary CQL/BC keep `shared_v1`. Joint observation dropout is training-only.
Coverage calibration remains an optional research analysis and does not alter runtime
abstention settings or latest model references.
