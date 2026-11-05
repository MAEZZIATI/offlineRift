# Architecture

```text
production/src/offline_rift/
  collection.py          Riot regional client, retries and raw collection
  catalogs.py            Exact-patch item catalogs
  workflow.py            One collect/build/train/use orchestration path
  training/trainer.py    Explicit discrete CQL and an imitation baseline
  inference/policy.py    Weights-only runtime and supported action ranking
  inference/live_client.py  Live adaptation, history and subject selection
  inference/transport.py    Bounded local / Windows loopback transport
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
  metrics.py             Behavior/support diagnostics
  artifacts.py           Atomic persistence, hashes and bundle integrity
research/src/rift_research/
  evaluation.py          Dataset audits and frozen-policy diagnostics
  cli.py                 Optional research entry point
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
