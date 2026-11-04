# Backend reference

Prefer `workflow.py` for normal operation. Lower-level interfaces remain available.
Implementation lives in `src/offline_rift/`; root modules preserve existing imports and
script commands. See [Project structure](STRUCTURE.md).

## Ingestion

```python
from offline_rift.pipeline.data_pipeline import RiotDataIngestor

api = RiotDataIngestor(api_key, "asia")
try:
    ids = api.fetch_match_ids(puuid, start=0, count=100, queue=420)
    match = api.fetch_match(ids[0])
    timeline = api.fetch_timeline(ids[0])
    snapshots = api.parse_timeline(match, timeline)
finally:
    api.close()
```

`TimelineProcessor(config).parse(match, timeline)` processes offline JSON without a key.
Only completed 5v5 Summoner's Rift matches are supported. Frames are causally forward-filled
onto a 10-second grid; events apply at their own timestamps. Exact game end adds a final
possibly shorter interval. No future frames or final economic statistics enter states.

Inventory tracks purchases, sales, destruction, and undo. Objective aliases include the
real `BARON_NASHOR`, `RIFTHERALD`, and `DRAGON` payload values. Ability cooldowns remain
unknown because the data sources do not provide enough cast history.

## Features

`FeatureEncoder(config, ItemCatalog(item_json))` exposes ordered `names` and
`encode(snapshot, team) -> numpy.float32[feature_count]`. Team is 100 or 200.
Features cover gold difference, role level gaps, per-champion levels and inventory/spikes,
coordinates, density grids, objective estimates, epic buff windows, frame freshness,
and availability masks. Added context includes masked team item/level advantages,
base/forward/lane fractions and team spread. Geometry stays fully masked in `live`.
Inventory value is a purchase-investment proxy, never team gold or combat strength.
Team player slots sort by canonical champion name.

`live` masks features absent from Live Client; `full` retains offline economic/spatial
features. Schema version, names, configuration, and catalog accompany datasets and policy
bundles. Policy loading checks the contract and model dimensions.

## Labels and MDP datasets

```python
from offline_rift.pipeline.action_labeler import MacroActionLabeler, MDPDatasetBuilder, load_dataset
from offline_rift.pipeline.features import FeatureConfig, ItemCatalog

action, confidence = MacroActionLabeler().classify(snapshots[0], team=100)
builder = MDPDatasetBuilder(FeatureConfig(), ItemCatalog(item_json), reward_mode="sparse")
builder.add_match(match, snapshots, catalog=ItemCatalog(patch_specific_item_json))
builder.save("custom.npz", provenance={"experiment": "my-experiment"})
arrays, metadata = load_dataset("custom.npz")
```

| ID | Action | Evidence |
|---|---|---|
| 0–2 | Dragon / Baron / Herald area setup | 3+ allies near the enabled pit, estimated ETA ≤60s |
| 3 | Cross-map pressure | Enemy objective cluster and distant forward side pressure |
| 4 | Side-lane split formation | 3+ mid positions and 1–2 forward side positions |
| 5–7 | Group mid / top side / bottom side | 3+ outside-base positions in a map band |
| 8 | Own-half defensive formation | 3+ allies in own half with 3+ enemies there |
| 9 | Base regroup | At least three allies within the base radius |
| 10 | Dispersed lane coverage | Active positions occupy all three map bands |
| 11 | Unclassified / insufficient evidence | Missing/stale coordinates or no matching formation |

Priority is base, objective, cross-map, split, defense, lane group, coverage, unclassified.
These map bands approximate formations; top/bottom side includes jungle positions.
Objective presence does not prove attacking, baiting, vision setup, or successful control.
`macro-v2` and `rift-v2` are incompatible with old artifacts. Rebuild and retrain;
there is one current implementation, and historical files are retained on disk.

Each outgoing interval uses its starting state's label.
`label_window(window, team)` majority-votes over supplied snapshots. Labels are weak
geometric supervision, not observed intent or optimality. Confidence is saved as a heuristic reliability weight for the companion classifier;
it is not a probability and does not reweight CQL transitions.

NPZ arrays: `observations`, `actions`, `rewards`, `next_observations`, `terminals`,
`episode_ids`, `match_ids`, `teams`, `timestamps`, `next_timestamps`, `label_confidence`.
Metadata is JSON. Loading uses `allow_pickle=False` and validates dimensions, finite
values, action IDs, terminal boundaries, contiguous episodes, and next-state continuity.
Both teams contribute separate episodes; duplicate match additions fail.

Sparse reward is zero until terminal, then +1 for a win or −1 for a loss. `gold` adds
team gold-advantage change / 10,000 while retaining terminal outcomes. Terminal next
observations are stored but never bootstrapped.

## Training and inference

```python
from offline_rift.training.rl_agent import MacroPolicy, OfflineMacroTrainer

trainer = OfflineMacroTrainer(device="cpu", batch_size=64, alpha=1.0, gamma=0.99, seed=42)
report = trainer.train("custom.npz", "models/my-run", n_steps=10000)
policy = MacroPolicy("models/my-run")
result = policy.recommend(policy.encoder.encode(snapshots[0], team=100))
```

The trainer uses d3rlpy 2.8.1 Discrete CQL with a shared 10→64→64 player network,
separate mean pools for the two teams, and a 128×128 context network. The `team_set`
encoder is invariant to player slot permutations within a team. `encoder_type="mlp"`
selects a 128×128 vector encoder for controlled comparison. Both use the same explicit
twelve-action space and match-level holdout. Reports include confusion matrices,
per-action support/recall, balanced accuracy, macro F1, coverage, collapsed predictions,
heuristic agreement, and majority baseline. These measure weak-label agreement, not
strategic value. References:
[MDPDataset](https://d3rlpy.readthedocs.io/en/v2.8.1/references/generated/d3rlpy.dataset.MDPDataset.html),
[algorithms](https://d3rlpy.readthedocs.io/en/v2.8.1/references/algos.html).

Direct commands remain available:

```bash
rift-train --dataset custom.npz --output models/my-run --steps 10000
rift-live --model models/my-run --team 100 --once
```

Direct CLIs retain explicit paths and original overwrite behavior. Prefer the workflow
wrapper for automatic IDs, reuse checks, and provenance.

Recommendations contain action ID/name, prediction method, selected `q_value`, all
twelve `q_values`, and separate `macro_scores` for supervised modes. The default hybrid
uses balanced neural pattern logits plus a bounded Q tie-breaker; it does not choose
the largest raw Q-value. Unsupported classes and objective timing violations are masked.
`prediction_mode="balanced_bc"` uses the pattern head; `"cql"` selects the raw CQL baseline.
See [model directions](MODEL_IMPROVEMENTS.md) for loss weighting, score formula and limits.
Q-values estimate discounted returns, not calibrated win probabilities. Live output
reports partial observations and unavailable features. Connection failures emit offline
status; malformed states emit invalid-state status. Event history reconstruction is
idempotent across polls and naturally resets for a new game. No stale recommendation
is returned after an error.
