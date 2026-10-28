# Backend reference

Prefer `workflow.py` for normal operation. Lower-level interfaces remain available.

## Ingestion

```python
from data_pipeline import RiotDataIngestor

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
and availability masks. Team player slots sort by canonical champion name.

`live` masks features absent from Live Client; `full` retains offline economic/spatial
features. Schema version, names, configuration, and catalog accompany datasets and policy
bundles. Policy loading checks the contract and model dimensions.

## Labels and MDP datasets

```python
from action_labeler import MacroActionLabeler, MDPDatasetBuilder, load_dataset
from features import FeatureConfig, ItemCatalog

action, confidence = MacroActionLabeler().classify(snapshots[0], team=100)
builder = MDPDatasetBuilder(FeatureConfig(), ItemCatalog(item_json), reward_mode="sparse")
builder.add_match(match, snapshots, catalog=ItemCatalog(patch_specific_item_json))
builder.save("custom.npz", provenance={"experiment": "my-experiment"})
arrays, metadata = load_dataset("custom.npz")
```

| ID | Action | Heuristic |
|---|---|---|
| 0 | Force/Bait Objective | 3+ allies near a pit ready within 60 seconds |
| 1 | Cross-map Play | Enemy pit cluster plus distant allied pressure/opposite pit |
| 2 | Split Push & Stall | One allied side pressure position and four allied mid positions |
| 3 | Group & Push Mid | 3+ allies in the mid corridor outside base |
| 4 | Defensive Reset & Setup Vision | Remaining states or missing/stale coordinates |

Rules use that priority order. Each outgoing interval uses its starting state's label.
`label_window(window, team)` majority-votes over supplied snapshots. Labels are weak
geometric supervision, not observed intent or optimality. Confidence is saved but is
not a probability or training weight.

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
from rl_agent import MacroPolicy, OfflineMacroTrainer

trainer = OfflineMacroTrainer(device="cpu", batch_size=64, alpha=1.0, gamma=0.99, seed=42)
report = trainer.train("custom.npz", "models/my-run", n_steps=10000)
policy = MacroPolicy("models/my-run")
result = policy.recommend(policy.encoder.encode(snapshots[0], team=100))
```

The trainer uses d3rlpy 2.8.1 Discrete CQL, a 128×128 vector encoder, an explicit
five-action space, and match-level holdout. References:
[MDPDataset](https://d3rlpy.readthedocs.io/en/v2.8.1/references/generated/d3rlpy.dataset.MDPDataset.html),
[algorithms](https://d3rlpy.readthedocs.io/en/v2.8.1/references/algos.html).

Direct commands remain available:

```bash
python train_macro_model.py --dataset custom.npz --output models/my-run --steps 10000
python live_client.py --model models/my-run --team 100 --once
```

Direct CLIs retain explicit paths and original overwrite behavior. Prefer the workflow
wrapper for automatic IDs, reuse checks, and provenance.

Recommendations contain action ID/name, selected `q_value`, and all five `q_values`.
Q-values estimate discounted returns, not calibrated win probabilities. Live output
reports partial observations and unavailable features. Connection failures emit offline
status; malformed states emit invalid-state status. Event history reconstruction is
idempotent across polls and naturally resets for a new game. No stale recommendation
is returned after an error.
