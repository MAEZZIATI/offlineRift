# V2 real-data experiments: 200 matches

Observed on 2026-11-04. These are exploratory development-set results against geometric weak labels, not verified strategy or win-rate improvement. Historical results in [RESULTS.md](RESULTS.md) are unchanged.

## Corpus and protocol

- Unique accepted matches: 200; skipped: 0.
- Episodes: 400; transitions: 62,168.
- Train/validation: 160/40 matches; 49,660/12,508 transitions.
- Patches: {"16.16": 16, "16.17": 56, "16.18": 22, "16.19": 71, "16.20": 35}.
- Catalogs: {"16.16": "16.16.1", "16.17": "16.17.1", "16.18": "16.18.1", "16.19": "16.19.1", "16.20": "16.20.1"}.
- Profiles: `live` masks geometry and team gold; `full` is an offline spatial comparison and is rejected by the stock live adapter.
- Feature/action schemas: `rift-v2` / `macro-v2`; 177 features and twelve labels.
- Each CQL branch uses 10,000 optimizer updates; each companion classifier uses another 10,000. Batch size 64, CPU, seed 42, CQL alpha 1 and gamma 0.99.
- Both teams from a match stay together. Repeated 10-second states are correlated; transitions are not independent experimental trials.
- The hybrid coefficient and class-weight formula are fixed design choices, not fitted to this holdout. This holdout has been consulted during development; confirm results on new matches and multiple seeds.

Chovy collection completed in the shared workspace: the corpus contains 100 matches with the configured `허거덩#0303` PUUID. It contains 104 matches with Faker’s PUUID, including four games where both accounts participated. Participation counts overlap; there are still only 200 unique matches. The agent did not independently exercise authenticated Riot requests; the user completed collection while implementation proceeded.

## Controlled prediction comparison: same live-profile model bundle

| Prediction method | Heuristic agreement | Balanced accuracy | Macro F1 | Predicted categories |
|---|---:|---:|---:|---:|
| Majority baseline | 27.87% | 8.33% | 3.63% | 1/12 |
| cql | 33.64% | 16.97% | 13.09% | 4/12 |
| balanced_bc | 36.50% | 27.09% | 22.67% | 10/12 |
| hybrid | 37.94% | 25.31% | 22.74% | 10/12 |

The published live bundle uses **hybrid** ranking. Balanced BC and hybrid use the same timing/support filters; raw CQL is an unfiltered baseline. Therefore the comparison combines classifier learning, reweighting and feasibility filtering, rather than isolating one component. More ablations are needed.

Balanced imitation has higher balanced accuracy than hybrid in this run; hybrid has higher overall agreement and slightly higher macro F1. The fixed hybrid default was chosen as a design before this 200-match evaluation, not because it wins every metric.

## Per-action performance of the published hybrid

| Action | Training labels | Validation labels | Predictions | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Dragon area setup | 348 | 77 | 155 | 9.03% | 18.18% |
| Baron area setup | 357 | 81 | 307 | 7.17% | 27.16% |
| Herald area setup | 93 | 39 | 30 | 0.00% | 0.00% |
| Cross-map pressure | 115 | 38 | 0 | 0.00% | 0.00% |
| Side-lane split formation | 2728 | 727 | 245 | 21.63% | 7.29% |
| Group mid | 1828 | 440 | 4 | 0.00% | 0.00% |
| Group top side | 4790 | 1244 | 1554 | 23.17% | 28.94% |
| Group bottom side | 11037 | 2760 | 2622 | 36.42% | 34.60% |
| Own-half defensive formation | 11214 | 2786 | 3318 | 38.37% | 45.69% |
| Base regroup | 2328 | 568 | 633 | 88.47% | 98.59% |
| Dispersed lane coverage | 13748 | 3486 | 3640 | 41.43% | 43.26% |
| Unclassified / insufficient evidence | 1074 | 262 | 0 | 0.00% | 0.00% |

Zero recall, low precision and absent predictions remain material failures. Increasing prediction diversity alone is not useful strategy. The objective classes have very few examples; class weighting can increase false positives. Labels describe current formation, not necessarily the next action to execute.

## Encoder and observation comparisons

All rows below use the same 200 match IDs, seed and match split. Full-profile results have access to positions used by the labeler, so high agreement can largely reflect reconstructing the heuristic. They are not a deployable substitute for the live profile.

| Representation | Prediction | Agreement | Balanced accuracy | Macro F1 |
|---|---|---:|---:|---:|
| Team set, live | cql | 33.64% | 16.97% | 13.09% |
| Team set, live | balanced_bc | 36.50% | 27.09% | 22.67% |
| Team set, live | hybrid | 37.94% | 25.31% | 22.74% |
| MLP, live | cql | 34.72% | 17.83% | 14.79% |
| MLP, live | balanced_bc | 34.92% | 29.26% | 22.38% |
| MLP, live | hybrid | 36.30% | 25.76% | 22.66% |
| Team set, full/offline | cql | 86.91% | 49.56% | 50.60% |
| Team set, full/offline | balanced_bc | 91.29% | 77.42% | 71.80% |
| Team set, full/offline | hybrid | 91.61% | 77.04% | 72.40% |

The set encoder does not uniformly outperform the MLP in this single-seed experiment. Its permutation invariance is verified, but architectural superiority is not established.

## Artifacts and reproduction

Primary dataset: `/home/amine/offlineRift/data/projects/faker/datasets/faker-fa3ec093d7f589e9/transitions.npz`.
Published model: `/home/amine/offlineRift/models/faker/20261104T163045Z-2e7c3564`.

```bash
rift status
rift run
rift ui
```

`rift run` reuses unchanged completed inputs. Comparison configurations are saved locally under `data/projects/faker-v2-mlp/workflow.json` and `data/projects/faker-v2-full/workflow.json`; run them with `rift --config PATH train` to reproduce their settings. Reports retain split IDs and confusion matrices. Artifacts and local configurations are ignored by Git.

## Earlier 100-match development checkpoint

Before the second account’s data arrived, the live set-encoder branch achieved 33.76% CQL agreement and 35.35% hybrid agreement; balanced accuracy was 17.47% and 25.79%, respectively. These are different-data results, not a paired comparison to the table above. The historical five-action 85.55% baseline uses a different label contract and cannot be compared directly to twelve-action agreement.

See [model improvements and papers](MODEL_IMPROVEMENTS.md) for the implemented design and proposed evidence-conditioned options, memory, factorization and abstention experiments. No active-game inference was exercised in this session.
