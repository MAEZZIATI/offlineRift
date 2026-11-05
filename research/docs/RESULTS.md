# First run on the 100 cached real matches

This is the initial real-data baseline, not a synthetic demonstration.

| Measure | Result |
|---|---|
| Source account observed in every match | Hide on bush#KR1 |
| Raw matches / accepted matches | 100 / 100 |
| Source patches | 16.18, 16.19, 16.20 |
| Data Dragon catalogs | 16.18.1, 16.19.1, 16.20.1 |
| Team episodes | 200 |
| Transitions | 30,742 |
| Training / validation matches | 80 / 20 |
| Training / validation transitions | 24,604 / 6,138 |
| CQL optimizer updates | 10,000 |
| Feature profile | live |
| Validation heuristic agreement | 85.55% |
| Majority-label baseline | 85.55% |

**The policy predicts action 4 for every validation observation.** Its agreement equals
the majority baseline; this run does not demonstrate useful discrimination between macro
actions. About 89.6% of dataset labels have heuristic confidence below 0.5. The low-confidence
fraction includes both stale-frame penalties and the fallback rule; it is not a probability
that the labels are wrong.

The pipeline, optimizer, and saved policy run end to end on the real data, but the
initial learned policy should be treated as a baseline. The combination of coarse weak
labels, strong action imbalance, and unavailable spatial features in the live profile
limits what it can learn. More history alone does not resolve these issues.

Before interpreting recommendations as useful, improve weak-label coverage using replay
inspection, evaluate per-action performance against the baseline, and compare offline
full-profile experiments. Full-profile policies still require another data adapter for
live serving. None of these agreement metrics establish win-rate improvement.

Run `python workflow.py status` to find the latest artifacts. The original run is:

```text
data/projects/faker/datasets/faker-72e508905205edcc/
models/faker/20261028T194510Z-5e332db0/
```

This document describes that specific run. Later runs preserve their own reports and
do not update these recorded results automatically.
