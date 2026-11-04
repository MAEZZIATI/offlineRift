# Causal memory experiment protocol

Completed 2026-11-04. This is a weak-label prediction experiment, not a strategic policy
validation. Final observed results are recorded below; historical
200-match and 243-new-match evaluations remain in their own records.

## Fixed development data and training

A frozen input manifest contains the 443 matches present before the official-roster
expansion. It combines the two original anchor histories and the ten-account pilot.
The workflow accepted all 443 matches, yielding 886 team episodes and 135,828 transitions.
The immutable dataset is `memory-dev-bf5bfc3cb398deda` under
`data/research/20261104-memory-dev/artifacts/datasets/`.

One match split uses seed 20261105 and remains independent of optimizer seeds
42, 43, 44, 45 and 46. It has 355 training and 88 development matches. Both team
episodes remain together. Since these matches have already informed development,
the independent-roster collection is reserved for a later frozen-model comparison.
No match from the complete 443-ID development manifest may enter that comparison.

Each model uses 5,000 AdamW updates, batch size 64, learning rate 0.0003, weight decay
0.0001, gradient clipping at five, and the existing confidence-weighted inverse-square-root
frequency cross entropy. Class weights use training rows only. The identical sequence
of sampled training row IDs is used for methods with the same optimizer seed.

| Method | Information available at decision time | Parameters |
|---|---|---:|
| Snapshot BC | Current masked state only | 48,335 |
| GRU 60s | Causal masked states over at most 60s | 48,332 |
| GRU 120s | Causal masked states over at most 120s | 48,332 |
| Repeat-current GRU 120s control | Current state repeated; same length, age/presence masks and recurrent compute | 48,332 |

All use the same shared player encoder and team pooling, with 64-value state embeddings.
The snapshot control uses a 66→193→64 network in place of GRU(66,64); this gives a
three-parameter difference. Equal update budgets do not give equal FLOPs. The repeated-current
control specifically tests whether a recurrent-compute/inductive-bias advantage can explain
an improvement without access to past state values. It was added after observing initial
development improvements, before evaluating any official-roster test predictions.

## Causal history and availability contract

Let `x_t` be the existing 177-value live-masked observation and `H` the history horizon.
Each sequence contains only rows in the same team episode with timestamps in `[t−H,t]`.
For observed row `u`, the recurrent input is

\[
e_u=\left[\operatorname{TeamSet}(x_u),\ (t-u)/\max(H,1),\ 1\right].
\]

Right padding has presence zero and is excluded by packed-sequence lengths. The final
valid hidden state is classified. For the repeat control, replace every `x_u` with
`x_t` while retaining valid sequence lengths and ages. It therefore contains only
snapshot information and known timestamp context, not simulated telemetry.

Histories reset at team/match boundaries and on non-increasing game clock. Elapsed age
and presence are known from observation timestamps; the historical source-frame age
is not unmasked in the live profile. Positions, team gold, unavailable cooldowns and
other masked values remain masked. Ten-second resampling still provides no new telemetry.

The implementation uses [PyTorch GRU](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html)
and [packed sequences](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html).
Unit tests cover padding exclusion, future-data exclusion, clock/episode resets,
fixed match grouping, feature-mask preservation, parameter counts, real optimizer
updates, save/reload, and exclusion of all development matches from frozen evaluation.

## Completed development comparison

All 20 fits completed: five optimizer seeds each for the three primary methods and
the repeated-current control. Means below use the same 88 development matches and
27,788 transitions. Standard deviations describe optimizer-seed variability, not
confidence intervals or variation across independent match samples.

| Method | Agreement, mean ± seed SD | Balanced accuracy, mean ± seed SD | Macro F1, mean ± seed SD |
|---|---:|---:|---:|
| Training-majority baseline | 28.02% | 8.33% | 3.65% |
| Snapshot BC | 31.85 ± 1.05% | 24.69 ± 1.72% | 16.95 ± 0.86% |
| GRU 60s | 34.29 ± 0.93% | 28.52 ± 2.82% | 21.01 ± 1.13% |
| GRU 120s | 35.67 ± 1.19% | 29.91 ± 2.04% | 22.41 ± 0.81% |
| Repeat-current GRU 120s | 34.31 ± 0.84% | 29.08 ± 1.29% | 21.33 ± 1.05% |

No trained fit collapses to one prediction category. Snapshot BC predicts only 5–7
of twelve categories across seeds; both real-history methods predict 8–10, and the
repeat control 8–10. Increased category coverage is not a correctness guarantee.
Training support remains highly imbalanced: 30,165 dispersed-coverage labels versus
179 Herald labels. Development support contains only 52 Herald states, which are
correlated observations rather than 52 independent expert decisions.

The 120s model exceeds the snapshot reference by 5.46 macro F1 points, but exceeds
the repeat-current control by only **1.08 points**. Consequently the full 5.46-point
gain cannot be attributed to historical information. The descriptive metric decomposition
is `5.46 ≈ 4.38 (repeat − snapshot) + 1.08 (real history − repeat)`. This is scalar
arithmetic under this protocol, not a causal mediation result.

Real 120s history beats the repeat control in point estimate for all five seeds.
Per-seed paired match-bootstrap macro F1 intervals, in percentage points, are:
42 [−0.47, +1.41], 43 [+1.03, +1.81], 44 [+0.28, +1.33],
45 [−0.06, +1.32], and 46 [+1.32, +2.83]. Two include zero. The effect is below the
provisional two-point practical target on average and awaits the reserved-cohort test.
Do not mistake the matched snapshot's score for the original published model's score:
its capacity, training corpus, split and update count differ from that earlier bundle.

The [objective-rule audit](LABEL_AUDIT.md) also identifies a Herald timing mismatch and
14 recorded kills after the configured cutoff. All methods here retain the same old
rules for a controlled comparison; none is thereby validated for strategic serving.

Artifacts: `data/research/20261104-memory-comparison-v1/` contains the 15 primary fits,
predictions, split, per-seed intervals and `development_control_bootstrap.json`.
`data/research/20261104-memory-repeat-control-v1/` contains the five repeat-control fits.

## Completed frozen evaluation: 846 new matches

The roster collection resolved all 900 requested history entries into 873 unique
queried matches, of which 846 were new. The cohort workflow accepted 923 cached
matches with no skips: 276,652 transitions, including 77 previously seen games in
which sampled accounts also participated. Excluding **all 443 development match IDs**
left **846 unseen matches /253,480 transitions**. No model, coefficient or threshold
was refitted on these predictions.

| Method | Agreement, mean ± seed SD | Balanced accuracy, mean ± seed SD | Macro F1, mean ± seed SD |
|---|---:|---:|---:|
| Training-majority baseline | 28.29% | 8.33% | 3.67% |
| Matched snapshot BC | 32.26 ± 1.12% | 23.99 ± 1.42% | 16.98 ± 0.83% |
| GRU 60s | 34.24 ± 0.97% | 27.54 ± 2.53% | 20.86 ± 0.98% |
| GRU 120s | 35.71 ± 1.22% | 28.52 ± 1.79% | 22.11 ± 0.55% |
| Repeat-current GRU 120s | 34.53 ± 1.14% | 28.00 ± 1.05% | 21.21 ± 0.88% |
| Original plugged bundle: BC ranking | 37.00% | 28.65% | 22.59% |
| Original plugged bundle: hybrid ranking | 38.30% | 25.32% | 22.71% |

The original bundle rows are a single frozen checkpoint, not five independent fits.
It trained on 160 matches with a different architecture and 10,000 classifier updates;
the new controlled fits used 355 matches and 5,000 updates. Their comparison is a
practical checkpoint benchmark, not an isolated architecture effect.

Real 120s history improves by **5.12 macro F1 points** over the matched snapshot but
only **0.90 points** over the same recurrent computation with repeated current input.
The latter paired match-bootstrap intervals, in percentage points, are:

| Optimizer seed | 846-match history − repeat interval | 212-match participant-disjoint interval |
|---|---:|---:|
| 42 | [+0.85, +1.22] | [+0.70, +1.31] |
| 43 | [+1.20, +1.49] | [+1.12, +1.80] |
| 44 | [+0.13, +0.65] | [+0.27, +0.90] |
| 45 | [+0.22, +0.68] | [+0.56, +1.45] |
| 46 | [+1.04, +1.50] | [+1.19, +2.15] |

Each conditional interval is positive, but the mean information-specific effect is
below the provisional two-point practical target. These are unadjusted match-cluster
intervals, not guarantees over independent source-account samples. Do not count five
seeds on the same test games as five independent populations.

**The new GRU does not outperform the original plugged hybrid on macro F1 or agreement.**
The original bundle's BC ranking also slightly exceeds the mean GRU in all three full-cohort
metrics. This branch therefore does not justify replacing the published model. Keep
the GRU as an experimental reference while correcting and auditing supervision.

No trained model collapses to a single category. Test prediction coverage is 5–7 for
snapshot BC, 8–10 for GRU60, 9–10 for GRU120 and 8–10 for the repeat control. Cross-map
and unclassified have zero recall in every real-history fit. All models still face
severe label imbalance; the test has 71,714 dispersed-coverage states versus 398 Herald
states. The plugged hybrid's objective precisions are 8.35% dragon, 8.03% baron and
0.84% Herald. None of these results establishes reliable strategic advice.

### Exploratory strict participant-disjoint subset

The provenance audit identified 212 new matches with no PUUID seen in **any** of the
443 development games. This criterion was saved before reserved-cohort predictions;
it is an exploratory subgroup, not an independently preregistered second study.
It contains 64,460 transitions. The other 634 new matches share development participants.

| Method | Agreement | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|
| Snapshot BC, five-seed mean | 32.54% | 24.02% | 16.94% |
| GRU60, five-seed mean | 34.79% | 27.77% | 20.98% |
| GRU120, five-seed mean | 36.28% | 29.00% | 22.29% |
| Repeat-current GRU120, five-seed mean | 34.93% | 27.94% | 21.16% |
| Original bundle, BC | 37.91% | 27.98% | 22.84% |
| Original bundle, hybrid | 39.31% | 26.01% | 23.14% |

The information-specific contrast is +1.14 macro F1 points, again below the practical
target; the old hybrid still has higher macro F1. Participant-disjointness between
development and test does not make test matches independent of one another. The
subgroup also has a different tier/network mixture from the full cohort.

Artifacts: `data/research/20261104-memory-unseen-v1/report.json`, `test_predictions.npz`
and `participant_disjoint.json`; collection provenance and quality reports are under
`data/research/20261104-independent-pilot/`. All 20 trained checkpoints were reloaded
and their saved development predictions reproduced exactly; per-model hashes and
software versions are saved in the training directories' `verification.json` files.

## Evaluation and interpretation

Every method uses the same training-support and objective-timing constraints, with
zero Q contribution. Record agreement, balanced accuracy, macro F1, majority baseline,
full per-action counts/precision/recall, coverage and collapse. Report all five optimizer
seeds with means and seed standard deviations, and paired match-bootstrap intervals
within each seed. These intervals are unadjusted exploratory intervals and do not include
all source-account dependence. Seeds are not independent draws of new matches.

The reserved cohort samples 15 accounts from each returned official Korean Master,
Grandmaster and Challenger roster and requests 20 consecutive ranked games per account.
It reduces dependence on the original discovery network, but equal current-tier quotas
are not representative of the population and current tier is not historical rank.
All old development match IDs are excluded, and remaining participant overlap must be
reported. A strict all-participant-disjoint test is not assumed.

Compare real history against both the matched snapshot and repeated-current control.
Also evaluate the original plugged bundle on exactly the same test matches; this is
a practical checkpoint comparison with different training data and capacity, not an
isolated architecture comparison. The provisional practical target is a two-point
macro F1 gain, alongside per-class coverage and error review. Weak-label improvement
does not replace the [independent label audit](LABEL_AUDIT.md).

Research artifacts use the experimental `history-bc-v1` format. They are not stock
`MacroPolicy` bundles and are not automatically published. Live deployment would require
a compatible observed-history adapter, interval/rollback/missing-history behavior,
reviewed label semantics, and separate serving validation. The current live UI remains
on its previous published hybrid.

## Reproduction

```bash
rift research-memory \
  --dataset data/research/20261104-memory-dev/artifacts/datasets/memory-dev-bf5bfc3cb398deda/transitions.npz \
  --output data/research/my-memory-experiment --steps 5000 --seeds 42 43 44 45 46

rift research-memory \
  --dataset data/research/20261104-memory-dev/artifacts/datasets/memory-dev-bf5bfc3cb398deda/transitions.npz \
  --output data/research/my-repeat-control --steps 5000 --seeds 42 43 44 45 46 --methods repeat120

# Frozen evaluation on the completed roster-cohort dataset:
rift research-memory \
  --dataset data/research/20261104-independent-pilot/artifacts/datasets/diversity-pilot-6cbb71319e1cdcbc/transitions.npz \
  --evaluate-from data/research/my-memory-experiment \
  --control-from data/research/my-repeat-control \
  --reference-bundle models/faker/20261104T163045Z-2e7c3564 \
  --output data/research/my-unseen-memory-evaluation
```

All output directories must be new. Interrupted reports and previously completed
weights are retained; no latest-model reference is changed by these research commands.
