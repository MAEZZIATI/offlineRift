# Research synthesis and prioritized experiments

Working record, 2026-11-04. Separate completed measurements, interpretation, and proposed
work. Historical results remain in [RESULTS.md](RESULTS.md) and [RESULTS_V2.md](RESULTS_V2.md).
The companion drafts are [the empirical paper](papers/01_OBSERVATION_LIMITS.md) and
[the mathematical research proposal](papers/02_EVIDENCE_CONDITIONED_OPTIONS.md).

The next phase executes the priorities in this synthesis: see the completed
[automated label/evidence audit](LABEL_AUDIT.md) and [causal-memory study](MEMORY_STUDY.md).
The audit now identifies a specific objective-timing contradiction, beyond generic
weak-label uncertainty. Correcting and validating those rules is a prerequisite to
promoting the experimental models. Historical measurements below retain their original
data and label contracts.

## Latest completed phase: audit, causal memory and broader sampling

The automated audit reproduces all 177 prepared labels but finds 27 threshold-sensitive
windows and 88 with a death event after the position frame. Independent human judgments
remain pending. The 443-match timing audit finds 14 Herald kills after the configured
cutoff and a first-spawn estimate requiring correction; see [the audit record](LABEL_AUDIT.md).

Twenty controlled fits compare snapshot BC, real 60/120s history and a repeated-current
GRU, across five optimizer seeds. On 846 new matches, mean macro F1 is **16.98%, 20.86%,
22.11% and 21.21%**, respectively. Real 120s history adds **0.90 points** over the
same recurrent computation without past state values; all five conditional paired
match-bootstrap intervals are positive. It does not reach the provisional two-point
practical target. Much of the larger snapshot-to-GRU contrast therefore cannot be
attributed to historical information alone.

The existing plugged hybrid scores **22.71% macro F1** on those exact matches. The
new GRU therefore **does not justify replacing it**. On the 212-match strict
participant-disjoint subgroup, GRU120 reaches 22.29% versus 21.16% for the repeat
control and 23.14% for the old hybrid. These remain weak-label metrics; audited
strategic utility and win-rate improvement are unestablished.

The cache now contains **1,289 unique matches** across 57 configured source anchors
in separate workflows. The official-roster phase sampled 15 Master, 15 Grandmaster
and 15 Challenger accounts and added 846 new games. This reaches the initial
1,000–2,000-match pilot range. Structural checks pass, while 634 new matches still
share participants with development. Current-tier quotas are not population sampling.

**Revised priorities:** correct/version objective rules and finish independent label
review first; retain the controlled models and strengthen comparisons on corrected
targets; plan the 5,000–10,000-match main study with untouched cohorts and known sampling
frames; add live history/cadence support only when a candidate warrants promotion.
The 30s live polling cadence was not validated by the 10s-grid memory experiment.
The original published model remains plugged in. Full methods, per-seed intervals,
coverage and limitations are in [the memory study](MEMORY_STUDY.md).

## What happened, and what the system currently does

The first five-action CQL model predicted its fallback on every validation state. Its
85.55% agreement equaled the majority baseline. Successful optimization and save/load
were established; useful macro discrimination was not.

The V2 implementation replaced that taxonomy with twelve weak formation labels, added
masked aggregate context and a team-invariant encoder, and fitted a class-balanced
companion classifier. The original two-account corpus has 200 unique ranked matches,
400 team episodes, and 62,168 transitions over five patches. Both perspectives stay
together in the 160/40 match split. Positions unavailable to live serving are masked.

In that setting, the live hybrid predicts ten categories and reaches 37.94% agreement,
25.31% balanced accuracy and 22.74% macro F1. It no longer predicts one category, but
objective precision remains poor and several labels have zero recall. The full-profile
offline hybrid reaches 72.40% macro F1 with access to geometry used to construct the
labels. That result primarily tests heuristic reconstruction; it is neither a live
result nor evidence that its recommendations improve wins.

## Completed here: frozen-model ablations and uncertainty

Three existing bundles were reloaded: live team set, live MLP, and full team set. The
saved confusion matrices were reproduced exactly. Each was evaluated with CQL,
balanced imitation (BC), and hybrid scores, with identical timing/support filters
either enabled or disabled. No new model training or latest-model publication occurred.

There are 2,000 paired bootstrap replicates, seed 20261104. A replicate resamples the
40 validation matches with replacement and pools their confusion matrices, preserving
all states and both teams within each selected match. Metrics retain the existing
supported-label definitions. Percentile intervals are exploratory 95% intervals,
conditional on fitted weights and this repeatedly consulted development split. They
do not include training-seed uncertainty, label error, or distribution shift. Multiple
comparisons have not been corrected.

| Live team-set method | Agreement | Balanced accuracy | Macro F1 | Predicted categories |
|---|---:|---:|---:|---:|
| Majority | 27.87% | 8.33% | 3.63% | 1/12 |
| CQL | 33.64% | 16.97% | 13.09% | 4/12 |
| Balanced imitation | 36.50% | 27.09% | 22.67% | 10/12 |
| Hybrid | 37.94% | 25.31% | 22.74% | 10/12 |

| Paired live team-set difference | Estimate (percentage points) | 95% match-bootstrap interval |
|---|---:|---:|
| BC − filtered CQL, macro F1 | +9.58 | [+8.26, +10.69] |
| Hybrid − BC, agreement | +1.44 | [+1.06, +1.80] |
| Hybrid − BC, balanced accuracy | −1.78 | [−3.36, −0.50] |
| Hybrid − BC, macro F1 | +0.07 | [−0.60, +0.65] |

Filtering changes no selected predictions for either live encoder on these states.
It does help full-profile BC: macro F1 increases from 70.51% to 71.80%, a +1.29 point
change with interval [+0.60, +2.01]. Full hybrid increases from 71.06% to 72.40%.
Therefore filters cannot explain the observed live BC–CQL difference in this sample.
The BC comparison still combines supervised training, confidence/class weighting and
a different objective; this experiment does not isolate those training components.

Live MLP hybrid − BC macro F1 has interval [−0.87, +1.56] points. Its balanced-accuracy
change has interval [−8.97, −0.76]. Full hybrid − BC macro F1 has interval
[−0.002, +1.23] points. None establishes a reliable macro F1 gain from adding Q ranking.
There is no paired encoder-superiority claim or win-rate test here.

**Recommendation:** use balanced imitation as the reference for new experiments;
retain hybrid and CQL as ablations. The current published hybrid is retained. Prioritize
auditing what the labels mean and testing observable history over larger RL sweeps.

## Completed extension: new accounts and unseen matches

After the analysis above, a deterministic role-stratified pilot collected the latest
25 ranked history entries for each of ten verified cached co-participants, excluding
both original anchors. All 250 entries were resolved: 243 unique new matches and seven
cache reuses from overlap between the sampled histories. None of the 243 new matches
contains either original anchor. The raw cache now has 443 unique matches. New patches
are 16.20 (46), 16.21 (172) and 16.22 (25); exact item catalogs were fetched for the new
patches. Basic payload checks found no ID mismatch, wrong queue, missing frames or
incorrect participant count. These are structural checks, not annotation validation.

The independent cohort build accepted 265 matches, skipped none, and produced 80,182
transitions; it also includes 22 original games featuring sampled accounts. Evaluation
explicitly removed those 22 matches and used only the 243 unseen games: 486 team
episodes and 73,660 transitions. The original live bundle's weights, masks, class
weights, filters and hybrid coefficient were frozen. No model was retrained or promoted.

| Frozen live team-set method | Agreement | Balanced accuracy | Macro F1 | Categories |
|---|---:|---:|---:|---:|
| Training-majority baseline | 28.12% | 8.33% | 3.66% | 1/12 |
| CQL | 34.79% | 17.38% | 13.32% | 5/12 |
| Balanced imitation | 36.31% | 27.23% | 21.90% | 10/12 |
| Hybrid | 37.80% | 24.77% | 22.27% | 10/12 |

Hybrid − BC is +0.37 macro F1 points, with a paired 2,000-draw match-bootstrap interval
of [+0.015, +0.713]. This is a small positive conditional result, unlike the original
40-match development comparison whose interval included zero. It is below the proposed
two-point practical-effect target and is unadjusted for multiple comparisons. Hybrid
also lowers balanced accuracy by 2.46 points [−4.33, −1.03]. BC − CQL macro F1 is
+8.58 points [+8.03, +9.17]. Filters again change no selected predictions.

Rare-label failures remain: hybrid objective precision is 7.22% for dragon, 7.27% for
baron and 0% for herald. Herald, cross-map, group-mid and unclassified have zero recall.
The new test includes 99 Herald-labeled states, versus 1,454 BC Herald predictions
(1.58% precision); hybrid makes 172 Herald predictions, all incorrect. A tiny aggregate
F1 gain does not resolve these failures or establish usefulness of advice.

This extension evaluates new matches and a shifted player/patch mix, not strict
all-participant independence or general ranked-play performance. Ten account histories
remain correlated and derive from the original matchmaking network. Match-bootstrap
intervals do not capture all of that dependence. The new results preserve the priority
of supervision audit, memory and data design, while providing limited positive evidence
for retaining bounded Q ranking as an experimental comparator.

Artifacts: `data/research/20261104-diversity-pilot/collection_summary.json`, its cohort
and batch manifests, the immutable dataset under `artifacts/datasets/`, and
`data/research/20261104-unseen-live-v1/report.json` plus frozen predictions.

## Supervision audit: prepared, not annotated

The sampler created 177 windows from training matches only, up to three distinct
matches per patch/label cell. The unit is a match then a state within that match's
label cell. Two rare cells
have fewer than three available matches. The reviewer sheet hides the weak label,
confidence, and stratum order; a separate key permits later comparison.

Give two reviewers separate copies of `audit_blinded.csv`. First assess the formation
at the marked instant using evidence at or before that instant. Record insufficient
evidence explicitly. Only afterward inspect the optional +30s window to discuss
formation versus subsequent behavior; do not rewrite the instant's state with future
information. The −60s/+30s bounds are review requests, not evidence that replay video
is available. Cached minute frames may be insufficient. Record that limitation.

Reviewers must work independently before adjudication. Report raw agreement, Cohen's
kappa with class prevalence, per-label precision against adjudication, missing-evidence
rates, and match-cluster intervals. Label 11 is a heuristic fallback; explicit model
abstention is a different event. The 177-window pilot is for failure discovery and
rubric refinement, not precise population-wide estimates. Equal strata do not preserve
natural frequencies, and label-cell sampling favors neither uniform timestamps nor
uniform population matches. Do not report an unweighted pooled precision as population
precision. Expand the audit before claiming rare-label reliability.

## Next experiments, ordered by what they can resolve

| Priority | Work | Fixed comparison and decision rule |
|---|---|---|
| 1 | Audit labels and collect a broader pilot | Complete independent review; revise or mark unreliable labels unknown. Preserve old schemas. Measure new unique matches and source overlap before training. |
| 2 | Causal memory | Compare a 60/120s GRU with single-state BC of matched capacity and update budget. Include freshness and history masks; reset at episode boundaries. Test on untouched later matches. |
| 3 | Supervised loss ablation and abstention | Cross entropy vs class weights vs confidence weights; fixed match splits, five training seeds. Tune rejection on a separate calibration set, report risk/coverage and per-label recall on test. |
| 4 | Verified public event context | Add only causal event-derived signals also supported by the live source; compare to memory alone. Distinguish unavailable signals from observed zeros. |
| 5 | Factorized evidence-conditioned options | Implement only after audit and memory baselines. Ablate factorization, evidence penalty, event auxiliaries and distillation individually. |
| 6 | More offline RL and speculative mechanisms | Test Q contribution only after label/action semantics and behavior support are defensible. Compare equal representations and compute budgets. |

For future confirmatory comparisons, predeclare audited-label macro F1 as primary and
require a positive lower bound for the paired test difference, no unexplained loss of
rare-label coverage, and an operationally useful effect (provisionally +2 macro F1
points). That effect threshold is a project decision, not a statistical theorem or a
guarantee of sufficient power. Report all seeds. Current observations cannot be relabeled
as preregistered. Freeze match IDs separately from optimizer seeds: the current trainer's
`seed` controls both initialization and split, so merely changing it does not implement
a fixed-split seed experiment.

## Reproduction and artifacts

```bash
rift research-evaluate \
  --dataset data/projects/faker/datasets/faker-fa3ec093d7f589e9/transitions.npz \
  --bundle models/faker/20261104T163045Z-2e7c3564 \
  --output data/research/my-new-live-evaluation --raw-dir data/raw
```

Observed final evaluations live under `data/research/20261104-{live,mlp,full}-frozen-v3/`.
Each contains an atomic completion report, input/model/source hashes, software versions,
saved split IDs, metrics and intervals, and frozen validation predictions. The live
directory also contains the blank annotation sheet and separate key. Earlier development
evaluation directories are retained. Generated payloads, weights and reports stay ignored
by Git. See [the data plan](DATA_PLAN.md) for sample-size estimates and collection status.

To reproduce the new-match extension in a fresh output directory:

```bash
rift research-evaluate \
  --dataset data/research/20261104-diversity-pilot/artifacts/datasets/diversity-pilot-ce108c3b20d1318b/transitions.npz \
  --bundle models/faker/20261104T163045Z-2e7c3564 \
  --output data/research/my-new-unseen-evaluation --unseen
```
