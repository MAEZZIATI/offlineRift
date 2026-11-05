# Observation Limits in Weakly Supervised Team Macro Modeling from Match-V5

**Empirical research draft, 2026-11-04.** Authors and affiliations pending. This is a
working technical manuscript, not a submitted or peer-reviewed paper. It reports the
original 200-match experiment, a frozen-model reanalysis and a separately reported
243-match extension. New collection does not retrospectively change the original
study's corpus. No claim of methodological priority.

Subsequent measurement-contract correction: [rules v3](../OBJECTIVE_RULES.md) rebuilds
1,289 matches under a new objective clock contract. Its 246-window independent review
has no completed reviewers yet. Historical estimates below retain their original labels;
no corrected-target model accuracy or reviewer agreement has been measured.

## Abstract

We study team macro formation prediction from coarse League of Legends Match-V5
timelines under a restricted live observation profile. A historical five-label CQL
baseline collapsed to its majority action. A revised twelve-label system combines
masked features, a team-invariant encoder, class-balanced imitation and bounded Q
ranking. On 200 unique matches split 160/40 by match, live balanced imitation achieves
22.67% macro F1, versus 13.09% for CQL and 22.74% for hybrid ranking. A paired bootstrap
over 40 validation matches gives a hybrid-minus-imitation macro F1 interval of
[−0.60, +0.65] percentage points. Timing/support filters leave live predictions unchanged
but improve offline spatial imitation by 1.29 points. Full spatial hybrid achieves
72.40% macro F1 against geometry-derived weak labels. On 243 newly collected unseen
matches, frozen live hybrid attains 22.27% macro F1 versus 21.90% for imitation, with
a small paired difference interval [+0.015, +0.713] points and lower balanced accuracy.
These findings favor auditing
supervision and observation adequacy before escalating offline RL complexity. They do
not establish optimal macro decisions, calibrated probabilities, or win-rate benefits.

An additional five-seed memory experiment, reported separately below, reaches 22.11%
macro F1 with 120s histories versus 21.21% with repeated-current recurrent inputs on
846 further new matches. It does not exceed the existing plugged hybrid's 22.71%.

## 1. Problem and scientific question

A strategy recommendation should concern a feasible next behavior. Our labels instead
describe simultaneous team formations inferred from locations and estimated objective
timing. This distinction makes heuristic agreement a representation diagnostic, not a
decision-quality measure. We ask whether the observed improvement over collapsed CQL
comes from supervised pattern learning, necessary feasibility constraints, or Q ranking,
and how much apparent performance depends on information unavailable at live inference.

We build on [CQL](https://arxiv.org/abs/2006.04779) for conservative offline value
estimation and [Deep Sets](https://arxiv.org/abs/1703.06114) for within-team permutation
invariance. Those papers motivate components; their guarantees do not certify the
weak-label advisory task studied here.

## 2. Observations, weak labels and learning objectives

Let `g` denote a match and `k ∈ {blue, red}` a perspective. A timeline provides coarse
observations at times `u_j`. For grid time `t`, define `j(t)=max{j:u_j≤t}` and construct

\[
x_{gkt}=\phi(o_{gk,u_{j(t)}},m_{gkt},t-u_{j(t)}).
\]

Here `m` is the availability mask and age records freshness. No future interpolation
is permitted. Ten-second resampling provides a consistent grid, not ten-second
telemetry. The default schema has 177 features, including masked per-player context,
team aggregates, inventory investment, levels and explicit objective estimates.
The live profile masks unavailable geometry and team gold. The full profile exposes
spatial information and is an offline comparison.

The weak target is `L_t=h(o_{u_j},estimated objective state)`, one of twelve categories:
three objective-area setups, cross-map pressure, split formation, three grouping
bands, defensive formation, base regroup, dispersed coverage and unclassified evidence.
The heuristic has priority ordering and cannot express simultaneous intentions.
It does not observe commands, vision control or verified next actions. Recently dead
players and stale frames can distort geometry. Its confidence `c_t` is a rule weight,
not a calibrated correctness probability.

For class counts `n_a` in training only, the companion predictor minimizes

\[
\mathcal L_{BC}(\theta)
=\mathbb E_{t\sim D_{train}}[-c_t w_{L_t}\log p_\theta(L_t\mid x_t)],
\qquad w_a\propto n_a^{-1/2},\quad w_a=0\text{ if }n_a=0.
\]

Positive class weights are normalized to mean one. Training uses AdamW, learning rate
`3×10⁻⁴`, weight decay `10⁻⁴`, and gradient clipping at five. The team-set encoder
shares player transformations, pools each team independently, and combines pooled
embeddings with global context. The MLP is an architecture comparison.

Schematically, discrete CQL combines a terminal-aware temporal-difference loss with
conservative regularization:

\[
\mathcal L_Q=\mathbb E[(Q(x_t,L_t)-y_t)^2]
+\alpha\mathbb E_x[\log\sum_a e^{Q(x,a)}-Q(x,L_t)],
\quad y_t=r_t+\gamma(1-d_t)\bar V(x_{t+1}).
\]

This expression summarizes the purpose of the regularizer; the exact optimizer and
target computation are the pinned d3rlpy 2.8.1 DiscreteCQL implementation. Crucially,
`L_t` is not a verified intervention, so a fitted `Q(x,L)` cannot automatically be
interpreted as the effect of choosing that formation.

Hybrid ranking uses classifier logit `z_a` and a bounded, standardized Q term:

\[
s_a=z_a+0.25\operatorname{clip}
\left(\frac{Q_a-\operatorname{mean}_b Q_b}
{\max(\operatorname{sd}_b Q_b,10^{-6})},-2,2\right).
\]

Serving constraints exclude unsupported training classes and objective-area labels
whose objective is disabled or more than 60 seconds from estimated availability.
If none remains, the output is unclassified. Neither logits, ranks nor Q-values are
calibrated win probabilities. These constraints do not establish combat feasibility.

## 3. Data and protocol

The historical study contains 200 unique Korean ranked matches drawn from two configured
accounts, spanning patches 16.16–16.20: 400 episodes and 62,168 transitions. The seed-42
split contains 160 training and 40 validation matches, with 49,660 and 12,508 transitions.
Both perspectives stay together. Both accounts occur in four shared matches, which are
stored only once. This is an elite-player ego-network sample, not population sampling.

Each CQL branch and each companion classifier receives 10,000 optimizer updates, batch
size 64, on CPU. The original comparisons use one training seed. The validation set
has been consulted repeatedly and is therefore a development set.

The reanalysis reloads three frozen bundles: live team set, live MLP and full team set.
For each, it toggles identical support/timing constraints for CQL, BC and hybrid scores.
It reproduces saved confusion matrices before reporting new comparisons. This isolates
filter application and the Q addition at inference, not weighting or training-objective
effects. There is no retraining in this reanalysis.

For uncertainty, let `C_g^a` be the confusion matrix for method `a` in match `g`.
Bootstrap replicate `b` draws 40 match indices with replacement and evaluates

\[
T_b^a=T\left(\sum_{j=1}^{40}C_{G_{bj}}^a\right),
\quad \Delta_b=T_b^a-T_b^{a'}.
\]

The same indices are used for all methods. Percentiles 2.5 and 97.5 across 2,000 draws
give exploratory intervals. Agreement is transition-weighted after pooling. Balanced
accuracy averages recall over truth-supported classes; macro F1 averages over classes
present in truth or predictions within that replicate. These intervals condition on
weights and available matches; bootstrap exchangeability is an approximation in a
shared player network. They do not measure training-seed or annotation uncertainty.

## 4. Results

| Representation/profile | Ranking | Agreement | Balanced accuracy | Macro F1 |
|---|---|---:|---:|---:|
| Live majority | Constant | 27.87% | 8.33% | 3.63% |
| Team set/live | CQL | 33.64% | 16.97% | 13.09% |
| Team set/live | BC | 36.50% | 27.09% | 22.67% |
| Team set/live | Hybrid | 37.94% | 25.31% | 22.74% |
| MLP/live | BC | 34.92% | 29.26% | 22.38% |
| MLP/live | Hybrid | 36.30% | 25.76% | 22.66% |
| Team set/full | BC, raw | 90.80% | 77.17% | 70.51% |
| Team set/full | BC, filtered | 91.29% | 77.42% | 71.80% |
| Team set/full | Hybrid, raw | 91.13% | 76.80% | 71.06% |
| Team set/full | Hybrid, filtered | 91.61% | 77.04% | 72.40% |

BC and hybrid live team-set models predict ten categories versus four for CQL. Coverage
does not imply correctness: the hybrid has 9.03% dragon precision, 7.17% baron precision,
and zero recall for herald, cross-map, group-mid and unclassified labels. Base regroup
is much easier: 88.47% precision and 98.59% recall. Full per-class counts remain in
[the run record](../RESULTS_V2.md).

Live team-set BC exceeds filtered CQL by 9.58 macro F1 points, interval [+8.26, +10.69].
Hybrid versus BC increases agreement by 1.44 points [+1.06, +1.80], lowers balanced
accuracy by 1.78 points [−3.36, −0.50], and changes macro F1 by +0.07 points
[−0.60, +0.65]. Live MLP hybrid-minus-BC macro F1 also spans zero [−0.87, +1.56].
Filters change no selected live predictions. For full BC they improve macro F1 by
1.29 points [+0.60, +2.01]. These are paired, unadjusted development-set intervals.

## 5. Interpretation and limitations

The live BC improvement cannot be attributed to timing filtering on these samples.
Supervised learning is a stronger reference than raw CQL here, though the experiment
does not isolate class reweighting from representation or loss design. Q ranking trades
supported-class recall for agreement without a demonstrated macro F1 gain.

The full/live gap is consistent with missing spatial information being important.
It does not estimate the Bayes ceiling for the live API: histories, improved targets
and other observable signals may still help. Because full inputs contain the geometry
used to derive targets, its high score is partly circular by construction.

Additional limitations include small biased samples, weak mutually exclusive labels,
single-seed training, repeated holdout use, sparse rare classes, coarse observation
cadence, uncertain objective rules and no human audit. A 177-window stratified training
audit is prepared with no completed annotations. Prospective win-rate improvement,
active-game integration and causal effects were not tested by this study.

## 6. Reproducibility and next study

The package command `rift-research evaluate` records saved split IDs, input/model/code
SHA-256 hashes, software versions, confusion matrices, frozen predictions and bootstrap
settings in a new output directory. The completed reanalysis directories are
`data/research/20261104-{live,mlp,full}-frozen-v3/`; commands and artifacts are documented
in [the synthesis](../RESEARCH_SYNTHESIS.md). No raw data or trained weights are committed.

A confirmatory extension should first audit labels, broaden source accounts and
reserve untouched later matches. Compare causal-memory BC to a matched single-state
reference, separately calibrate abstention, then test Q contributions with fixed splits
and multiple optimizer seeds. See [the data plan](../DATA_PLAN.md) for a staged budget
of 1,000–2,000 pilot and 5,000–10,000 primary-study unique matches. These counts are
planning targets, not evidence that a strategic policy will emerge.

## 7. Extension on newly collected matches

After the original frozen-model analysis, we collected 25 ranked-history entries from
each of ten role-stratified accounts verified in the original cached participants.
The sampling seed is 20261104; the original two anchor PUUIDs were excluded. Unique
accounts were sampled uniformly within dominant historical-role strata, not in proportion
to their number of appearances. This yields 250 history entries and 243 unique new
games; the seven cache reuses are overlaps between the ten histories. None of these
new games contains either original anchor. The original ego-network discovery frame
nevertheless remains a source of selection bias. Historical ranks were not verified.

The new games span patches 16.20 (46), 16.21 (172) and 16.22 (25); exact patch catalogs
were cached before processing. The separate cohort build accepted 265 matches with no
skips and 80,182 transitions, including 22 old games in which sampled accounts appeared.
We excluded all old training and development match IDs before evaluation, leaving 243
unseen matches, 486 episodes and 73,660 transitions. The original live team-set weights,
training-majority label, filtering rules and hybrid coefficient were unchanged.

| Frozen live method | Agreement | Balanced accuracy | Macro F1 | Categories |
|---|---:|---:|---:|---:|
| Training majority | 28.12% | 8.33% | 3.66% | 1/12 |
| CQL | 34.79% | 17.38% | 13.32% | 5/12 |
| Balanced imitation | 36.31% | 27.23% | 21.90% | 10/12 |
| Hybrid | 37.80% | 24.77% | 22.27% | 10/12 |

With the same 2,000-draw paired match bootstrap, hybrid − imitation macro F1 is
+0.37 points [+0.015, +0.713]. This is limited positive evidence for the Q addition on
this new-match sample. It is small, unadjusted, conditional on one fit, and accompanied
by −2.46 balanced-accuracy points [−4.33, −1.03]. Imitation − CQL macro F1 is +8.58
points [+8.03, +9.17]. Filtering changes none of the selected predictions.

The hybrid still has zero recall for herald, cross-map, group-mid and unclassified.
Its dragon/baron precision is 7.22%/7.27%, and its 172 herald predictions are all wrong.
The test includes 99 Herald-labeled states across 27 distinct matches. BC's Herald
recall is 23.23% but precision is only 1.58%, illustrating the weighting trade-off.
These failures prevent interpreting higher aggregate F1 as reliable strategic advice.

The extension is exploratory, not preregistered, and uses weak labels. It measures a
new player/patch mixture rather than independent effects of population and patch.
New match IDs and absence of the original anchors do not imply all-participant-disjoint
data. Ten histories and their co-participants remain dependent; a match bootstrap does
not include all source-account dependence. No weights were refitted and no model was
promoted. Artifacts are `data/research/20261104-unseen-live-v1/` and the cohort's
`collection_summary.json`; the raw cache now contains 443 unique matches in total.

## 8. Subsequent supervision and memory study

The next phase operationalizes the proposed priorities. An [automated evidence audit](../LABEL_AUDIT.md)
reproduces 177 labels across 96 training matches but flags 27 threshold-sensitive
windows and 88 windows where a recorded death follows the position frame. A scan of
443 development matches finds 14 Herald kills after the configured hard cutoff and
a first-spawn estimate inconsistent with the documented 15:00 change. No independent
human-label correctness estimate follows from these diagnostics.

A separate [causal-memory experiment](../MEMORY_STUDY.md) freezes those 443 matches,
uses a fixed 355/88 match split and five optimizer seeds, and matches model capacity
to within three parameters. Snapshot BC, real 60s history, real 120s history, and a
120s GRU repeating current information achieve development macro F1 means of 16.95%,
21.01%, 22.41%, and 21.33%. With scalar metric `M`, the exact decomposition

\[
M(R_{history})-M(S)
=[M(R_{repeat})-M(S)]+[M(R_{history})-M(R_{repeat})]
\]

gives approximately `5.46 = 4.38 + 1.08` percentage points. It is descriptive arithmetic,
not a causal mediation claim. Two of five paired match-bootstrap intervals for the
history-versus-repeat contrast include zero. This limits attribution of the total
gain to historical information alone. These scores are not directly comparable to
the earlier published model because training data, split, capacity and update count
differ. Frozen evaluation on a roster-sampled cohort then uses 846 new matches:
snapshot, 60s history, 120s history and repeated-current control reach mean macro F1
of 16.98%, 20.86%, 22.11% and 21.21%. All five paired match-bootstrap intervals for
120s history minus the repeat control are positive, but the mean contrast is only
0.90 points. The original plugged hybrid reaches 22.71% on the same matches, so
replacement is not justified. In a 212-match subset sharing no participants with
development, the corresponding history/control/plugged scores are 22.29%/21.16%/23.14%.
These findings retain a modest history signal while prioritizing supervision quality.
All original results above retain their specific historical data and target contracts.

## References

1. Kumar, Zhou, Tucker and Levine (2020). [Conservative Q-Learning for Offline Reinforcement Learning](https://arxiv.org/abs/2006.04779).
2. Zaheer et al. (2017). [Deep Sets](https://arxiv.org/abs/1703.06114).
3. Geifman and El-Yaniv (2017). [Selective Classification for Deep Neural Networks](https://arxiv.org/abs/1705.08500).
4. Riot Games. [Official API reference](https://developer.riotgames.com/apis/).
