# Macro model improvements and research directions

Status: 2026-11-04. Implemented changes and proposals are separated below. Observed
runs are recorded in [V2 experiments](RESULTS_V2.md); the historical five-action
baseline remains in [RESULTS.md](RESULTS.md).

## What was wrong

The original live-profile CQL model predicted one fallback label for every validation
state. The fallback mixed defense, recall, and vision setup with states the heuristic
could not classify. Its 85.55% agreement simply matched the majority baseline. More
training updates or another hundred games alone cannot fix unobserved map state or
an action taxonomy that confuses missing evidence with tactical intent.

There is also a modeling mismatch: locations describe what a team currently looks
like, whereas a recommendation concerns what it should do next. The current weak
labels are simultaneous formation labels, not verified executed decisions. Offline
CQL does not remove that mismatch or establish causal benefits of an action.

## Implemented: states

`rift-v2` extends the shared feature contract from 159 to 177 values with the default
4×4 grid and two buff types. It adds team-relative inventory investment and level
advantages, masks for their availability, base and forward presence fractions, three
map-band occupancy fractions, and spatial spread for both teams. A scalar aggregate
can be easier to learn on this small corpus than reconstructing it from ten slots.
Inventory investment is not gold income, damage, or fighting strength.

All geometry remains masked in the live profile. Masks distinguish unknown from
zero. Full-profile spatial context uses only the current causally forward-filled
frame. No future movement, final gold, or final inventory is introduced. Objective
rules remain configurable estimates, not authoritative patch mechanics. No new
degrees of telemetry are implied by the 10-second grid.

## Implemented: action modeling

`macro-v2` has twelve mutually exclusive weak labels:

| ID | Label | What the evidence can support |
|---|---|---|
| 0 | Dragon area setup | Three allies near an enabled dragon pit within 60s of estimated availability |
| 1 | Baron area setup | The corresponding baron geometry and timing |
| 2 | Herald area setup | The corresponding herald geometry and timing |
| 3 | Cross-map pressure | Enemy pit cluster plus distant forward side pressure |
| 4 | Side-lane split formation | At least three mid positions plus one or two forward side positions |
| 5 | Group mid | Three active positions in the diagonal mid band |
| 6 | Group top side | Three active positions above the mid band |
| 7 | Group bottom side | Three active positions below the mid band |
| 8 | Own-half defensive formation | Three allies and three enemies in the allied half |
| 9 | Base regroup | Three allies within the base-radius approximation |
| 10 | Dispersed lane coverage | Active allies occupy all three map bands |
| 11 | Unclassified / insufficient evidence | Missing/stale positions or unmatched geometry |

Base takes priority, followed by objective, cross-map, split, defense, group, coverage,
and unclassified rules. Team-relative halves swap with perspective. Top/bottom bands
include jungle and river; they are not precise lane detectors. Pit proximity cannot
prove attacking, baiting, denying vision, or controlling the objective. Base presence
cannot prove a recall. Positions may include recently dead players because death
state is not reconstructed. The unclassified label is a transparent fallback, not
proof that every classified state has enough evidence for advice.

There is one current feature/action implementation. Old artifacts remain stored,
but loaders reject incompatible contracts: rebuild and retrain with `rift run`.

## Implemented: deep learning and prediction

CQL already used deep learning. The new default `team_set` encoder applies a shared
10→64→64 network to each player's masked features, pools the five players separately
for each team, then fuses those team embeddings with all remaining global features
through a 128→128 network. This removes dependence on champion alphabetical slot
ordering while preserving the ally/enemy distinction. The original MLP architecture
is an explicit ablation option, not a legacy schema implementation.

A companion neural classifier learns weak macro patterns from training matches only.
Its cross entropy uses square-root inverse class frequency weights, multiplied by
heuristic label confidence. This tempers majority dominance without giving an isolated
rare, noisy label unbounded inverse-frequency influence. Missing classes have zero
weight. Confidence is a heuristic weight, not calibrated correctness. The classifier
uses AdamW, learning rate 0.0003, weight decay 0.0001, gradient clipping at 5, and the
same requested update count/batch size as CQL. It does not resample or break RL episodes.

Default `prediction_mode=hybrid` ranks actions using:

```text
score(a) = balanced_classifier_logit(a)
         + 0.25 × clip((Q(a) − mean(Q)) / max(std(Q), 1e−6), −2, 2)
```

The bounded Q term is a tie-breaker: unsupported return estimates cannot grow without
bound and overwhelm the pattern predictor. Classes absent from training are excluded.
An objective-area action is excluded when its configured objective is disabled or its
ETA exceeds 60s. If all supported options are excluded, output unclassified. These are
necessary timing constraints, not safety, visibility, or combat-success guarantees.
`balanced_bc` removes the Q term; `cql` retains the raw research baseline. Neither the
weighting nor the fixed hybrid coefficient was fitted on the reported holdout.

The bundle saves both `policy.d3` and, for supervised modes, `macro_predictor.pt`.
Predictions identify their method and expose macro scores separately from Q-values.
Neither is a win probability. Balanced logits need not represent the natural action
frequency, and can overpredict rare actions. The dashboard displays the ranking scores,
with Q estimates available on each row, and reports balanced accuracy and macro F1.

## Implemented: evaluation and collection

Each split keeps both team episodes from a match together. Reports retain the actual
label counts, validation support, predicted counts, confusion matrix, per-action
precision/recall/F1, balanced accuracy, macro F1, prediction coverage, majority baseline,
and collapse detection. Balanced accuracy averages recall over supported truth labels;
macro F1 includes classes present in truth or predictions. All measure agreement with
weak labels, not strategic utility. The hybrid run records raw CQL, balanced BC, and
hybrid predictions on the same split, so a better majority score cannot hide collapse.

The live display refresh defaults to 30 seconds. Training remains on a causal 10-second
observation grid. A 30-second display interval is not a 30-second learned option or a
claim that the suggested play remains appropriate for its whole display duration.

`add-player` resolves the user-supplied `허거덩#0303` account through Riot Account-V1 and
stores its PUUID separately from the configured primary account. `collect --player`
scans that verified account; raw matches are cached once, and combined builds deduplicate
shared games. No PUUID is inferred from public name matching. Account lookup verifies
the account, not the professional's identity. Authenticated collection requires a valid `RIOT_API_KEY` visible to the running
command. Newly cached data can be built without the key. See [workflow commands](WORKFLOW.md).

## Proposed next experiments, in priority order

| Priority | Direction | Concrete experiment and success criterion |
|---|---|---|
| 1 | Audit the supervision | Review replay windows stratified by all twelve labels and patches. Record independent annotator agreement, intent vs formation ambiguity, and per-label precision. Do not optimize to a bad heuristic. |
| 2 | Causal memory | Add a GRU or small causal Transformer over 60–120s of observed history, including freshness and history masks. Reset on every match/team episode and game-clock rollback. Compare against a matched-capacity single-state model on a new match holdout. |
| 3 | Factor macro decisions | Predict formation, destination/objective, phase, and commitment separately, allowing simultaneous side pressure and objective preparation. Add explicit unknown targets where evidence is absent. Measure per-factor metrics and joint feasibility. |
| 4 | Event-based public context | Reconstruct verified structure destruction, inhibitor state and observed deaths from timestamped events, with patch rules and availability masks shared with live. Add only signals exposed by each source. Measure gains on audited labels. |
| 5 | Multi-task learning | Add masked future-event prediction as a training target (never an input), such as observed objective/structure events in 30/60/120s. Keep match splits fixed and test whether representation improves audited macro classification. |
| 6 | Time-extended options | Learn initiation/termination and elapsed-time-aware rewards instead of treating formation labels as primitive controls. Compare 30/60s options and event boundaries. Aggregate rewards and discount by actual elapsed time; do not just drop two of every three transitions. |
| 7 | Confidence and abstention | Fit action-specific thresholds on a separate calibration set. Report selective risk vs recommendation coverage and match-bootstrap intervals on a fresh test set. Compare ensemble disagreement; never softmax Q-values into win probabilities. |
| 8 | Data diversity | Add the requested Chovy history; report counts and overlap by PUUID/patch. Evaluate source-only, combined, forward-time/patch, and player-disjoint splits where overlapping matches stay grouped. Record role/lineup bias. |
| 9 | Offline RL comparison | Compare balanced imitation, CQL, discrete-compatible IQL, and recurrent variants with the same representation and audited labels. Report seeds, action coverage, support, uncertainty and sensitivity to reward scaling; no win-rate claims from agreement alone. |

## Proposed original research direction

Investigate **evidence-conditioned macro options**: a factorized option predictor with
an explicit representation of which strategic claims can be supported by the available
observation. A full-profile teacher learns masked auxiliary event targets and spatial
relations; a live student learns from public history, observes an availability mask,
and must abstain on factors that cannot be inferred reliably. Distill only targets with
validated teacher evidence and measure teacher errors separately. The student must never
present estimated coordinates as observed facts.

Each option would expose a destination, participating group, estimated duration and
termination condition, along with supported/unknown factors. Option scoring would
combine observed-behavior support, bounded conservative value, and an evidence penalty.
For example, it could suggest preparing an objective window while explicitly declining
to claim a safe engage path when positions/vision are unavailable. Joint options can
express a side-lane threat plus a grouped objective setup, which the current flat labels
cannot express simultaneously.

This is a project-specific hypothesis, not an established novel algorithm or a claim of
priority over the literature. The implemented masked context, set encoder, balanced
predictor and bounded hybrid ranking are a testable first step, not the complete method.
Falsify it with ablations removing evidence masks, confidence weighting, memory,
factorization, distillation and Q ranking one at a time. Predeclare data splits and
primary audited-label macro F1 plus selective-risk/coverage metrics. Keep unseen matches
for final testing; repeated development on the current holdout makes it a development
set. A learned simulator is not trustworthy counterfactual evidence for winning plays.

## Research used

These primary papers informed the implementation and the proposed experiments. Their
game environments, data access and evaluation settings differ from this Match-V5 tool.

1. Kumar, Zhou, Tucker & Levine (2020), [Conservative Q-Learning for Offline Reinforcement Learning](https://arxiv.org/abs/2006.04779).
   Basis of the existing CQL branch; motivates caution about unsupported actions and
   value extrapolation. This project's weak action labels do not inherit its guarantees.
2. Zaheer et al. (2017), [Deep Sets](https://arxiv.org/abs/1703.06114).
   Informs the shared per-player network and within-team invariant pooling implemented
   here. It does not establish that pooling improves League strategic utility.
3. Wu et al. (2019), [Hierarchical Macro Strategy Model for MOBA Game AI](https://arxiv.org/abs/1812.07887).
   Informs explicit macro structure, spatial goals and hierarchy. Its controlled MOBA
   agents have capabilities and observations absent from this advisory backend.
4. Gao et al. (2021), [Learning Diverse Policies in MOBA Games via Macro-Goals](https://arxiv.org/abs/2110.14221).
   Informs the goal-conditioned, multi-option direction. Diversity of predictions alone
   is not evidence of useful diversity or better decisions.
5. Chen et al. (2021), [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/abs/2106.01345).
   Informs causal history modeling. A Transformer/return-conditioned policy is proposed,
   not implemented, and observed future returns must never enter live state features.
6. Ajay et al. (2021), [OPAL: Offline Primitive Discovery for Accelerating Offline Reinforcement Learning](https://arxiv.org/abs/2010.13611).
   Informs recurring time-extended behavior and option support. OPAL is not implemented.
7. Kostrikov, Nair & Levine (2022), [Offline Reinforcement Learning with Implicit Q-Learning](https://arxiv.org/abs/2110.06169).
   Informs the proposed offline algorithm comparison; IQL is not implemented here.
