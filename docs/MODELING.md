# Next-action modeling contract

## Objective

Learn an advisory policy that ranks supported next behaviors toward terminal team victory.
The first implementation supports one player with an explicit role, or a team-level
perspective. It is an offline, partially observed, cooperative decision problem. A model
successfully fitting the dataset does not prove an optimal or useful policy.

## State and decision unit

`decision-state-v1` is a 64-value public-observation vector: time buckets, actor/team scope,
role identity and its mask, focal level/inventory/CS/KDA/health/dead state, bounded recent
CS and level changes, both teams' aggregate public levels/inventory/CS/kills, recent deaths,
structure-loss history and pinned objective estimates. Every optional numerical value
has an availability mask. Positions are deliberately absent from training/live inputs.

Dataset states use **actual observed frames**, typically about 60 seconds apart, rather
than interpolating or treating the old ten-second resampling as new telemetry. Each player
and each team has a separate episode. The first eligible decision begins after 90 seconds;
there must be a following observed frame. Team actions require three matching player
options, otherwise continuing the current distribution is the recorded team behavior.

Historical roles come from Match-V5's retrospective `teamPosition` and can be imperfect.
Live roles must be reported by the API or explicitly selected. No role is guessed from a
champion name. Other champions' roles are not required as live state features.

## Executed options and targets

`next-action-v1` reconstructs an option from the following interval's movement, CS changes,
base arrival, objective-area trajectory and timestamped combat participation. Future
observations are **label/transition evidence only** and never enter the preceding state.
Targets describe inferred executed behavior, not verified intention or ideal play.

The complete implemented taxonomy is in [ACTIONS.md](ACTIONS.md). Stationary farming is
included; routine play is not mislabeled as defense. Rotating to another lane does not
assert a gank, and staying around an objective does not prove vision control or commitment.
Combat labels include deaths as well as kills/assists, avoiding a successful-kills-only
sample; involuntary/ambushed combat remains an important confounding limitation. Coarse
endpoints may conceal multiple activities within the interval. Geometry thresholds are
explicit heuristics, not ground-truth lane/route detection.

If behavior cannot be reconstructed, the **whole actor episode** is excluded and counted.
Rows are never deleted and stitched across an unknown action, nor is missing evidence
invented as “continue.” Short games with no eligible decision are excluded without
selecting on victory. Selection bias from exclusions remains to be measured.

## Rewards, transitions and CQL

`option-transitions-v1` stores `(state, option, reward, next_state, terminal, elapsed_seconds)`
explicitly. Rewards are zero except +1 for a terminal team win and -1 for a terminal loss.
Final match outcomes are reward targets only. No final gold/items are input features.

A sampled Bellman target is:

```text
y = r + (1 - terminal) * gamma ** (elapsed_seconds / 60) * Q_target(s_next, a_next)
```

The next action uses the online network's best supported, feasible action, evaluated by
a periodically copied target network. If future support is wholly absent, the bootstrap
uses -1, the worst terminal-win return bound. Terminal rows never bootstrap. The loss is
Huber TD error plus `alpha * (logsumexp(Q_feasible) - Q_recorded)` and a behavior-support
auxiliary loss. Default alpha is 0.1, gamma 0.99, AdamW learning rate 0.0003, gradient norm
cap 5, target refresh every 200 updates. One shared 128×128 MLP has value and behavior heads.
This is a small explicit discrete CQL variant, not a new algorithm or an inherited guarantee
for this observational, partially observed dataset.

At inference the behavior head restricts weakly supported choices; it is **not blended
into the return score**. Among role-supported, feasible choices above the behavior-support
floor, CQL selects the largest Q estimate. No supported choice produces abstention. Dead
players are restricted to waiting; team views cannot issue individual farming options.
Objective actions require known modeled availability within 120 seconds. These checks
are necessary conditions, not evidence that a route or fight is safe.

The behavior head uses tempered class weights. Training randomly masks selected optional
history/health/structure fields to represent incomplete live observations. Accelerator
memory scales with batches; NPZ corpora are still loaded in memory. No distributed or
out-of-core scale claim is made.

## Discrete implicit Q-learning comparison

The optional `iql` algorithm fits two independent action critics and a scalar state value.
The value minimizes asymmetric squared error against the minimum target critic on the
**recorded action**, with expectile 0.7 by default. Critic targets are terminal rewards
plus the same elapsed-time-discounted next-state value. Terminal rows never bootstrap.
Target critics move toward online weights with a Polyak coefficient of 0.005 each update.

A separate categorical actor minimizes masked negative log likelihood of the recorded
action weighted by `exp(3 * (min(Q_target) - V))`, capped at 100. Actor and critic encoders
have independent parameters. Unlike the CQL/BC support auxiliary objective, this actor
uses no class balancing: reweighting actions again would change the IQL extraction loss.
Optional observation masking, action feasibility, grouped splits and terminal rewards
remain identical across the comparison. This is a discrete adaptation, not a replication
of continuous-control benchmark results in the [IQL paper](https://arxiv.org/abs/2110.06169).

IQL inference ranks categorical actor logits within the same observed role/action counts
and feasibility masks. `behavior_floor` applies to the advantage-weighted actor's masked
mass; that mass is **not** an independently calibrated behavior probability or a win
probability. No critic/actor score blending occurs. The UI identifies IQL and labels its
scores as policy scores. The extra training networks are not needed for inference.

The study runner fixes `split_seed=42` independently of optimizer seeds. All eligible
cached matches participate in either training or development; training on the development
rows would invalidate this comparison. Results are exploratory because these cached games
have been consulted previously. No model is automatically promoted by the study runner.

## Optional robustness experiments

The research robustness suite uses the same trainer with `joint_history_dropout=0.2`
and/or `bounded_values=True`; both default to disabled. Joint masking removes recent
rates/deaths/structures and objective estimates together, recomputes feasibility, and
rejects current-state omissions that would invalidate the recorded action. Rejection
counts expose the resulting action-dependent augmentation bias. Sampled current/next
inputs may be masked; stored transitions, boundaries, labels and rewards are unchanged.

Bounded CQL applies tanh only to the value head and declares `bounded_shared_v1` explicitly.
It alters CQL's optimization/pessimism and does not make values calibrated. Existing
bundles are never clipped or reinterpreted. It is unavailable for IQL or BC.

Research-only coverage calibration uses disjoint halves of development matches and
confidence ranks per role/team, retaining whole ties. It evaluates both separately fitted
cold-start thresholds and transfer of observed-state thresholds into cold-start states.
Thresholds are not installed in serving. Confidence mass is neither action correctness
nor win probability; previously consulted development partitions do not supply a fresh
confirmatory test. See the [observed results](../research/docs/ROBUSTNESS_20261105.md).

## Evaluation and deployment boundaries

All actors and both teams from each match share a split. Reports include action support,
per-role distributions, abstentions, collapse/coverage, behavior agreement and majority
baseline. The research evaluator clusters bootstrap samples by match. None estimates
improved win rate. Q estimates are not probabilities, and a higher predicted Q does not
establish a causal advantage in this data.

The serving contract intentionally rejects formation bundles. An optimizer-completed run
can become the latest **experimental** next-action policy only after dataset/bundle
validation and reload. Broad recommendation concentration in a role is shown in the UI.
Prospective outcomes, action evidence quality, hidden confounding and user adherence
require a separate evaluation before claiming useful winning strategy.

References: [Conservative Q-Learning](https://arxiv.org/abs/2006.04779),
[Riot's Game Client API](https://developer.riotgames.com/docs/lol#game-client-api),
and the source URLs preserved in `common/src/rift_common/objective_rules.json`.
