# Decision problem and robustness study

## What is optimized

Let H_t contain only observations available by t, including availability flags. Hidden
vision, wave state, intentions and mechanical details make this a partially observed
problem. Our encoder x_t = phi(H_t) is a lossy causal summary, not a sufficient belief
state. Roles in historical Match-V5 data are retrospective metadata. A team policy and
five role policies share contracts but fit different rows.

An option lasts measured duration d_t. Reward r_t is zero until terminal team outcome
Y in {-1,+1}. For an observed episode ending at T, compute recursively

    G_T = Y;  G_t = r_t + gamma^(d_t / 60) G_(t+1).

The strategic objective is E_pi[Y] (or its explicitly discounted variant). The observed
conditional mean E[G | x,a] is not Q^pi: future teammates, opponent decisions, hidden
confounders and the continuation policy differ. Neither regression nor action agreement
identifies which alternative action would increase a player's chance to win.

For example, if only action A is observed at x, two worlds can agree on every recorded
(x,A,Y) and disagree completely on the result of B. No learner can distinguish them
from those records. Global action counts do not establish conditional overlap at x.
Partial observability can also violate exchangeability even when both actions occur.
Thus more same-distribution games, a new network or a conservative penalty alone cannot
prove an alternative strategy better. Imitation is a supervised surrogate here, not a
direct optimizer of terminal victory. Prospective outcome evidence remains necessary.

## Baselines and conservative regression

The imitation baseline minimizes weighted cross entropy

    L_BC(batch) = sum_i[-w_(a_i) log softmax(z(x_i))_(a_i)] / sum_i w_(a_i),

using inverse-square-root class frequencies, normalized over supported training classes.
Its outputs are behavior scores; class weighting prevents interpreting them as calibrated
behavior probabilities. Feasibility masks and empirical scope support restrict serving.

The return variant uses bounded q(x,a) in [-1,1] and minimizes

    L_return = E[Huber(q(x,a), G)]
             + alpha E[logsumexp_(b in F(x)) q(x,b) - q(x,a)]
             + 0.25 L_BC.

F(x) is the explicit feasibility set. The conservative penalty discourages unsupported
alternatives, but does not correct confounding. This observed-return adaptation is not
a theorem or faithful replication of Bellman CQL. The existing Bellman alternative uses
r + gamma^(d/60) (1-terminal) Q_target(x_next,a_next), preserving exact stored next states.

Reference: [Kumar et al., Conservative Q-Learning](https://arxiv.org/abs/2006.04779).
Conservative smoothing is also studied by
[Yang et al., RORL](https://arxiv.org/abs/2206.02829). Our discrete availability omission
method below is a project-specific experiment, not an implementation of RORL.

## Availability-aware consistency

### European representation and checkpoint experiment

`decision-state-v2` preserves all 64 v1 features and adds 24 masked static features:
four team-mean differences, three focal-minus-ally and three focal-minus-enemy
differences, and two measured alive fractions. Linear differences use the operands'
existing scales and require both known flags. Alive fractions require all five
current dead flags. No new event-history derivative can preserve omitted information.

The optional phase screen intersects serving support with `N_train(scope, phase, a)
>= 20`, using identified training rows and three fixed time bins (<14, 14–25, >=25
minutes). It does not alter the stored feasibility masks or Bellman target support.
It is a coarse empirical screen; conditional exchangeability and state-local overlap
remain unestablished. Report the resulting coverage and action loss.

The softer imitation candidate replaces inverse-square-root class weights with
inverse-quarter-power weights, retaining normalization over supported classes.
IQL uses its own advantage-weighted categorical actor loss. Sparse identified terminal
rows, especially TEAM, remain a limitation of its TD supervision.

Every 500 updates (up to 4,000), evaluate the frozen clean/cold development criterion
below on the fitted scope. Restore the best network parameters and stop after three
nonimproving checkpoints. Record both completed updates and the restored checkpoint.
Optimizer state is not resumed after restoration. Candidate and checkpoint selection
never consult reserved games. These changes form a product experiment, not an isolated
feature/dropout ablation or evidence of increased win rate.

The subsequent adaptive regional selection reuses these saved checkpoints. Its score
is the equal-weight mean of the European and Korean development scores defined below,
with both regional gates and at least twenty identified targets per region. It changes
candidate selection only, not checkpoint fitting or optimizer weights. A revision
motivated by a held-out result needs fresh evaluation; the consulted cohort cannot
independently establish its quality. The failed first-study TEAM comparison is retained.

### Shared omission regularizer

Let C(x) zero the history-dependent values AND their known/enabled flags. It removes
information without filling it with predictions. Recompute feasibility F(C(x)). Training
also samples joint omission with probability 0.5. Supervised omission is rejected if the
recorded target becomes infeasible; exclusions and rejection counts remain visible.

For consistency, define I = F(x) intersect F(C(x)) intersect empirical scope support.
Use rows only when the recorded action is in I and |I| > 1. Let p be the clean network's
softmax over I, detached from differentiation; let p_c be the corrupted network's softmax
over I. The additional loss is

    L_consistency = mean_(eligible rows) KL(stopgrad(p) || p_c)
                    + 1_(critic) mean_(eligible row/action entries)
                      Huber(q(C(x),a), stopgrad(q(x,a))).

Fit L_base + 0.25 L_consistency. The coefficient and omission rate are frozen before this
study. This penalizes reliance on removable history while permitting some justified
adaptation. It assumes nearby information patterns should have some shared ranking;
that assumption may hurt decisions where history is essential. Thus ordinary baselines,
clean agreement, cold agreement, conditional switching and abstention must all be reported.
Ordinary candidates retain the previous 20% joint dropout and optional-field dropout;
the new versions combine 50% joint dropout with consistency. This four-way comparison
does not isolate the causal contribution of stronger dropout versus consistency.
This regularizer does not create missing data, calibrated uncertainty or causal values.

## Coherence and abstention

Independently predict macro m and specialist a from their causal summaries. Let D be
the set of explicit lane-rotation/objective destinations. Define conflict(m,a) when
m != a and both are in D. If conflict, retain m and withhold a; do not choose the next
highest score merely to manufacture agreement. Recheck after advice persistence.

This is a conservative feasibility constraint on the delivered pair, not joint-value
optimization. Nonconflicting actions can still be strategically incompatible in ways our
coarse contract cannot observe. Matching actions establish symbolic agreement only.
Withholding must be counted separately from aligned proposals and existing abstentions.
A stronger learned coordinator needs causal joint-action supervision and useful team
evidence; current sparse macro labels do not establish that capability.

## Selection and evidence boundaries

All candidate algorithms may be selected. For each scope use development rows only:

    S = (F1_clean + F1_cold-feasible) / 2 - 0.1 switching_still-feasible.

F1_clean uses identified targets. F1_cold-feasible excludes recorded actions invalidated
by omission, and reports its denominator. Switching conditions on the original predicted
action remaining feasible; this separates prerequisite loss from unstable choices.
Candidates need clean/cold retention >= 0.5, voluntary top share <= 0.9, and nonempty
identified clean/cold evidence. Prefer passing candidates, then S; keep failed-gate runs
for diagnosis. These are exploratory engineering gates, not thresholds proving good play.
The frozen switching score includes forced respawn rows, which can dilute instability
among voluntary decisions. Final omission diagnostics additionally report voluntary-only
switching and its denominator. That added measurement does not change frozen selection.

Freeze newly collected IDs before fitting; old tests are now development evidence.
Report match counts, repeated-account dependence and pooled support inspection. Fresh
downloaded IDs may still contain recurring players and are not independent-player tests.
If the bounded collection yields no new IDs, report development-only evidence explicitly.
Never choose models or coefficients by inspecting the reserved test.

The default should change only if every scope passes declared development gates and
engineering verification. Even then, improved terminal win rate needs a prospective,
predefined outcome comparison after later live acceptance.
