# Evidence-Conditioned Macro Options under Partial Observation

**Theory and methods proposal, 2026-11-04.** Authors and affiliations pending.
This is a mathematical research draft. The proposed factorization, evidence penalty,
variational objective and option learner are not implemented or validated. A separate
causal-memory BC research branch has since been implemented and trained; see
[the controlled memory study](../MEMORY_STUDY.md). It is not the proposed option learner.
Elementary arguments below are deductions under stated assumptions, not claims of new
theorems. A comprehensive novelty review remains outstanding.

## Abstract

Macro advice from sparse public telemetry combines three distinct uncertainties:
incomplete observation, ambiguous behavioral labels and unsupported intervention values.
We propose a factorized option predictor that conditions each strategic claim on its
available evidence and may abstain separately on destination, formation and commitment.
The framing combines partial observation, information-preserving compression, temporal
abstraction and selective prediction. We derive an observation-aliasing bound and a
support-constrained entropy-regularized ranking distribution, and specify experiments
that could reject the method. Statistical-mechanics analogies motivate bounded energy
scores and hysteresis; no physical law or quantum computational advantage is assumed.

## 1. Formal problem: observations are not the underlying game

Let `S_t` be the latent game state and `A_t` the teams' executed controls. A live source
provides `O_t=g(S_t,M_t)+noise`, with availability mask `M_t`. The available history is

\[
H_t=(O_{0:t},M_{0:t},\Delta_{0:t}),\qquad
b_t(s)=P(S_t=s\mid H_t).
\]

`Δ` contains elapsed time and freshness. We do not assume that the current feature
vector is Markov. A causal recurrent encoder `Z_t=fθ(H_t)` approximates a task-relevant
history representation; it must reset between team episodes and clock rollbacks.

A geometric weak label `L_t=h(S_t^{observed})` is neither `A_t` nor an optimal next
decision. A model trained on `L_t` learns its conditional distribution, subject to
sampling and weighting. Interpreting `E[R|L,H]` as `E[R|do(A),H]` would additionally
require valid intervention labels, consistency, confounding control, support and a
defensible environment model. Those assumptions have not been established here.

### Proposition 1: indistinguishable histories limit classification

For any categorical audited target `Y` and observation history `H`, the best measurable
classifier has accuracy

\[
\sup_f P(f(H)=Y)=\mathbb E_H\max_y P(Y=y\mid H).
\]

**Argument.** Conditional on a history `h`, choosing the most probable label maximizes
the success probability. Integrate that conditional maximum. Randomization cannot
improve a linear average of conditional success probabilities.

If a history class has total probability mass `p` and contains two incompatible target
labels with conditional probabilities `q` and `1−q`, every history-only classifier makes
at least `p min(q,1−q)` error on that class. This does not numerically bound the current
system: `p,q` are unknown, continuous histories need meaningful equivalence classes,
and richer causal history can distinguish states that a single frame cannot.
It explains why larger networks cannot guarantee recovery of missing evidence.

### A second limitation: weighting changes the learned conditional distribution

Even with infinite data and a sufficiently expressive model, weighted cross entropy
does not generally recover the natural label posterior. At a fixed history, minimizing
`−E[c w_Y log q_Y | H]` over categorical `q` yields

\[
q_y^*(H)=\frac{w_y\,\mathbb E[c\mid Y=y,H]P(Y=y\mid H)}
{\sum_j w_j\,\mathbb E[c\mid Y=j,H]P(Y=j\mid H)},
\]

when the denominator is positive. This follows from the same constrained log-loss
minimization as ordinary categorical estimation. It explains why confidence-weighted,
balanced logits require separate natural-frequency calibration and cannot be read as
formation probabilities, let alone action-success probabilities. If all conditional
weights vanish, this objective provides no identification of `q` at that history.

## 2. Factorized options with evidence requirements

Following the established [options framework](https://www.sciencedirect.com/science/article/pii/S0004370299000521),
an option has initiation set `I`, intra-option policy `π` and termination rule `β`.
We propose a descriptive factorization

\[
o=(d,f,g,c,\tau),
\]

where `d` is destination/objective, `f` formation, `g` participating group, `c` commitment,
and `τ` duration/termination context. Factor values include unknown. A partial suggestion
may support an objective window while declining to claim a safe route or engage.
This differs from the existing twelve mutually exclusive formation labels.

For factor `j`, let `E_j(H)` denote observed evidence satisfying a declared requirement,
and let `u_j(H)` be a learned uncertainty estimate validated on held-out audited labels.
Absence of required direct evidence may still permit statistical inference, but the
UI must distinguish inferred from observed claims. Define acceptance

\[
g_j(H)=\mathbf1\{u_j(H)\le t_j\}\mathbf1\{\text{claim type permitted by }E_j(H)\}.
\]

Masks indicate availability, not certainty. A timing-compatible option can still lack
position, vision and combat evidence. The system therefore needs factor-specific
evaluation rather than a universal confidence number.

## 3. Learning objective and train-only future information

For audited factor targets `Y_j` and target-availability mask `v_j`, propose

\[
\mathcal L=\sum_j\mathbb E[v_j w_j\ell(p_j(Z),Y_j)]
+\lambda_{aux}\mathcal L_{event}
+\lambda_{IB}\mathcal L_{compression}
+\lambda_{dist}\mathcal L_{distillation}.
\]

`L_event` predicts verified events in future 30/60/120s windows as labels only; future
frames never enter `H`. Events outside an observable recording horizon are censored,
not negative examples. `L_distillation` uses a full-profile training teacher only where
its relevant target is independently validated; teacher error is reported separately.
No teacher coordinate estimate may be exposed as observed live geometry.

The [information bottleneck](https://arxiv.org/abs/physics/0004057) motivates retaining
predictive information while limiting irrelevant detail. For stochastic encoder
`qθ(z|H)` and reference distribution `p₀(z)`, the variational penalty

\[
\mathcal L_{compression}=\mathbb E_H\mathrm{KL}(q_\theta(Z\mid H)\Vert p_0(Z))
\ge I(H;Z)
\]

follows from the decomposition into `I(H;Z)+KL(q(Z)||p₀(Z))`, assuming the densities and
KL are well-defined. A factor-prediction loss encourages relevance. This formulation
requires a stochastic latent model; it is not a justified finite information penalty
for an arbitrary deterministic continuous encoder. Smaller latent dimension alone is
not proof of a bottleneck benefit. Start with `λ_IB=0` and add this component only if
the audited memory baseline works.

## 4. A support-constrained energy distribution

Let `μ(o|H)` be a behavior-support reference, estimated from executed options only after
their annotations exist. Current weak formation frequencies are not such a behavior
policy. For a future valid option dataset, define score

\[
F(o,H)=B(o,H)+\eta\,\operatorname{clip}(\widetilde Q(o,H),-c,c)
-\lambda U(o,H)-\kappa C(o,H),
\]

with audited pattern evidence `B`, a bounded value contribution, evidence uncertainty
`U`, and an explicit claim-cost `C` for relying on unsupported factors. The proposed
ranking solves, over options satisfying declared necessary constraints,

\[
\max_p\sum_o p_o F(o,H)-T\mathrm{KL}(p\Vert\mu),\quad T>0.
\]

### Proposition 2: Gibbs form and hard support

On a finite nonempty feasible set with `μ_o>0`, the optimizer is

\[
p_o^*=\frac{\mu_o\exp(F_o/T)}{\sum_{o'}\mu_{o'}\exp(F_{o'}/T)}.
\]

**Argument.** Add a multiplier for `Σp=1`; stationarity gives
`F_o−T(log(p_o/μ_o)+1)+λ=0`. Normalize to obtain the expression. Strict concavity gives
uniqueness on the positive support. If `μ_o=0`, assigning positive mass incurs infinite
KL, so the optimizer cannot assign that option. Empty feasible support requires an
explicit abstention outcome instead of silently normalizing an empty set.

This is a classical entropy-regularized optimization identity, not a new RL guarantee.
Its resemblance to a Gibbs distribution is useful for separating evidence, support and
temperature. Its normalized values express ranking preferences, not win probabilities.
Unknown support cannot be made known by exponentiating logits. The existing hybrid is
a bounded score rule; it does not implement this objective or estimate `μ`.

## 5. Temporal abstraction and a control-inspired switching cost

For an option lasting `k` ten-second intervals, consistent discrete discounting gives

\[
R_o=\sum_{j=0}^{k-1}\gamma^j r_{t+j},\qquad
Q(H_t,o)=\mathbb E[R_o+\gamma^k V(H_{t+k})].
\]

For irregular times, use `exp(−λ Δt)` with `λ=−log(γ)/10` in inverse seconds and discount
each reward by its elapsed timestamp. Terminal episodes never bootstrap. Dropping two
out of every three transitions while retaining `γ` does not produce equivalent options.
Event-based rewards must be assigned once at their actual boundaries.

A control-inspired extension adds `−ξ 1[o_t≠o_{t−1}]` to reduce suggestion flicker.
Switch only when new evidence exceeds the cost, with mandatory termination on invalidity
or unavailable required evidence. This resembles hysteresis in physical systems but
does not prove control stability or strategic value. It could dangerously preserve a
bad suggestion; evaluate reaction delay after verified context changes and compare
against no switching cost. The system is advisory and does not execute controls.

## 6. Selective risk and falsifiable claims

Following [selective classification](https://arxiv.org/abs/1705.08500), define factor
coverage `C_j=P(g_j=1)` and conditional error

\[
R_j=\frac{\mathbb E[g_j\mathbf1\{\hat Y_j\ne Y_j\}]}{\mathbb E[g_j]},\quad C_j>0.
\]

At zero coverage, conditional risk is undefined, not zero. Fit thresholds on a separate
calibration cohort, then report complete risk–coverage curves and match-cluster intervals
on untouched tests. Classical independent-example guarantees do not automatically
apply to correlated game states, repeated players or patch shifts.

The research is rejected or simplified if these tests fail:

| Hypothesis | Required comparison | What would count against it |
|---|---|---|
| History resolves useful aliasing | GRU 60/120s vs matched single-state BC | No audited-label gain on later matches across fixed-split seeds |
| Factorization represents simultaneous behavior | Joint and per-factor audited targets vs flat labels | Improved marginal scores but incoherent joint suggestions or unacceptable annotation disagreement |
| Evidence conditioning limits unsupported claims | With/without evidence penalty at matched coverage | No reduction in audited unsupported claims, or only trivial near-zero coverage |
| Auxiliary futures improve representation | With/without event targets, same inference features | Gains vanish after censoring, patch-shift testing or label audit |
| Teacher helps the live student | Validated teacher targets vs no teacher | Student inherits teacher errors or relies on unavailable live signals |
| Q ranking contributes | BC vs identical BC plus bounded Q | No primary-metric gain or worsened rare-factor coverage |

Predeclare audited macro F1 and selective risk at specified positive coverage, use
five optimizer seeds with frozen match splits, and report source/patch performance.
Current development-set reanalysis already fails to establish a macro F1 benefit from
Q ranking; this is motivation to keep that component optional.
A later frozen-model evaluation on 243 new matches observes a small +0.37-point macro
F1 gain, interval [+0.015, +0.713], while balanced accuracy decreases by 2.46 points.
That exploratory extension supports retaining the comparator; it does not validate
the proposed option method or supply audited evidence of better decisions.

## 7. What quantum inspiration would and would not add

A classical belief vector `b` can be written as a diagonal matrix `ρ=diag(b)`. Then
`−tr(ρ log ρ)=−Σ b_s log b_s`; for a diagonal event projector, its trace probability
equals an ordinary sum of classical belief masses. This notation supplies no extra
predictive information or computational advantage. Off-diagonal terms cannot be
identified from these diagonal observations alone. Introducing complex amplitudes or
an uncertainty-principle analogy without an observation model would add unsupported
parameters. Quantum-inspired representations are therefore a low-priority hypothesis,
with no planned empirical claim. Classical filtering and entropy regularization provide
the useful mathematics for this dataset first.

## 8. Status and relation to the empirical paper

Implemented: masks, causal resampling, a team-set encoder, weighted BC, CQL, bounded
hybrid ranking and frozen-model uncertainty analysis. An experimental causal GRU branch
now compares 60/120s histories with matched snapshot and repeated-current controls
across five optimizer seeds; it is not deployed. Proposed: option, factorized,
variational, distillation and calibrated-rejection components above.
There is no result establishing those proposals, their novelty, a safe engage detector,
or improved win rates. The appropriate next contribution may be a careful negative
result about observation limits rather than a new policy algorithm.

The development experiment illustrates why computation controls matter: mean macro
F1 is 16.95% for the matched snapshot, 21.33% for a GRU repeating current information,
and 22.41% for real 120s history. The information-specific contrast is therefore
1.08 points, not the full 5.46-point snapshot contrast. This is a descriptive ablation,
not proof of a latent belief representation. The [label audit](../LABEL_AUDIT.md) also
finds objective-rule contradictions; improving prediction against those rules does
not establish that the proposed strategic factors have been learned correctly.

On the subsequent 846-match frozen evaluation, the real-history contrast over the
repeated-current control is +0.90 macro F1 points, with positive conditional intervals
for each of five fits. The experimental history model nevertheless averages 22.11%,
below the existing plugged hybrid's 22.71%. This is limited evidence about representation
under weak supervision, not evidence for option-value identification or superiority
of the proposed option method. Full results, including the 212-match participant-disjoint
sensitivity analysis, are in [the memory record](../MEMORY_STUDY.md).

## References

1. Sutton, Precup and Singh (1999). [Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning](https://www.sciencedirect.com/science/article/pii/S0004370299000521).
2. Tishby, Pereira and Bialek (1999; arXiv 2000). [The information bottleneck method](https://arxiv.org/abs/physics/0004057).
3. Kumar et al. (2020). [Conservative Q-Learning for Offline Reinforcement Learning](https://arxiv.org/abs/2006.04779).
4. Geifman and El-Yaniv (2017). [Selective Classification for Deep Neural Networks](https://arxiv.org/abs/1705.08500).
5. Zaheer et al. (2017). [Deep Sets](https://arxiv.org/abs/1703.06114).
