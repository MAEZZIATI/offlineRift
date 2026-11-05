# Offline Rift
## Mathematical synthesis | 5 November 2026

An experimental next-action policy for a selected League of Legends player/role or team. The goal is terminal team victory. We have a working offline learning and advisory pipeline, evidence of better robustness to missing history, and a new decision-rule candidate. Improved live win rate, causal decision quality and algorithmic novelty remain unestablished. No experimental model or threshold was promoted.

### 1. The problem we actually observe

Each decision uses a 64-value causal public-observation vector, including time, role, public statistics, recent history and availability masks. The 15 options describe coarse activities such as farming, rotation, objective preparation, fighting and waiting to respawn. The true game state is only partly observed; the vector is not assumed to be a sufficient Markov state.

```math
o_t = encode(causal public observations and history)
D = {(o_t, a_t, r_t, o_next, d_t, Δt, match_id)}
G_t = Σ[k ≥ t] γ^((T_k − T_t)/60) r_k ;  J(π) = E_π[G_0]
```

Here a_t is a reconstructed executed option, d_t marks a terminal transition, and Δt is measured elapsed time in seconds. Rewards are zero until the terminal team result: +1 for a win and −1 for a loss. With γ = 0.99 and only terminal rewards, true discounted returns lie in [−1, 1]. Learned critics need not respect that interval unless explicitly bounded. Neither Q nor a classifier score is a win probability.

Future frames identify action/reward/next-state targets only. They never enter the preceding input. Intervals are actual measured frames, usually about one minute apart; there are no invented intermediate commands, movement or vision. Complete actor episodes and exact next states are preserved. Unresolved behavior excludes the episode rather than joining rows across a gap.

### 2. Data and comparison discipline

The cached corpus yields 1,280 accepted matches, 15,360 player/team episodes and 365,040 transitions. All perspectives from one match stay together: 1,024 training matches and 256 development matches. The latter contain 71,472 transitions and have already been consulted, so these are exploratory development results, not a fresh confirmatory test.

Twelve BC/CQL/IQL runs compared six variants across two optimizer seeds. Eight further runs compared ordinary CQL, joint history masking, bounded critics and their combination. Each trained for 10,000 updates on the same match split. Serving reloads reproduce saved predictions. Research reuses the maintained builder/trainer and never publishes a latest policy automatically.

### Reading the evidence

Behavior agreement asks whether recommendations match reconstructed recorded behavior. Action coverage counts distinct predictions; retention coverage counts states receiving advice. Concentration measures repetition. These diagnose fit and support, not the effect of following advice. A successful optimizer run does not establish useful strategy.

---PAGE---

# Learning the policy
## Losses, support constraints and missing observations

### 3. Three learners, three different roles

BC (behavior cloning) is an imitation baseline: predict recorded behavior using masked categorical probabilities μ(a|o). Tempered class weights reduce domination by common options. BC does not optimize the terminal reward.

```math
L_BC ∝ E_D[−w(a) log μ(a|o)]
y_CQL = r + (1 − d) γ^(Δt/60) Q_target(o_next, a_star)
a_star = argmax[b ∈ A_next] Q_online(o_next, b)
L_CQL = E_D[Huber(Q(o,a) − y_CQL)]
          + α E_D[log Σ[b ∈ F(o)] exp Q(o,b) − Q(o,a)]
          + 0.25 L_BC
```

F(o) is the necessary-feasibility set. A_next additionally requires training role/action support and sufficient behavior-head mass. If no supported next action exists, the bootstrap uses −1; terminal rows never bootstrap. The conservative term discourages unsupported high values, but can push learned estimates below −1. Defaults include α = 0.1 and a target-network copy every 200 updates.

Discrete IQL uses twin critics, an expectile state value and a separate categorical actor. It learns the value from recorded actions, then fits the actor using advantage weights. Unlike BC/CQL's auxiliary head, this actor is not class balanced.

```math
q_bar = min(Q_target1(o,a), Q_target2(o,a)) ; u = q_bar − V(o)
L_V = E_D[|τ − 1{u < 0}| u²]
y_IQL = r + (1 − d) γ^(Δt/60) V(o_next)
L_actor = E_D[−min(exp(β(q_bar − V(o))), W) log π(a|o)]
```

The comparison uses τ = 0.7 or 0.9, β = 3 and W = 100. IQL serves actor logits; CQL ranks supported Q estimates. Both can learn a repetitive policy. IQL's roughly 91% fighting concentration in jungle/support is a warning, not evidence that fighting is optimal.

### 4. What improved robustness

Joint masking removes recent rates/deaths/structures and objective estimates together with probability 0.2. Masks and feasibility are recomputed. Current-state omissions that invalidate the recorded target are rejected: about 3.6% of proposals, creating explicit action-dependent augmentation bias. Stored targets and transitions remain unchanged. Bounded CQL replaces its value head by tanh, preserving unbounded behavior logits and an explicit bundle architecture.

| Variant | Cold-start prediction change | Seed agreement | Jungle dominant action |
| Ordinary CQL | 58.51–72.35% | 74.63% | 82.77–85.38% |
| Joint history masking | 13.93–17.16% | 74.66% | 72.35–83.60% |
| Bounded + joint | 12.41–12.67% | 82.43% | 68.30–73.13% |

Ranges span two seeds. Lower sensitivity is measurable progress; it can still reflect shared repetitive bias. Bounding values enforces a numerical range, not value accuracy or calibration.

---PAGE---

# Beyond replication
## A testable decision rule, with its negative results

### 5. Minimax regret across evidence views

The proposed candidate chooses a supported compromise across critic seeds and controlled losses of causal evidence. Each critic is evaluated on four views: observed input, missing health, missing recent numerical history, and both omissions. Objective prerequisites are retained in this decision-rule experiment; already-missing evidence is never restored.

```math
A_v(o) = {a ∈ F_v(o): N_role,a ≥ 20 and μ_v(a|o) ≥ 0.02}
I(o) = intersection[v] A_v(o)
R_v(o,a) = max[b ∈ A_v(o)] Q_v(o,b) − Q_v(o,a)
a_hat(o) = argmin[a ∈ I(o)] max[v] R_v(o,a)
abstain if I(o) is empty or max[v] R_v(o,a_hat) > ε
```

The view index v combines optimizer seed and omitted evidence. Support counts come only from training data. Independent additive critic offsets cancel: replacing Q_v by Q_v + c_v leaves R_v unchanged. A compromise can therefore beat both critics' individual winners in worst-case relative gap. This invariance does not remove multiplicative scale differences or estimation error. R_v is a learned decision gap, not observed causal regret.

The study separates seed-only, omission-only and combined effects. Fixed tolerances ε ∈ {0, 0.05, 0.1, 0.25} were specified before inspection; zero requires a common maximizing action. With bounded + joint critics, the combined ε = 0.05 filter retains 92.73% of all development decisions and predicts all 15 actions. But jungle/support dominant-action fractions remain 77.4%/82.8%. Stable consensus can amplify a shared fighting bias.

### 6. Compare against the simpler explanation

An adaptive follow-up fits confidence-only thresholds to the regret rule's per-role retention on 128 calibration matches, then evaluates both on the other 128. Coverage is approximately matched; thresholds use no behavior/outcome labels. The table gives retained-state behavior agreement, not policy quality.

| Variant | Regret: coverage / agreement | Confidence: coverage / agreement | Actions: regret / confidence |
| Ordinary CQL | 75.05% / 56.75% | 75.69% / 57.15% | 12 / 11 |
| Joint history | 80.20% / 55.85% | 80.33% / 55.70% | 15 / 13 |
| Bounded critic | 89.09% / 50.69% | 88.20% / 54.03% | 15 / 13 |
| Bounded + joint | 92.98% / 51.09% | 92.12% / 52.67% | 15 / 15 |

The regret filter sometimes preserves more action types, but has no consistent agreement advantage. The candidate has not earned promotion. Two optimizer seeds do not define a calibrated uncertainty distribution, and legitimately useful evidence can change the best action when removed.

### Contribution boundary

The research idea is the combination of executable-option support, exact evidence removal, relative regret and explicit abstention under coarse observations. Ensemble pessimism and regret have substantial prior literature. This is a falsifiable candidate with ablations and retained failures, not an established novelty claim or field advance. Independent tasks with known outcomes and prospective evidence are needed.

---PAGE---

# What the evidence permits next
## Calibration, label ambiguity and a practical research agenda

### 7. Conditional coverage is not correctness calibration

For group g and availability signature z, fit a confidence threshold on calibration states only. Confidence c(o) is feasible-action-normalized behavior-head mass of the serving-selected action; it does not replace CQL's Q ranking.

```math
t_(g,z)(ρ) = Quantile_(1−ρ){c(o_i): i ∈ calibration cell (g,z)}
retain advice when c(o) ≥ t_(g,z)(ρ) and the cell is supported
coverage = retained states / evaluated states
```

Cells require at least 200 distinct source rows and 20 calibration matches; repeated row/signature pairs count once. Otherwise abstain. At target ρ = 0.90, the eight policies retain 89.29–90.44% overall in four controlled availability scenarios and pass the predefined 85–95% screen in sufficiently supported cells. Whole confidence ties stay together. The calibration set includes these omissions: this is not evidence of transfer to unseen live missingness.

Action retention means keeping states labeled with an action, not recommending that action correctly. After full cold-start omission, all recorded objective-preparation targets become infeasible. For example, all 273 jungle dragon-preparation rows in the evaluation half lose feasibility, although one model keeps advice on 95.2% of those states by choosing other actions. Good coverage can conceal zero objective-action recall.

### 8. Labels are a central scientific bottleneck

The full audit contains 135,146 fighting labels among 304,200 player transitions. Within fight intervals, 25.7% are death-only participation, 69.4% also gain lane CS, and 20.2% first record combat in the final quarter. These overlapping facts do not reveal initiation, forced response or duration. Reviewing 75 stratified intervals exposed three death-containing cases that become RESET when combat events are removed: base arrival is not sufficient evidence of a voluntary reset. The labels remain unchanged.

### 9. Next steps, in order

1. Improve the evidence contract. Represent overlapping or ambiguous executed-option evidence explicitly before defining finer action targets. Preserve causal inputs, terminal boundaries and exact transitions; do not infer intention from endpoints.
2. Challenge the candidate. Add more optimizer seeds and independent tasks with known outcomes. Repeat matched-coverage comparisons and report per-role/action support, concentration, rare-action recall and abstention. Diagnose shared critic/label bias before adding more complexity.
3. Measure real missingness. The new opt-in live recorder stores aggregate availability patterns, role/time buckets and observation-session counts, excluding identities, state values, actions and outcomes. It is off by default and tested with fixtures; no active-game capture was exercised. Polls are correlated, and sessions are not verified unique matches.
4. Establish usefulness prospectively. Freeze the policy and eligibility rules; predefine match/team-level assignment, adherence, effect size, power/sample size, exclusions and stopping rules. Terminal team victory is the outcome. Offline agreement cannot supply a policy win-rate estimate.

### Engineering status and traceability

One maintained collection/build/train path; runtime loading imports no training/research orchestration. Experimental architecture versions, atomic reports and artifact integrity checks are explicit. Validation: 57 tests passed, six optional browser tests skipped; lint, formatting, CLI help and diff checks passed. Raw data/models/reports stay local and ignored. Detailed sources: ROBUSTNESS_20261105.md, DECISION_RESEARCH_20261105.md and NEXT_STUDY_PROTOCOL.md in research/docs/; related methods include IQL (arXiv:2110.06169), MSG (arXiv:2205.13703) and Jin et al., ICML 2018 (PMLR 80).
