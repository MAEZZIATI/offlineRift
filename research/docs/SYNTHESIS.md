# Offline Rift: six-policy macro coach

Current product synthesis. One team-direction model and separate TOP, JUNGLE, MIDDLE, BOTTOM and UTILITY specialists share causal state/action contracts and terminal team victory as the objective. The ADC-only pilot is historical.

This phase builds and tests the product offline. No active-game test was run. Saved-model correctness and behavior diagnostics do not establish optimal strategy, calibrated victory probabilities or increased win rate.

Architecture: verified raw cache → causal v3 evidence and complete episodes → one maintained trainer → six independent selected networks → integrity-checked inference router → team proposal, role proposal and symbolic alignment/conflict. Production imports no research; inference imports no training orchestration.

Coordination compares actual proposals. A conflicting role destination is explicitly withheld while the macro proposal remains visible. No compatible alternative is invented. This constraint is not learned plan conditioning or evidence of strategic compatibility.

| Layer | Boundary |
| --- | --- |
| Common | Causal state, evidence, transitions, scopes, compatibility |
| Production training | Optimizer and explicit TD/observed-return targets |
| Production inference | Six verified independent networks and routing |
| Workflow / UI | One orchestration path and optional future delivery |
| Research | Offline diagnostics and reproducible reporting |

# Data, targets and exclusions

The frozen dataset is /home/amine/offlineRift/data/projects/faker/next-actions/datasets/next-actions-6ea423fe0e6de2b8/transitions.npz. Its SHA-256 is 8ab20f604026659a32cb2d8259af81799a154d5741d8f769a0a841ddb612f4c8. There are 1296 training, 323 development and 21 reserved matches. Every perspective of a match remains together.

Bounded collection: 57 configured verified accounts; 549 cache references and 21 new payloads from 7 accounts; route asia. All new IDs were reserved before fitting. Historical tests are now development evidence. Rebuilding uses retained raw payloads, not synthetic games. Configured-account distribution and recurring players limit generalization.

V3 retains representative labels, candidate activities, identification flags and complete stored transitions. Death-only combat and overlapping activity do not identify intent. TEAM targets require three identified matching executed macro options. No coherent direction is unresolved, not a positive continue strategy.

Only identified rows in the requested scope fit its action-specific losses/support counts. Exclusions remain a selection bias. Complete episodes, exact next states, measured elapsed discounts and terminal rewards are preserved without stitching gaps.

| Scope | Identified rows (pooled) | Matches with evidence |
| --- | --- | --- |
| TEAM | 2525 | 1114 |
| TOP | 49206 | 1640 |
| JUNGLE | 44059 | 1640 |
| MIDDLE | 47087 | 1640 |
| BOTTOM | 46457 | 1640 |
| UTILITY | 47734 | 1640 |

![Identified activity support](SYNTHESIS_support.svg)

# Training and development comparison

Each scope compares ordinary imitation and conservative observed-return regression with availability-consistent versions: four candidates per scope, 24 total. Every algorithm is eligible for selection. The frozen score averages clean and cold-feasible identified F1 and subtracts 0.1 times still-feasible prediction switching. Gates require nonempty evidence, clean/cold retention at least 50% and voluntary concentration at most 90%. These exploratory gates do not prove policy quality.

cql-td uses supported next-action Bellman targets. cql-return variants regress complete observed episode returns with the same conservative penalty. Macro candidates use observed returns because identified macro intervals rarely coincide with terminal frames. This is an explicit conservative return-regression adaptation, not a replication or new theorem for standard CQL.

Terminal team result is used only as supervision. Observed trajectory returns remain confounded by teammates, opponents and hidden state; they are not identified causal alternative-action values. Bounded heads constrain numerical range but do not calibrate values.

A selected imitation child learns next-activity agreement, not terminal win/loss directly. The product's victory objective does not convert this supervised surrogate into a demonstrated winning policy. Return losses apply to critic candidates only.

F1 measures reconstructed-activity agreement. A higher imitation baseline or concentrated policy must be reported, not reinterpreted as a victory-rate comparison. All candidates, failures and source snapshots are retained in the study directory.

| Scope | Selected model | Identified F1 | BC F1 | Voluntary top share |
| --- | --- | --- | --- | --- |
| TEAM | bc-consistent | 0.344 | 0.325 | 43.6% |
| TOP | bc | 0.177 | 0.177 | 55.2% |
| JUNGLE | bc | 0.197 | 0.197 | 34.3% |
| MIDDLE | bc | 0.203 | 0.203 | 50.8% |
| BOTTOM | bc-consistent | 0.179 | 0.184 | 45.6% |
| UTILITY | bc-consistent | 0.173 | 0.183 | 30.6% |

# Mathematical design and robustness

Partial observability: x = phi(H) summarizes causal history with explicit known flags; it is not a sufficient belief state. Terminal reward is Y in {-1,+1}. Complete-episode return satisfies G_t = r_t + gamma^(d_t/60) G_next. Hidden teammates, opponents and continuation behavior prevent identifying counterfactual action values from observed returns.

Imitation minimizes weighted cross entropy. Conservative regression minimizes Huber(q(x,a),G) + alpha[logsumexp q(x,b) - q(x,a)] + 0.25 cross entropy, with bounded q. The history-consistent variants add 0.25 times detached-teacher KL plus critic Huber disagreement on supported actions feasible before AND after omission. Joint history dropout is 50%; neither values nor known flags are imputed.

The regularizer assumes some ranking stability after information loss. This can suppress justified adaptation; the table therefore reports both cold-target agreement and switching conditional on the warm action remaining feasible. Recorded targets invalidated by omission are excluded from the cold F1 denominator, never silently relabeled.

Coherence is a constrained delivery rule: when macro and role choose different explicit destinations, withhold the role action. Other pairs remain independent rather than being claimed aligned. This adds abstention, not a learned alternative or a joint-value theorem. Recheck after advice persistence.

The full equations, assumptions and primary references are in docs/MATHEMATICAL_DESIGN.md. Related work: Kumar et al., CQL (arxiv.org/abs/2006.04779); Yang et al., RORL (arxiv.org/abs/2206.02829). Our availability omission experiment is not a replication of RORL.

| Scope | Selected | Cold F1 | Feasible switching | Gate |
| --- | --- | --- | --- | --- |
| TEAM | bc-consistent | 0.227 | 20.3% | pass |
| TOP | bc | 0.200 | 6.2% | pass |
| JUNGLE | bc | 0.190 | 13.4% | pass |
| MIDDLE | bc | 0.235 | 12.4% | pass |
| BOTTOM | bc-consistent | 0.210 | 4.0% | pass |
| UTILITY | bc-consistent | 0.197 | 14.9% | pass |

# Reserved-match results

This fixed selection was evaluated on 21 reserved match IDs and 5664 measured transitions. These IDs were excluded from optimizer and development selection; repeated players and team dependence remain possible. This is an offline retrospective test.

Identified F1 uses identified activities only. Retention here is raw per-policy support coverage BEFORE coordination withholding, not final paired-advice coverage. Action diversity counts predicted classes. Top share includes forced respawn outputs in this table, while the development table separates voluntary concentration. None of these measures is policy win rate.

Macro evidence is much thinner than role evidence. Even coherent executed team activity does not establish deliberate coordination. Sparse targets, action reconstruction ambiguity, hidden positioning/vision and selection bias limit interpretation.

Controlled missing-health and missing-history tests are saved in the evaluation report when requested. Changes after removing evidence can be legitimate. They do not establish the frequency of real live missingness or uncertainty calibration.

Changes after cold-history omission: TEAM 73.4%; TOP 28.4%; JUNGLE 52.2%; MIDDLE 34.3%; BOTTOM 31.0%; UTILITY 66.5%. These include actions made infeasible by lost prerequisites; the JSON report separates that mechanism. Omission-test retention: TEAM 100.0%; TOP 100.0%; JUNGLE 100.0%; MIDDLE 100.0%; BOTTOM 100.0%; UTILITY 100.0%. Retention does not establish reliable advice.

Previous F1 is the former default evaluated on exactly the same frozen match IDs and dataset. This measures a product comparison, not an isolated algorithm ablation: the historical model used different training partitions and update counts. Test results were measured after candidate selection and did not change it. Baseline bundle: /home/amine/offlineRift/models/six-policy/20261109/release

The reserved macro F1 uses only 25 identified TEAM rows. This is thin evidence, not a reliable estimate of strategic utility. Read all positive and negative differences as exploratory. Activation, if performed, retains an experimental offline candidate selected by the frozen development rule; it does not establish superiority on every metric.

| Scope | Identified F1 | Previous F1 | Retention | Top share | Action types |
| --- | --- | --- | --- | --- | --- |
| TEAM | 0.201 | 0.225 | 100.0% | 45.2% | 8 |
| TOP | 0.224 | 0.156 | 100.0% | 50.7% | 10 |
| JUNGLE | 0.220 | 0.166 | 100.0% | 32.0% | 10 |
| MIDDLE | 0.189 | 0.172 | 100.0% | 47.6% | 14 |
| BOTTOM | 0.159 | 0.136 | 100.0% | 44.0% | 8 |
| UTILITY | 0.195 | 0.125 | 100.0% | 28.3% | 11 |

# History omission on reserved matches

Both fixed products are evaluated on the same measured states after controlled history omission. Switch rates condition on the original action remaining feasible; voluntary rates additionally remove forced respawn choices. The conditioning cohorts can differ because the two models choose different original actions.

Lower switching is not automatically better strategy: history can change the correct choice. Read this table together with identified agreement, action diversity and support coverage. The dropout experiment does not establish how often information will be absent in a real game.

The previous default used return critics; the new selection may use imitation. Gains therefore cannot be attributed solely to the consistency penalty. Same-study ordinary versus consistent candidates provide the controlled development comparison; no test retuning was performed.

| Scope | Previous switch | New switch | Previous voluntary | New voluntary | New voluntary rows |
| --- | --- | --- | --- | --- | --- |
| TEAM | 53.2% | 22.0% | 53.2% | 22.0% | 322 |
| TOP | 27.8% | 6.1% | 31.0% | 7.0% | 631 |
| JUNGLE | 42.7% | 14.3% | 48.0% | 16.9% | 443 |
| MIDDLE | 28.6% | 12.9% | 32.8% | 14.8% | 620 |
| BOTTOM | 28.4% | 3.3% | 31.2% | 3.7% | 598 |
| UTILITY | 9.1% | 19.4% | 9.9% | 24.1% | 315 |

# Engineering, cleanup and next tests

Development gates: all six passed. Coordination audit over 4720 measured role rows: 1534 matching actions, 2362 independent proposals, 408 conflicting role directions withheld and 416 forced-respawn pairs. Missing macro rows: 0; unsupported pairs: 0. These are instantaneous proposal counts; persistence and real-game benefit are not simulated.

The coach manifest binds six child manifests, aggregate metadata and the completed reload-parity record. Children must agree on data digest, feature ordering/scaling and match partitions, and declare independent fitting/serving scopes. Corruption or incompatible children fail loading.

Final validation recorded 103 passing tests, including 6 synthetic Chromium cases. Tests exercise real small optimizer/save/reload runs, scope isolation, holdout exclusion, terminal discounting, routing, corruption rejection, coordination and synthetic local UI behavior. These are engineering checks, not evidence of strategic benefit.

Superseded ADC-only utilities, dated reports and the old specialized synthesis renderer were checksummed into the local archive. Current documentation is concise; historical raw data, datasets, weights and failed/interrupted reports were not deleted. See research/docs/HISTORY.json for the archive manifest.

Later live acceptance should validate each selected role and the macro view, actual input availability, advice expiry, conflicts and opt-in recording. A predefined prospective comparison with terminal victory and measured adherence is still required before an increased-win-rate claim. No live test was needed or performed for this phase.

Reproduce offline: rift verify --model /home/amine/offlineRift/models/coherent-robust/20261109/model; rift-research evaluate --new-matches --stress --dataset /home/amine/offlineRift/data/projects/faker/next-actions/datasets/next-actions-6ea423fe0e6de2b8/transitions.npz --bundle /home/amine/offlineRift/models/coherent-robust/20261109/model --output NEW_DIRECTORY. Current source and contracts: README.md, AGENTS.md, docs/ARCHITECTURE.md, docs/MODELING.md and research/docs/PROTOCOL.md.
