# Evidence-conditioned decision research — 2026-11-05

This study tests a proposed decision rule beyond the preceding CQL/IQL replication:
**minimize worst-case supported decision regret across optimizer seeds and controlled
losses of causal evidence**. It also completes availability-conditioned calibration,
reviews the existing combat samples, and implements opt-in aggregate live availability
capture. No policy or threshold was promoted. The result is a falsifiable research
candidate with measured failure modes, not an established field advance or winning policy.

## Hypothesis and prior work

A single high Q can reflect an unstable ranking or a common offset in critic estimates.
For model/view `v`, retain its existing feasible, role-supported, behavior-supported
action set `A_v`. Define:

```text
regret_v(a) = max_{b in A_v} Q_v(b) - Q_v(a)
choose argmin_{a in intersection(A_v)} max_v regret_v(a)
```

Empty intersections abstain. A fixed tolerance on the selected maximum regret supplies
an additional abstention rule. Independent additive offsets in each view's Q values
cancel exactly. Multiplicative scale differences do not cancel; results should be compared
within an architecture/variant. Tests verify offset invariance, a compromise action that
is neither critic's winner, unsupported-action exclusion and empty-intersection abstention.

The candidate combines executable-option feasibility, causal evidence removal, relative
value gaps and abstention. Ensemble pessimism and regret already have substantial prior
work, including [MSG](https://arxiv.org/abs/2205.13703),
[pessimistic offline RL](https://proceedings.mlr.press/v139/jin21e.html), and
[regret minimization under partial observation](https://proceedings.mlr.press/v80/jin18c.html).
No algorithmic priority or theorem follows from this combination. A broader literature
review and independent benchmarks would be needed before a novelty claim.

## Protocol

The [protocol](NEXT_STUDY_PROTOCOL.md) specifies the rule, omissions, tolerances and
calibration screens before these analyses. No new training run was needed: all four CQL
robustness variants supply two optimizer-seed critics each, evaluated on the same 256
development matches and 71,472 transitions. These are previously consulted matches.

The four views are observed input, missing health, missing recent numerical history, and
both omissions. History here means rates/deaths/structure statistics. Objective
prerequisites are retained: making every objective unknown in every stress set would
eliminate objective actions by construction. Already-missing evidence is never restored.
There is no invented movement, telemetry, simulated rollout or future-state input.

Ablations use the single observed critic, both seeds on observed input, four omission
views of the first critic, and all eight seed/view combinations. Fixed tolerances are
0, 0.05, 0.1 and 0.25 learned return units. No action/outcome labels select them.
The single-view method exactly reproduces its serving CQL baseline.

## Observed decision-regret results

The table uses the combined seed-and-omission method at tolerance 0.05. Retained behavior
agreement compares selected actions to reconstructed recorded actions, not good actions.

| Critic variant | Retained decisions | Distinct predicted actions | Retained behavior agreement | Strict zero-regret coverage |
|---|---:|---:|---:|---:|
| Ordinary CQL | 75.34% | 12/15 | 57.08% | 36.10% |
| Joint history masking | 80.33% | 15/15 | 56.45% | 48.94% |
| Bounded critic | 88.84% | 15/15 | 50.85% | 47.75% |
| Bounded + joint masking | 92.73% | 15/15 | 51.14% | 61.89% |

All support intersections were nonempty on these observed development rows. This is not
a support guarantee for live inputs. At zero tolerance, ordinary CQL never recommends
any objective preparation option; bounded + joint predicts all 15 options but rarely
selects some rotations. All 5,290 respawn decisions survive every filter, contributing
easy stable observations to overall coverage and agreement.

For bounded + joint at tolerance 0.05, jungle coverage is 92.7% but its largest prediction
fraction is 77.4%; support coverage is 92.2% with largest fraction 82.8%. Both remain
fight-heavy. Jungle concentration exceeds the first critic's ordinary 68.3% baseline.
Strict consensus further concentrates these roles. Robustness can therefore amplify a
shared repetitive bias. A common decision across critics is not evidence of correctness.

Unfiltered minimax selection changes 15.5–23.9% of actions for the two bounded variants,
while reducing behavior agreement relative to their first critics. This is neither proof
of worse strategy nor evidence of improvement. The diagnostic discourages promoting the
new rule based on stability or selected-row agreement alone. There is no causal evaluation
or realized-regret estimate.

An adaptive follow-up fits confidence-only thresholds to the regret rule's per-group
calibration coverage, then evaluates both on the other 128 development matches. At
tolerance 0.05, the results are:

| Variant | Regret coverage / agreement | Confidence coverage / agreement | Predicted actions, regret / confidence |
|---|---:|---:|---:|
| Ordinary CQL | 75.05% / 56.75% | 75.69% / 57.15% | 12 / 11 |
| Joint history masking | 80.20% / 55.85% | 80.33% / 55.70% | 15 / 13 |
| Bounded critic | 89.09% / 50.69% | 88.20% / 54.03% | 15 / 13 |
| Bounded + joint masking | 92.98% / 51.09% | 92.12% / 52.67% | 15 / 15 |

Here agreement is among retained rows. Coverage matching is approximate because of
finite-sample variation and ties. The regret filter sometimes retains more action types,
but does not consistently improve behavior agreement over the simpler filter. This
negative comparison is retained; the exploratory candidate has not earned promotion.
Neither filter's behavior agreement estimates decision quality.

## Conditional calibration and objective-action bias

Exact availability signatures combine health/history masks and objective enabled flags.
The latter conflate game rules with modeled usability and are not a complete missingness
model. Calibration uses the existing 128 development matches; evaluation uses the other
128. Each calibration row/signature counts once despite repeated scenarios. A cell
requires 200 source rows and 20 matches; unsupported cells abstain. The full four-scenario
grid also includes omitted objective evidence for this separate stress analysis.

At 90% target coverage, all eight policies pass the predefined 85–95% coverage screen for
supported evaluation cells in the four controlled scenarios. Overall retained coverage
is 89.29–90.44%. All sufficiently supported role/action state-retention checks pass the
50% screen. This improves the earlier threshold-transfer diagnostic, but the calibration
set explicitly includes those omission patterns. It does not demonstrate transfer to
unseen missingness, and many rare role/objective pairs lack 100 rows or 20 matches for
the retention screen. Sparse cells remain abstentions, as verified with small fixtures.

Retaining a state labeled with an objective action does not mean recommending that action:
after full cold-start omission, every recorded objective target is infeasible, so its
behavior recall is necessarily zero. The report separately counts these targets by role
and action. For example, 273 jungle dragon-preparation rows occur in the evaluation half;
all become infeasible, though 95.2% of those states retain some other recommendation in
the first ordinary CQL run. This illustrates why coverage/retention screens alone cannot
qualify a policy for deployment and why current-state augmentation rejects those targets.

## Combat interval desk review

All 75 stored stratified intervals were reviewed at the available resolution. Event
membership/timing checks confirm that each contains recorded participation; 50 also
gain lane CS, 14 gain jungle CS, and 14 first record combat in the final quarter. These
counts describe the deliberately balanced review sample, not prevalence in the corpus.

Three intervals become RESET when combat events are removed. Each includes a death and
ends near base, so the alternative label cannot establish a voluntary return. Examples:

| Match, actor, measured interval | Available evidence | Interpretation limit |
|---|---|---|
| `KR_8443743098`, 9, 1260.364–1320.379s | Death at 1276.667s, lane CS 242→245, endpoint near red base | Base arrival cannot distinguish voluntary reset from death-related return |
| `KR_8434149228`, 3, 300.175–360.189s | Kill at 338.572s, death at 341.383s, endpoint near blue base | Combat-event deletion creates an unreliable RESET alternative |
| `KR_8445027983`, 9, 1680.453–1740.486s | Jungle CS 12→16; kills at 1740.419s, almost exactly the interval end | One fighting label hides overlapping activity; initiation and duration remain unknown |

No forced/initiated distinction or replacement action schema is justified. A promising
future direction is explicitly representing ambiguity in executed-option evidence, with
new dataset contracts and transition semantics studied before changing training targets.

## Prospective availability capture

`rift live --capture-availability NEW_DIRECTORY` now opts into atomic aggregate counts.
It records availability patterns, role/scope and role-selection source, coarse time
buckets, previous-observation/event-history availability and observation-session counts.
It omits identities, raw payloads, state values, actions and outcomes. Default live use
and the dashboard remain unrecorded. Existing output directories are rejected; interrupted
reports remain available. Session counts are not verified unique matches and polls are
correlated. Fixture tests cover missing history, duplicate polls, clock reset, privacy,
disabled defaults and interruption. No active-game capture was exercised.

## Next experiments and retained outputs

The next useful experiment repeats the matched-coverage comparison with more optimizer
seeds, role/action diagnostics, independent tasks, and prospective availability patterns.
Test whether the rule's concentration penalty is
intrinsic or reflects shared critic/target bias. Add independent tasks with known outcomes
before treating this as a general offline-RL contribution. Prospective policy usefulness
still requires a separately powered and predefined outcome study.

Local ignored artifacts in `data/research/`:

- `next-actions-20261105-conditional-coverage-v2/report.json`: eight policies, exact
  signatures, thresholds, per-cell screens and per-role/action retention/infeasibility.
- `next-actions-20261105-decision-regret-v3/report.json`: four variants, all four ablations,
  regret quantiles, support intersections, fixed-tolerance curves, per-role diagnostics
  and the separate-half confidence-only comparator.
- Earlier `v1`/`v2` reports are retained; the matched-coverage follow-up was specified
  after the initial results and before running `v3`.

Reports identify the source study, dataset, source/protocol digests and bundle manifests.
No capture of actual players or authenticated collection was performed during this work.
