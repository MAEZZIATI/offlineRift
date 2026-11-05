# Availability-conditioned development study protocol

This exploratory protocol is specified before running the new conditional analysis.
The cached matches have already been consulted. No result is a confirmatory policy
quality or win-rate test, and the analysis cannot publish serving thresholds.

## Conditional coverage

Keep the existing 128/128 development-match calibration/evaluation partition. Fit
confidence quantiles by role/team and exact input availability signature. Use four
controlled scenarios: observed, missing health, cold-start history, and both omissions.
Signatures use the health/history `_known` masks and objective `_enabled` flags. An
objective flag indicates modeled usability, not proof of actual objective availability;
disabled and unobserved objectives can share a signature. These scenarios do not model
their real-world frequency. Deduplicate the same source row/signature in calibration.

A cell requires at least 200 distinct source rows and 20 distinct calibration matches;
otherwise abstain without borrowing a threshold from another cell. At target 90%, a
development coverage screen passes only if every evaluation cell with at least 200 rows
and 20 matches retains 85–95% of rows and has sufficient calibration support. Whole
confidence ties remain together; deterministic respawn predictions can fail this screen.
Report unsupported and evaluation-sparse cells separately, without hiding them in a mean.

For every role/action with at least 100 evaluation rows and 20 matches, report how many
rows remain after abstention. Require at least 50% retention for a diversity screen.
This is retention of states labeled with the action, not correctness, action recall or
strategic utility. Report prediction distributions and behavior recall as well. Count
recorded targets made infeasible by omission, by action and role. This exposes the
selection mechanism behind rejected current-state training augmentations.

Passing these screens is necessary for further review, never sufficient for deployment.
Do not adjust these criteria after observing the results without declaring a new study.

## Combat evidence review

Review all 75 existing stratified intervals with their kill/death/assist timestamps,
CS endpoints and coarse movement endpoints. Record factual ambiguity and concrete
examples. Do not label initiation, ambush, intent, action duration or a preferred
intervention from endpoints. Do not replace targets with the label obtained by removing
combat events, especially when death can explain a base arrival.

## Prospective availability capture

Provide an explicit `rift live --capture-availability NEW_DIRECTORY` option. Default live
use and the dashboard remain unrecorded. Persist only aggregate feature availability,
role/scope, selection source, coarse game-time bucket, and local observation-session
counts. Do not persist player identities, raw payloads, items, coordinates, state values,
recommendations or outcomes. Treat polls as correlated observations and reconnects as
observation sessions, not verified unique matches. Refuse an existing output directory;
update progress atomically and preserve interrupted reports. Validate with local fixtures.

No active-game capture is assumed to be available during development. Before estimating
real missingness rates, collect opt-in prospective sessions and define eligibility,
sampling and match/session grouping. Aggregate capture cannot evaluate policy outcomes.

## Later outcome study

Keep the policy and abstention rules frozen before enrollment. Define eligibility and
user adherence, randomize advice availability at the match/team level where feasible,
and use terminal team victory as the primary outcome with match-level analysis. Prevent
both teams receiving conflicting assignments; keep repeat-player dependence explicit.
Specify effect size, sample-size/power calculation, exclusions, stopping rules and analysis
before recruitment. An adequately powered study cannot be designed from the current
offline behavior diagnostics alone. No intervention or outcome study is run here.

## Research hypothesis: decision regret under evidence removal

Instead of choosing the highest value from one critic, choose the action minimizing
the largest supported-action value gap across optimizer seeds and causal evidence
omissions. For view/model `v`, let `A_v` be its feasible, behavior-supported options:

```text
regret_v(a) = max_{b in A_v} Q_v(b) - Q_v(a)
action = argmin_{a in intersection(A_v)} max_v regret_v(a)
```

An empty intersection abstains. Compare a single observed view, seed-only, omission-only
and combined variants. Omit health and/or recent numerical history, retaining objective
feasibility prerequisites; already-missing objective evidence is never restored. Do not
require invariance to removal of prerequisites for performing an action. Compare fixed
regret tolerances 0, 0.05, 0.1 and 0.25 return units, before looking at outcome labels.
Also report the unfiltered minimax action, original CQL, predicted action diversity,
per-role support and behavior diagnostics. These tolerances are exploratory, not deployable.

The hypothesis is that relative decision regret separates unstable action rankings from
arbitrary common offsets in critic values, while support intersection prevents filling
missing evidence with imagined telemetry. It may instead select repetitive actions or
abstain on useful rare options; report those failures. Two optimizer seeds do not provide
a statistical uncertainty bound, and removed evidence can legitimately change the best
action. This is a proposed decision rule for this observation contract, not a proof of
causal robustness or a claim of algorithmic priority. Ensemble pessimism and regret have
substantial prior literature, including [MSG](https://arxiv.org/abs/2205.13703) and
[regret minimization under partial observation](https://proceedings.mlr.press/v80/jin18c.html).
The contribution to test is the combination of executable-option feasibility, exact
evidence removal, relative regret and explicit abstention under coarse observational data.

### Matched-coverage follow-up

After inspecting the first decision-regret results, add a confidence-only comparator
before running the next report version. Use the same 128/128 calibration/evaluation match
halves. For each group, measure the regret rule's retained fraction on calibration states
and fit the first critic's confidence quantile to that fraction. Evaluate both filters
on the other half; fit no thresholds or tolerances to behavior/outcome labels. Report
achieved coverage because finite-sample variation and confidence ties prevent exact
matching. This is an adaptive exploratory follow-up, not a fresh confirmatory comparison.
