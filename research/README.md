# Next-action policy research

The objective is a player/role- or team-conditioned policy that recommends executable
next behaviors toward winning. Formation classification is no longer the objective.
Previous source and research records are archived locally under `data/archive/`; they
are not evidence for this new action contract or new policies.

See the [matched CQL/IQL/BC comparison](docs/POLICY_COMPARISON_20261105.md) for twelve
completed runs, seed and missing-history diagnostics, action replays and the cleanup audit.

## Initial study

1. Reconstruct behavior from **real adjacent frames**, with future evidence restricted
   to action/reward/next-state targets. Keep all ten players and both teams grouped.
2. Audit which inferred options are supported, durations, role/action balance, unknown
   episodes and patch coverage. Inspect evidence at action boundaries, especially combat,
   base arrival, lane/jungle classification and objective approaches.
3. Fit CQL with terminal team win/loss rewards and elapsed-time discount. Fit imitation
   on the identical match split as a baseline. Do not select solely on behavior agreement.
4. Record role-specific concentration, unsupported-state abstentions, value stability,
   action coverage and sensitivity to conservative/support settings. No Q-to-win conversion.
5. Test connection, subject/role selection and availability masks in a real game. Capture
   opt-in sessions only through a separately designed recording feature; the dashboard
   currently records nothing.
6. Design a later prospective outcome study with predefined eligibility, adherence,
   comparison, sample size and stopping rules before claiming a win-rate improvement.

## Open modeling questions

- Do inferred choices correspond to meaningful interventions, rather than incidental
  movement, forced reactions or overlapping activities in a minute-long interval?
- How should failed/ambushed combat be represented without conditioning labels on success?
- Does shared team reward identify useful individual choices under hidden positioning,
  teammate policies and selection bias? Current data do not establish that assumption.
- When can finer temporal events justify shorter option boundaries? How should a current
  activity terminate? Do not fabricate telemetry between frames.
- Does causal history improve actionable value estimates over the same current features?
- Can audited gank attempts, siege/defense, Grubs and wave decisions be added with reliable
  evidence and compatible live observations? See [the action catalog](../docs/ACTIONS.md).
- How should uncertainty/abstention thresholds be calibrated on a separate split?
- Which off-policy estimators are defensible under the observed support and confounding?
  An offline value score is not automatically a trustworthy counterfactual evaluation.

## Commands

```bash
rift-research dataset --dataset /path/to/transitions.npz
rift-research evaluate --dataset /path/to/transitions.npz \
  --bundle /path/to/policy --output /path/to/new/evaluation
rift-research study --config workflow.json --output data/research/next-actions-study \
  --steps 10000 --seeds 42 43
rift-research sensitivity --study data/research/next-actions-study \
  --output data/research/next-actions-study-sensitivity
```

Evaluation reproduces serving predictions on the saved match split and reports a
match-cluster bootstrap for behavior diagnostics. `policy_winrate` and
`policy_improvement` remain unset. Initial retrospective development uses already
consulted cached matches; it is not an untouched confirmatory test.

The maintained study runner calls the existing workflow builder, trainer and frozen
evaluator. It compares BC, CQL alpha 0.03/0.1/0.3 and discrete IQL expectile 0.7/0.9,
each with two optimizer seeds and one fixed match split. Every completed run is reloaded
through serving, with behavior metrics, per-role concentration, score quantiles and
match-cluster bootstrap intervals persisted atomically. Failed/interrupted study reports
and partial runs remain stored. It never updates the latest model reference. The
[IQL paper](https://arxiv.org/abs/2110.06169) and
[author implementation](https://github.com/ikostrikov/implicit_q_learning) motivate the
recorded-action expectile backup and advantage-weighted actor. Discrete masks and
elapsed-time targets adapt those ideas to this observation contract.

Sensitivity analysis requires a completed study and checks the exact dataset digest. It
reports agreement across optimizer seeds, recorded-action critic ranges and prediction
changes when health or cold-start history is masked. Masks represent missing evidence;
they do not invent observations or alter target outcomes. These controlled omissions
do not measure the actual live input distribution or policy usefulness.
