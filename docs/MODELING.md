# Modeling contract

## Objective and scope

Learn supported next-action policies toward terminal team victory: one TEAM policy
and five role specialists. All perspectives from one match remain in one partition.
Successful training or behavior agreement does not establish useful or optimal strategy.

## Observations and actions

`decision-state-v1` contains 64 causal public-observation values: time, perspective,
role masks, public stats/equipment, available health/dead state, recent measured rates,
public event history and pinned objective estimates. Every optional value has a mask.
Final economic statistics and future observations are never input features. Historical
roles are retrospective. Positions, waves, vision, cast history and camp safety are not
invented for serving. Live testing is deferred; the same contract will be checked later.

`next-action-v1` contains fifteen coarse activities with feasibility masks and human
checks. See ACTIONS.md. Team views cannot select individual farming or respawn options.
The taxonomy is an observed activity vocabulary, not a set of verified intentions.

## Versioned evidence and complete episodes

New builds use `option-transitions-v3` / `executed-evidence-v3`. V1/v2 data remain
readable under their own contracts. V3 preserves representative descriptive labels,
candidate activity sets, identification flags and death/credit evidence. Overlapping
combat/farming is unresolved; death-only participation never identifies a chosen fight.
Removing a death event cannot establish a voluntary base return.

Macro fitting requires three identified matching executed macro activities. No coherent
team direction is unresolved, not a positive "continue" target; combat is not treated
as an identified voluntary team strategy. This remains heuristic execution evidence,
not proof of coordinated intent or a known best plan.

Each actor episode retains its exact states, next states, elapsed durations, terminal
boundary and terminal team result. Unreconstructable episodes are excluded and counted.
Ambiguous identified-flag-false rows remain stored; they are never removed and stitched.
Role-specialist fitting uses only that role's identified rows, with shared match splits.
Exclusions and retained terminal coverage are reported separately for each scope.

## Rewards and explicit critic targets

Rewards are zero except terminal +1 for a team victory and -1 for a loss. No shaping
or final-match economic input is introduced. Discounting uses measured elapsed seconds:

`G_t = r_t + (1-terminal_t) * gamma ** (duration_t/60) * G_next`

Bellman-target CQL uses online supported-action selection and a periodically copied
target critic. Necessary feasibility, per-scope training support and behavior mass
constrain next choices. Missing future support uses -1; terminal rows never bootstrap.
The loss combines Huber target error, a conservative feasible-action penalty and a
behavior-support auxiliary loss. Critics can be bounded with tanh; this controls range,
not accuracy or calibration.

The separate `episode_return` variant regresses observed complete-episode discounted
returns on identified activity rows, with the same conservative penalty and auxiliary
loss. It uses terminal results as targets, including when an identified macro interval
is followed by unresolved activity. It does not create intermediate commands, join
episodes or treat the observed trajectory return as a causal alternative-action value.
Call it conservative observed-return regression, not a replication of standard CQL.

The macro comparison uses observed-return variants because its identified intervals
rarely coincide with terminal frames. Role comparisons include Bellman and observed-
return variants. Behavior cloning is an imitation baseline; it is not selected as an
RL winner automatically. Discrete IQL remains an optional diagnostic implementation.

## Composition and evaluation

The six child policies are fitted independently with one trainer, common encoder and
fixed match partitions. The serving router verifies compatible schema, data digest and
scope, then returns the macro proposal plus role proposal. Symbolic alignment/conflict
reporting does not claim learned conditioning on a hypothetical team plan.

Use development data for per-scope candidate selection. Evaluate the fixed selection
once on reserved unseen match IDs. Report identified and all-row descriptive agreement,
training-derived majority baseline, action support, abstention, concentration, collapsed
predictions, score ranges and missing-input sensitivity. None is policy win rate.
Repeated players and shared team outcomes remain dependent and confounded.

Serving ranks supported feasible learned scores without blending actor and critic values.
Advice persistence requires continued eligibility and clears on death or lost support.
Optional following-evidence review reports supported, contradicted, consistent or
unverifiable measured activity. Attention, intent and causal adherence remain unknown.

Prospective active-game acceptance and a predefined outcome comparison remain later
work. Do not claim precise mechanics, safe fights, calibrated victory probabilities,
algorithmic novelty or improved winning strategy from these offline checks.
