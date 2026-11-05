# Initial next-action study — 2026-11-05

This is the first retrospective study of the new objective. No formation-classification
metrics are reused as evidence for it. No new authenticated collection or live-game
outcome study was performed. The existing verified cache supplies 57 source accounts
and all player/team perspectives; the selected matches remain a biased elite-KR sample.

## Data and splits

The builder considered 1,289 cached matches over seven pinned patches (16.16–16.22).
It accepted **1,280 matches, 15,360 complete actor episodes and 365,040 transitions**.
Nine short matches had no complete eligible decision episode after the opening cutoff.
All five roles occur in each accepted game. There were no missing-role actor exclusions.

These are real measured intervals, with median **60.017s**
and range **0.338–64.189s**.
The short tail includes terminal fragments, not invented telemetry. Rewards are terminal
team win +1 / loss -1. All actors from a match share the split: **1,024 training matches /
293,568 transitions** and **256 development matches / 71,472 transitions** (seed 42).
These cached matches were consulted in earlier work; this is not an untouched final test.

Executed options are inferred from subsequent movement, CS changes, area maintenance,
base arrival and combat participation. This is not a command log or an intent annotation.
Combat includes deaths as well as kills/assists. Minute windows can mix multiple actions,
and forced/ambushed combat remains confounded. Team options are majority reconstructions,
not an observed centralized controller's commands.

## Training and serving checks

Both models used the same 64 public/history features, actor/role conditioning, 128×128
shared MLP, 15 action heads, 10,000 optimizer updates, batch 256 and seed 42. The CQL
variant used alpha 0.1, gamma 0.99 per minute, elapsed-time Bellman targets, a target
network refreshed every 200 steps and a behavior-support auxiliary head. The BC run
is an independently optimized imitation baseline, not a claim of action value.

| Diagnostic | CQL | Imitation baseline |
|---|---:|---:|
| Behavior agreement | 51.64% | 48.82% |
| Behavior macro F1 | 23.13% | 28.72% |
| Predicted action coverage | 15/15 | 14/15 |
| Unsupported-state abstentions | 0 | 0 |

The global most-frequent-training-action baseline has **44.08%**
agreement. These numbers diagnose reconstructed behavior and support; they do not rank
policies by winning. BC's higher macro F1 does not establish a better decision policy.
CQL is the experimental default because next-action return optimization is the objective,
not because these diagnostics prove superiority. No policy win rate is reported.

## Concentration and evidence gaps

| Subject / role | CQL predicted coverage | Most selected action | Fraction |
|---|---:|---|---:|
| TOP | 14/15 | Participate in a nearby fight | 58.1% |
| JUNGLE | 7/15 | Participate in a nearby fight | 85.8% |
| MIDDLE | 10/15 | Participate in a nearby fight | 42.5% |
| BOTTOM | 8/15 | Participate in a nearby fight | 42.4% |
| UTILITY | 11/15 | Participate in a nearby fight | 78.0% |
| TEAM | 2/15 | Continue current activity | 58.0% |

The jungler concentration is still a substantial problem, and the team perspective
predicts only two options despite wider nominal support. Some objective/reset choices
are selected only a handful of times. Overall 15-action coverage must not hide that.
The UI warns about role-specific dominant predictions. Do not interpret repeated fight
advice as evidence that fighting is safe, reachable or favorable.

The next research priorities are evidence quality and temporal/action overlap (especially
combat), prospective live-state distribution checks, role-specific support/concentration,
conservative/support sensitivity and a later outcome study. Finer wave, gank-intent,
vision, camp-route, Grubs and micro actions remain explicitly unsupported; see the
[current action catalog](../../docs/ACTIONS.md).

## Artifacts

- Dataset: `data/projects/faker/next-actions/datasets/next-actions-2d23a67940ee221b/`.
- Selected CQL bundle: `/home/amine/offlineRift/models/next-actions/faker/20261105T142329Z-d084b9d4`.
- BC baseline: `models/next-actions/faker/bc-20261105-v1/`.
- Dataset audit and frozen evaluations: `data/research/next-actions-20261105/`.
- Completed metadata, model hashes and interrupted/pilot reports remain local and ignored.

The frozen evaluator reproduced both bundles' saved serving predictions. It uses
match-cluster bootstrap for behavior diagnostics and leaves `policy_winrate` and
`policy_improvement` unset. The default model pointer is in the new next-action namespace;
the old formation pointer and historical artifacts remain stored and are not loaded.

Validation: 41 tests passed, including real CQL/BC optimization and save/reload,
causal/live feature parity, boundary/reward checks, mocked Riot/catalog behavior, local
HTTP handling and desktop/mobile browser interactions. Ruff lint, formatting, CLI help,
local documentation links and git diff checks passed. Production packaging excludes
research and old formation modules. An installed wheel served the new policy with
research, training and d3rlpy imports blocked. A single real Windows-bridge local API
check returned offline; active-game integration and authenticated Riot collection were
not verified.
