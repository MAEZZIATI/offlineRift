# Combat evidence, history robustness and abstention — 2026-11-05

The follow-up to the [algorithm comparison](POLICY_COMPARISON_20261105.md) finds
substantially better missing-history stability with joint observation masking.
It also confirms ambiguous combat targets and threshold transfer failures. No model
or abstention threshold was promoted. Strategic usefulness and improved win rate remain
unestablished.

## Protocol and implementation

The existing completed robustness study contains eight runs: ordinary CQL alpha 0.1,
joint history masking, bounded critics, and both changes together, each with optimizer
seeds 42 and 43. Each run used 10,000 updates and the same 1,024 training / 256 development
match split and 365,040 transitions as the algorithm comparison. Baseline metrics reproduce
the earlier CQL runs. The frozen source snapshots and their digests were verified.

`joint_history_dropout=0.2` jointly omits recent rates, deaths, structure losses and
objective estimates using the same named fields as the cold-start stress test. Ordinary
independent optional-field dropout remains active. Feasibility is recomputed after joint
omission. A current-state omission that makes the recorded objective action infeasible
is rejected without changing its target. There were 18,257–18,336 such rejections among
512,100–512,785 proposed current-state joint omissions (about 3.6%). This action-dependent
augmentation selection limits claims about robustness on objective states. Next-state
omissions retain no recorded-action constraint. Stored transitions and outcomes remain
unchanged; masks are applied to sampled training inputs.

`bounded_values=True` applies tanh to the CQL value head, with an explicit
`bounded_shared_v1` bundle architecture. Behavior logits remain unbounded. Bounding is
an experimental change to CQL optimization and its pessimism, not calibration of returns.
Existing bundles retain their original architecture and values. Both settings default
to disabled and use the maintained trainer; the study never publishes a latest model.

## Observed robustness diagnostics

Ranges span optimizer seeds. Macro F1 measures agreement with reconstructed behavior.
Cold-start change compares recommendations on the same rows before and after history
omission; it does not measure actual live-distribution performance.

| Variant | Behavior macro F1 | Cold-start prediction change | Seed agreement | Jungle dominant-action fraction |
|---|---:|---:|---:|---:|
| Ordinary CQL | 22.72–26.36% | 58.51–72.35% | 74.63% | 82.77–85.38% |
| Joint history masking | 24.61–26.42% | 13.93–17.16% | 74.66% | 72.35–83.60% |
| Bounded critic | 23.32–25.22% | 42.46–51.85% | 76.60% | 66.79–78.16% |
| Bounded + joint masking | 25.22–25.72% | 12.41–12.67% | 82.43% | 68.30–73.13% |

Every experimental variant predicted all 15 options somewhere in development; ordinary
CQL predicted 14–15. All eight had zero abstentions on unmodified development inputs.
Recorded-action values outside [-1,1] were 47.79–51.27% for ordinary CQL and 45.95–52.10%
for joint masking. Bounded variants had zero excursions by construction. That mechanical
property is not evidence of better value accuracy. Fight concentration and disagreement
across seeds remain substantial even for the combined variant.

## Combat target audit

The cached-input audit verified raw payload digests against dataset provenance and
reconstructed the exact measured intervals. Of 304,200 player transitions, 135,146
(44.4%) were labeled fighting. Within those fight intervals:

- 34,673 (25.7%) contained death participation without a credited kill or assist.
- 93,844 (69.4%) also gained lane CS; 23,430 (17.3%) also gained jungle CS.
- 73,585 (54.4%) changed coarse endpoint area.
- 27,264 (20.2%) had their first recorded combat participation in the final quarter.

These overlap measures are not mutually exclusive. Death-only events do not identify
forced combat, and CS or endpoint movement cannot reveal precise activity timing.
Removing combat-event priority for diagnosis exposes farming, rotations, resets and
objective options underneath. Those alternatives are not validated replacement targets;
in particular, removing a death event can make a base endpoint look like a voluntary reset.
The maintained labels were not changed. A stratified reservoir sample retains five real
intervals per role and combat-evidence category for manual review.

## Separate coverage calibration and evaluation

The analysis partitions the 256 development matches deterministically into 128 calibration
matches (36,096 transitions) and 128 evaluation matches (35,376 transitions). Every actor
and both teams remain together. Both halves are previously consulted development data,
not an untouched confirmatory test. Model training matches are excluded and split/corpus
compatibility is validated for each bundle.

For each role/team group, confidence is the feasible-action-normalized behavior-head mass
of the serving-selected action. CQL still selects by supported Q; confidence is only an
abstention filter. Thresholds use calibration confidence ranks at retained-coverage targets
100%, 90%, 75% and 50%, without behavior or outcome labels. Groups absent in calibration
abstain. Ties are retained together and existing support abstentions remain abstentions,
so target and achieved coverage can differ. Reports include action support, majority
baseline, abstention, concentration and per-group evaluation diagnostics.

At the 90% target, ordinary observed-state thresholds retained 89.44–90.12% of evaluation
rows. Applying those **same thresholds** after cold-start omission retained only
54.22–62.13%. With joint masking, observed-state coverage was 89.58–89.75%, while the
transferred cold-start coverage was 95.98–96.24%. The combined variant transferred to
94.78–95.15%. Augmentation therefore reduces prediction instability but does not make
coverage thresholds invariant to missing evidence.

Separately fitting thresholds on cold-start calibration observations restores roughly
90% coverage in that controlled scenario. At a 50% target, these cold-start-specific
filters retain only seven distinct actions for the joint and combined variants. Higher
retained behavior agreement can coexist with lost action diversity. This is coverage
calibration, not correctness calibration, a finite-sample risk guarantee, calibrated win
probability or demonstrated policy improvement. No threshold is installed in serving.

## Next supported steps

1. Review the stratified real combat intervals before defining a new action schema.
   Keep initiation/forced-response distinctions unresolved unless telemetry supports them.
2. Test calibration conditional on observed missing-evidence patterns, including per-role
   rare-action retention and the bias from rejected objective-action augmentations.
   Predefine acceptance criteria before using thresholds for serving.
3. Measure prospective live input availability using a separately designed opt-in capture
   feature. Controlled omissions do not establish the frequency or combinations of real
   missing inputs. No authenticated collection or active-game integration was exercised.
4. Evaluate decision usefulness with a predefined prospective protocol. Neither bounded
   values, greater stability nor behavior agreement is a policy win-rate estimate.

## Retained local artifacts

All outputs are ignored local research artifacts under `data/research/`:

- `next-actions-20261105-combat-audit-v1/`: counts and real review samples.
- `next-actions-20261105-robustness-v1/`: eight bundles, evaluations and frozen sources.
- `next-actions-20261105-robustness-sensitivity-v1/`: seed, range and omission diagnostics.
- `next-actions-20261105-robustness-coverage-v1/`: fitted thresholds, disjoint match IDs,
  coverage curves and observed-to-cold-start threshold transfer.

Validation includes real small optimizer/save/reload checks for the bounded architecture,
augmentation feasibility and target preservation, both study suites, disjoint calibration
and evaluation, fixed-threshold transfer, invalid split rejection, and raw provenance
failure retention. Optional browser tests require `RIFT_TEST_BROWSER`.
