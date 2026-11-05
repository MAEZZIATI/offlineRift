# Matched policy comparison — 2026-11-05

Twelve full training runs completed: BC, three CQL penalties and two discrete IQL
expectiles, each with optimizer seeds 42 and 43. No new policy was promoted. The
experiment improves our modeling and stability evidence; it does **not** establish
improved decision utility or win rate. The current experimental CQL policy remains active.

## Available data and protocol

The inventory found 1,289 raw matches. Older frozen-data folders contain 1,732 copies
of existing match IDs and no additional match IDs. The maintained builder accepted
1,280 matches, 15,360 complete player/team episodes and 365,040 measured transitions;
nine short games had no eligible decision episode. All 57 configured source identities
are retained. No authenticated collection or new synthetic training data was used.

All eligible games participated across the same grouped split: 1,024 training matches
with 293,568 transitions and 256 development matches with 71,472 transitions. The split
seed is fixed at 42 independently of the optimizer seed. Every model trained for 10,000
updates, batch 256, using terminal team outcomes and elapsed-time discount gamma 0.99.
The separate split RNG changes minibatch sequences from the initial study, so comparisons
should use the matched runs here. These are previously consulted development games.

CQL alpha values were 0.03/0.1/0.3. IQL expectiles were 0.7/0.9, with temperature 3,
weight cap 100 and Polyak coefficient 0.005. IQL uses independent actor/critic encoders,
twin critics and a training-only scalar value function. Its categorical actor extraction
is not class balanced; the CQL support auxiliary and BC retain their existing tempered
weights. This difference is explicit and precludes attributing every diagnostic change
to the backup alone. See [the modeling contract](../../docs/MODELING.md).

The approach follows the [IQL paper](https://arxiv.org/abs/2110.06169) and
[author implementation](https://github.com/ikostrikov/implicit_q_learning), alongside
[Conservative Q-Learning](https://arxiv.org/abs/2006.04779). Those papers' benchmark
results do not transfer into claims about this observational game corpus.

## Observed diagnostics

Ranges span the two optimizer seeds. Macro F1 measures agreement with inferred recorded
behaviors. Seed agreement compares actions on identical development rows. Cold-start
change compares predictions after jointly masking rates, recent death/structure history
and objective availability; it is a controlled omission, not observed live performance.

| Variant | Behavior macro F1 | Predicted coverage | Seed agreement | Cold-start prediction change |
|---|---:|---:|---:|---:|
| bc | 28.93–29.30% | 14–14/15 | 92.8% | 24.7–34.2% |
| cql-alpha-0.03 | 19.03–22.69% | 15–15/15 | 47.6% | 61.8–79.2% |
| cql-alpha-0.1 | 22.72–26.36% | 14–15/15 | 74.6% | 58.5–72.4% |
| cql-alpha-0.3 | 24.97–26.15% | 11–12/15 | 90.5% | 27.9–42.7% |
| iql-expectile-0.7 | 22.47–24.03% | 7–8/15 | 93.8% | 17.1–19.9% |
| iql-expectile-0.9 | 22.29–24.15% | 7–8/15 | 93.0% | 17.5–21.4% |

All twelve models had zero abstentions on the unmodified development states. That does
not establish reliable support in live games. Each serving reload reproduced the saved
training predictions, and evaluation used 500 match-cluster bootstrap samples per run.
All team models selected only two or three distinct options.

IQL's higher behavior agreement (56.25–56.52%) conceals poor diversity: about 91% of its
jungle and support recommendations are fighting. CQL alpha 0.1 selects fighting on
82.8–85.4% of jungle observations. BC is less concentrated in that role (47.8–54.1%
for its dominant option) but is an imitation baseline, not evidence of return optimization.
Higher agreement or lower concentration alone is not grounds to select a winning policy.

CQL alpha 0.1 changes 58.5–72.4% of predictions in the cold-start omission test.
IQL changes 17.1–21.4%, but stability largely coexists with repetitive fight predictions.
Removing health alone changes roughly 1.8–11.3% across the compared models. These
stress results justify prospective live-distribution checks and joint missing-history
augmentation studies before assuming the existing independent field dropout is adequate.

Recorded-action CQL estimates outside the theoretical discounted terminal-return range
[-1,1] occur in 12.5–78.2% of development rows across penalties; for IQL's first critic,
the fraction is 8.8–12.9%. CQL regularization can intentionally push estimates downward;
these are unbounded learned estimates, not calibrated outcome values. Range excursions,
seed sensitivity and confounding remain reasons against treating Q as a winning probability.

## Action replay and live boundary

The serving loader generated 108 recommendations from one real same-patch development
match, `KR_8445027983`: BC/CQL/IQL at actual times 600.222, 1200.377 and 1800.487 seconds,
covering ten players and both teams. Each recommendation contains subject, retrospective
role, instruction, prerequisite checks and horizon. Inputs use only the selected causal
frame and previous frame. This is a cached-observation replay, not a rollout that predicts
what would happen after obeying the advice.

A real Windows-bridge `rift live --once --actor active --role JUNGLE` poll returned
`offline`. No active-game integration, issued game commands, prospective outcomes or
policy win-rate improvement were verified. The application remains advisory.

## Cleanup and retained artifacts

The maintained source already excluded formation modules; no old objective was restored.
Thirty-eight legacy paths were moved into
`data/archive/20261105T145112Z-54d34f98-legacy-artifacts/`. The archive manifest verified
all 2,292 files after movement (2,914,594,796 logical bytes; hard links mean this is not
unique disk usage). No disk-reclamation claim is made. Raw matches, exact-patch catalogs,
active next-action datasets/models and earlier source archives remain available.
Historical report path strings refer to original locations; the manifest maps them to
the archive. `data/archive/legacy-artifacts-latest.json` locates that manifest.

Current outputs, all local and ignored:

- `data/research/next-actions-20261105-iql-study-v1/study.json`: twelve completed runs.
- The study's `sources/` and `source-sha256.json`: exact training/evaluation source snapshot.
- Per-run `model/` and `evaluation/`: bundles, losses, match splits, bootstrap diagnostics and predictions.
- `data/research/next-actions-20261105-iql-sensitivity-v1/report.json`: seed, critic and missing-state diagnostics.
- The study's `cached-action-replay.json`: actual recommendation examples.

Future study runs automatically preserve source snapshots. Study and sensitivity commands
retain failed/interrupted reports, refuse to overwrite prior output directories and do
not publish a latest model. Existing CQL/BC bundles remain compatible; IQL uses an explicit
`independent_actor_v1` architecture contract. All three runtime loaders worked with
training and research imports blocked.

## Next experiments supported by these results

1. Audit mixed/forced combat labels against real intervals before adding finer action
   classes. Fight concentration is not evidence that initiating a fight is appropriate.
2. Compare joint history-mask augmentation and conservative abstention calibration on a
   separate development/calibration partition, preserving action feasibility and labels.
3. Study bounded critics or return-scale regularization explicitly, including sensitivity
   of CQL's pessimism and action coverage; do not silently clip existing saved models.
4. Collect prospective live-distribution evidence and design an outcome study before
   selecting a model on strategic usefulness. Previously consulted matches are not a
   fresh confirmatory test.

Validation: 50 tests including CQL/BC/IQL optimizer/reload parity, independent actor
gradients, fixed split experiments, stress diagnostics and desktop/mobile browser
interactions. Ruff lint/format, affected CLI help and git diff checks also pass.
