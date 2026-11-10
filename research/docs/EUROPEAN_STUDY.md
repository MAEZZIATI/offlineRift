# European development and game review — 2026-11-10

The [executed notebook](../notebooks/european_high_elo_game.ipynb) contains 16 figures,
development comparisons, learning curves, the first reserved-cohort results, the
subsequent regional diagnosis, and one measured-frame game replay for both teams and
all ten players. Its final cell verifies saved recommendations against the raw game
and model hashes, without downloads or fitting.

The first candidate improved UTILITY activity agreement but regressed on TEAM. After
regional development reselection, a fresh forty-game comparison supports improved
MIDDLE and UTILITY activity agreement. The other four differences remain uncertain,
with slightly lower TEAM/JUNGLE point estimates. The verified revision is activated
as an experimental offline candidate, retaining the previous pointer and weights.
Neither comparison establishes increased win rate.

## Data and modeling changes

- Registered eight current EUW Challenger roster PUUIDs through authenticated Riot
  requests, preserving the primary account. Collected 313 distinct ranked solo games;
  310 have complete supported episodes. Current roster membership is not historical
  rank verification for every participant.
- Rebuilt 1,950 accepted Korean/European matches into 559,200 measured transitions.
  Thirteen raw matches had no complete supported episodes. All forty reserved European
  matches and the earlier twenty-one reserved matches survived validation.
- State v2 adds 24 masked relative-public-stat/alive features to the unchanged 64-value
  v1 prefix. No positions, hidden vision, waves or future match statistics are inputs.
  V1 bundles and datasets remain loadable; the previous coach still reproduces its
  frozen serving replay.
- Trained eight candidates independently for each of six scopes, up to 4,000 updates:
  ordinary/consistent imitation, ordinary/consistent conservative observed-return
  regression, phase-supported imitation, softer class weights, a second consistency
  seed and discrete IQL. All algorithms could win selection.
- Checked development every 500 updates, restoring the best eligible checkpoint and
  allowing stopping after three nonimproving checkpoints. Phase-supported candidates
  require twenty identified training examples for an action in the current role/time
  bin. This is a coarse overlap screen, not calibrated confidence.

All six first-study selections passed the frozen development gates. All selected
networks were imitation variants. Their behavior surrogate does not directly optimize
terminal victory. Critic candidates retain terminal team win/loss rewards and their
observational/confounding limitations.

## First reserved European comparison

Forty unseen match IDs, held out before fitting and selection; 1,000 paired whole-match
bootstrap samples. Every perspective of a match stays together. Repeated players can
remain dependent. Intervals are exploratory and are not corrected for multiple scopes.

| Policy | Identified rows | Previous F1 | First candidate F1 | Difference, 95% interval |
| --- | ---: | ---: | ---: | --- |
| TEAM | 61 | 0.316 | 0.272 | -0.045 [-0.125, -0.011] |
| TOP | 1,209 | 0.168 | 0.175 | +0.007 [-0.007, +0.021] |
| JUNGLE | 1,052 | 0.210 | 0.214 | +0.003 [-0.005, +0.012] |
| MIDDLE | 1,127 | 0.196 | 0.194 | -0.002 [-0.014, +0.007] |
| BOTTOM | 1,145 | 0.178 | 0.186 | +0.009 [-0.003, +0.019] |
| UTILITY | 1,105 | 0.187 | 0.216 | +0.029 [+0.012, +0.048] |

F1 measures agreement with identified reconstructed activity, not which alternative
would win. This compares complete products with different data and representations;
it does not isolate the effect of any single change. The prior model uses an explicit,
recorded canonical v2-to-v1 feature projection for this evaluation.

## Adaptive regional selection

The European TEAM development subset had only 87 identified targets against 485 Korean
targets. The revised selection gives the two regional development scores equal weight,
requiring both regional gates and at least twenty identified targets in each region.
It reuses the original trained networks and checkpoints, performs zero further optimizer
updates, and preserves their common match partitions. No candidate is chosen using
test predictions. The revision was nevertheless motivated by the first test, so that
cohort is now consulted retrospective evidence and cannot independently validate it.

| Policy | Revised selected network |
| --- | --- |
| TEAM | bc-phase |
| TOP | bc |
| JUNGLE | bc |
| MIDDLE | bc-phase-softweights |
| BOTTOM | bc |
| UTILITY | bc-phase-softweights |

All revised selections pass both regional development gates and saved serving replay.
Weights are retained at `models/european-balanced/20261110/model`; the first study is
retained at `models/european-context/20261110/`.

## Fresh evaluation after the adaptive revision

A temporary Riot HTTP 401 was resolved after the local key changed. An additional
bounded account window (`start=40,count=10`) yielded 73 new distinct European games,
bringing total European collection to 386 unique games. Forty new IDs were frozen
after the revised selection, before evaluating it; all forty passed dataset validation.
This is an earlier-date cohort: 19 games on patch 16.19, five on 16.20 and sixteen on
16.21. Eligibility required a duration of at least fifteen minutes. These shifts and
recurring players limit interpretation; fresh IDs do not establish player independence.

| Policy | Identified rows | Previous F1 | Revised F1 | Difference, 95% paired interval |
| --- | ---: | ---: | ---: | --- |
| TEAM | 68 | 0.334 | 0.322 | -0.012 [-0.062, +0.031] |
| TOP | 1,180 | 0.170 | 0.175 | +0.005 [-0.012, +0.022] |
| JUNGLE | 1,130 | 0.199 | 0.192 | -0.007 [-0.019, +0.006] |
| MIDDLE | 1,184 | 0.185 | 0.208 | +0.023 [+0.009, +0.039] |
| BOTTOM | 1,145 | 0.174 | 0.181 | +0.006 [-0.003, +0.015] |
| UTILITY | 1,175 | 0.181 | 0.210 | +0.029 [+0.008, +0.054] |

No candidate was changed after this fresh comparison. A second comparison on exactly
these rows contrasts the initial and revised selection rules with the same trained
candidate bank: TEAM F1 increases by 0.013 and MIDDLE by 0.017; the other four policies
are identical. The notebook retains both positive findings and the first regression.
The final revision was activated as an experimental candidate after integrity/reload
verification, with the previous reference retained. It is not a certified winning policy.

## One-game notebook

`EUW1_8034380954`, EUW ranked solo, patch 16.21, created 2026-11-03 01:25 UTC. A verified
collection roster account is present. Both teams and every role/player are
replayed with the revised selection and previous coach. Roles are retrospective.
Measured input frames precede every recommendation; following frames label observed
activity only. No intermediate polls or movement are fabricated.

Plots distinguish raw role proposals, team directions, delivered actions after conflict
withholding, identified observed activities, support, concentration and coverage. The
recorded game was reserved in the fresh second cohort before evaluation. It remains
one explanatory game, not standalone evidence of strategic benefit. Unresolved macro
behavior stays unresolved. Its exact 16.21 catalog is saved with a digest and used for
both models; no neighboring-patch or frozen-current-patch substitution occurs.

Each model's 288 raw replay proposals exactly match its independently saved batch
evaluation predictions for the same game: both teams, ten players and both macro views.

Local reproduction uses the existing maintained commands:

```bash
.venv/bin/rift verify --model models/european-balanced/20261110/model
.venv/bin/rift reselect-coach --study models/european-context/20261110 \
  --output NEW_SELECTION_DIRECTORY \
  --holdout-manifest data/research/european-balanced-protocol-20261110.json
.venv/bin/rift-research review-game --match data/raw/EUW1_8034380954.json \
  --bundle models/european-balanced/20261110/model \
  --baseline models/coherent-robust/20261109/model \
  --study models/european-balanced/20261110/study.json \
  --protocol data/research/european-fresh-protocol-20261110.json \
  --holdout-comparison data/research/european-fresh-comparison-20261110.json \
  --followup data/research/european-final-followup-20261110.json \
  --output NEW_REVIEW_DIRECTORY --notebook NEW_NOTEBOOK.ipynb --execute-notebook
```

Raw data, source snapshots, all fitted candidates, protocols, first-test predictions,
failure reports and prior models remain local. Prior notebook/default snapshots have
checksums under `data/archive/european-improvement-20261110/`.
