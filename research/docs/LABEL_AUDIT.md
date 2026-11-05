# Weak-label evidence audit

Update: [versioned objective rules and review handoff](OBJECTIVE_RULES.md) now record
the corrected `sr-2026-v3` build and a 246-window blinded packet covering all seven
patches. Independent judgments remain pending; workflow retraining is gated on them.
The original 177-window results below remain historical measurements.

Observed 2026-11-04. This completes an automated audit of the prepared 177 windows;
independent human annotation is still outstanding. The diagnostics measure reproducibility
and risks in the evidence, not label accuracy, intention, or strategic usefulness.

## Results

| Diagnostic | Observed result |
|---|---:|
| Windows / distinct training matches | 177 / 96 |
| Reconstructed labels differing from the saved dataset | 0 |
| Label changes under geometric-threshold perturbations | 27 / 177 (15.25%) |
| A recorded death after the position frame and before the marked instant | 88 / 177 (49.72%) |
| Missing positions / positions outside map bounds | 0 / 0 |
| Mean / maximum age of positions | 32.52s / 60s |
| Completed independent human annotations | 0 |

The sensitivity check crosses pit radii 1,800/2,000/2,200 and mid-band widths
1,620/1,800/1,980 map units, retaining all other rules. It asks whether a ±10% change
in these design thresholds changes the label. It is not a probabilistic coordinate
noise model and does not estimate the probability that a label is wrong.

Death flags use only timestamped kills occurring after the observed position frame
and at or before the review instant. They can concern either team, including players
irrelevant to a particular grouping label. They show that some positions predate a
recorded death; they do not establish that the player is still dead, has respawned,
or should be assigned an invented current coordinate.

| Weak label | Windows | Threshold-sensitive | Windows with death flag |
|---|---:|---:|---:|
| Dragon area setup | 15 | 6 | 9 |
| Baron area setup | 15 | 4 | 8 |
| Herald area setup | 14 | 4 | 9 |
| Cross-map pressure | 13 | 3 | 5 |
| Side-lane split formation | 15 | 0 | 10 |
| Group mid | 15 | 3 | 7 |
| Group top side | 15 | 1 | 8 |
| Group bottom side | 15 | 3 | 7 |
| Own-half defensive formation | 15 | 0 | 10 |
| Base regroup | 15 | 0 | 0 |
| Dispersed lane coverage | 15 | 0 | 9 |
| Unclassified / insufficient evidence | 15 | 3 | 6 |

The same 177 windows were sampled from training matches with approximately equal
patch/label allocation. These percentages must not be reported as population-wide
failure rates. The audit does not use a test set to revise the labeler. Existing
`macro-v2` labels and historical datasets remain unchanged.

## Objective-rule consistency: an additional concrete limitation

The illustrative configuration uses Herald first spawn 840s (14:00) and a hard
disable at 1,170s (19:30). Riot documented a **15:00** Herald spawn in
[Patch 25.09](https://www.leagueoflegends.com/en-au/news/game-updates/patch-25-09-notes/).
The [2026 season update](https://www.leagueoflegends.com/en-us/news/game-updates/patch-26-1-notes/)
states that epic-monster spawn times remain unchanged except where explicitly noted
and moves Baron back to 20:00. This is evidence for a mechanics review, not automatic
certification of every intervening target patch or of an exact despawn time.

An additional audit scans all **443 frozen development matches**. It finds 390 Herald
kill events in 390 matches and no multiple-Herald kills. The earliest is 911.543s and
the latest is 1,190.933s. **Fourteen recorded kills occur at or after the configured
1,170s cutoff.** A hard disable before a recorded kill is a direct contradiction of
the estimated availability rule. Kill times alone do not identify exact spawn or
despawn mechanics. Full match/timestamp traces are retained in
`herald_timing_audit.json` in the audit directory.

Among the frozen corpus's 231 Herald-labeled states, 54 occur before 840s. Under a
900s spawn and the existing 60s setup-window definition, eligibility would begin at
840s; these states are therefore a specifically identifiable sensitivity to correcting
the first-spawn estimate. This is conditional on confirming that rule for each target
patch, not a human-verified count of wrong tactical intentions.

The objective rules must therefore be reviewed and versioned before promoting a new
model. Preserve the current controlled experiment under its original target definition;
do not silently relabel its holdout. Corrected timing configurations require their own
dataset build, training and evaluation. Other objective estimates, including the
elemental-to-Elder transition, still need a patch-specific audit. No exact respawn or
despawn value is invented to fill the evidence gap.

## Interpretation and priority

Label reproduction is sound for the sampled inputs; semantic validity remains unproven.
Objective labels deserve early manual attention because their small support, low model
precision and threshold sensitivity all affect interpretation. A pit cluster supports
an area-formation description, not a commitment to attack or proof of vision control.
Base presence supports regrouping geometry, not verified recall intent.

The stale-position/death interaction is a concrete supervision limitation. A future
label revision should represent unavailable life-state evidence explicitly and be
audited before schema migration. Do not silently reconstruct respawn times or relabel
every death-flagged example as incorrect. Keep such changes separate from the present
memory experiment so the comparison retains one target definition.

## Blinded review materials

The local directory `data/research/20261104-label-audit-v1/` contains:

- `review.html`: a self-contained local viewer with causal maps, preceding observations,
  source-position age, and timestamped kill/objective/building events. It reveals
  neither weak labels nor the sensitivity diagnostics.
- `review_packets.json`: the same causal evidence in machine-readable form.
- `diagnostics.json`: the unblinded audit key and per-window automated findings.
- `report.json`: aggregate results and limitations.

Give two reviewers separate blank copies of the HTML. Each records a formation,
evidence sufficiency, whether intent is observable, and notes, then downloads their
own JSON before closing the page. The viewer stores answers only in page memory.
Reviewers must not inspect the diagnostics/key or each other's answers before completion.
The map is an approximate coordinate plot, not replay video or vision reconstruction.

After both reviews, adjudicate disagreements and report per-label precision against
the adjudicated rubric, reviewer agreement, Cohen's kappa with prevalence, missing-evidence
rates and match-cluster uncertainty. Do not force an intention label when observations
cannot support it. Equal strata require an explicitly defined weighting target for
any pooled metric; the unweighted audit average is not population precision.

```bash
rift-research audit \
  --dataset data/projects/faker/datasets/faker-fa3ec093d7f589e9/transitions.npz \
  --sheet data/research/20261104-live-frozen-v3/audit_blinded.csv \
  --key data/research/20261104-live-frozen-v3/audit_key.csv \
  --reference-report data/research/20261104-live-frozen-v3/report.json \
  --raw-dir data/raw --output data/research/my-label-audit
```

The command verifies each keyed label against the original dataset, requires all
windows to belong to the saved training split, and refuses to overwrite an output
directory. Future events are excluded from both diagnostic flags and reviewer packets.
