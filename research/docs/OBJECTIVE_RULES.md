# Versioned objective rules and review handoff

Observed 2026-11-04. The active specification is **`sr-2026-v3`**, covering ranked
Summoner's Rift (queue 420, map 11), client patches **16.16–16.22** only. Unsupported
patches fail before processing; there is no nearest-patch fallback. Rules remain
estimates reconstructed from kill events, not direct spawn telemetry.

## Corrected mechanics

| Quantity | Active specification |
|---|---|
| First elemental dragon / Herald / Baron | 300 / 900 / 1,200 seconds |
| Elemental dragon / Baron respawn | 300 / 360 seconds after kill |
| First Elder | 360 seconds after one team's fourth elemental dragon |
| Subsequent Elder | 360 seconds after Elder kill |
| Baron / Elder buff estimate | 180 / **150** seconds after kill |
| Herald | One spawn; unknown availability from 1,170 until 1,200 seconds if not killed |

Four elemental kills across both teams do not trigger Elder. Explicit Elder kills
update the dragon-pit clock and buff without incrementing elemental counts. Missing
kill subtype, or enough unattributed kills to make the soul transition ambiguous,
masks dragon availability rather than inventing a phase. An explicit Elder event
resolves that uncertainty.

Herald's 19:30–20:00 interval is a **conservative uncertainty envelope**, not an assertion
of an exact despawn time. Combat-dependent expiry cannot be recovered from these
observations. The existing `enabled` feature acts as the evidence mask; unknown objective
fields are zeroed. Feature order, scaling and the 177-dimensional contract are retained;
the semantic change is explicitly pinned in feature configuration and dataset provenance.
Buff features remain kill-window estimates and do not establish per-player buff retention.

Sources: Riot documents Herald's 15:00 spawn in [25.09](https://www.leagueoflegends.com/en-au/news/game-updates/patch-25-09-notes/)
and Baron's 20:00 return in [26.1](https://www.leagueoflegends.com/en-us/news/game-updates/patch-26-1-notes/).
[9.23](https://www.leagueoflegends.com/en-us/news/game-updates/patch-9-23-notes/)
describes the same-team soul transition, six-minute Elder intervals and historical
conditional Herald expiry; [14.3](https://www.leagueoflegends.com/en-us/news/game-updates/patch-14-3-notes/)
corrects a first-Elder timing bug. **[9.24b](https://www.leagueoflegends.com/en-us/news/game-updates/patch-9-24b-notes/)
supersedes 9.23's Elder duration, reducing it to 150 seconds.** Current patch-note review
found no later relevant timer change for the supported range. This supports a scoped
working ruleset, not blanket certification of every mechanic.

The unpromoted `sr-2026-v2` draft incorrectly used a 180-second Elder buff. Its exact
specification and generated dataset are retained for reproduction, explicitly superseded
and blocked from promotion. Historical bundles with no rules version retain `legacy-v1`
behavior; their reported metrics and weights have not changed. New workflow initialization
uses v3. Existing local root configuration was migrated with a backup.

## Observed rebuild and checks

Artifacts are under `data/research/20261104-objective-rules-v3/`. The immutable build
`objective-rules-v3-c5e982ca11d8a1fb` contains **1,289 matches, 2,578 team episodes and
389,308 transitions** over seven patches. No new collection or optimizer training was
performed in this phase.

A causal pre-kill check covers **6,418** modeled objective kills: 4,188 dragons,
1,106 Heralds and 1,124 Barons. It finds **zero contradictions while availability is
known**, and **34 kills in masked uncertainty windows**. This one-sided check cannot
prove exact expiry, absence, label correctness or strategic utility. Void Grubs remain
outside the modeled objective taxonomy (3,828 recorded kills).

The corrected blinded packet contains **246 windows from 147 development matches**:
the original 177 windows plus 69 supplements from patches 16.21–16.22. No supplement
comes from the later 846-match evaluation cohort. Only **one** original window changes
weak label under the corrected rules. All 246 labels reproduce; 34 are sensitive to
geometric thresholds and 117 contain a death after the position frame. These diagnostics
are not human validation or population error rates. Completed independent reviewers: **0**.

## Complete the independent review

Give each reviewer a separate blank copy of:

`data/research/20261104-objective-rules-v3/label-review/review.html`

They should receive only that viewer and the accompanying `REVIEWER_INSTRUCTIONS.md`.
Do not send diagnostics, keys, lineage, model predictions or the other review. Each
reviewer records a distinct ID, human/AI kind and independence attestation, labels every
window and exports JSON. Unknown evidence is allowed; blank answers remain unfinished.
Reviewing the old labels first cannot be undone by renaming a reviewer.

An adjudicator then resolves conflicting scorable labels and explicitly chooses `hold`,
`revise_labels` or `accepted_for_retraining`, with a rationale. Agreement alone never
automatically authorizes retraining. Example, after actual judgments are supplied:

```bash
rift-research review \
  --audit data/research/20261104-objective-rules-v3/label-review \
  --review /path/to/reviewer-1.json --review /path/to/reviewer-2.json \
  --adjudication /path/to/adjudication.json \
  --output data/research/20261104-objective-rules-v3/completed-review
```

Set `label_review.report` in the intended workflow configuration to the resulting
`report.json`. The gate binds the review to dataset bytes/fingerprint, rules and labeler
hashes, all patch coverage, complete distinct exports and an explicit acceptance decision.
It reports human and AI judgments separately, agreement/kappa, per-action weak-label
metrics and match-bootstrap intervals. Unknowns are excluded, never counted as correct.
Identity/independence are attestations, not authenticated human credentials.

A real call to the corrected workflow's training method was rejected before optimizer
construction because the review is missing; the latest artifact reference was unchanged.
This gate covers normal workflow training/publication. Explicit research/direct training
entry points remain experimental and must not be treated as promotion. Accepted review
would authorize a controlled retraining study, not establish model superiority: promotion
still needs corrected-target comparisons, coverage, majority baselines and an untouched
sampling cohort. The plugged model remains `models/faker/20261104T163045Z-2e7c3564`.
