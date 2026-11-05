# Data volume, quality and selection bias

Planning estimates as of 2026-11-04, not guarantees of model quality or formal power
calculations. The current target is team-formation research in Korean ranked solo/duo.
Advice for all ranks or regions requires a separately defined target population.

## How much data is a reasonable target?

Count unique matches, source accounts, patches and independent audited situations.
Do not count 62,168 correlated transitions as 62,168 independent demonstrations.

| Stage | Unique matches | Source accounts | Purpose |
|---|---:|---:|---|
| Original model corpus | 200 | 2 configured anchors | Development and pipeline validation; inadequate for broad generalization |
| First pilot checkpoint | 443 | 12 anchors across separate configurations | 243 newly collected games; original published model retained |
| Official-roster expansion | 1,289 | 57 anchors across separate configurations | 846 further new games; 212 have no participant seen in the 443-game development corpus |
| Diversity/learning-curve pilot | 1,000–2,000 | 50–100 across five historical roles | Find label failures; estimate learning curves, overlaps and cluster variance |
| Main narrow-scope study | 5,000–10,000 | 100–200+, capped histories | Compare memory/abstention with roughly 1,000–2,000 held-out matches |
| Broader rank/region claims | 10,000–30,000+ | Several hundred stratified accounts | Separate evaluation in declared rank, region and patch strata; revise size using pilot uncertainty |

These are engineering budgets. Five thousand poorly sampled games can be less useful
than one thousand well-audited, diverse games. Start with the pilot, then fit learning
curves on 200/500/1,000/2,000 training matches with fixed development/test cohorts.
Add data where rare-label coverage and held-out uncertainty remain limiting. Stop
scaling when additional matches do not improve audited metrics or reduce uncertainty
enough to justify collection. Quantity cannot restore spatial telemetry absent at live time.

### A concrete rare-label calculation

In the original 200 matches, weak labels occur in these numbers of distinct matches:

| Label | Positive matches | Positive 10s states |
|---|---:|---:|
| Dragon area setup | 82 | 425 |
| Baron area setup | 73 | 438 |
| Herald area setup | 22 | 132 |
| Cross-map pressure | 37 | 153 |

Herald appears in 11% of matches. If that frequency transported to a new sample,
100 positive test matches would require about `100 / 0.11 = 909` test matches,
or about 4,545 total matches with a 20% test allocation. For 385 positive test
matches, the same arithmetic gives about 17,500 total matches. Actual prevalence
will change with patches, player pools and corrected labels. These are expected counts,
not lower confidence bounds or recommendations to preserve an unreliable label.
The new 243-match extension contains Herald labels in 27 distinct matches (11.1%),
close to the original occurrence rate; this still does not validate those labels.

For a single binomial proportion, the conservative normal-approximation planning
formula is `n ≈ 1.96² × 0.25 / ε²`: about 97 independent examples for ±10 percentage
points and 385 for ±5. Precision requires examples predicted as the action; recall
requires true positive-label examples. Their denominators are different. Repeated
states, both teams, overlapping windows, and repeated players reduce independence.
For `m` equally sized observations per cluster with intracluster correlation `ρ`,
the textbook approximation `design effect ≈ 1 + (m−1)ρ` illustrates why extra timestamps
cannot substitute for new matches. Here `ρ` is unknown and cluster sizes differ, so do
not insert an invented effective sample size. Use pilot cluster bootstrap or simulation
to plan the actual paired macro F1 comparison, including training-seed variability.

## Sampling design and remaining bias

1. Declare the target first: initially Korean high-level ranked play, not the general
   player population. Metadata currently does not establish historical rank. Champion
   mastery, streamer identity, final KDA, wins and highlight quality are not valid
   substitutes for a sampling frame.
2. Sample accounts uniformly within declared strata, then cap matches per account and
   collect consecutive ranked history independent of outcome. Equal-role quotas aid
   coverage but differ from population sampling; save their inclusion probabilities.
3. Expand beyond the original ego network. The present pilot samples co-participants
   of the original anchors, so elite matchmaking, activity and social/network bias
   remain. A second phase needs independent account frames across the intended rank
   strata and regions using official League endpoints or a documented consented panel.
   Rank measured today must not be described as rank at a historical match.
4. Save anchor identity, discovery match, role stratum, sampling seed, queue, match time,
   patch, fetch failures and deduplicated overlap. Sampling frequently encountered
   accounts proportional to their appearances would overrepresent active players;
   this pilot samples unique accounts instead.
5. Keep a natural-frequency test set. Oversampling rare situations belongs in training
   or a separately reported stress test, not silently in the population test metric.
   Reweight only with defensible target/source probabilities; weights cannot repair
   populations that have no observed support. Report weight effective sample size
   `(sum w)² / sum(w²)` if weighting is used.
6. Use later matches and anchor-disjoint cohorts for shift tests. Both teams from each
   match always stay together. Anchor-disjoint does not imply all-participant-disjoint:
   report shared participants and remove cross-split games for a strict player-disjoint
   test if feasible. Dense matchmaking may make that test small or impossible; report
   the retained sizes rather than pretending the condition holds.

## Quality gates

- Require matching match/timeline IDs, queue 420, ten valid participants, complete
  usable frame structure and monotone timestamps. Deduplicate match IDs globally.
- Log short games, remakes, surrender, missing frames and parsing failures. Predeclare
  exclusions; avoid removing losses or mistakes merely because they look low quality.
- Pin patch-matched catalogs and objective rules; audit whether each weak-label rule
  remains meaningful on each patch. No nearest-patch fallback.
- Retain causal forward-fill age and availability masks. Audit stale positions and
  labels; do not use final items, gold, casts or future coordinates as state features.
- Audit at least about 100 independent situations per important label as a first
  precision goal, preferably from different matches, with two independent reviewers.
  Increase toward 385 when ±5-point precision is material. Stratify by patch and source
  where feasible, record unknowns, and report disagreement rather than forcing intent.
- Keep live and full profiles distinct. A spatial teacher may supervise representations
  only under explicit train-only targets; its inaccessible inputs cannot become live facts.

## Prepared expansion pilot

The original cache contains 700 other eligible unique PUUIDs: 167 TOP, 149 JUNGLE,
92 MIDDLE, 151 BOTTOM and 141 UTILITY under dominant historical role assignment.
Two accounts per role were sampled with seed 20261104, excluding both configured
anchors. Identity comes from the verified PUUID in a cached Match-V5 payload; no
professional identity was inferred. The role assignment is a sampling convention.

The independent configuration and cohort manifest are in
`data/research/20261104-diversity-pilot/`. The initial request is ten accounts ×25
ranked matches, at most 250 history entries before duplicates and cache reuse. This is
a collection pilot, not the full 1,000–2,000-match target. The original workflow config
and latest model references are preserved. Collection status and actual counts are
recorded in `cohort.json` and its individual batch manifests; requested count must not
be substituted for downloaded or accepted matches.

```bash
# Prepare a new cohort; the command never overwrites an existing directory.
rift-research cohort --output data/research/my-cohort --players-per-role 2 --count 25

# With RIOT_API_KEY exported in the calling environment, collect/resume that exact cohort.
rift-research cohort --resume data/research/my-cohort/workflow.json --collect
```

The CLI reads `RIOT_API_KEY` from its environment; it does not automatically execute
`.env`. Use your trusted local environment loader. Do not paste credentials into chat
or commit `.env`. A completed player is skipped on resume; an interrupted player's
downloaded games are reused by the existing workflow collector. Each match's raw
payload is cached once. Match-V5 uses regional routing and token headers; bounded
retries and Retry-After handling remain in the existing transport. Official sources:
[Riot API reference](https://developer.riotgames.com/apis/) and
[key and rate-limit documentation](https://developer.riotgames.com/docs/portal).

### Observed collection and evaluation outcome

The pilot completed with authenticated Riot requests after `RIOT_API_KEY` was supplied
through a local `.env` loader. It resolved 250 requested history entries into **243 new
unique matches**, with seven within-cohort cache reuses and zero matches from the
original 200 among the queried histories. None of the new matches contains either
original configured anchor. Total raw cache size is now **443 unique matches**.
The new-patch distribution is 46 matches on 16.20, 172 on 16.21 and 25 on 16.22.
All passed basic ID/queue/participant/frame-presence checks. These do not assess intent,
label correctness, timeline reconstruction precision or historical rank.

The cohort build accepted 265 matches, including 22 older cached co-participant games,
and skipped none. Its 80,182 transitions are not 80,182 independent trials. Frozen-model
evaluation excluded all original training/development match IDs and used 243 games /
73,660 transitions; see [the synthesis](RESEARCH_SYNTHESIS.md). No training or model
promotion occurred. Histories were capped at 25 queried entries per account; incidental
co-participation means total observed appearances per sampled account range from 25 to 32.

This is an initial ten-account pilot, below the recommended 50–100-account diversity
study. Full collection provenance is saved locally in `collection_summary.json`,
`cohort.json` and individual batch manifests under the pilot directory. Remaining bias
requires independent account frames, not just more histories from this same network.

## Official-roster expansion: executed next phase

To reduce the initial discovery-network bias, official authenticated League-V4 responses
returned 10,000 Master, 700 Grandmaster and 300 Challenger Korean accounts. The Master
response may be truncated; do not treat it as a complete population census. Sample
15 unique PUUIDs uniformly from each returned tier frame with seed 20261105, excluding
all twelve previous source anchors. Current tier is not historical game rank. Selection
probabilities and roster SHA-256 hashes are retained in the cohort manifest.

All **45 accounts ×20 history entries** were resolved. The 900 entries contain 873
unique queried matches: 27 already present in development and **846 new matches**.
Another 27 entries repeat games within the queried histories, giving 54 cache-reuse
entries in total. The raw cache now contains **1,289 unique matches**, reaching the
1,000–2,000-match pilot range. The 5,000–10,000-match main-study goal remains a later
stage after correcting the objective rules and reviewing labels.

The new-match patch distribution is 16.20: 120, 16.21: 605 and 16.22: 121. Structural
checks find no ID, queue/map, participant-count or frame-order/completeness failures.
Eight early-surrender games are retained and reported; losses and short games were
not removed merely for looking uninformative. These checks do not validate tactics,
objective mechanics or historical ranks.

Participant overlap remains substantial: **634 of 846 new matches** contain at least
one PUUID present in the 443-game development corpus. Only **212 new matches** are
strictly participant-disjoint from that entire corpus. This is why account-frame
sampling and new match IDs must not be confused with statistically independent games.
The 212-match subgroup is reserved as an exploratory sensitivity analysis using the
same frozen models; its rank/network mix differs from the full cohort.

Frozen API rosters and their provenance are under
`data/research/20261104-independent-rosters/`. Collection configuration, inclusion
probabilities, source exclusions, atomic batch manifests, `collection_summary.json`
and `corpus_quality.json` are under `data/research/20261104-independent-pilot/`.
The original workflow identity and latest model reference are preserved.

```bash
# Reuse frozen, verified official roster responses; this does not fetch names or guess identities.
rift-research cohort --output data/research/my-ranked-pilot \
  --roster-dir data/research/20261104-independent-rosters \
  --accounts-per-tier 15 --count 20 --seed 20261105 \
  --exclude-config data/research/20261104-diversity-pilot/workflow.json --collect
```

The package method `RiotDataIngestor.fetch_ranked_accounts("kr", TIER)` fetches new
official returned frames when needed, using platform routing for League-V4 and the
same bounded token-header retry handling as Match-V5. New frames obtained later will
not reproduce the same sampled accounts merely by reusing the random seed. Preserve
the responses and their hashes. Exact patch catalogs are required for every build.
