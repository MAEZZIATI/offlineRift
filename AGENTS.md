# Offline Rift repository instructions

## Objective and structure

The final product is a coordinated six-policy macro coach: one team-direction model
and one specialist for each of TOP, JUNGLE, MIDDLE, BOTTOM and UTILITY. All six target
terminal team victory. An ADC pilot must never redefine or replace this product scope.
Train and evaluate each policy separately using shared causal contracts and match splits.
Return the team proposal and selected role's proposal together, with explicit alignment,
conflict or missing support. Do not invent that a role model was trained conditional on
a proposed plan when coordination is only a serving-time compatibility check.

Build next-action policies for a team or a selected player/role toward terminal victory.
Do not turn current formation labels into recommendations. A functioning offline RL run
does not establish optimality, strategic utility or calibrated win probabilities.

- `production/src/offline_rift/`: collection, catalogs, workflow, maintained training,
  inference and local UI. Keep one orchestration path through `rift` / `python workflow.py`.
- `common/src/rift_common/`: shared actions, causal state, timeline reconstruction,
  executed-option evidence, explicit transitions, network and artifact handling.
- `research/src/rift_research/`: optional new-objective studies and frozen diagnostics.
- `tests/`: every test, in this one folder. Synthetic fixtures belong only in tests.
- `docs/`: current contracts; `research/docs/`: results from specific observed runs.

Common must import neither application nor research. Production must never import
research. Runtime policy loading must not import training orchestration. Use composition,
small explicit interfaces and dataclasses where useful; avoid speculative frameworks.

Keep domain contracts in common, optimizer code in production/training, inference-only
bundle loading/routing in production/inference, workflow orchestration in workflow.py,
and optional offline comparisons/report generation in research. Six independent model
bundles share one dataset pipeline; do not duplicate collection or training implementations.
Make per-policy support, abstention and collapse visible, including failed macro targets.
Keep mathematical assumptions and losses in docs/MATHEMATICAL_DESIGN.md. All algorithms,
including imitation, may compete on frozen development criteria; never exclude a better
baseline solely to promote RL. Report robustness/coverage tradeoffs and failed gates.
Withhold conflicting role destinations after persistence rather than inventing a compatible
alternative. Independent proposals are not proof of strategic coherence.
Do not turn "no coherent observed team action" into a positive learned strategy label.
Archive superseded prototypes and dated reports with checksums; keep current docs concise.

## Modeling invariants

- Inputs contain only causal observations/history. Future observations may identify
  executed-option/reward targets, never preceding input features. Never use final match
  economic statistics as state. Role metadata is retrospective and must be documented.
- Current training intervals use measured frames, usually about a minute apart. Do not
  invent intermediate movement, commands, casts, waves, cooldowns, coordinates or vision.
- Version incompatible state/action/dataset schemas explicitly. Preserve masks and shared
  ordering/scaling. Live player/role selection must be reported or explicitly provided.
- Preserve actor episodes, terminal boundaries, exact stored next states and elapsed-time
  discount. Never stitch across unresolved actions. Log exclusions and their bias.
- Deduplicate matches and keep all player/team perspectives from a match in one split.
- Rewards are terminal team win/loss unless an explicitly studied version changes them.
  Keep shaping studies separate. Q values and behavior scores are not win probabilities.
- Report action and role support, behavior agreement, majority baseline, abstention,
  concentration and collapsed predictions. These are diagnostic, not policy win rate.
- Do not claim precise micro, gank intent, camp/vision safety or optimal choices from
  coarse observations. Keep unsupported action families explicit in the research roadmap.

## APIs, data and artifacts

Use configured verified PUUIDs as collection identities. Do not guess accounts from names
or silently change the primary account. Collection defaults to the primary player;
collection across configured accounts is explicit. Keep Riot keys only in RIOT_API_KEY,
use the token header and correct regional/platform routing. Preserve bounded retries and
honor Retry-After without early retry. Cache raw payloads by match ID.

Resolve item catalogs to the game's exact patch, without nearest-version fallback.
Objective rules are pinned, patch scoped estimates. Persist progress/metadata atomically.
Publish latest references only after successful validation and complete training/reload.
Retain historical raw data, datasets/models and failed/interrupted reports. The old
objective's source/docs were archived in data/archive at the user's request; do not
reintroduce that objective or delete the archived artifacts as cleanup.

Never commit keys, .env, root workflow.json, data/, root models/, catalogs, logs or .venv.
Use Python 3.10+ and the existing .venv; dependencies are in pyproject.toml. Do not add
one-off scripts or duplicate collection/build/train paths.

## Validation and handoff

Run from the repository root:

```bash
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/python -m pytest -q
git diff --check
```

Tests must not access Riot or Data Dragon. Mock pagination, transport, rate limits,
duplicates, interrupted collection and catalog mismatch. Keep real Torch optimizer /
save / reload checks small. Check affected CLI --help and installed rift entry points.
Optional browser checks use a local fixture server and RIFT_TEST_BROWSER, never live
training data. Explain changes, validation and material limits. Never claim authenticated
collection or live-game integration without actually exercising it.
