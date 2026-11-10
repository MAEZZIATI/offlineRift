# Later ranked acceptance: show, record and review advice

The product contains one macro coach and five role specialists.
Live testing is deferred during the current offline development phase. These commands
are retained for later acceptance sessions; victory improvement is unestablished.
Use the active-player view so your side is resolved from the live roster.
Replace BOTTOM below with your actual role; all five roles are part of the product.

## Before loading the map

From the repository root, in one terminal:

```bash
.venv/bin/rift verify
RIFT_SESSION="data/research/ranked-$(date -u +%Y%m%dT%H%M%SZ)"
.venv/bin/rift ui --actor active --role BOTTOM --interval 5 \
  --capture-session "$RIFT_SESSION"
```

Open **http://localhost:8765**, keep the dashboard open, and select **Me**. The
command prints the recording directory. Start a ranked Summoner's Rift game.
The Live Client API needs no Riot key. Advice waits for the game; known unsupported
modes such as Swiftplay withhold advice while still allowing opted-in observation.
CLASSIC mode alone does not verify ranked queue membership.

The same process serves and records recommendations, issuance/expiry times, measured
stats/items/events and feature availability. Player names are omitted in saved samples.
Persistence requires continued support; death, abstention or lost support withdraws
advice. Conflicting explicit role/team destinations withdraw the role direction while
the team proposal remains visible; no compatible alternative is inferred. Independent
proposals are not proof of strategic alignment. Scores are not win probabilities. Objective and route checks still require
your assessment; unobserved vision, waves and safe paths are not reconstructed.

## After the game

Stop the UI with **Ctrl+C**, then use the same terminal:

```bash
.venv/bin/rift-research audit-session --session "$RIFT_SESSION" \
  --output "${RIFT_SESSION}-audit"
.venv/bin/rift-research adherence --session "$RIFT_SESSION" \
  --output "${RIFT_SESSION}-review"
```

Read `${RIFT_SESSION}-review/summary.md`. It groups five-second polls into
nonoverlapping decision windows and reports **supported**, **contradicted**,
**consistent evidence**, or **unverifiable** activity. Missing endings, changed
advice, observation gaps and unresolved subject selection stay explicit.

Live CS gains can establish some farming evidence, not the lane or camp. A death
does not establish that you chose to fight. Forced respawn advice is separate from
voluntary adherence. A dashboard response does not prove that you read it. The
report therefore does not invent a global "followed recommendations" percentage.
You can add `--reported-outcome win` or `loss` to the audit command, using a new
output directory; it remains explicitly self-reported if no terminal evidence exists.

## Stronger review using the completed Match-V5 game

For your own match collection, your Riot ID must be explicitly configured with a
verified PUUID. This does not change the existing primary collection account.
If your account is not yet configured, replace both placeholders below:

```bash
.venv/bin/rift --env-file .env add-player \
  --riot-id 'YOUR_NAME#TAG' --region YOUR_REGIONAL_ROUTE
.venv/bin/rift --env-file .env collect --player 'YOUR_NAME#TAG' --count 1
```

Use the correct regional route for your account, for example `europe` or `asia`.
The collection report identifies the downloaded/reused match ID. Then:

```bash
.venv/bin/rift-research adherence --session "$RIFT_SESSION" \
  --match data/raw/MATCH_ID.json --output "${RIFT_SESSION}-match-review"
```

The review refuses a different roster or game clock. Only aligned, measured adjacent
frames can provide reconstructed activity evidence; it never invents intermediate
movement. Ambiguous activities remain unresolved, even after a win. This is evidence
of consistency with advice, not proof that the advice caused your actions or result.
