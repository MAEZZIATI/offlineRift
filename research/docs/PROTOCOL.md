# Current offline protocol

The European `relative-context-v2` experiment adds 313 unique ranked solo games from
eight currently API-verified EUW Challenger accounts. Forty European matches,
including one explanatory notebook game, and the previous 21 reserved matches are
excluded before fitting. Current roster membership does not certify each game's
historical ranks. The frozen manifest is
`data/research/european-improvement-protocol-20261110.json`.

The experiment compares eight candidates per scope: the previous four families,
phase-supported imitation, softer class weights with phase support, a second
consistency seed and discrete IQL. State v2 adds masked public-stat differences and
measured alive fractions. Checkpoints every 500 updates, up to 4,000, use the same
development rule, stopping after three nonimproving checkpoints. Report the complete
candidate set and restored steps; one extra seed is not a full seed-robustness study.
Evaluation compares fixed products once on the forty European matches and replays
the reserved notebook game for every policy and participant. No test retuning.

That first comparison showed a TEAM regression and is retained. A subsequent
development-only reselection equally weights European/Korean scores, reusing the same
trained networks without additional optimizer updates. Its frozen study hash precedes
the additional bounded collection and forty fresh reserved IDs in
`data/research/european-fresh-protocol-20261110.json`. These older-date games span
patches 16.19–16.21; duration eligibility is at least fifteen minutes. All forty passed
validation. No candidate changed after this fresh comparison. Final replay uses one
freshly reserved game and its exact patch catalog; the older consulted replay is archived.

The following describes the preceding robustness study, retained for reproduction.

The product contains six separate networks: TEAM, TOP, JUNGLE, MIDDLE, BOTTOM and
UTILITY. Freeze corpus membership, match-level partitions and scope-specific
training settings before fitting. Do not select on reserved-match diagnostics.

The current robustness comparison freezes split seed 42, optimizer seed 42 and 6,000
updates per candidate. Per scope compare ordinary imitation and conservative observed-
return regression with availability-consistent versions (four per scope). All rewards
remain terminal team victory/loss. The previous 18-run study remains historical.

All algorithms may be selected. See docs/MATHEMATICAL_DESIGN.md for the exact development
score, gates and losses. The frozen machine-readable study protocol is retained in
data/research/coherent-robust-protocol-20261109.json. A bounded latest-ten collection across
57 configured verified accounts found 21 new IDs, reserved before fitting. Previously
viewed test matches are development evidence. No fresh-test retuning is allowed.
Selection on behavior diagnostics remains exploratory and does not establish strategic value.

Report source hashes, evidence exclusions, action/role support, majority baseline,
abstention, action concentration and isolated saved-model serving parity for all scopes.
Compare selected policies once on frozen unseen match IDs. Repeated players and shared
teams can remain dependent; new IDs do not guarantee independent-player evaluation.

The packaged `rift_research/protocol.json` records machine-readable defaults and is
hashed into optional calibration/regret diagnostic reports. Those tools do not publish
serving thresholds. Their older observed results and protocol versions remain archived.

No live-game test is performed in this phase. Future active-game acceptance and a
predefined prospective comparison are required before claims of improved victory rates.
