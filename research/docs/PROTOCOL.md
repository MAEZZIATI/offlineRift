# Current offline protocol

The product contains six separate networks: TEAM, TOP, JUNGLE, MIDDLE, BOTTOM and
UTILITY. Freeze corpus membership, match-level partitions and scope-specific
training settings before fitting. Do not select on reserved-match diagnostics.

The current comparison uses split seed 42, optimizer seed 42 and 10,000 updates per
candidate. Per scope compare two conservative critics and one imitation baseline.
Macro critics use complete observed episode-return targets because identified macro
intervals rarely coincide with terminal frames. Role critics compare Bellman targets
with observed return regression. All rewards remain terminal team victory/loss.

BC measures imitation and is not automatically promoted as an RL policy. Select
noncollapsed CQL candidates first, then identified development macro F1, with lower
voluntary concentration as a tie breaker. Report failures and thin support explicitly.
Selection on behavior diagnostics is exploratory and does not establish strategic value.

Report source hashes, evidence exclusions, action/role support, majority baseline,
abstention, action concentration and isolated saved-model serving parity for all scopes.
Compare selected policies once on frozen unseen match IDs. Repeated players and shared
teams can remain dependent; new IDs do not guarantee independent-player evaluation.

The packaged `rift_research/protocol.json` records machine-readable defaults and is
hashed into optional calibration/regret diagnostic reports. Those tools do not publish
serving thresholds. Their older observed results and protocol versions remain archived.

No live-game test is performed in this phase. Future active-game acceptance and a
predefined prospective comparison are required before claims of improved victory rates.
