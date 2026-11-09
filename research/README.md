# Offline policy research

Research evaluates the six-policy product through the maintained production trainer
and inference loader. It supplies diagnostics, not another collection or training path.
The terminal team outcome remains the objective; behavior metrics are not win rates.

## Maintained tools

- `rift-research dataset`: inspect roles, action support and evidence exclusions.
- `rift-research evaluate`: reproduce frozen development predictions.
- `rift-research evaluate --new-matches`: evaluate only IDs absent from the model's
  training and development partitions. Freeze these matches before selecting models.
- `rift-research study` and `sensitivity`: optional matched optimizer and omission studies.
- `audit-combat`, `calibrate-coverage`, `decision-regret`: optional evidence/decision diagnostics.
- `audit-session` and `adherence`: explicitly opted-in sessions and measured following evidence.
- `synthesize`: generate the current Markdown/PDF from a completed coach and evaluation.

Train the six policies with `rift train-coach`, not a separate research trainer.
Each candidate records its configuration, data digest, scope, match split, optimizer
progress and serving parity. Selection uses development diagnostics only. Report
macro and each role separately, including baseline, abstention and concentrated output.

Coarse action evidence can be ambiguous or involuntary. Identified-only fitting creates
selection bias. Complete episodes remain stored, including unresolved intervals and
terminal rewards. Conservative return regression on complete observed outcomes is a
separate variant from Bellman-target CQL; neither identifies causal action values here.

[Current synthesis](docs/SYNTHESIS.pdf) and [editable summary](docs/SYNTHESIS.md) describe
observed results. [Historical archive](docs/HISTORY.md) retains superseded reports and
ADC-only prototypes. No current result establishes improved victory rates. Live-game
acceptance and a predefined prospective comparison remain later work.
