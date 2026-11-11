# ADR 0002 — Offline script training via the Kaggle CLI

Status: accepted for Phase 1; authenticated account checks remain pending.

Established decision: use Kaggle as remote compute, only through user-run CLI
Makefile targets. Use the owner's Riot data, privately uploaded as a transport
input; never source third-party Kaggle datasets. Collect and minimize identifiers
locally. Neither token nor Riot key is read by application code, tests or CI.

Established decision: typed config generates private script-kernel metadata with
internet disabled. Local and remote runs invoke
`python -m offlinerift.training.run --config ...`. Bootstrap code only installs
offline wheels and invokes that entry point; no logic lives in notebooks.
No model-fitting implementation belongs to this phase.

Established: the [official metadata specification](https://github.com/Kaggle/kaggle-cli/wiki/Kernel-Metadata)
defines script, privacy, GPU, internet, dataset and previous-kernel source fields.
The [official dataset CLI documentation](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets.md)
states that create defaults to private. Our targets never request public creation.
Source review date: 2026-11-11. Existing dataset visibility must be checked by
the user before versioning; versioning does not guarantee a dataset is private.

Established alternatives rejected: the VS Code extension, agent-side authentication,
live experiment tracking, and notebook training logic conflict with the locked
workflow. Internet-enabled training is excluded. No hosted tracking service is
needed: local output folders plus hashed receipts suffice for Phase 1.

Plausible planning assumptions for the synthetic smoke config only: CPU, no GPU,
0 GB VRAM, 1 GB RAM, 0.1 session hours, 0 GPU-hours, 0.01 GB output, one session
if submitted. These are budget requests, not verified Kaggle limits. The job has
not been submitted. GPU type/VRAM, RAM, weekly quota, maximum duration, disk/output
limits and current account permissions remain **unverified**. Before a real run,
the user checks account/runtime limits, records the date and actual assumptions
in its typed budget, and budgets GPU-hours/session count. No fixed platform
quota is assumed here.

Established lineage layout: config hash, git commit, input version/content digest,
declared kernel version and hardware/library observations in `lineage.json`;
actual numeric Kaggle output version in the import receipt. A source tree change
requires a new code artifact and commit reference; uncommitted code is not a
reproducible training source. Phase 1 stores declared input hashes but does not
verify data contents: dataset manifests and hashing are Phase 2.

Established checkpoint boundary: atomic step/cursor snapshots and deterministic
seed replay are tested for the smoke stream. Future model jobs must add model
weights, optimizer/scheduler state, all RNG states, and data-loader/shard cursor
before claiming model-training resume. Resume must reject config mismatches.

Established policy requirement: features are not assumed anonymous. Private
hosting is not proof of GDPR or Riot-policy compliance. Rights to upload Riot
data, processing terms, retention and deletion of Kaggle versions/outputs require
review before real uploads. This phase uploads nothing. Minimal-retention
alternative: run locally, or upload only the needed pseudonymized shards for a
single private training run and remove remote data/output copies once validated
local artifacts are recovered, subject to confirmed deletion behavior. Exact
retention remains open; the raw-data 30–90-day window is not resolved by this ADR.
