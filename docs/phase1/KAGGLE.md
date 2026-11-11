# User-run Kaggle training workflow

Established: all authenticated commands below are run by the user locally.
Do not paste credentials into configs, shell arguments, this repository, or
agent-visible files. CLI credentials stay outside the workspace. CI never invokes
Kaggle. Install/authenticate the official Kaggle CLI yourself; its version is an
external prerequisite, not a training dependency.

Established: Kaggle is compute for your own Riot data. Private datasets are only
the transport for your data and offline runtime wheels, not a source of external
training data. Phase 1 needs no real data and performs no upload.

1. Established: run `make check smoke build` locally. For a real future run,
   commit code and replace smoke placeholders with actual commit, data manifest
   digest/version, run ID, owner/kernel ID, declared submission version and
   verified compute assumptions in a new JSON config. The input manifest and
   pseudonymization pipeline belong to Phase 2.
2. Established: run `make wheelhouse` on a Python/platform matching the selected
   Kaggle runtime. It builds our packages and downloads pinned dependencies from
   `uv.lock` with hashes. It requires network during preparation only. Upload
   just `build/runtime/wheels` plus reviewed dataset metadata as a private runtime
   dataset. Prepare `dataset-metadata.json` manually in `build/runtime/` using
   the [official format](https://github.com/Kaggle/kaggle-cli/wiki/Dataset-Metadata),
   choosing an appropriate license only after checking distribution rights.
   Never upload the repository wholesale. No credentials belong in the bundle.
3. Established: for each reviewed staging directory, use
   `make kaggle-dataset-create DATASET_DIR=...`, or
   `make kaggle-dataset-version DATASET_DIR=... MESSAGE='...'` for an existing
   **verified private** dataset. Upload only your required pseudonymized Riot
   inputs; no raw player identifiers or local identity map.
4. Established: set `dataset_sources` to your private runtime/data datasets;
   set `wheelhouse` to their actual Kaggle mount's `wheels` directory. The
   metadata sources identify Kaggle inputs, while `dataset.version`/hash and
   `kernel_version` record exact intended versions. Kaggle metadata attachments
   alone do not enforce a content digest; real input verification is Phase 2.
5. Established: `make kernel-prepare CONFIG=configs/your-run.json` writes only
   `build/kernel/entry.py` and generated `kernel-metadata.json`. Inspect them.
   `make kaggle-push` submits the script. It bootstraps with `pip --no-index`,
   then invokes the same typed module entry point as local execution. Internet
   is off, kernel private, GPU controlled by config. No notebook logic.
6. Established: poll `make kaggle-status KERNEL_ID=owner/slug` manually, then
   download with `make kaggle-output KERNEL_ID=owner/slug`. Do not push another
   version while fetching the current run's outputs. Record the actual numeric
   version from Kaggle; `make outputs-sync KERNEL_VERSION=1` imports the allowlisted
   artifacts into `tracking/<run-id>/1/` and writes a content-hashed receipt.
   This folder is local experiment tracking; there is no live remote connection.
7. Established: for resumption, preserve the exact computation settings and attach
   the prior output through `kernel_sources`; update the declared `kernel_version`.
   Use `make kernel-prepare CONFIG=... RESUME=/kaggle/input/.../artifacts/checkpoint.json`.
   The complete config hash changes and is recorded in lineage. A separate resume
   hash excludes only `kernel_sources` and `kernel_version`, allowing a new
   submission to attach the previous output. All other config changes are rejected
   on resume; record each actual Kaggle version in receipts.
   Full model/optimizer/RNG/data-cursor checkpoint support is a later-phase task.

Established artifact layout:

```text
/kaggle/working/run-config.json
/kaggle/working/artifacts/
  lineage.json       # source/config/input refs + observed hardware/libraries
  checkpoint.json    # atomic cursor/seed-stream state (smoke only)
  metrics.json       # progress, explicitly model_trained=false
  events.jsonl       # structured completion record
tracking/<run-id>/<actual-kernel-version>/
  lineage.json, checkpoint.json, metrics.json, events.jsonl, receipt.json
```

Established: all quotas, available GPU/VRAM, session duration, output limits and
remote deletion behavior require account/runtime verification. No real Kaggle
session or training result has been validated in Phase 1.
