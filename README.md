# OfflineRift

Established: a post-game League of Legends decision-review coach, currently at
**Phase 1: foundation only**. Official Riot data only; no live features.
Kaggle supplies remote training compute through its CLI, using only the owner's
privately uploaded Riot data. No third-party Kaggle datasets are imported.

Established commands (Python 3.11+, `uv`, GNU Make):

```sh
make setup
make check
make smoke
make build
```

Established: `make setup` installs the locked workspace and pre-commit hooks.
Tests use tiny synthetic inputs, block socket connections, and require no
credentials, Riot calls, Kaggle calls, GPU, or real dataset. Dependency installation
needs network access once; test execution does not.

Established: the runner is a deterministic infrastructure smoke test, not a
model trainer. Its fixture has explicit zero placeholders for commit/data hashes;
these are not provenance claims. Real runs require real immutable references.

See [Phase 1 spec](docs/phase1/SPEC.md), [ADRs](docs/adr/0001-foundation.md),
[Kaggle workflow](docs/phase1/KAGGLE.md), [implementation tasks and backlog](docs/phase1/TASKS.md),
and [repository instructions](AGENTS.md).

Established: Phase 2 collection/storage pipelines, baseline and recommendation
models, transfer experiments, and the web application have not been implemented.
