# offlinerift-models

Established: independently buildable Python distribution with typed namespace
`offlinerift.models`. Install with `uv pip install ./packages/models` after
building/installing its declared sibling dependencies, or use `uv sync --all-packages`.

Established: Phase 1 contains only contracts and workflow plumbing. Domain has
no I/O; outer adapters depend on domain. No collection, features, model fitting,
evaluation claims, inference service, or product UI is implemented here.

Validation: `make check` from the repository root; see `docs/phase1/SPEC.md`.
