# ADR 0001 — Small Python workspace and inward contracts

Status: accepted for Phase 1.

Established decision: nine independently buildable distributions share the
`offlinerift` namespace. One root `uv.lock`, Python 3.11 minimum, Ruff, strict
mypy, pytest and local pre-commit hooks provide the development gates.
CI runs CPU checks and a CLI smoke test without credentials.

Established dependency rule: domain imports only typing and pydantic. Outer
packages depend on domain; training also composes the storage adapter.
No outer package may be imported into domain. AST tests enforce the current
small graph. The adapters are injected through typed Protocol contracts.

Established alternatives rejected: a service per package, orchestration platform,
tracking server and large ML dependency stack add no Phase 1 value. A single
distribution would be smaller but would not meet the brief's independent-package
requirement. Empty package boundaries are scaffolds, not implemented subsystems.

Plausible consequence: a namespace workspace keeps later work separable while
maintaining one toolchain. Revisit the layout only when a concrete import or
deployment requirement demonstrates friction.

Established: strict, frozen V1 pydantic contracts reject undeclared fields.
Breaking schema changes require a new version and acceptance tests. Dataset
identity is match ID plus participant slot; no PUUID is accepted by the key.
The foundation review contract abstains unconditionally.
