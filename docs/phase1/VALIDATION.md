# Phase 1 local validation

Established, measured on 2026-11-11 using Python 3.11.15 on CPU:

- `make check`: Ruff lint/format passed, strict mypy passed, 10 offline tests passed.
- Pytest execution: 0.36 seconds in the recorded acceptance run. This is a tiny
  fixture check, not a data-processing or model-performance benchmark.
- `make smoke`: the CLI interrupted after four synthetic steps and resumed to
  eight steps successfully in a temporary directory.
- `make kernel-prepare`: generated private, internet-disabled script metadata and
  a syntactically valid bootstrap without contacting Kaggle.
- `make build`: all nine independent distribution wheels built successfully.
- Locked runtime requirements exported successfully for offline wheel preparation.

Established limitations: no credential was read; no Riot or Kaggle service was
called; no upload, account-limit verification, remote bootstrap execution, model
fitting or predictive evaluation was performed. The optional online wheel download
and actual Kaggle runtime compatibility remain user-run deployment preparation.

Established: foundation validation supports repository/workflow behavior only.
It does not support model quality, information-set completeness, coaching
effectiveness or production readiness. Phase 1 is complete; later phases remain
unstarted.
