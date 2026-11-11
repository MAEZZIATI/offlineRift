UV ?= uv
RUN = $(UV) run --frozen --all-packages
CONFIG ?= configs/smoke.json
KERNEL_DIR ?= build/kernel
DOWNLOAD_DIR ?= runs/download
TRACKING_DIR ?= tracking

.PHONY: setup check lint types test smoke build wheelhouse kernel-prepare outputs-sync \
 kaggle-dataset-create kaggle-dataset-version kaggle-push kaggle-status kaggle-output

setup:
	$(UV) sync --frozen --all-packages
	$(UV) run --frozen pre-commit install

check: lint types test

lint:
	$(RUN) ruff check .
	$(RUN) ruff format --check .

types:
	$(RUN) mypy

test:
	$(RUN) pytest

smoke:
	$(RUN) python scripts/smoke.py $(CONFIG)

build:
	$(UV) build --all-packages --wheel --out-dir build/runtime/wheels

# Online dependency download, no credentials; user builds for Kaggle's Python/platform.
wheelhouse: build
	$(UV) export --frozen --all-packages --no-dev --no-emit-workspace --format requirements-txt --output-file build/runtime/requirements.txt
	$(UV) run --with pip==25.3 python -m pip download --require-hashes --only-binary=:all: -r build/runtime/requirements.txt -d build/runtime/wheels

kernel-prepare:
	$(RUN) python -m offlinerift.training.workflow prepare --config $(CONFIG) --destination $(KERNEL_DIR) $(if $(RESUME),--resume $(RESUME),)

outputs-sync:
	@test -n "$(KERNEL_VERSION)" || (echo 'Set KERNEL_VERSION from Kaggle status/output'; exit 1)
	$(RUN) python -m offlinerift.training.workflow sync --source $(DOWNLOAD_DIR)/artifacts --tracking $(TRACKING_DIR) --kernel-version $(KERNEL_VERSION)

# USER-RUN ONLY: these invoke an authenticated CLI. Agent/tests/CI never run them.
# DATASET_DIR contains only your own reviewed pseudonymized Riot data OR runtime wheels.
# New datasets default to PRIVATE; never add --public. Verify privacy before versioning.
kaggle-dataset-create:
	@test -n "$(DATASET_DIR)" || (echo 'Set DATASET_DIR to a reviewed private staging folder'; exit 1)
	kaggle datasets create -p "$(DATASET_DIR)"

kaggle-dataset-version:
	@test -n "$(DATASET_DIR)" -a -n "$(MESSAGE)" || (echo 'Set DATASET_DIR and MESSAGE'; exit 1)
	kaggle datasets version -p "$(DATASET_DIR)" -m "$(MESSAGE)"

kaggle-push:
	kaggle kernels push -p "$(KERNEL_DIR)"

kaggle-status:
	@test -n "$(KERNEL_ID)" || (echo 'Set KERNEL_ID=owner/slug'; exit 1)
	kaggle kernels status "$(KERNEL_ID)"

kaggle-output:
	@test -n "$(KERNEL_ID)" || (echo 'Set KERNEL_ID=owner/slug'; exit 1)
	kaggle kernels output "$(KERNEL_ID)" -p "$(DOWNLOAD_DIR)"
