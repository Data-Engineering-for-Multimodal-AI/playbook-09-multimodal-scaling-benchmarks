.PHONY: setup run smoke smoke-full train demo predictive triage check-data score try-scaledown-window try-batch-size try-max-workers modal-deploy modal-benchmark modal-predictive clean

# A bare `make` runs the smoke test, as it did before the uniform targets were added.
.DEFAULT_GOAL := smoke

# Uniform targets for The Playbooks: setup, run, smoke, smoke-full and clean. The book's commands
# (pip install -r requirements.txt, python train_models.py, python run_serverless_demo.py,
# modal deploy serverless_multimodal_pipeline.py, modal run ...) still work as printed.

# Install the locked dependencies, then train the three models if models/ doesn't have them yet.
setup:
	uv sync --frozen
	@if [ -f models/radiology.joblib ] && [ -f models/labs.joblib ] && [ -f models/notes.joblib ]; then \
		echo "Models already trained (models/)"; else uv run --frozen python train_models.py; fi

# The four clinical scenarios on local elastic workers, small and fast (--quick). No Modal account.
# Runs from results/, so serverless_cost_analysis.json goes to results/ and the committed copy at
# the repository root is not rewritten.
run: setup
	mkdir -p results
	cd results && uv run --frozen python ../run_serverless_demo.py --quick

# Score one case by hand: make triage XRAY=path/to/xray.jpeg NOTE="a clinical note" (either or both)
triage: setup
	@uv run --frozen python scripts/triage.py $(if $(XRAY),--xray "$(XRAY)") $(if $(NOTE),--note "$(NOTE)")

# Your own cases: cases.csv and xrays/ in DIR (see README, "Use your own data").
# check-data checks every file and runs nothing; score runs the cases on local elastic workers.
#   make check-data DIR=my_cases
#   make score DIR=my_cases
check-data:
	@test -n "$(DIR)" || { echo "Usage: make check-data DIR=path/to/folder"; exit 2; }
	@uv run --frozen python scripts/own_cases.py --dir "$(DIR)" --check-only

score: setup
	@test -n "$(DIR)" || { echo "Usage: make score DIR=path/to/folder"; exit 2; }
	@uv run --frozen python scripts/own_cases.py --dir "$(DIR)"

# Try this: three measured experiments on local workers (see README, "Try this"). No Modal account.
try-scaledown-window: setup
	@uv run --frozen python experiments/scaledown_window.py

try-batch-size: setup
	@uv run --frozen python experiments/batch_size.py

try-max-workers: setup
	@uv run --frozen python experiments/max_workers.py

# Smoke tests: local mode on CPU. Skips the slow local LLM advisor test (make smoke-full runs it).
smoke:
	uv run pytest -q -m "not slow"

# All smoke tests, including the local LLM advisor demo (Qwen3-1.7B on CPU, up to an hour)
smoke-full:
	uv run pytest -q

train:
	uv run python train_models.py

# The four clinical scenarios on local elastic workers
demo:
	uv run python run_serverless_demo.py

predictive:
	uv run python run_serverless_demo.py --predictive

# Modal (needs `modal setup`; GPUs need a payment method on the Modal account)
modal-deploy: train
	uv run modal deploy serverless_multimodal_pipeline.py

modal-benchmark:
	uv run modal run serverless_multimodal_pipeline.py::simulate_clinical_workload_bursts

modal-predictive:
	uv run modal run predictive_autoscaler.py::demo_predictive_autoscaling

clean:
	rm -rf .cache results models .pytest_cache
