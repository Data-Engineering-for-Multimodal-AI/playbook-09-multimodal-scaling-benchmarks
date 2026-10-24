.PHONY: smoke train demo predictive modal-deploy modal-benchmark modal-predictive clean

# Smoke tests: local mode on CPU (about 10 minutes the first time)
smoke:
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
