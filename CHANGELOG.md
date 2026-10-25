# Changelog

## 2026.10 (2026-10-24)

Maintenance update. The code as printed in the book is tagged `first-edition`.

### Fixed
- `modal deploy serverless_multimodal_pipeline.py` failed at import on Modal 1.x
  (`container_idle_timeout` was removed; it is `scaledown_window`).
- The "parallel" encoders ran one after another (`.remote()` blocks); they now run in parallel
  (`.spawn()`), and concurrent requests make Modal scale out containers per service.
- Every container image uploaded the whole `sample-data/` folder (2.3 GB of X-rays no function read).
  It now carries only what the services need.
- `run_serverless_demo.py` counted every patient twice (it concatenated two identical files).
- `predictive_autoscaler.py` imported `openai` and `anthropic` it didn't use. Deploying it started a
  schedule every 5 minutes; the schedule now needs `ATLAS_ENABLE_SCHEDULE=1`.

### Changed
- **Real inputs, real models.**
  - Before: the radiology encoder was an untrained network over synthetic metadata, the vitals were
    random numbers, and the triage classifier was trained on labels computed from its own inputs.
  - Now the X-rays, lab results and notes are encoded by pretrained models (ResNet-18,
    all-MiniLM-L6-v2) with classifiers trained on their real labels.
  - The classifiers are tested on held-out records: pneumonia AUC 0.933, admission 0.708,
    pulmonary note 0.919.
  - `python train_models.py` trains them.
- **Measured instead of simulated.**
  - Before: the local demo slept for made-up cold starts and printed random scores and a fixed
    $750 monthly cost.
  - The local mode now runs the services on elastic worker processes (real cold starts,
    scale-out, scale to zero).
  - Both backends run the same benchmark and report measured latencies, cold starts,
    containers/workers, accuracy and cost at Modal's current rates.
  - `serverless_cost_analysis.json` and `serverless_scaling_results.json` hold measured results.
- **Real predictive autoscaling.**
  - Pre-warming sets `min_containers` on the deployed services (`autoscaler.update`) for a
    forecast surge.
  - The demos compare a surge at scaled-to-zero services with one at pre-warmed services.
  - Forecasts come from the first edition's schedule rules, evaluated for a given time instead of
    random metrics.
- **Real LLM advisor.**
  - Before: `local_llm_scaling_demo.py` and `comprehensive_llm_demo.py` printed canned "LLM"
    text from if-rules.
  - Now the same metrics go to the rules and to an open model (Qwen3-1.7B locally, Qwen3-4B on a
    Modal GPU).
  - The model's actions are checked against what the platform can do and compared with the rules.
- **GPU choice.**
  - `ATLAS_GPU` picks the radiology GPU (A100 40 GB by default) or `none` for CPU containers.
  - `ATLAS_ADVISOR_GPU` does the same for the advisor (L4 by default).
- **Install and packaging.**
  - Python 3.12+, with `pyproject.toml`, `uv.lock` and a pinned `requirements.txt`.
  - The first edition's install line also needs `torchvision`.

### Added
- `atlas_pipeline/` (shared code), `train_models.py`, `sample-data/README.md` (source and license of
  every file), smoke tests (`make smoke`), CI on Python 3.12 and 3.13, a Makefile.

### Removed (with the author's approval)
- `sample-data/chest_xray/chest_xray/`, a byte-identical second copy of the X-rays (1.15 GB), and
  `sample-data/telemetry_2.xlsx`, a byte-identical copy of `telemetry_1.xlsx` (checked by SHA-256).
- `GITHUB_SETUP.md` (notes from preparing the first upload).
