# Serverless GPU Scaling Implementation - Summary

How the pipeline is built and how each number it prints is measured. For setup and results, see the
[README](README.md).

## Services

| Service | Input | Model (loaded once per container) | Output | Modal resources | Scale-down |
|---|---|---|---|---|---|
| `RadiologyService` | chest X-ray paths | ResNet-18 (ImageNet) embedding + logistic regression | probability of pneumonia | A100 40 GB (ATLAS_GPU), 8 GiB | 60 s idle |
| `LabsService` | blood-count vectors | logistic regression | probability of hospital admission | 2 CPU cores, 4 GiB | 30 s idle |
| `NotesService` | note texts | all-MiniLM-L6-v2 embedding + logistic regression | probability the note is cardiovascular/pulmonary | 4 CPU cores, 8 GiB | 45 s idle |

The three classifier heads are trained by `python train_models.py` on real labels and tested on
records they didn't see (pneumonia AUC 0.933, admission 0.708, pulmonary note 0.919). The triage
score is a weighted mean of the three probabilities (0.5, 0.3, 0.2); a case above 0.6 is urgent.
There is no joint label to train the weights on, because the three datasets describe different
patients.

The shared code is in `atlas_pipeline/`: `data.py`, `encoders.py`, `models.py`, `benchmark.py`,
`forecast.py`, `advisor.py`, and `local_runtime.py` for the local backend.

## Two backends, one benchmark

- **Modal** (`serverless_multimodal_pipeline.py`): each service is a Modal class. `@modal.enter`
  loads the model when a container starts. Modal starts more containers when requests queue up and
  stops idle ones after `scaledown_window`.
- **Local** (`atlas_pipeline/local_runtime.py`): each service is an `ElasticPool` of worker
  processes. A request that finds no free worker starts one, which loads its model; idle workers
  exit after the same scale-down windows; `prewarm(n)` is the local `min_containers`.

`atlas_pipeline/benchmark.py` drives either backend with the first edition's four scenarios:

- Morning imaging surge: 50 cases at once, all three services.
- Telemetry stream: 30 lab panels, one batch every 0.5 s.
- Evening mixed burst: 80 cases at once.
- Weekend light load: 15 cases, one batch per second.

Scenarios are separated by 30-second pauses. Requests carry batches of 5 cases.

## What is measured

- **Latency**: from sending a request to receiving its result, per request (p50, p95, max).
- **Cold start**: a request counts as cold if its container or worker finished loading after the
  request was sent. Per container, the model load time is measured inside it.
- **Workers/containers**: distinct containers (Modal task IDs) or worker processes that served a
  scenario.
- **Accuracy**: the services' predictions against the real labels of the held-out records.
- **Worker-seconds and cost**:
  - Local: each worker's lifetime, from start to exit.
  - Modal: estimated from each container's start to its last request plus the scale-down window.
  - Both are priced at Modal's published per-second rates (`modal billing rates`, recorded 2026-10-24
    in `benchmark.RATES`).
  - The billed amount for a Modal run is in `modal billing report`.

## Predictive autoscaling

`forecast.py` keeps the first edition's schedule rules:

- A PACS upload at 08:00 on weekdays (200 radiology cases).
- Shift changes at 07:00, 15:00 and 23:00 (dictation).
- Scheduled procedures at 09:00 on weekdays (75 mixed cases).

It also keeps the first edition's pre-warming policy: confidence of at least 0.6, 2 to 15 minutes
ahead, and GPU work or more than 100 cases.

`prewarm_plan(time)` turns the forecast into warm containers per service. On Modal,
`predictive_autoscaler.py` applies the plan with `autoscaler.update(min_containers=n)` on the
deployed services and sets it back to 0 afterwards.

The demos run the same kind of surge twice, once at scaled-to-zero services and once at pre-warmed
ones, and compare latency and cold starts. On Modal (CPU containers), p95 fell from 14.7 s to 2.0 s
for radiology and from 18.3 s to 2.5 s for notes; see `sample_autoscaler_output.md`.

## LLM scaling advisor

`advisor.py` sends the same scenario metrics to the first edition's rules and to an open language
model. The model is Qwen3-1.7B by default on a laptop, and Qwen3-4B on a Modal GPU.

The model must answer with JSON in the chapter's format, `{"prediction": ..., "action": ...}`. Each
action must be one the platform can carry out, or it is rejected. Agreement with the rules is scored
as precision and recall.

At these model sizes the advisor matches the rules on clear signals but also proposes unneeded
scaling. Treat its output as advice to check, not commands to execute.

## Costs and the schedule

Containers that are not running cost nothing; Modal bills running containers per second.

Deploying `predictive_autoscaler.py` with `ATLAS_ENABLE_SCHEDULE=1` creates a job that runs every
5 minutes and can keep containers warm. It costs money until it is stopped
(`modal app stop atlas-health-predictive-autoscaler`). Without that variable, the deploy creates no
schedule.
