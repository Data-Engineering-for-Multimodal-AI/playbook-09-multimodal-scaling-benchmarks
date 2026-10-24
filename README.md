# Serverless Multimodal Medical Pipeline

**Chapter 9 playbook** from *Data Engineering for Multimodal AI* (O'Reilly, 2026). Page numbers to be confirmed.
Book: https://oreil.ly/data-engineering-multimodal

> Want the code exactly as printed? Use the `first-edition` tag. This branch is the 2026 maintenance update; see [Differences from the printed book](#differences-from-the-printed-book).

## What you'll build

A multimodal triage pipeline for Atlas Health, a fictional hospital network, that scales elastically: GPU capacity appears when a burst of X-rays arrives and disappears when it's done, so you pay only while it works. You run it two ways, on Modal's serverless GPUs and on your own machine, and measure what elasticity costs and buys.

- **Three services, three modalities, real data**:
  - Chest X-rays (ResNet-18 plus a pneumonia classifier, on a GPU).
  - Hospital lab results (an admission-risk classifier).
  - Clinical notes (MiniLM plus a pulmonary-note classifier).
  - Each is trained on real labels and tested on held-out records.
- **Elastic scaling**: each service scales out with demand, loads its model once per container (the cold start), and scales to zero after an idle window.
- **The same benchmark on two backends**:
  - Workloads: the chapter's four clinical scenarios (morning imaging surge, telemetry stream, evening mixed burst, weekend light load).
  - Backends: Modal containers, or local elastic worker processes.
  - Measured for each: latency, cold starts, containers, accuracy and cost at Modal's rates.
- **Predictive autoscaling**: forecast surges from the hospital schedule, pre-warm containers before they arrive, and measure the difference.
- **An LLM scaling advisor**: a small open language model reads the same signals as the chapter's rules. Its recommendations are checked and compared with the rules.

## Quick start

You need Python 3.12 or newer and about 4 GB of disk space (the environment, the 1.2 GB of X-rays and the pretrained encoders), plus 4 GB for the LLM demos.

```bash
# Clone the repository
git clone https://github.com/Data-Engineering-for-Multimodal-AI/playbook-09-multimodal-scaling-benchmarks.git
cd playbook-09-multimodal-scaling-benchmarks

# Create and activate a virtual environment, then install dependencies
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The book's install line still works with one addition, `torchvision` (for the X-ray encoder):
`pip install modal sentence-transformers torch scikit-learn pandas openpyxl torchvision`. With [uv](https://docs.astral.sh/uv/): `uv sync`, then prefix commands with `uv run`.

### 1. Train the three models (once)

```bash
python train_models.py
```

This embeds about 1,600 X-rays and 5,000 notes and trains the classifier heads (about 2 minutes on a laptop CPU). It downloads ResNet-18 (45 MB) and all-MiniLM-L6-v2 (90 MB) and writes `models/`, which both backends load.

### 2. Run it locally

```bash
python run_serverless_demo.py                 # the four clinical scenarios (about 5 minutes)
python run_serverless_demo.py --predictive    # a surge at scaled-to-zero workers vs pre-warmed ones
python local_llm_scaling_demo.py              # the scaling advisor: rules vs Qwen3-1.7B
python comprehensive_llm_demo.py              # the same, all five scenarios
```

Each service runs as a pool of worker processes that behaves like a serverless function: a request with no free worker starts one (it loads its model: a cold start), and idle workers exit after the service's scale-down window. `--quick` runs smaller scenarios with short windows. The LLM demos download Qwen3-1.7B (about 4 GB) and need about 4 GB of free memory.

### 3. Run it on Modal

You need a [Modal](https://modal.com) account. Run `modal setup` once. GPUs need a payment method on the account; without one, use `ATLAS_GPU=none` (below).

```bash
modal deploy serverless_multimodal_pipeline.py
modal run serverless_multimodal_pipeline.py::multimodal_inference_dag --cases 10
modal run serverless_multimodal_pipeline.py::simulate_clinical_workload_bursts
modal run predictive_autoscaler.py::demo_predictive_autoscaling
modal run predictive_autoscaler.py::demo_llm_scaling_advisor
modal billing report --for today                # what it cost
```

- **Services**: radiology runs on an A100 40 GB, labs on 2 CPU cores, notes on 4 CPU cores. The predictive demo uses the deployed services.
- **Choosing hardware**:
  - `ATLAS_GPU=T4` (or `L4`, `A10G`, `H100`, ...) picks another GPU for radiology, and `ATLAS_GPU=none` runs it on 4 CPU cores.
  - The advisor uses an L4; `ATLAS_ADVISOR_GPU=none` runs it on CPU.
  - Set these variables for `modal deploy` and `modal run` alike.
- **Schedule**: `modal deploy predictive_autoscaler.py` deploys the autoscaler agent without a schedule. `ATLAS_ENABLE_SCHEDULE=1` adds the every-5-minutes schedule, which pre-warms containers and costs money until you run `modal app stop atlas-health-predictive-autoscaler`.

All the Modal commands are in [`modal_deployment_commands.txt`](modal_deployment_commands.txt); how the services and measurements work is in [`SERVERLESS_IMPLEMENTATION_SUMMARY.md`](SERVERLESS_IMPLEMENTATION_SUMMARY.md).

### Environment variables

| Variable | Default | What it sets |
|---|---|---|
| `ATLAS_GPU` | `A100-40GB` | GPU for the radiology service on Modal (`none` = 4 CPU cores) |
| `ATLAS_ADVISOR_GPU` | `L4` | GPU for the advisor on Modal (`none` = 8 CPU cores) |
| `ATLAS_MODAL_ADVISOR_MODEL` | `Qwen/Qwen3-4B` | Advisor model on Modal |
| `ATLAS_ADVISOR_MODEL` | `Qwen/Qwen3-1.7B` | Advisor model locally |
| `ATLAS_XRAY_TRAIN_PER_CLASS` | `500` | X-rays per class used to train the pneumonia model |
| `ATLAS_ENABLE_SCHEDULE` | unset | `1` deploys the autoscaler with its 5-minute schedule |

No keys are needed other than your Modal login (`modal setup`).

## Tested on

| | |
|---|---|
| Date | 2026-10-24 |
| Local | macOS 26.5.1, Apple M3 (8 cores, 8 GB), Python 3.12.15 |
| CI | Ubuntu 24.04 (x86-64, 4 cores), Python 3.12.3 and 3.13.17 |
| Modal | client 1.6.2, cloud CPU containers (see the note below) |
| Key packages | torch 2.14.1, torchvision 0.29.1, sentence-transformers 6.2.0, transformers 5.20.2, scikit-learn 1.9.1, pandas 3.1.0, modal 1.6.2 |

Fresh clone on the Mac, `pip install -r requirements.txt`: install 62 s, `python train_models.py` 2 min 56 s, `python run_serverless_demo.py` 3 min 3 s, `--predictive` 1 min 30 s. `make smoke` in CI: about 10 minutes, including the model downloads.

**What was not run on a GPU.** The Modal account used for testing had no payment method, and Modal allows GPU functions only with one. So the Modal runs below used `ATLAS_GPU=none` (radiology on 4 CPU cores) and `ATLAS_ADVISOR_GPU=none` (the advisor on 8 CPU cores). With a GPU, radiology cold starts and compute times will differ, and the cost will be higher (an A100 40 GB is $2.10 per hour).

The local LLM demos did not finish on the 8 GB Mac: Qwen3-1.7B was swapped out while other apps were running. Their output below is from CI (Linux, 16 GB). Allow at least 8 GB of free memory, or use the Modal version.

The billed cost of every Modal run made while testing (deploys, image builds, the DAG, benchmark and demo runs, advisor runs) was $0.85 (`modal billing report --for today`). An earlier run of the four-scenario benchmark was billed $0.1325, against $0.1276 estimated from measured container time; the run below was estimated at $0.1140.

## Differences from the printed book

| Printed book | This version | Why |
|---|---|---|
| Python 3.8+; `pip install modal sentence-transformers torch scikit-learn pandas openpyxl` | Python 3.12+; `pip install -r requirements.txt` (pinned), or the book's line plus `torchvision` | Reproducible installs; the X-ray encoder needs torchvision |
| `container_idle_timeout=` | `scaledown_window=` | Removed in Modal 1.0; the app failed at import |
| Radiology "encoder": an untrained network over synthetic study metadata | ResNet-18 embeddings of the chest X-rays plus a pneumonia classifier (test AUC 0.93) | The 5,856 X-rays in the repository were never used; random weights encode nothing |
| Telemetry: lab values plus random heart rate and SpO2 | Blood counts of the 603 patients who have them, predicting admission (test AUC 0.71) | No random inputs |
| Triage classifier trained on labels computed from its own inputs | Per-modality classifiers trained on real labels; triage = weighted mean of their probabilities | The old score was meaningless |
| Encoders called with `.remote()` in sequence | `.spawn()` in parallel; concurrent requests | `.remote()` blocks, so nothing ran in parallel |
| `run_serverless_demo.py`: slept for "cold starts", random scores, fixed $750 monthly cost | Local elastic worker processes; measured latency, cold starts, workers, accuracy and cost | Show real behavior, not a simulation |
| `multimodal_inference_dag` took hand-written JSON records | `--cases N` held-out cases (X-ray, labs, note) | The services take real X-rays |
| Predictive autoscaler: random metrics; pre-warmed a separate A100 function | Schedule-based forecasts for a given time; sets `min_containers` on the deployed services; measured before/after | Pre-warm what serves the traffic, and show the effect |
| `modal deploy predictive_autoscaler.py` started a 5-minute schedule | Schedule only with `ATLAS_ENABLE_SCHEDULE=1` | Avoid recurring cost by surprise |
| "LLM" advisor: if-rules with canned reasoning text and fixed confidences | Rules and a real open model (Qwen3-1.7B / 4B) on the same metrics; actions validated; agreement scored | Make the comparison real |
| `openai`, `anthropic` in the autoscaler image (unused) | Removed | Not used |
| Container images included all of `sample-data/` (2.3 GB) | The 624 test X-rays, trained heads and pretrained weights | Faster builds and cold starts |
| `sample-data/chest_xray/chest_xray/`, `telemetry_2.xlsx` | Removed | Byte-identical copies (checked by SHA-256) |

The full list is in [CHANGELOG.md](CHANGELOG.md).

## Data and licenses

| Data | Source | License |
|---|---|---|
| `sample-data/chest_xray/` (5,856 chest X-rays) | Kermany, Zhang and Goldbaum, Mendeley Data 2018 (Kaggle: paultimothymooney/chest-xray-pneumonia) | CC BY 4.0 |
| `sample-data/telemetry_1.xlsx` (5,644 patients, lab results) | Hospital Israelita Albert Einstein, Kaggle: einsteindata4u/covid19 | As published on Kaggle by the hospital |
| `sample-data/mtsamples.csv` (4,999 transcriptions) | Kaggle: tboyle10/medicaltranscriptions (from mtsamples.com) | CC0 1.0 on Kaggle |
| `sample-data/radiology_metadata.csv` | Synthetic, created by the author | MIT |
| ResNet-18 ImageNet weights | torchvision | BSD-3-Clause |
| all-MiniLM-L6-v2 | sentence-transformers (Hugging Face) | Apache-2.0 |
| Qwen3-1.7B, Qwen3-4B | Qwen (Hugging Face) | Apache-2.0 |

Details, and how each file is used, are in [`sample-data/README.md`](sample-data/README.md). The three datasets describe different people; a triage case pairs one held-out record of each. The pairing is synthetic, but the records and labels are real. Atlas Health is fictional. The models are teaching examples, not medical devices.

## Expected output

All from runs on 2026-10-24. Local runs: fresh clone, macOS 26.5.1 (Apple M3), Python 3.12.15. Modal runs: Modal 1.6.2, CPU containers (`ATLAS_GPU=none`). Download and progress messages are left out. Latencies and cold starts depend on the machine and load; the scores and labels don't.

**`python train_models.py`**

```
Test scores (records no model trained on):
  radiology pneumonia vs normal chest X-ray          AUC 0.933  accuracy 0.840  F1 0.884  (624 records, 390 positive)
  labs      hospital admission from blood counts     AUC 0.708  accuracy 0.646  F1 0.418  (181 records, 39 positive)
  notes     cardiovascular/pulmonary note            AUC 0.919  accuracy 0.867  F1 0.488  (994 records, 74 positive)
```

**`python run_serverless_demo.py`** (local elastic workers)

```
Atlas Health Serverless Multimodal Pipeline Demo (local)
============================================================
Models (test scores on records they didn't train on):
   radiology pneumonia vs normal chest X-ray          AUC 0.933  accuracy 0.840
   labs      hospital admission from blood counts     AUC 0.708  accuracy 0.646
   notes     cardiovascular/pulmonary note            AUC 0.919  accuracy 0.867
Morning imaging surge - GPU intensive (50 cases, burst)
  50 cases in 12.3 s; urgent: 20; mean triage score 0.492
  modality   requests workers  cold   p50 s   p95 s   max s accuracy
  radiology        10       3    10    7.89    8.61    8.72    0.860
  labs             10       1    10    2.12    2.13    2.14    0.580
  notes            10       2    10   11.94   12.23   12.24    0.980
  pausing 30 s (idle workers scale down after their scaledown window)
Telemetry stream - CPU steady state (30 cases, stream)
  30 cases in 2.5 s; urgent: 4; mean triage score 0.350
  modality   requests workers  cold   p50 s   p95 s   max s accuracy
  labs              6       2     4    0.44    1.57    1.70    0.800
  pausing 30 s (idle workers scale down after their scaledown window)
Evening mixed burst - Multi-modal (80 cases, burst)
  80 cases in 13.5 s; urgent: 43; mean triage score 0.543
  modality   requests workers  cold   p50 s   p95 s   max s accuracy
  radiology        16       3    16    8.18    9.23    9.40    0.875
  labs             16       1     0    0.00    0.02    0.02    0.662
  notes            16       2    16   12.88   13.53   13.53    0.825
  pausing 30 s (idle workers scale down after their scaledown window)
Weekend light load - Minimal scaling (15 cases, stream)
  15 cases in 2.2 s; urgent: 8; mean triage score 0.530
  modality   requests workers  cold   p50 s   p95 s   max s accuracy
  radiology         3       1     0    0.21    0.26    0.27    0.933
  labs              3       1     2    0.73    1.64    1.74    0.600
  notes             3       1     0    0.14    0.19    0.20    0.933
Waiting for every worker to scale down...
  modality   started cold start s  worker-s  at Modal rates
  radiology        6         6.42     450.7         $0.2717
  labs             7         1.84     254.7         $0.0090
  notes            4        11.63     264.4         $0.0186
  total $0.2993 at Modal's rates of 2026-10-24 (radiology on A100-40GB)
Results saved to results/local_benchmark.json and serverless_cost_analysis.json
```

How to read it:
- **cold**: requests that waited for a worker to start and load its model. In the first burst every request did. The 30-second pauses outlast the labs window (30 s), so labs workers scale down between scenarios, and the radiology workers (60 s window) are gone by the evening burst.
- **accuracy**: the services' predictions against the real labels of held-out records.
- **at Modal rates**: what these worker-seconds would cost as Modal containers with the same resources (radiology priced as an A100). Most of it is idle time before scale-down, which is why scale-down windows matter.

**`python run_serverless_demo.py --predictive`**

```
1. Reactive: the surge arrives at scaled-to-zero workers
  50 cases in 15.6 s; urgent: 20; mean triage score 0.492
  modality   requests workers  cold   p50 s   p95 s   max s accuracy
  radiology        10       3    10    8.75    9.58    9.78    0.860
  labs             10       2    10    3.06    3.08    3.09    0.580
  notes            10       2    10   14.93   15.49   15.57    0.980
   waiting for the workers to scale down...
2. Predictive: at Tue 08:50 the schedule predicts:
   scheduled_procedures: 75 cases at 09:00 (confidence 0.90) -> pre-warm: True
   pre-warming {'radiology': 2, 'labs': 1, 'notes': 2}...
   warm after 14 s
  50 cases in 2.7 s; urgent: 28; mean triage score 0.553
  modality   requests workers  cold   p50 s   p95 s   max s accuracy
  radiology        10       2     0    1.56    2.64    2.64    0.860
  labs             10       1     0    0.02    0.03    0.03    0.740
  notes            10       2     0    1.51    2.43    2.44    0.820
            reactive p95 s  pre-warmed p95 s  cold starts
radiology             9.58              2.64    10 -> 0   
labs                  3.08              0.03    10 -> 0   
notes                15.49              2.43    10 -> 0   
Surge wall time: 15.6 s reactive, 2.7 s pre-warmed
Results saved to results/local_predictive.json
```

**`modal run serverless_multimodal_pipeline.py::multimodal_inference_dag --cases 10`** (`ATLAS_GPU=none`)

```
Starting multimodal inference DAG for 10 cases
   radiology cold container ta-01M5QWZC1DHE1CHN089YWJ389R, model load 11.8 s, compute 0.47 s
   labs      cold container ta-01M5QWZCXNX9CXGKQFGH7QYT7R, model load 3.2 s, compute 0.02 s
   notes     cold container ta-01M5QWZFKE18BS8HPMKT2WE0TR, model load 12.8 s, compute 0.35 s
DAG completed in 21.09s
   C0000: radiology 1.000, labs 0.355, notes 0.197 -> triage 0.646  URGENT
   C0001: radiology 0.960, labs 0.552, notes 0.102 -> triage 0.666  URGENT
   C0002: radiology 0.000, labs 0.274, notes 0.960 -> triage 0.274
   C0003: radiology 0.013, labs 0.468, notes 0.079 -> triage 0.163
   C0004: radiology 0.408, labs 0.530, notes 0.932 -> triage 0.549
   C0005: radiology 0.996, labs 0.611, notes 0.124 -> triage 0.706  URGENT
   C0006: radiology 0.375, labs 0.292, notes 0.086 -> triage 0.292
   C0007: radiology 0.746, labs 0.271, notes 0.020 -> triage 0.458
   C0008: radiology 0.991, labs 0.675, notes 0.155 -> triage 0.729  URGENT
   C0009: radiology 0.984, labs 0.306, notes 0.122 -> triage 0.608  URGENT
```

The three services ran in parallel, each on a fresh (cold) container; the 21 s is mostly model loading.

**`modal run serverless_multimodal_pipeline.py::simulate_clinical_workload_bursts`** (`ATLAS_GPU=none`)

```
Atlas Health Clinical Workload Simulation (Modal)
============================================================
Radiology: {'gpu': None, 'cpu': 4}
Morning imaging surge - GPU intensive (50 cases, burst)
  50 cases in 15.6 s; urgent: 20; mean triage score 0.492
  modality   requests workers  cold   p50 s   p95 s   max s accuracy
  radiology        10       2    10   12.65   12.96   13.02    0.860
  labs             10       1    10    5.50    5.50    5.50    0.580
  notes            10       2    10   15.27   15.60   15.60    0.980
  pausing 30 s (idle workers scale down after their scaledown window)
Telemetry stream - CPU steady state (30 cases, stream)
  30 cases in 2.8 s; urgent: 4; mean triage score 0.350
  modality   requests workers  cold   p50 s   p95 s   max s accuracy
  labs              6       6     0    0.21    0.28    0.29    0.800
  pausing 30 s (idle workers scale down after their scaledown window)
Evening mixed burst - Multi-modal (80 cases, burst)
  80 cases in 1.5 s; urgent: 43; mean triage score 0.543
  modality   requests workers  cold   p50 s   p95 s   max s accuracy
  radiology        16       3     0    0.99    1.44    1.45    0.875
  labs             16       1     0    0.51    0.61    0.61    0.662
  notes            16       4     0    0.76    1.16    1.26    0.825
  pausing 30 s (idle workers scale down after their scaledown window)
Weekend light load - Minimal scaling (15 cases, stream)
  15 cases in 2.3 s; urgent: 8; mean triage score 0.530
  modality   requests workers  cold   p50 s   p95 s   max s accuracy
  radiology         3       3     0    0.42    0.58    0.60    0.933
  labs              3       3     0    0.22    0.26    0.26    0.600
  notes             3       3     0    0.30    0.32    0.32    0.933
Waiting for every worker to scale down...
  modality   started cold start s  worker-s  at Modal rates
  radiology        6         8.21     686.3         $0.0483
  labs             8         2.64     621.4         $0.0219
  notes            5        12.61     623.4         $0.0438
  total $0.1140 at Modal's rates of 2026-10-24 (radiology on 4 CPU cores)
Results saved to results/modal_benchmark.json and serverless_scaling_results.json
Billed cost (appears within a few minutes): modal billing report --for today
```

Only the first surge met cold containers: Modal kept containers a little past their scale-down windows through the 30-second pauses, so later requests found them warm (well under 1.5 s). The platform also decides how many containers to use: the telemetry stream's six requests went to six containers.

**`modal run predictive_autoscaler.py::demo_predictive_autoscaling`** and **`demo_llm_scaling_advisor`**: see [`sample_autoscaler_output.md`](sample_autoscaler_output.md). A surge at scaled-to-zero services had a p95 latency of 14.7 s (radiology) and 18.3 s (notes); after pre-warming for the forecast surge, 2.0 s and 2.5 s.

**`python local_llm_scaling_demo.py`** (CI, Ubuntu 24.04, Python 3.12.3)

```
Atlas Health LLM-Based Scaling Advisor Demo
==================================================

LLM advisor: Qwen/Qwen3-1.7B on cpu (loaded in 1.4 s)

Scenario: morning_pacs_surge
  Rules:
    {"prediction": "Radiology surge in 5 minutes", "action": "Pre-warm 2 A100 GPUs"}
    {"prediction": "Immediate radiology processing needed", "action": "Pre-warm 1 A100 GPU"}
  LLM (89.0 s):
    {"prediction": "15 minutes from now", "action": "Pre-warm 1 A100 GPU"}
    {"prediction": "10 minutes from now", "action": "Emergency scale: 3 A100 GPUs + 12 CPU cores"}
  Agreement with the rules: 1 of the LLM's 2 actions (precision 0.50), 1 of the rules' 2 (recall 0.50)

Scenario: shift_change_dictation
  Rules:
    {"prediction": "Text processing surge in 15 minutes", "action": "Scale CPU text processors to 6 cores"}
  LLM (94.1 s):
    {"prediction": "14:00", "action": "Pre-warm 1 A100 GPU"}
    {"prediction": "14:15", "action": "Emergency scale: 3 A100 GPUs + 12 CPU cores"}
  Agreement with the rules: 0 of the LLM's 2 actions (precision 0.00), 0 of the rules' 1 (recall 0.00)

Scenario: weekend_minimal
  Rules:
    {"prediction": "Minimal scaling needed", "action": "Maintain baseline CPU only"}
  LLM (91.2 s):
    {"prediction": "14:00", "action": "Emergency scale: 3 A100 GPUs + 12 CPU cores"}
    {"prediction": "14:15", "action": "Pre-warm 1 A100 GPU"}
  Agreement with the rules: 0 of the LLM's 2 actions (precision 0.00), 0 of the rules' 1 (recall 0.00)

Mean precision 0.17, mean recall 0.17 over 3 scenarios
```

The rules are the chapter's heuristics. The model reads the same metrics and answers in the chapter's format; actions outside the allowed list are rejected. Small models often disagree with the rules, so treat their output as advice to check.

## Troubleshooting

- **"Please add a payment method to use ... GPU functions"**: the Modal account has no payment method. Add one, or run with `ATLAS_GPU=none` (and `ATLAS_ADVISOR_GPU=none`).
- **`models/radiology.joblib not found`**: run `python train_models.py` first; the Modal image includes `models/`, so train before `modal deploy`.
- **Slow LLM demos or swapping**: Qwen3-1.7B needs about 4 GB of free memory; `ATLAS_ADVISOR_MODEL=Qwen/Qwen3-0.6B` is smaller but rarely reads the signals (it tends to list every action).
- **Stop everything on Modal**: `modal app stop atlas-health-multimodal` and `modal app stop atlas-health-predictive-autoscaler`. Idle containers stop by themselves after their scale-down window.

## Repository structure

```
├── serverless_multimodal_pipeline.py   # Modal app: the three services, the DAG, the burst benchmark
├── predictive_autoscaler.py            # Modal app: schedule-based pre-warming, the LLM advisor
├── run_serverless_demo.py              # Local: the burst benchmark and the predictive demo
├── local_llm_scaling_demo.py           # Local: rules vs LLM advisor (3 scenarios)
├── comprehensive_llm_demo.py           # Local: rules vs LLM advisor (5 scenarios)
├── train_models.py                     # Trains the three classifier heads -> models/
├── atlas_pipeline/                     # Shared code: data, encoders, models, benchmark, forecast, advisor, local runtime
├── sample-data/                        # X-rays, lab results, notes, synthetic radiology metadata
├── serverless_scaling_results.json     # Triage scores from the last Modal benchmark run
├── serverless_cost_analysis.json       # Measured cost analysis from the last local benchmark run
├── sample_autoscaler_output.md         # Recorded output of the Modal predictive and advisor demos
├── SERVERLESS_IMPLEMENTATION_SUMMARY.md
├── modal_deployment_commands.txt
└── tests/                              # Smoke tests (make smoke)
```

## Contributing

Contributions are welcome. The architecture is modular: a new modality is a new service class and encoder.

## License

The code is licensed under the MIT License; see the [LICENSE](LICENSE) file. The data has its own licenses (see [Data and licenses](#data-and-licenses)).

---

**Atlas Health AI - Serverless Multimodal Pipeline**
*Elastic scaling for healthcare AI workloads*
