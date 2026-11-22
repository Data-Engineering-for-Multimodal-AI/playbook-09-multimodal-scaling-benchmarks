# Playbook 09: Scale a hospital pipeline on serverless GPUs, and measure it
**How to deploy a multimodal inference pipeline on serverless GPUs.** Chapter 9 of [*Data Engineering for Multimodal AI*](https://oreil.ly/data-engineering-multimodal) (O'Reilly, 2026), pages 357–367.

Hospital workloads arrive in bursts, and GPUs left running between bursts cost money for nothing. In this repo, we explore this with Atlas Health, a fictional hospital network, using real chest X-rays, lab results and clinical notes (the book's version used synthetic inputs). Follow along to run the same pipeline on serverless GPUs and on your own machine, and measure what scaling to zero costs and saves.

**The problem.** Each modality needs different compute at a different time, and keeping GPUs on all the time wastes most of the money.

**What you'll work with.** You'll scale each modality on its own, run on serverless GPUs that scale to zero, and compare that with always-on clusters such as Spark, Ray and Beam.

> Following the book? Its Steps 1–4 are in [Run it the way the book does](#run-it-the-way-the-book-does) and still work as printed. Want the code exactly as printed? Use the `first-edition` tag. This branch is the 2026 maintenance update; see [Differences from the printed book](#differences-from-the-printed-book).

## Run it

You need [uv](https://docs.astral.sh/uv/) and `make`. No Docker, no Modal account, no API keys: every step runs on your machine. Or open the repo in a GitHub Codespace, where both are ready. On Windows without `make`, use a Codespace or the book's steps below.

The times below are from a Mac (Apple M3, 8 GB) that was running other jobs at the same time (load average about 35), so they are slower than on an idle machine, and cold starts most of all. Times from CI (Ubuntu, 4 cores) are given where they exist.

**1. Get the code.** The clone includes the 1.2 GB of sample data (X-rays, lab results, notes).

```bash
git clone https://github.com/Data-Engineering-for-Multimodal-AI/playbook-09-multimodal-scaling-benchmarks.git
cd playbook-09-multimodal-scaling-benchmarks
```

**2. Train the models and run the four clinical scenarios.** Installs the locked dependencies, trains the three models if `models/` is empty (`python train_models.py`), then runs the chapter's four scenarios on local elastic workers in their small form (`python run_serverless_demo.py --quick`). On a fresh clone this took 9 min 17 s on the Mac, most of it training; a second run, with the models trained, took 3 min 6 s. In CI, with the models trained, it took 46 s.

```bash
make run
```

You should see the test scores of the three models, then one table per scenario and the cost of the worker time:

```
  radiology pneumonia vs normal chest X-ray          AUC 0.933  accuracy 0.840  F1 0.884  (624 records, 390 positive)
...
Morning imaging surge - GPU intensive (20 cases, burst)
  modality   requests workers  cold   p50 s   p95 s   max s accuracy
  radiology         4       3     4   22.80   22.93   22.94    0.850
...
  total $0.2074 at Modal's rates of 2026-10-24 (radiology on A100-40GB)
```

Every request in the first burst waited for a cold start: a worker process starting and loading its model. The last line prices the measured worker-seconds as Modal containers with the same resources; most of that time is cold starts and idle time before scale-down.

**3. Pre-warm for a surge.** Sends a 50-case surge to scaled-to-zero workers, then pre-warms the workers the hospital schedule predicts and sends the surge again (`python run_serverless_demo.py --predictive`). This took 7 min 7 s on the busy Mac.

```bash
make predictive
```

```
            reactive p95 s  pre-warmed p95 s  cold starts
radiology           138.69             13.07    10 -> 0
labs                 24.75              0.66    10 -> 0
notes               190.54             10.31    10 -> 0
Surge wall time: 190.9 s reactive, 13.4 s pre-warmed
```

On the busy Mac, cold starts took most of the reactive surge; after pre-warming, no request waited for one. [Expected output](#expected-output) has the same demo from 2026-10-24 (15.6 s against 2.7 s).

**4. Score a case yourself.** Give an X-ray, a note, or both. The models load in this process, so you see what a cold start costs one request.

```bash
make triage XRAY=sample-data/chest_xray/test/PNEUMONIA/person100_bacteria_475.jpeg \
            NOTE="Productive cough and fever for three days, crackles at the right base."
```

```
radiology P(pneumonia) = 0.990   model load 24.7 s, scoring 0.57 s
notes     P(cardiovascular/pulmonary note) = 0.055   model load 30.8 s, scoring 0.77 s
triage score 0.723 (weights radiology 0.5, notes 0.2)  URGENT
```

Loading each model took about 40 times longer than scoring with it. The note model gave this short pulmonary note 0.055: it was trained on long transcriptions (see [Before production](#before-production)). Try X-rays from `sample-data/chest_xray/test/NORMAL/` and your own sentences.

**5. Clean up.**

```bash
make clean
```

This deletes the trained models (`models/`), the embedding cache (`.cache/`), the results (`results/`) and the test cache. The sample data and the downloaded encoders stay. The next `make run` trains the models again.

## How much time do you have?

| Time | What to do |
|---|---|
| 15 minutes | [Run it](#run-it), steps 1, 2 and 4. On the busy Mac, step 2 took 9 min 17 s on a fresh clone and 3 min 6 s after that. |
| 1 hour | Step 3, then the three [Try this](#try-this) experiments (under a minute each in CI, 2 to 6 minutes on the busy Mac). |
| 1 day | [Use your own data](#use-your-own-data), the full four scenarios (`make demo`, 3 min 3 s on a Mac in October), the Modal steps in [Run it the way the book does](#run-it-the-way-the-book-does) if you have an account, and [Before production](#before-production). |

## Try this

Each experiment changes one thing and measures the effect, on local worker processes. None needs a Modal account. The tables below are from CI on 2026-11-22 (Ubuntu, 4 cores, nothing else running); each experiment took under a minute there. On the busy Mac the same experiments took 2 to 6 minutes, and where the results differ, the text says how.

**1. What does the scale-down window cost, and what does it save?** Sends two bursts of 10 X-rays to the radiology service with 15 quiet seconds between them, with scale-down windows of 5, 30 and 90 seconds.

```bash
make try-scaledown-window
```

```
 window  1st burst  2nd burst  2nd burst cold  workers started  worker-s  at A100 rate
     5s       3.7s       3.8s          2 of 2                4      35.4       $0.0213
    30s       4.1s       0.3s          0 of 2                2      98.8       $0.0596
    90s       4.2s       0.4s          0 of 2                2     219.3       $0.1322
```

With a 5-second window the workers stopped between the bursts, and the second burst waited for new ones: 3.8 s instead of 0.3 s. The longer windows kept it fast and paid for idle workers: the 90-second window cost 6 times as much as the 5-second one. On the busy Mac, where a cold start took about 30 s, the second burst waited 51.4 s and the 5-second window cost about as much as the 30-second one ($0.1107 against $0.1177), because the platform bills the cold start too. The repo's windows are 60 s (radiology), 30 s (labs) and 45 s (notes); change `GAP` in `experiments/scaledown_window.py` to the gap in your traffic.

**2. How many X-rays per request?** Sends the same 40 X-rays to 2 warm radiology workers in requests of 1, 5, 10 and 40.

```bash
make try-batch-size
```

```
per request  requests  all done  X-rays/s  slowest request  compute per X-ray
          1        40      1.4s      28.5             1.4s              68 ms
          5         8      1.3s      30.0             1.3s              66 ms
         10         4      1.3s      29.7             1.3s              65 ms
         40         1      1.8s      22.7             1.8s              44 ms
```

One request of 40 used the least compute per X-ray (44 ms) and still had the lowest throughput, because one request goes to one worker and the other one waits. On the busy Mac the gap was wider: 9.2 X-rays/s for one request of 40, 21.5 for requests of 5. The benchmark sends 5 per request: keep at least as many requests as workers.

**3. How far should one service scale out?** Sends a burst of 40 X-rays to a radiology pool that starts at zero, with a cap of 1, 2 and 4 workers sharing the machine's cores.

```bash
make try-max-workers
```

```
max workers  threads each   burst  workers started  mean cold start  worker-s  at A100 rate
          1             4    4.9s                1             3.2s      10.0       $0.0060
          2             2    4.9s                2             3.5s      20.0       $0.0121
          4             1    6.7s                4             5.5s      46.3       $0.0279
```

On one machine, more workers didn't make the burst faster, and four cost 4.6 times as much as one: each loads its own model, and they compete for the same cores. On the busy Mac (8 cores), four workers made the burst 2.4 times slower (45.0 s against 18.7 s) and 8 times more expensive. On Modal each container gets its own resources, but each one still pays its own cold start. Change the caps in `experiments/max_workers.py`, or set `max_workers` in `LocalBackend`.

## Use your own data

Put your cases in a folder, one row per case in `cases.csv`, with the X-rays next to it:

```
my_cases/
  cases.csv    case_id, then any of:
                 xray        a file name in xrays/ (.jpeg .jpg .png)
                 note        the clinical note text (the first 1,000 characters are used)
                 the 14 blood counts in atlas_pipeline/config.py (LAB_COLUMNS) and
                 "Patient age quantile" (0-19), standardized as in the Einstein dataset
                 xray_label, labs_label, note_label   true labels, 0 or 1 (optional)
  xrays/       patient_01.jpeg  patient_02.jpeg ...
```

Each case needs at least one modality; the triage score uses the ones it has. Lab column names may leave out the dataset's extra and non-breaking spaces. Check the folder first. This reads every file and runs nothing:

```bash
make check-data DIR=my_cases
```

```
my_cases: 7 cases, 6 X-rays, 4 lab panels, 4 notes. All checks passed.
```

It reports every problem with the file and the reason, and runs nothing if there is one. These are from a copy of that folder with mistakes put in on purpose:

```
9 problem(s) in broken; nothing was run:
  xrays/patient_01.jpeg: Git LFS pointer, not the image (run git lfs pull)
  cases.csv line 3: duplicate case_id P01
  cases.csv line 4: Platelets 250 doesn't look standardized (expected a z-score, about -5 to 5)
  cases.csv line 5: Patient age quantile 35 is not a whole number from 0 to 19
  xrays/patient_05.dcm: not an image type the encoder reads (.jpeg .jpg .png)
  cases.csv line 6: case P05 has no X-ray, note or lab values
  xrays/patient_06.jpeg: too small (40x40), needs at least 64x64
  cases.csv line 7: xray_label 'yes' is not 0 or 1
  cases.csv line 9: case P08 has no X-ray, note or lab values
```

It also checks for a missing or non-UTF-8 `cases.csv`, unknown columns, empty IDs, X-rays that are missing, empty or won't open, lab values that aren't numbers, and lab values without the age quantile. A raw platelet count would not fail in the model; it would give a meaningless score, which is why the check exists. When everything passes, score the cases:

```bash
make score DIR=my_cases
```

```
case         radiology   labs  notes  triage
P01              0.981  0.355  0.119   0.621  URGENT
P02              0.973  0.552  0.096   0.671  URGENT
P03              0.997  0.274  0.962   0.773  URGENT
P04              0.001  0.468      -   0.176
P05              1.000      -      -   1.000  URGENT
P06              1.000      -      -   1.000  URGENT
P07                  -      -  0.495   0.495
radiology: 5 of 6 labelled cases right
labs: 2 of 4 labelled cases right
notes: 3 of 3 labelled cases right

Scored in 46.1 s. Workers: radiology 2 (cold start 30.42 s), labs 1 (cold start 13.31 s), notes 1 (cold start 45.74 s). $0.0483 at Modal's rates.
Results saved to results/own_cases.json
```

This folder was made from held-out sample records (cases P01 to P06) plus one note written by hand (P07). `make score` uses the same local worker pools as the benchmark, with 5-second scale-down windows. Nothing is sent to Modal.

## What you end up with

Three trained models behind elastic services that scale to zero, a benchmark that measures their cold starts, latency and cost, and a scorer for your own cases.

Skills you'll practice:
- Scaling each modality's service on its own, with its own resources and scale-down window
- Measuring cold starts, latency percentiles and worker-seconds instead of estimating them
- Pricing measured worker time at a serverless platform's rates
- Pre-warming capacity from a forecast, and measuring what it saves
- Checking input data against what the models were trained on

## Before production

This playbook runs on CPU with public sample data. Before you run something like it for real:

1. **Make cold starts shorter.** Loading the X-ray and note models took 24.7 s and 30.8 s on the busy Mac, against under 1 s to score a case ([Run it](#run-it), step 4). Bake the weights into the container image, load them once per container, and keep the image small. *Chapter 9, "Dynamic Model Parking with Checkpoint Streaming" (page 344).*
2. **Set the scale-down window from your traffic.** A window shorter than the gap between bursts made the second burst start cold (3.8 s instead of 0.3 s in CI, 51.4 s on the busy Mac); on the busy Mac it saved almost nothing, because cold starts are billed too ([Try this 1](#try-this)). Measure the gaps in your traffic, and pre-warm from the schedule where you can (`make predictive`). *Chapter 9, "Auto-Scaling Strategies for Multimodal Workloads" (page 332).*
3. **Batch, but keep enough requests for every worker.** One 40-X-ray request ran at 22.7 X-rays/s in CI, against 30.0 in requests of 5, because it kept the second worker idle ([Try this 2](#try-this)). *Chapter 9, "Batching and Inference Tiering" (page 344).*
4. **Cap scale-out where workers share hardware.** Four workers on one machine cost 4.6 times as much as one in CI without finishing sooner, and on the busy Mac made the burst 2.4 times slower ([Try this 3](#try-this)). *Chapter 9, "Modality-Specific Scaling Profiles" (page 332).*
5. **Check that new data looks like the training data.** The note model scores the first 1,000 characters of long transcriptions; a short pulmonary note scored 0.055 and another 0.495. Track the score distribution per modality and alert when it shifts. *Chapter 10, "Detecting Drift" (page 384).*
6. **Validate inputs at the service.** The services score any lab vector and open any path they are sent; the lab model expects standardized values. `make check-data` checks a folder; the services should check each request too. *Chapter 4, "Validation, Transformation, Observability, and Annotation Across Modalities" (page 119).*
7. **Retry and isolate failures.** When one local worker fails to start, `ElasticPool` fails every request waiting for that service, and nothing is retried. *Chapter 9, "Fault Tolerance and Recovery" (page 327).*
8. **Version the models with the triage rule.** The triage weights (0.5, 0.3, 0.2) and the urgent threshold (0.6) are fixed in `atlas_pipeline/config.py`, and `models/metrics.json` records scores but no version. Record which models and which rule produced each triage score. *Chapter 8, "Data Versioning and Experiment Tracking for Multimodal AI" (page 276).*
9. **Protect patient data.** The sample records are public and de-identified; real ones are not. The Modal image carries the test X-rays and the trained heads. Encrypt data in transit and at rest, and limit who can deploy. *Chapter 11, "Protecting Data Across the Pipeline" (page 421).*
10. **Track cost per case.** Each run prints its cost at Modal's rates once; record it per run and per modality, with the rates' date. *Chapter 9, "Cost Optimization for Cloud-Based Multimodal AI Infrastructure" (page 342).*

## What it costs

| Where you run it | Cost |
|---|---|
| Your machine | $0. Everything in [Run it](#run-it), [Try this](#try-this) and [Use your own data](#use-your-own-data) runs locally: no API keys, no Modal account. About 2.3 GB of disk on a Mac: 1.2 GB of sample data and 1 GB for the environment; the encoders add about 135 MB, and the LLM demos another 4 GB. |
| A GitHub Codespace (2 cores, 8 GB, the default size; this repo asks for none) | Estimated $0.18 per hour at GitHub's list price (November 2026). A free personal account includes 120 core-hours a month, about 60 hours on this machine. In CI, `make run` in a clean devcontainer (with the models trained while the container was created) took 36 s; the whole job took 4 to 7 minutes. |
| Modal | Estimated $0.07 to $0.21 for the `make run` workload (75 cases) as Modal containers, from the worker-seconds measured locally: $0.0714 in CI, and $0.2074 and $0.1882 in two runs on the busy Mac, where cold starts were longer; radiology priced as an A100 40 GB ($0.000583 per second), CPU at $0.0000131 per core per second and memory at $0.00000222 per GiB per second ([Modal pricing](https://modal.com/pricing), November 2026). Most of it is radiology's GPU time. The local workers ran on CPU, so on a GPU the compute and cold-start times, and with them the total, will differ. Modal's Starter plan includes $30 of free credits a month. The billed costs of the edition's Modal runs on CPU containers are in [Tested on](#tested-on). |

## How it works

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

## Run it the way the book does

These are the book's Steps 1–4 (pages 359–366), with the fixes listed in [Differences from the printed book](#differences-from-the-printed-book). Steps 1 and 2 run locally; step 3 needs a Modal account.

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

You need a [Modal](https://modal.com) account. Run `modal setup` once (the book's `modal auth new`, page 359, is not a command in Modal 1.6). Train the models first (step 1): the image includes them, and the book's `modal deploy` (page 360) stops with a message if they're missing. GPUs need a payment method on the account; without one, use `ATLAS_GPU=none` (below).

> **GPU path not run for this edition.** The default configuration (radiology on an A100 40 GB, the advisor on an L4) is written for Modal 1.x but was not run while testing this update: the test account had no payment method, which Modal requires for GPUs. Everything else on Modal was run on CPU containers (`ATLAS_GPU=none`, `ATLAS_ADVISOR_GPU=none`); see [Tested on](#tested-on). The code paths are the same apart from the device; on a GPU, expect shorter model compute times and a higher cost per second (A100 40 GB: $2.10 per hour, L4: $0.80 per hour at Modal's rates of 2026-10-24).

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

The left column is the first edition: the code at the `first-edition` tag, which the book describes.

| First edition | This version | Why |
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

All from runs on 2026-10-24. This is the output of this version; it may differ from the output printed in the book, and that's okay. Local runs: fresh clone, macOS 26.5.1 (Apple M3), Python 3.12.15. Modal runs: Modal 1.6.2, CPU containers (`ATLAS_GPU=none`). Download and progress messages are left out. Latencies and cold starts depend on the machine and load; the scores and labels don't.

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
├── scripts/
│   ├── triage.py                       # make triage: score one case by hand
│   └── own_cases.py                    # make check-data, make score: your own cases
├── experiments/                        # make try-*: the three Try this experiments
├── Makefile                            # make run, predictive, triage, check-data, score, try-*, smoke, clean
└── tests/                              # Smoke tests (make smoke); test_own_data.py checks make check-data
```

## Contributing

Contributions are welcome. The architecture is modular: a new modality is a new service class and encoder.

## License

The code is licensed under the MIT License; see the [LICENSE](LICENSE) file. The data has its own licenses (see [Data and licenses](#data-and-licenses)).

---

**Atlas Health AI - Serverless Multimodal Pipeline**
*Elastic scaling for healthcare AI workloads*
