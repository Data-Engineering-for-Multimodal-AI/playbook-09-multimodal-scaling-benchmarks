# Sample output: predictive autoscaling and the LLM advisor on Modal

Recorded on 2026-10-24 with Modal 1.6.2 (Python 3.12.15 on macOS 26.5.1). The Modal account used had
no payment method, so GPUs were unavailable: the radiology service ran on 4 CPU cores
(`ATLAS_GPU=none modal deploy serverless_multimodal_pipeline.py`) and the advisor on 8 CPU cores
(`ATLAS_ADVISOR_GPU=none`). Run URLs are left out.

## Predictive autoscaling

```bash
modal run predictive_autoscaler.py::demo_predictive_autoscaling
```

```
Atlas Health Predictive Autoscaling Demo (Modal)
============================================================
Making sure the services have scaled to zero (75 s)...

1. Reactive: the surge arrives at scaled-to-zero services
  50 cases in 18.5 s; urgent: 20; mean triage score 0.492
  modality   requests workers  cold   p50 s   p95 s   max s accuracy
  radiology        10       4    10   14.48   14.74   14.78    0.860
  labs             10       1    10    5.39    5.41    5.41    0.580
  notes            10       1    10   17.39   18.33   18.48    0.980
Scaling back to zero before the predictive run (75 s)...

2. Predictive: at Tue 08:50 the schedule predicts:
   scheduled_procedures: 75 cases at 09:00 (confidence 0.90) -> pre-warm: True
   min_containers {'radiology': 2, 'labs': 1, 'notes': 2}; waiting until the containers have loaded...
   warm after 19 s

   The surge arrives at pre-warmed services
  50 cases in 2.5 s; urgent: 28; mean triage score 0.553
  modality   requests workers  cold   p50 s   p95 s   max s accuracy
  radiology        10       2     0    1.43    1.97    2.02    0.860
  labs             10       1     0    0.95    0.96    0.96    0.740
  notes            10       2     1    1.78    2.51    2.51    0.820
   min_containers back to 0

            reactive p95 s  pre-warmed p95 s  cold starts
radiology            14.74              1.97    10 -> 0   
labs                  5.41              0.96    10 -> 0   
notes                18.33              2.51    10 -> 1   
Surge wall time: 18.5 s reactive, 2.5 s pre-warmed
```

The same kind of surge, 50 cases: arriving at scaled-to-zero services every request waits for a
container to start and load its model (p95 13-18 s); after the 08:50 forecast pre-warmed the services
(radiology 2, labs 1, notes 2 containers, ready 19 s later), p95 is 1-2.5 s.

## LLM scaling advisor

```bash
ATLAS_ADVISOR_GPU=none modal run predictive_autoscaler.py::demo_llm_scaling_advisor
```

Qwen3-4B on 8 CPU cores (bfloat16, about 70 s per answer):

```
Atlas Health LLM-Based Scaling Advisor Demo (Modal)
============================================================
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
[1ALoading weights:   0%|          | 0/398 [00:00<?, ?it/s]
[1ALoading weights:   5%|▍         | 19/398 [00:00<00:02, 159.60it/s]
[1ALoading weights:  11%|█         | 42/398 [00:00<00:01, 190.25it/s]
[1ALoading weights:  17%|█▋        | 67/398 [00:00<00:01, 209.75it/s]
[1ALoading weights:  26%|██▌       | 104/398 [00:00<00:01, 234.21it/s]
[1ALoading weights:  32%|███▏      | 128/398 [00:00<00:01, 213.97it/s]
[1ALoading weights:  38%|███▊      | 150/398 [00:00<00:01, 178.73it/s]
[1ALoading weights:  44%|████▍     | 176/398 [00:00<00:01, 197.89it/s]
[1ALoading weights:  49%|████▉     | 197/398 [00:01<00:01, 189.87it/s]
[1ALoading weights:  56%|█████▌    | 222/398 [00:01<00:00, 196.60it/s]
[1ALoading weights:  62%|██████▏   | 245/398 [00:01<00:00, 175.32it/s]
[1ALoading weights:  67%|██████▋   | 268/398 [00:01<00:00, 177.39it/s]
[1ALoading weights:  74%|███████▍  | 295/398 [00:01<00:00, 197.35it/s]
[1ALoading weights:  79%|███████▉  | 316/398 [00:01<00:00, 160.25it/s]
[1ALoading weights:  89%|████████▉ | 356/398 [00:01<00:00, 211.65it/s]
[1ALoading weights:  95%|█████████▌| 380/398 [00:01<00:00, 205.34it/s]
[1ALoading weights: 100%|██████████| 398/398 [00:01<00:00, 202.35it/s]
LLM advisor: Qwen/Qwen3-4B on cpu (loaded in 4.1 s)
Scenario: morning_pacs_surge
  Rules:
    {"prediction": "Radiology surge in 5 minutes", "action": "Pre-warm 2 A100 GPUs"}
    {"prediction": "Immediate radiology processing needed", "action": "Pre-warm 1 A100 GPU"}
  LLM (75.6 s):
    {"prediction": "In the next 15 minutes, there will be a burst of X-rays due to scheduled PACS uploads", "action": "Pre-warm 1 A100 GPU"}
    {"prediction": "There will be a hospital-wide surge due to multiple queues", "action": "Emergency scale: 3 A100 GPUs + 12 CPU cores"}
  Agreement with the rules: 1 of the LLM's 2 actions (precision 0.50), 1 of the rules' 2 (recall 0.50)
Scenario: shift_change_dictation
  Rules:
    {"prediction": "Text processing surge in 15 minutes", "action": "Scale CPU text processors to 6 cores"}
  LLM (68.5 s):
    {"prediction": "Next 15 minutes", "action": "Pre-warm 1 A100 GPU"}
    {"prediction": "Next 15 minutes", "action": "Scale CPU text processors to 6 cores"}
  Agreement with the rules: 1 of the LLM's 2 actions (precision 0.50), 1 of the rules' 1 (recall 1.00)
Scenario: night_emergency_surge
  Rules:
    {"prediction": "Emergency imaging spike in 10 minutes", "action": "Pre-warm 1 A100 GPU"}
  LLM (71.0 s):
    {"prediction": "In the next 15 minutes, radiology queue depth is high and emergency department census is elevated.", "action": "Emergency scale: 3 A100 GPUs + 12 CPU cores"}
    {"prediction": "Telemetry queue depth is high and CPU resources may need to be scaled up.", "action": "Scale CPU telemetry workers to 4 workers"}
  Agreement with the rules: 0 of the LLM's 2 actions (precision 0.00), 0 of the rules' 1 (recall 0.00)
Scenario: weekend_minimal
  Rules:
    {"prediction": "Minimal scaling needed", "action": "Maintain baseline CPU only"}
  LLM (67.6 s):
    {"prediction": "Next 15 minutes", "action": "Emergency scale: 3 A100 GPUs + 12 CPU cores"}
    {"prediction": "Next 15 minutes", "action": "Scale CPU telemetry workers to 4 workers"}
  Agreement with the rules: 0 of the LLM's 2 actions (precision 0.00), 0 of the rules' 1 (recall 0.00)
Scenario: high_volume_crisis
  Rules:
    {"prediction": "Radiology surge in 5 minutes", "action": "Pre-warm 2 A100 GPUs"}
    {"prediction": "Text processing surge in 15 minutes", "action": "Scale CPU text processors to 6 cores"}
    {"prediction": "Multi-modal surge crisis in 2 minutes", "action": "Emergency scale: 3 A100 GPUs + 12 CPU cores"}
  LLM (65.6 s):
    {"prediction": "Next 15 minutes", "action": "Pre-warm 1 A100 GPU"}
    {"prediction": "Next 15 minutes", "action": "Scale CPU telemetry workers to 4 workers"}
  Agreement with the rules: 0 of the LLM's 2 actions (precision 0.00), 0 of the rules' 3 (recall 0.00)
Mean precision 0.20, mean recall 0.30 over 5 scenarios
```

The model found the GPU need on the PACS upload and the text surge at the shift change, but it also
recommended emergency scaling at night and on a quiet weekend, and missed most of the crisis plan
(mean precision 0.20, recall 0.30 against the rules). That is why its recommendations are checked
and compared with the rules rather than applied directly.
