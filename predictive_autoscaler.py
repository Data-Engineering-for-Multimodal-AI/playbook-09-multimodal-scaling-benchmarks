#!/usr/bin/env python3
"""
Predictive Autoscaler for Multimodal Medical Pipeline
Anticipates workload spikes based on clinical patterns and schedules.

Works with the deployed pipeline (modal deploy serverless_multimodal_pipeline.py):

    modal run predictive_autoscaler.py::demo_predictive_autoscaling   # surge latency, cold vs pre-warmed
    modal run predictive_autoscaler.py::demo_llm_scaling_advisor      # rules vs a language model, on a GPU
    modal run predictive_autoscaler.py::demo_integrated_predictive_scaling   # today's plan, applied
    modal deploy predictive_autoscaler.py                               # see the note on the schedule

Pre-warming sets min_containers on the deployed services (Function.autoscaler.update), so containers
are loaded before the surge arrives, and sets it back to 0 afterwards.

Deploying this app does not start the every-5-minutes schedule unless ATLAS_ENABLE_SCHEDULE=1 is
set when you deploy: a running schedule (and the containers it pre-warms) costs money until you stop
it with `modal app stop atlas-health-predictive-autoscaler`.
"""
import os
import time
from datetime import datetime
from typing import Dict, List

import modal

app = modal.App("atlas-health-predictive-autoscaler")
PIPELINE_APP = "atlas-health-multimodal"
SERVICES = {"radiology": "RadiologyService", "labs": "LabsService", "notes": "NotesService"}

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("pandas==3.1.0", "numpy==2.5.4")
    .add_local_python_source("atlas_pipeline")
)

# On a GPU the advisor can be larger than on a laptop: Qwen3-4B (Apache-2.0) by default
ADVISOR_MODEL = os.environ.get("ATLAS_MODAL_ADVISOR_MODEL", "Qwen/Qwen3-4B")
advisor_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.14.1", "transformers==5.20.2")
    .env({"ATLAS_ADVISOR_MODEL": ADVISOR_MODEL})
    .run_commands(f"python -c \"from huggingface_hub import snapshot_download; snapshot_download('{ADVISOR_MODEL}')\"")
    .add_local_python_source("atlas_pipeline")
)


def _service(modality: str):
    return modal.Cls.from_name(PIPELINE_APP, SERVICES[modality])()


def apply_plan(plan: Dict[str, int]) -> None:
    """Set min_containers on each deployed service (0 lets it scale to zero)."""
    for modality, n in plan.items():
        _service(modality).score.autoscaler.update(min_containers=n)


@app.function(image=image, cpu=1, memory=1024,
              schedule=modal.Cron("*/5 * * * *") if os.environ.get("ATLAS_ENABLE_SCHEDULE") == "1" else None)
def predictive_autoscaler_agent(now_iso: str = None) -> Dict:
    """
    Forecast the next 15 minutes from the hospital schedule and pre-warm the deployed services.
    """
    from atlas_pipeline import forecast
    now = datetime.fromisoformat(now_iso) if now_iso else datetime.now()
    predictions = [p.to_dict() for p in forecast.predictions(now)]
    plan = forecast.prewarm_plan(now)
    apply_plan(plan)
    print(f"Predictive Autoscaler at {now:%a %H:%M}: {len(predictions)} predictions, min_containers {plan}")
    return {"time": now.isoformat(timespec="minutes"), "predictions": predictions, "min_containers": plan}


# ATLAS_ADVISOR_GPU=none runs the advisor on 8 CPU cores (about 2 minutes per answer with Qwen3-4B)
ADVISOR_GPU = os.environ.get("ATLAS_ADVISOR_GPU", "L4")
ADVISOR_OPTIONS = {"gpu": None, "cpu": 8, "memory": 16384} if ADVISOR_GPU.lower() in ("none", "cpu", "") else {"gpu": ADVISOR_GPU}


@app.cls(image=advisor_image, **ADVISOR_OPTIONS, timeout=1800, scaledown_window=60)
class AdvisorService:
    @modal.enter()
    def load(self):
        from atlas_pipeline.advisor import LLMAdvisor
        self.advisor = LLMAdvisor()

    @modal.method()
    def compare(self, scenarios: List[str] = None) -> List[Dict]:
        from atlas_pipeline.advisor import compare
        return compare(scenarios, self.advisor)


@app.local_entrypoint()
def demo_llm_scaling_advisor():
    """The first edition's rules and a language model (Qwen3-4B) on the same five scenarios, on a Modal GPU."""
    print("Atlas Health LLM-Based Scaling Advisor Demo (Modal)")
    print("=" * 60)
    results = AdvisorService().compare.remote()


def _surge(label: str, cases: int, offset: int = 0) -> Dict:
    """The morning imaging surge against the deployed services; returns its benchmark summary."""
    from atlas_pipeline import benchmark
    from serverless_multimodal_pipeline import ModalBackend
    backend = ModalBackend({m: _service(m) for m in SERVICES})
    scenario = dict(benchmark.SCENARIOS[0], cases=cases)
    print(f"\n{label}")
    summary = benchmark.run_scenario(backend, scenario, offset=offset)
    benchmark.print_scenario(summary)
    return summary


@app.local_entrypoint()
def demo_predictive_autoscaling(cases: int = 50):
    """
    A mixed surge twice: once arriving at scaled-to-zero services (reactive scaling, cold starts),
    once after the predictive autoscaler pre-warmed them for the 09:00 scheduled procedures.
    """
    from atlas_pipeline import forecast

    print("Atlas Health Predictive Autoscaling Demo (Modal)")
    print("=" * 60)
    apply_plan({m: 0 for m in SERVICES})
    wait = 75
    print(f"Making sure the services have scaled to zero ({wait} s)...")
    time.sleep(wait)
    reactive = _surge("1. Reactive: the surge arrives at scaled-to-zero services", cases)

    print(f"Scaling back to zero before the predictive run ({wait} s)...")
    time.sleep(wait)
    clock = datetime(2026, 10, 27, 8, 50)    # a Tuesday, 10 minutes before the scheduled procedures
    print(f"\n2. Predictive: at {clock:%a %H:%M} the schedule predicts:")
    for p in forecast.predictions(clock):
        print(f"   {p.prediction_type}: {p.expected_volume} cases at {p.expected_start:%H:%M} "
              f"(confidence {p.confidence_score:.2f}) -> pre-warm: {forecast.should_prewarm(p, clock)}")
    plan = forecast.prewarm_plan(clock)
    print(f"   min_containers {plan}; waiting until the containers have loaded...")
    started = time.time()
    apply_plan(plan)
    from serverless_multimodal_pipeline import ModalBackend
    backend = ModalBackend({m: _service(m) for m in SERVICES})
    for modality, n in plan.items():
        if n:
            backend.prewarm(modality, n)
    print(f"   warm after {time.time() - started:.0f} s")
    predictive = _surge("   The surge arrives at pre-warmed services", cases, offset=cases)
    apply_plan({m: 0 for m in SERVICES})
    print("   min_containers back to 0")

    print(f"\n{'':10s} {'reactive p95 s':>15s} {'pre-warmed p95 s':>17s} {'cold starts':>12s}")
    for m in reactive["modalities"]:
        r, p = reactive["modalities"][m], predictive["modalities"][m]
        print(f"{m:10s} {r['latency_p95']:>15.2f} {p['latency_p95']:>17.2f} "
              f"{r['cold_start_requests']:>5d} -> {p['cold_start_requests']:<4d}")
    print(f"Surge wall time: {reactive['wall_seconds']:.1f} s reactive, {predictive['wall_seconds']:.1f} s pre-warmed")


@app.local_entrypoint()
def demo_integrated_predictive_scaling(now: str = ""):
    """Run the autoscaler agent once (in the cloud) for a given time (default: now) and show its plan."""
    result = predictive_autoscaler_agent.remote(now or None)
    print(f"Plan for {result['time']}: min_containers {result['min_containers']}")
    for p in result["predictions"]:
        print(f"   {p['prediction_type']}: {p['expected_volume']} cases at {p['expected_start']} -> {p['recommended_action']}")
    if any(result["min_containers"].values()):
        print("Services stay warm until the agent runs again with nothing predicted; "
              "to release them now: modal run predictive_autoscaler.py::demo_integrated_predictive_scaling --now 2026-10-25T12:00")
