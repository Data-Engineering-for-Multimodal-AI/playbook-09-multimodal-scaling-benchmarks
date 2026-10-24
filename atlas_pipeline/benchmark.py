"""Clinical workload scenarios and the scaling benchmark, the same for both backends.

A backend has call(modality, items) -> result dict (probabilities, worker_id, container_ready,
latency_seconds, compute_seconds), prewarm(modality, n), workers() -> worker lifetimes, wait_for_scale_down().
The local backend is atlas_pipeline.local_runtime.LocalBackend; the Modal backend is in
serverless_multimodal_pipeline.py.
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import numpy as np

from atlas_pipeline import config, data

MODALITIES = ("radiology", "labs", "notes")

# The first edition's four scenarios. 'burst' sends every batch at once; 'stream' sends one batch
# every `interval` seconds.
SCENARIOS = [
    {"name": "morning_imaging_surge", "description": "Morning imaging surge - GPU intensive",
     "cases": 50, "modalities": MODALITIES, "arrival": "burst"},
    {"name": "telemetry_stream", "description": "Telemetry stream - CPU steady state",
     "cases": 30, "modalities": ("labs",), "arrival": "stream", "interval": 0.5},
    {"name": "evening_mixed_burst", "description": "Evening mixed burst - Multi-modal",
     "cases": 80, "modalities": MODALITIES, "arrival": "burst"},
    {"name": "weekend_light_load", "description": "Weekend light load - Minimal scaling",
     "cases": 15, "modalities": MODALITIES, "arrival": "stream", "interval": 1.0},
]

# Resources per modality, as requested from Modal (serverless_multimodal_pipeline.py). Used to price
# worker/container time at Modal's rates. scaledown_window is the idle time before a container stops.
RESOURCES = {
    "radiology": {"gpu": "A100-40GB", "cpu": 0.125, "memory_gib": 8, "scaledown_window": 60},
    "labs": {"gpu": None, "cpu": 2, "memory_gib": 4, "scaledown_window": 30},
    "notes": {"gpu": None, "cpu": 4, "memory_gib": 8, "scaledown_window": 45},
}

# Modal's rates on 2026-10-24 (`modal billing rates`), US dollars per hour
RATES = {"date": "2026-10-24", "cpu_core": 0.0473, "memory_gib": 0.008,
         "gpu": {"A100-40GB": 2.10, "A100-80GB": 2.50, "A10G": 1.10, "L4": 0.80, "T4": 0.59, "H100": 3.95}}


def resources_for(gpu: str) -> Dict:
    """RESOURCES with the radiology service on another GPU type, or on 4 CPU cores ('none')."""
    resources = {m: dict(r) for m, r in RESOURCES.items()}
    if gpu.lower() in ("none", "cpu", ""):
        resources["radiology"].update(gpu=None, cpu=4)
    else:
        resources["radiology"]["gpu"] = {"A100": "A100-40GB", "A100-80GB": "A100-80GB"}.get(gpu, gpu)
    return resources


def triage(probabilities: Dict[str, float]) -> Dict:
    """Weighted mean of the available modality scores; 'urgent' above config.URGENT_THRESHOLD."""
    weights = {m: config.TRIAGE_WEIGHTS[m] for m in probabilities}
    score = sum(weights[m] * p for m, p in probabilities.items()) / sum(weights.values())
    return {"triage_score": score, "urgent": score > config.URGENT_THRESHOLD}


def cost_per_second(modality: str, rates: Dict = RATES, resources: Dict = None) -> float:
    r = (resources or RESOURCES)[modality]
    hourly = r["cpu"] * rates["cpu_core"] + r["memory_gib"] * rates["memory_gib"]
    if r["gpu"]:
        hourly += rates["gpu"][r["gpu"]]
    return hourly / 3600


def _items(modality: str, batch: List[Dict]) -> List:
    return [c["xray"] if modality == "radiology" else c["labs"] if modality == "labs" else c["note"] for c in batch]


def run_scenario(backend, scenario: Dict, offset: int = 0, batch_size: int = 5, log=print) -> Dict:
    cases = data.cases(scenario["cases"], offset)
    batches = [cases[i:i + batch_size] for i in range(0, len(cases), batch_size)]
    requests = []
    started = time.time()
    with ThreadPoolExecutor(max_workers=64) as pool:
        for k, batch in enumerate(batches):
            if scenario["arrival"] == "stream":
                time.sleep(max(0.0, started + k * scenario["interval"] - time.time()))
            for modality in scenario["modalities"]:
                submitted = time.time()
                requests.append({"modality": modality, "batch": batch, "submitted": submitted,
                                 "future": pool.submit(backend.call, modality, _items(modality, batch))})
        for r in requests:
            r["result"] = r.pop("future").result()
    wall = time.time() - started

    # Fuse per case
    scores = {c["case_id"]: {} for c in cases}
    for r in requests:
        for case, p in zip(r["batch"], r["result"]["probabilities"]):
            scores[case["case_id"]][r["modality"]] = p
    triaged = [{"case_id": cid, **{m: round(p, 4) for m, p in probs.items()}, **triage(probs)}
               for cid, probs in scores.items()]

    per_modality = {}
    for modality in scenario["modalities"]:
        rs = [r for r in requests if r["modality"] == modality]
        latency = np.array([r["result"]["latency_seconds"] for r in rs])
        labels = np.array([c[{"radiology": "xray_label", "labs": "labs_label", "notes": "note_label"}[modality]]
                           for r in rs for c in r["batch"]])
        probs = np.array([p for r in rs for p in r["result"]["probabilities"]])
        per_modality[modality] = {
            "requests": len(rs), "items": int(len(probs)),
            "latency_p50": float(np.percentile(latency, 50)), "latency_p95": float(np.percentile(latency, 95)),
            "latency_max": float(latency.max()),
            # A request hit a cold start if its worker/container finished loading after it was sent
            "cold_start_requests": int(sum(r["result"]["container_ready"] > r["submitted"] for r in rs)),
            "workers_used": len({r["result"]["worker_id"] for r in rs}),
            "accuracy": float(np.mean((probs > 0.5) == labels)),
        }
    summary = {"scenario": scenario["name"], "description": scenario["description"], "cases": len(cases),
               "wall_seconds": wall, "urgent_cases": int(sum(t["urgent"] for t in triaged)),
               "mean_triage_score": float(np.mean([t["triage_score"] for t in triaged])),
               "modalities": per_modality, "triage": triaged}
    return summary


def print_scenario(s: Dict, log=print) -> None:
    log(f"  {s['cases']} cases in {s['wall_seconds']:.1f} s; urgent: {s['urgent_cases']}; "
        f"mean triage score {s['mean_triage_score']:.3f}")
    log(f"  {'modality':10s} {'requests':>8s} {'workers':>7s} {'cold':>5s} {'p50 s':>7s} {'p95 s':>7s} {'max s':>7s} {'accuracy':>8s}")
    for m, r in s["modalities"].items():
        log(f"  {m:10s} {r['requests']:>8d} {r['workers_used']:>7d} {r['cold_start_requests']:>5d} "
            f"{r['latency_p50']:>7.2f} {r['latency_p95']:>7.2f} {r['latency_max']:>7.2f} {r['accuracy']:>8.3f}")


def worker_report(backend, rates: Dict = RATES, resources: Dict = None) -> Dict:
    """Cold-start durations, worker-seconds and their price at Modal's rates, per modality."""
    report = {}
    for modality, workers in backend.workers().items():
        seconds = sum(w["stopped"] - w["started"] for w in workers)
        cold = [w["ready"] - w["started"] for w in workers if w.get("ready")]
        report[modality] = {
            "workers_started": len(workers), "worker_seconds": round(seconds, 1),
            "cold_start_mean_seconds": round(float(np.mean(cold)), 2) if cold else None,
            "cost_at_modal_rates_usd": round(seconds * cost_per_second(modality, rates, resources), 4),
        }
    return report


def run_benchmark(backend, scenarios: Optional[List[Dict]] = None, pause: float = 30.0, batch_size: int = 5,
                  log=print, resources: Optional[Dict] = None) -> Dict:
    scenarios = scenarios or SCENARIOS
    results, offset = [], 0
    for i, scenario in enumerate(scenarios):
        log(f"\n{scenario['description']} ({scenario['cases']} cases, {scenario['arrival']})")
        s = run_scenario(backend, scenario, offset, batch_size, log)
        offset += scenario["cases"]
        print_scenario(s, log)
        results.append(s)
        if i < len(scenarios) - 1 and pause:
            log(f"  pausing {pause:.0f} s (idle workers scale down after their scaledown window)")
            time.sleep(pause)
    log("\nWaiting for every worker to scale down...")
    backend.wait_for_scale_down()
    resources = resources or RESOURCES
    workers = worker_report(backend, resources=resources)
    total = sum(w["cost_at_modal_rates_usd"] for w in workers.values())
    log(f"\n  {'modality':10s} {'started':>7s} {'cold start s':>12s} {'worker-s':>9s} {'at Modal rates':>15s}")
    for m, w in workers.items():
        log(f"  {m:10s} {w['workers_started']:>7d} {w['cold_start_mean_seconds'] or 0:>12.2f} "
            f"{w['worker_seconds']:>9.1f} {'$' + format(w['cost_at_modal_rates_usd'], '.4f'):>15s}")
    gpu = resources["radiology"]["gpu"] or f"{resources['radiology']['cpu']} CPU cores"
    log(f"  total ${total:.4f} at Modal's rates of {RATES['date']} (radiology on {gpu})")
    return {"backend": backend.name, "scenarios": results, "workers": workers, "resources": resources,
            "total_cost_at_modal_rates_usd": round(total, 4), "rates": RATES}


def save(results: Dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
