#!/usr/bin/env python3
"""
Local demo of the serverless multimodal pipeline: the same services, workload and benchmark as the
Modal version, with elastic worker processes on your machine instead of cloud containers.

    python run_serverless_demo.py                 # the four clinical scenarios (about 5 minutes)
    python run_serverless_demo.py --predictive    # a surge: reactive vs pre-warmed workers
    python run_serverless_demo.py --quick         # small scenarios and short idle windows (about 1 minute)

Trains the models first if models/ is empty. Every number printed is measured on this machine:
worker cold starts (process start + model load), latencies, workers started, worker-seconds, and
what those worker-seconds would cost at Modal's rates. Results: results/local_benchmark.json and
serverless_cost_analysis.json.
"""
import argparse
import json
import os
import time
from datetime import datetime

from atlas_pipeline import benchmark, config


def ensure_models():
    if not all(os.path.exists(os.path.join(config.MODELS_DIR, f"{m}.joblib")) for m in benchmark.MODALITIES):
        print("Training the three models first (python train_models.py)...")
        from atlas_pipeline.models import train_all
        train_all()


def write_cost_analysis(results, path="serverless_cost_analysis.json"):
    """Measured worker-seconds priced at Modal's rates (replaces the first edition's fixed $750 estimate)."""
    analysis = {
        "source": f"run_serverless_demo.py ({results['backend']} backend), {datetime.now():%Y-%m-%d}",
        "rates_usd_per_hour": results["rates"],
        "resources": results["resources"],
        "per_modality": results["workers"],
        "total_cost_at_modal_rates_usd": results["total_cost_at_modal_rates_usd"],
        "cases": sum(s["cases"] for s in results["scenarios"]),
    }
    analysis["cost_per_case_usd"] = round(analysis["total_cost_at_modal_rates_usd"] / analysis["cases"], 6)
    with open(path, "w") as f:
        json.dump(analysis, f, indent=2)


def predictive_demo(backend, cases: int):
    """A mixed surge at scaled-to-zero workers, then after pre-warming for the 09:00 scheduled procedures."""
    from atlas_pipeline import forecast
    scenario = dict(benchmark.SCENARIOS[0], cases=cases)
    print("\n1. Reactive: the surge arrives at scaled-to-zero workers")
    reactive = benchmark.run_scenario(backend, scenario)
    benchmark.print_scenario(reactive)
    print("   waiting for the workers to scale down...")
    backend.wait_for_scale_down()

    clock = datetime(2026, 10, 27, 8, 50)    # a Tuesday, 10 minutes before the scheduled procedures
    print(f"\n2. Predictive: at {clock:%a %H:%M} the schedule predicts:")
    for p in forecast.predictions(clock):
        print(f"   {p.prediction_type}: {p.expected_volume} cases at {p.expected_start:%H:%M} "
              f"(confidence {p.confidence_score:.2f}) -> pre-warm: {forecast.should_prewarm(p, clock)}")
    plan = forecast.prewarm_plan(clock)
    print(f"   pre-warming {plan}...")
    started = time.time()
    for modality, n in plan.items():
        if n:
            backend.prewarm(modality, n)
    print(f"   warm after {time.time() - started:.0f} s")
    predictive = benchmark.run_scenario(backend, scenario, offset=cases)
    benchmark.print_scenario(predictive)
    for modality in plan:
        backend.pools[modality].prewarm(0)

    print(f"\n{'':10s} {'reactive p95 s':>15s} {'pre-warmed p95 s':>17s} {'cold starts':>12s}")
    for m in reactive["modalities"]:
        r, p = reactive["modalities"][m], predictive["modalities"][m]
        print(f"{m:10s} {r['latency_p95']:>15.2f} {p['latency_p95']:>17.2f} "
              f"{r['cold_start_requests']:>5d} -> {p['cold_start_requests']:<4d}")
    print(f"Surge wall time: {reactive['wall_seconds']:.1f} s reactive, {predictive['wall_seconds']:.1f} s pre-warmed")
    return {"reactive": reactive, "predictive": predictive, "plan": plan}


def main():
    """
    Main demo orchestration showing serverless multimodal pipeline concepts.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictive", action="store_true", help="reactive vs pre-warmed morning surge")
    parser.add_argument("--quick", action="store_true", help="small scenarios, 5 s idle windows")
    parser.add_argument("--pause", type=float, default=None, help="seconds between scenarios (default 30)")
    args = parser.parse_args()

    from atlas_pipeline.local_runtime import LocalBackend

    print("Atlas Health Serverless Multimodal Pipeline Demo (local)")
    print("=" * 60)
    ensure_models()
    with open(os.path.join(config.MODELS_DIR, "metrics.json")) as f:
        metrics = json.load(f)
    print("\nModels (test scores on records they didn't train on):")
    for m, r in metrics.items():
        print(f"   {m:9s} {r['task']:40s} AUC {r['auc']:.3f}  accuracy {r['accuracy']:.3f}")

    windows = {"radiology": 5, "labs": 5, "notes": 5} if args.quick else None
    backend = LocalBackend(scaledown_windows=windows)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    try:
        if args.predictive:
            results = predictive_demo(backend, 20 if args.quick else 50)
            benchmark.save(results, os.path.join(config.RESULTS_DIR, "local_predictive.json"))
            print("\nResults saved to results/local_predictive.json")
            return
        scenarios = [dict(s, cases=min(s["cases"], 20)) for s in benchmark.SCENARIOS] if args.quick else None
        pause = args.pause if args.pause is not None else (3 if args.quick else 30)
        results = benchmark.run_benchmark(backend, scenarios, pause=pause)
        benchmark.save(results, os.path.join(config.RESULTS_DIR, "local_benchmark.json"))
        write_cost_analysis(results)
        print("\nResults saved to results/local_benchmark.json and serverless_cost_analysis.json")
        print("\nOn Modal (GPU containers in the cloud; needs a Modal account):")
        print("   modal deploy serverless_multimodal_pipeline.py")
        print("   modal run serverless_multimodal_pipeline.py::simulate_clinical_workload_bursts")
    finally:
        backend.close()


if __name__ == "__main__":
    main()
