"""Smoke tests: data, models, the local elastic runtime, the benchmark, forecasting and the advisors.

Run with `make smoke` (or `uv run pytest -q`). CPU only, about 10 minutes the first time: trains the
models on a small X-ray sample (ATLAS_XRAY_TRAIN_PER_CLASS=100) and downloads ResNet-18,
all-MiniLM-L6-v2 and Qwen3-0.6B. The Modal app definitions are checked without running anything in
the cloud (that needs a Modal account; see the README).
"""
import json
import os
import subprocess
import sys
from datetime import datetime

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("ATLAS_XRAY_TRAIN_PER_CLASS", "100")


def run(*args, timeout=1800, env=None):
    result = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True, timeout=timeout,
                            env={**os.environ, **(env or {})})
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    return result.stdout


# ------------------------------------------------------------------ data

def test_sample_data():
    from atlas_pipeline import data
    assert {s: len(data.xray_files(s)) for s in ("train", "test", "val")} == {"train": 5216, "test": 624, "val": 16}
    assert not os.path.exists(os.path.join(ROOT, "sample-data", "chest_xray", "chest_xray"))   # duplicate removed
    assert not os.path.exists(os.path.join(ROOT, "sample-data", "telemetry_2.xlsx"))
    labs = data.labs()
    assert len(labs) == 603 and labs["admitted"].sum() == 129
    assert len(data.notes()) == 4966 and len(data.radiology_metadata()) == 100
    with open(os.path.join(ROOT, "sample-data", "README.md")) as fh:
        readme = fh.read()
    for name in os.listdir(os.path.join(ROOT, "sample-data")):
        if name != "README.md":
            assert f"`{name}" in readme, name


def test_cases_use_held_out_records():
    from atlas_pipeline import data
    from atlas_pipeline.models import split_indices
    cases = data.cases(30)
    assert len({c["case_id"] for c in cases}) == 30
    assert all(c["xray"].startswith("test/") for c in cases)
    assert set(split_indices("labs")["train"]).isdisjoint(split_indices("labs")["test"])


# ------------------------------------------------------------------ models

@pytest.fixture(scope="session")
def models():
    out = run("train_models.py")
    with open(os.path.join(ROOT, "models", "metrics.json")) as fh:
        return json.load(fh), out


def test_models_learn_real_labels(models):
    metrics, out = models
    assert metrics["radiology"]["auc"] > 0.85 and metrics["radiology"]["n"] == 624
    assert metrics["notes"]["auc"] > 0.85
    assert metrics["labs"]["auc"] > 0.6
    assert "Test scores" in out


# ------------------------------------------------------------------ local runtime and benchmark

def test_elastic_pool_scales_out_and_to_zero(models):
    from atlas_pipeline.local_runtime import ElasticPool
    pool = ElasticPool("labs", max_workers=2, scaledown_window=1.0)
    try:
        from atlas_pipeline import data
        items = [c["labs"] for c in data.cases(4)]
        results = [f.result(timeout=300) for f in [pool.submit(items[:2]), pool.submit(items[2:])]]
        assert all(len(r["probabilities"]) == 2 for r in results)
        assert len(pool.snapshot()) == 2                      # two pending requests -> two workers
        pool.wait_until_idle_stopped(timeout=60)
        assert not pool.workers                               # scaled to zero after the idle window
        pool.prewarm(1)
        pool.wait_until_ready()
        assert len(pool.workers) == 1
        r = pool.submit(items[:1]).result(timeout=60)
        assert r["latency_seconds"] < 5                       # warm: no model load
    finally:
        pool.close()


def test_run_serverless_demo_quick(models):
    out = run("run_serverless_demo.py", "--quick")
    assert "Morning imaging surge" in out and "Weekend light load" in out
    with open(os.path.join(ROOT, "results", "local_benchmark.json")) as fh:
        results = json.load(fh)
    assert len(results["scenarios"]) == 4
    surge = results["scenarios"][0]["modalities"]
    assert surge["radiology"]["cold_start_requests"] > 0 and surge["radiology"]["accuracy"] > 0.7
    assert all(w["workers_started"] > 0 for w in results["workers"].values())
    with open(os.path.join(ROOT, "serverless_cost_analysis.json")) as fh:
        assert json.load(fh)["total_cost_at_modal_rates_usd"] > 0


def test_predictive_prewarming_cuts_latency(models):
    run("run_serverless_demo.py", "--predictive", "--quick")
    with open(os.path.join(ROOT, "results", "local_predictive.json")) as fh:
        results = json.load(fh)
    for modality in ("radiology", "notes"):
        reactive = results["reactive"]["modalities"][modality]
        prewarmed = results["predictive"]["modalities"][modality]
        assert reactive["cold_start_requests"] > 0 and prewarmed["cold_start_requests"] == 0
        assert prewarmed["latency_p95"] < reactive["latency_p95"]


# ------------------------------------------------------------------ forecasting and advisors

def test_forecast_plans():
    from atlas_pipeline import forecast
    assert forecast.prewarm_plan(datetime(2026, 10, 27, 8, 50)) == {"radiology": 2, "labs": 1, "notes": 2}
    assert forecast.prewarm_plan(datetime(2026, 10, 27, 7, 50))["radiology"] == 3
    assert forecast.prewarm_plan(datetime(2026, 10, 25, 12, 0)) == {"radiology": 0, "labs": 0, "notes": 0}


def test_rule_advisor_and_action_check():
    from atlas_pipeline.advisor import SCENARIOS, parse_recommendations, rule_advisor
    assert [r["action"] for r in rule_advisor(SCENARIOS["weekend_minimal"])] == ["Maintain baseline CPU only"]
    assert len(rule_advisor(SCENARIOS["high_volume_crisis"])) == 3
    valid, error = parse_recommendations('[{"prediction": "x", "action": "Pre-warm 2 A100 GPUs"},'
                                         ' {"prediction": "y", "action": "Buy more GPUs"}]')
    assert [v["action"] for v in valid] == ["Pre-warm 2 A100 GPUs"] and "Buy more GPUs" in error


def test_llm_advisor_runs():
    out = run("local_llm_scaling_demo.py", timeout=3600)
    assert out.count("Scenario:") == 3 and "LLM (" in out


# ------------------------------------------------------------------ Modal app definitions (no cloud calls)

def test_modal_apps_define():
    out = run("-c", "import serverless_multimodal_pipeline as p, predictive_autoscaler as a; "
                    "print(sorted(p.SERVICES), a.app.name, a.predictive_autoscaler_agent is not None)",
              env={"ATLAS_GPU": "A100-40GB"}, timeout=300)
    assert "['labs', 'notes', 'radiology'] atlas-health-predictive-autoscaler True" in out
