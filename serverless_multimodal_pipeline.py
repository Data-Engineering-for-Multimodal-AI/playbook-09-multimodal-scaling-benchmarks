#!/usr/bin/env python3
"""
Serverless Multimodal Medical Inference Pipeline using Modal
Demonstrates elastic GPU scaling with zero idle cost.

    python train_models.py                       # once: trains models/ (the containers load them)
    modal deploy serverless_multimodal_pipeline.py
    modal run serverless_multimodal_pipeline.py::simulate_clinical_workload_bursts
    modal run serverless_multimodal_pipeline.py::multimodal_inference_dag --cases 10

Each modality is a Modal class with its own resources and autoscaling:
- RadiologyService: GPU (A100 40 GB by default; ATLAS_GPU=T4, L4, ... or none), ResNet-18 + pneumonia model
- LabsService: 2 CPU cores, admission model on blood counts
- NotesService: 4 CPU cores, MiniLM + pulmonary-note model
Containers load their model once (@modal.enter: the cold start), scale out with the number of
pending requests and stop after their scaledown_window of idle time (zero idle cost).

The same benchmark runs locally without Modal: python run_serverless_demo.py
"""
import json
import os
import time
from typing import Dict, List

import modal

app = modal.App("atlas-health-multimodal")

# GPU for the radiology service. "none" runs it on 4 CPU cores instead (Modal accounts without a
# payment method can't use GPUs).
GPU = os.environ.get("ATLAS_GPU", "A100-40GB")
GPU_OPTIONS = {"gpu": None, "cpu": 4} if GPU.lower() in ("none", "cpu", "") else {"gpu": GPU}
REMOTE_ROOT = "/root/atlas"


def _download_weights():
    """Bake the pretrained weights into the image so cold starts don't download them."""
    import torchvision
    from sentence_transformers import SentenceTransformer
    torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# One image for all three services: the libraries, the pretrained weights, the trained models,
# the X-ray test images the workload sends (not the 1.1 GB training set) and the shared code
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.14.1", "torchvision==0.29.1", "sentence-transformers==6.2.0",
                    "transformers==5.20.2", "scikit-learn==1.9.1", "pandas==3.1.0", "openpyxl==3.1.5",
                    "joblib==1.6.0", "pillow==13.0.0", "numpy==2.5.4")
    .run_function(_download_weights)
    .env({"ATLAS_ROOT": REMOTE_ROOT, "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir("models", f"{REMOTE_ROOT}/models")
    .add_local_dir("sample-data/chest_xray/test", f"{REMOTE_ROOT}/sample-data/chest_xray/test")
    .add_local_python_source("atlas_pipeline")
)


class _Container:
    """What each service does: load the model once per container, then score batches."""
    modality = None

    def _load(self):
        from atlas_pipeline.encoders import ModalityService
        self.started = time.time()
        self.service = ModalityService(self.modality)
        self.ready = time.time()
        self.served = 0

    def _score(self, items: List) -> Dict:
        out = self.service.score(items)
        out.update(worker_id=os.environ.get("MODAL_TASK_ID", "local"), first_request=self.served == 0,
                   container_started=self.started, container_ready=self.ready,
                   load_seconds=self.service.load_seconds)
        self.served += 1
        return out


@app.cls(**GPU_OPTIONS, memory=8192, timeout=300, scaledown_window=60, max_containers=10, image=image)
class RadiologyService(_Container):
    modality = "radiology"

    @modal.enter()
    def load(self):
        self._load()

    @modal.method()
    def score(self, xray_paths: List[str]) -> Dict:
        return self._score(xray_paths)


@app.cls(cpu=2, memory=4096, timeout=120, scaledown_window=30, max_containers=10, image=image)
class LabsService(_Container):
    modality = "labs"

    @modal.enter()
    def load(self):
        self._load()

    @modal.method()
    def score(self, lab_vectors: List[List[float]]) -> Dict:
        return self._score(lab_vectors)


@app.cls(cpu=4, memory=8192, timeout=180, scaledown_window=45, max_containers=10, image=image)
class NotesService(_Container):
    modality = "notes"

    @modal.enter()
    def load(self):
        self._load()

    @modal.method()
    def score(self, notes: List[str]) -> Dict:
        return self._score(notes)


SERVICES = {"radiology": RadiologyService, "labs": LabsService, "notes": NotesService}
SCALEDOWN = {"radiology": 60, "labs": 30, "notes": 45}


class ModalBackend:
    """The benchmark's interface (see atlas_pipeline/benchmark.py) on Modal containers."""

    name = "modal"

    def __init__(self, services: Dict = None):
        self.services = services or {m: cls() for m, cls in SERVICES.items()}
        self.containers = {m: {} for m in self.services}

    def call(self, modality: str, items: List) -> Dict:
        submitted = time.time()
        out = self.services[modality].score.remote(items)
        finished = time.time()
        out["latency_seconds"] = finished - submitted
        c = self.containers[modality].setdefault(out["worker_id"], {
            "id": out["worker_id"], "started": out["container_started"], "ready": out["container_ready"],
            "load_seconds": out["load_seconds"]})
        c["last_active"] = max(c.get("last_active", 0), finished)
        return out

    def prewarm(self, modality: str, n: int) -> None:
        """Keep n containers running (min_containers), then wait until they have loaded."""
        self.services[modality].score.autoscaler.update(min_containers=n)
        # Each warm container answers immediately; send n parallel requests to make sure they're up
        if n:
            list(self.services[modality].score.map([[]] * n))

    def workers(self) -> Dict[str, List[Dict]]:
        """Container lifetimes: from start until the last request plus the scaledown window."""
        return {m: [dict(c, stopped=c["last_active"] + SCALEDOWN[m]) for c in cs.values()]
                for m, cs in self.containers.items()}

    def wait_for_scale_down(self) -> None:
        """Containers stop on their own; their idle time is counted in workers()."""


@app.local_entrypoint()
def simulate_clinical_workload_bursts(pause: int = 30, batch_size: int = 5):
    """
    The four clinical workload scenarios on Modal: morning imaging surge (GPU burst), telemetry
    stream (CPU steady), evening mixed burst and weekend light load, with a pause between them so
    containers can scale down. Watch the app in the Modal dashboard while it runs.
    """
    from atlas_pipeline import benchmark

    print("Atlas Health Clinical Workload Simulation (Modal)")
    print("=" * 60)
    print(f"Radiology: {GPU_OPTIONS}")
    results = benchmark.run_benchmark(ModalBackend(), pause=pause, batch_size=batch_size,
                                      resources=benchmark.resources_for(GPU))
    os.makedirs("results", exist_ok=True)
    benchmark.save(results, "results/modal_benchmark.json")
    with open("serverless_scaling_results.json", "w") as f:
        json.dump([t | {"scenario": s["scenario"], "scenario_time": s["wall_seconds"]}
                   for s in results["scenarios"] for t in s["triage"]], f, indent=2)
    print("\nResults saved to results/modal_benchmark.json and serverless_scaling_results.json")
    print("Billed cost (appears within a few minutes): modal billing report --for today")
    return results


@app.local_entrypoint()
def multimodal_inference_dag(cases: int = 10):
    """
    Score `cases` triage cases: the three encoders run in parallel (.spawn), then the scores are fused.
    """
    from atlas_pipeline import benchmark, data

    batch = data.cases(cases)
    print(f"Starting multimodal inference DAG for {len(batch)} cases")
    started = time.time()
    calls = {
        "radiology": RadiologyService().score.spawn([c["xray"] for c in batch]),
        "labs": LabsService().score.spawn([c["labs"] for c in batch]),
        "notes": NotesService().score.spawn([c["note"] for c in batch]),
    }
    scores = {m: call.get() for m, call in calls.items()}
    for m, out in scores.items():
        print(f"   {m:9s} {'cold' if out['first_request'] else 'warm'} container {out['worker_id']}, "
              f"model load {out['load_seconds']:.1f} s, compute {out['compute_seconds']:.2f} s")
    results = []
    for i, case in enumerate(batch):
        probs = {m: scores[m]["probabilities"][i] for m in scores}
        results.append({"case_id": case["case_id"], **{m: round(p, 3) for m, p in probs.items()},
                        **benchmark.triage(probs)})
    print(f"\nDAG completed in {time.time() - started:.2f}s")
    for r in results:
        print(f"   {r['case_id']}: radiology {r['radiology']:.3f}, labs {r['labs']:.3f}, notes {r['notes']:.3f} "
              f"-> triage {r['triage_score']:.3f}{'  URGENT' if r['urgent'] else ''}")
    return results


if __name__ == "__main__":
    print("Run this file with the Modal CLI:")
    print("   modal run serverless_multimodal_pipeline.py::simulate_clinical_workload_bursts")
    print("Or the same pipeline locally: python run_serverless_demo.py")
