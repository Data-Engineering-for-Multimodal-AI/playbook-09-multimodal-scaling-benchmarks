"""A small serverless runtime on your machine: elastic worker processes per modality.

Each ElasticPool behaves like a serverless function with autoscaling:
- a request that finds no free worker starts a new worker process (up to max_workers); the worker
  loads its encoder and model first, which is a real cold start
- a worker that has been idle for scaledown_window seconds exits (scale to zero)
- prewarm(n) keeps n workers running ahead of an expected surge (like Modal's min_containers)

Every worker's start, ready and stop times are recorded, so the benchmark can report cold starts,
peak workers and worker-seconds (what a serverless platform bills).
"""
import multiprocessing as mp
import os
import queue
import threading
import time
from concurrent.futures import Future
from typing import Dict, List


def _worker_main(modality: str, worker_id: int, threads: int, tasks, results):
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    started = time.time()
    try:
        if modality != "labs":           # the lab model is plain scikit-learn
            import torch
            torch.set_num_threads(threads)
        from atlas_pipeline.encoders import ModalityService
        service = ModalityService(modality, device="cpu")
    except Exception as e:  # report the failure instead of hanging the pool
        results.put(("failed", worker_id, repr(e)))
        return
    ready = time.time()
    results.put(("ready", worker_id, {"started": started, "ready": ready, "load_seconds": service.load_seconds}))
    served = 0
    while True:
        task = tasks.get()
        if task is None:
            break
        task_id, items = task
        try:
            out = service.score(items)
            out.update(worker_id=worker_id, first_request=served == 0, container_ready=ready)
            served += 1
            results.put(("done", task_id, out))
        except Exception as e:
            results.put(("error", task_id, repr(e)))


class ElasticPool:
    def __init__(self, modality: str, max_workers: int = 4, scaledown_window: float = 30.0, threads_per_worker: int = 1):
        self.modality = modality
        self.max_workers = max_workers
        self.scaledown_window = scaledown_window
        self.threads = threads_per_worker
        self.ctx = mp.get_context("spawn")
        self.results = self.ctx.Queue()
        self.lock = threading.Lock()
        self.workers: Dict[int, Dict] = {}     # id -> {process, tasks, state, last_active, started, ...}
        self.pending: List = []                # (task_id, items, future, submitted)
        self.futures: Dict[int, tuple] = {}
        self.next_worker = 0
        self.next_task = 0
        self.min_workers = 0
        self.history: List[Dict] = []          # finished workers
        self.closed = False
        threading.Thread(target=self._collect, daemon=True).start()
        threading.Thread(target=self._reaper, daemon=True).start()

    # ------------------------------------------------------------------ public
    def submit(self, items: List) -> Future:
        future = Future()
        with self.lock:
            task_id = self.next_task
            self.next_task += 1
            self.pending.append((task_id, items, future, time.time()))
            self._dispatch()
        return future

    def prewarm(self, n: int) -> None:
        """Keep at least n workers running (0 lets the pool scale to zero again)."""
        with self.lock:
            self.min_workers = min(n, self.max_workers)
            while len(self.workers) < self.min_workers:
                self._start_worker()

    def wait_until_ready(self, timeout: float = 600) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self.lock:
                if all(w["state"] != "starting" for w in self.workers.values()):
                    return
            time.sleep(0.1)

    def wait_until_idle_stopped(self, timeout: float = 600) -> None:
        """Wait until every worker has scaled down (so idle time is counted)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self.lock:
                if not self.workers:
                    return
            time.sleep(0.2)

    def close(self) -> None:
        with self.lock:
            self.closed = True
            for wid in list(self.workers):
                self._stop_worker(wid)

    def snapshot(self) -> List[Dict]:
        """Lifetimes of all workers so far (running ones end now)."""
        with self.lock:
            now = time.time()
            live = [{k: w[k] for k in ("id", "started", "ready", "load_seconds")} | {"stopped": now}
                    for w in self.workers.values()]
            return self.history + live

    # ------------------------------------------------------------------ internals (hold self.lock)
    def _start_worker(self) -> None:
        wid = self.next_worker
        self.next_worker += 1
        tasks = self.ctx.Queue()
        process = self.ctx.Process(target=_worker_main, args=(self.modality, wid, self.threads, tasks, self.results),
                                   daemon=True)
        process.start()
        self.workers[wid] = {"id": wid, "process": process, "tasks": tasks, "state": "starting",
                             "started": time.time(), "ready": None, "load_seconds": None, "last_active": time.time()}

    def _stop_worker(self, wid: int) -> None:
        worker = self.workers.pop(wid)
        worker["tasks"].put(None)
        self.history.append({k: worker[k] for k in ("id", "started", "ready", "load_seconds")} | {"stopped": time.time()})

    def _dispatch(self) -> None:
        idle = [w for w in self.workers.values() if w["state"] == "idle"]
        while self.pending and idle:
            worker = idle.pop()
            task_id, items, future, submitted = self.pending.pop(0)
            self.futures[task_id] = (future, submitted, worker["id"])
            worker["state"] = "busy"
            worker["tasks"].put((task_id, items))
        starting = sum(w["state"] == "starting" for w in self.workers.values())
        while len(self.pending) > starting and len(self.workers) < self.max_workers:
            self._start_worker()
            starting += 1

    def _collect(self) -> None:
        while not self.closed:
            try:
                kind, key, payload = self.results.get(timeout=0.2)
            except queue.Empty:
                continue
            with self.lock:
                if kind == "ready" and key in self.workers:
                    self.workers[key].update(state="idle", ready=payload["ready"],
                                             load_seconds=payload["load_seconds"], last_active=time.time())
                elif kind == "failed":
                    self.workers.pop(key, None)
                    for _, _, future, _ in self.pending:
                        future.set_exception(RuntimeError(f"{self.modality} worker failed to start: {payload}"))
                    self.pending.clear()
                elif kind in ("done", "error"):
                    future, submitted, wid = self.futures.pop(key)
                    if wid in self.workers:
                        self.workers[wid].update(state="idle", last_active=time.time())
                    if kind == "done":
                        payload["latency_seconds"] = time.time() - submitted
                        future.set_result(payload)
                    else:
                        future.set_exception(RuntimeError(payload))
                self._dispatch()

    def _reaper(self) -> None:
        while not self.closed:
            time.sleep(0.2)
            with self.lock:
                now = time.time()
                idle = sorted((w for w in self.workers.values() if w["state"] == "idle"), key=lambda w: w["last_active"])
                for worker in idle:
                    if len(self.workers) <= self.min_workers:
                        break
                    if now - worker["last_active"] >= self.scaledown_window:
                        self._stop_worker(worker["id"])


class LocalBackend:
    """One ElasticPool per modality; the same interface as the Modal backend."""

    name = "local"

    def __init__(self, max_workers: Dict[str, int] = None, scaledown_windows: Dict[str, float] = None):
        """scaledown_windows defaults to the Modal settings (radiology 60 s, labs 30 s, notes 45 s)."""
        from atlas_pipeline.benchmark import RESOURCES
        cores = os.cpu_count() or 4
        max_workers = max_workers or {"radiology": 3, "labs": 2, "notes": 2}
        windows = scaledown_windows or {m: r["scaledown_window"] for m, r in RESOURCES.items()}
        threads = max(1, cores // sum(max_workers.values()))
        self.pools = {m: ElasticPool(m, n, windows[m], threads) for m, n in max_workers.items()}

    def call(self, modality: str, items: List) -> Dict:
        return self.pools[modality].submit(items).result()

    def prewarm(self, modality: str, n: int) -> None:
        self.pools[modality].prewarm(n)
        self.pools[modality].wait_until_ready()

    def workers(self) -> Dict[str, List[Dict]]:
        return {m: pool.snapshot() for m, pool in self.pools.items()}

    def wait_for_scale_down(self) -> None:
        for pool in self.pools.values():
            pool.wait_until_idle_stopped()

    def close(self) -> None:
        for pool in self.pools.values():
            pool.close()
