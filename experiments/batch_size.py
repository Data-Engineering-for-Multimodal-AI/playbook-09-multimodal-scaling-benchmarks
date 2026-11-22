"""Try this 2: how many X-rays per request? (make try-batch-size)

The benchmark sends 5 cases per request (batch_size=5 in atlas_pipeline/benchmark.py). Each
request is one call to a worker, and the worker encodes its items together. This sends the
same 40 X-rays to 2 warm radiology workers (pre-warmed, so no cold starts are measured) in
requests of 1, 5, 10 and 40, and measures the time for all 40 and the slowest request.
Local worker processes on CPU; no Modal account.
"""

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from atlas_pipeline import data  # noqa: E402
from atlas_pipeline.local_runtime import ElasticPool  # noqa: E402

N = 40
WORKERS = 2


def main():
    paths = data.xray_files("test").sample(n=N, random_state=2)["path"].tolist()
    pool = ElasticPool("radiology", max_workers=WORKERS, scaledown_window=600,
                       threads_per_worker=max(1, (os.cpu_count() or 4) // WORKERS))
    try:
        pool.prewarm(WORKERS)
        pool.wait_until_ready()
        [f.result() for f in [pool.submit(paths[:2]) for _ in range(WORKERS)]]  # first-call warm-up
        print(f"{N} X-rays to {WORKERS} warm radiology workers\n")
        print(f"{'per request':>11} {'requests':>9} {'all done':>9} {'X-rays/s':>9} {'slowest request':>16} "
              f"{'compute per X-ray':>18}")
        for size in (1, 5, 10, 40):
            start = time.time()
            futures = [pool.submit(paths[i:i + size]) for i in range(0, N, size)]
            results = [f.result() for f in futures]
            wall = time.time() - start
            compute = sum(r["compute_seconds"] for r in results) / N
            print(f"{size:>11} {len(results):>9} {wall:>8.1f}s {N / wall:>9.1f} "
                  f"{max(r['latency_seconds'] for r in results):>15.1f}s {compute * 1000:>15.0f} ms")
    finally:
        pool.close()


if __name__ == "__main__":
    main()
