"""Try this 3: how far should one service scale out? (make try-max-workers)

The local radiology pool may start up to 3 workers (LocalBackend in atlas_pipeline/local_runtime.py);
on Modal the platform decides. Each new worker is a cold start you pay for. This sends a burst
of 40 X-rays (8 requests of 5) to a radiology pool that starts at zero, with a cap of 1, 2 and 4
workers sharing this machine's cores, and measures the burst, the workers started and
the worker-seconds, priced at Modal's A100 rate. Local worker processes on CPU; no Modal account.
"""

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from atlas_pipeline import benchmark, data  # noqa: E402
from atlas_pipeline.local_runtime import ElasticPool  # noqa: E402

N = 40
WINDOW = 5   # short scale-down window, so idle time adds little to the totals


def main():
    paths = data.xray_files("test").sample(n=N, random_state=3)["path"].tolist()
    rate = benchmark.cost_per_second("radiology")
    cores = os.cpu_count() or 4
    print(f"A burst of {N} X-rays (8 requests of 5) at a scaled-to-zero radiology pool, {cores} cores\n")
    print(f"{'max workers':>11} {'threads each':>13} {'burst':>7} {'workers started':>16} {'mean cold start':>16} "
          f"{'worker-s':>9} {'at A100 rate':>13}")
    for cap in (1, 2, 4):
        threads = max(1, cores // cap)
        pool = ElasticPool("radiology", max_workers=cap, scaledown_window=WINDOW, threads_per_worker=threads)
        try:
            start = time.time()
            [f.result() for f in [pool.submit(paths[i:i + 5]) for i in range(0, N, 5)]]
            wall = time.time() - start
            pool.wait_until_idle_stopped()
            workers = pool.snapshot()
        finally:
            pool.close()
        seconds = sum(w["stopped"] - w["started"] for w in workers)
        cold = sum(w["ready"] - w["started"] for w in workers) / len(workers)
        print(f"{cap:>11} {threads:>13} {wall:>6.1f}s {len(workers):>16} {cold:>15.1f}s {seconds:>9.1f} "
              f"{'$' + format(seconds * rate, '.4f'):>13}")


if __name__ == "__main__":
    main()
