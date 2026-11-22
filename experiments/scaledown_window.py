"""Try this 1: what does the scale-down window cost, and what does it save? (make try-scaledown-window)

A serverless worker that has been idle for its scale-down window stops, so the next request
pays a cold start; a longer window keeps it warm, and you pay for the idle time. This sends
two bursts of 10 X-rays to the radiology service, with 15 quiet seconds between them, with three windows,
and measures the second burst and the worker-seconds, priced at Modal's A100 rate as the
benchmark does. Local worker processes on CPU; no Modal account, nothing is sent anywhere.
"""

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from atlas_pipeline import benchmark, data  # noqa: E402
from atlas_pipeline.local_runtime import ElasticPool  # noqa: E402

GAP = 15            # seconds between the two bursts
BURST = 10          # X-rays per burst, sent as 2 requests of 5
WORKERS = 2


def burst(pool, paths):
    submitted = time.time()
    futures = [pool.submit(paths[i:i + 5]) for i in range(0, len(paths), 5)]
    results = [f.result() for f in futures]
    cold = sum(r["container_ready"] > submitted for r in results)
    return max(r["latency_seconds"] for r in results), cold


def main():
    paths = data.xray_files("test").sample(n=2 * BURST, random_state=1)["path"].tolist()
    rate = benchmark.cost_per_second("radiology")
    threads = max(1, (os.cpu_count() or 4) // WORKERS)
    print(f"Two bursts of {BURST} X-rays, {GAP} s of quiet between them, radiology on up to {WORKERS} local workers\n")
    print(f"{'window':>7} {'1st burst':>10} {'2nd burst':>10} {'2nd burst cold':>15} {'workers started':>16} "
          f"{'worker-s':>9} {'at A100 rate':>13}")
    for window in (5, 30, 90):
        pool = ElasticPool("radiology", max_workers=WORKERS, scaledown_window=window, threads_per_worker=threads)
        try:
            first, _ = burst(pool, paths[:BURST])
            time.sleep(GAP)  # quiet time after the first burst's last result
            slowest, cold = burst(pool, paths[BURST:])
            pool.wait_until_idle_stopped()
            workers = pool.snapshot()
        finally:
            pool.close()
        seconds = sum(w["stopped"] - w["started"] for w in workers)
        print(f"{window:>6}s {first:>9.1f}s {slowest:>9.1f}s {cold:>10} of 2 {len(workers):>16} {seconds:>9.1f} "
              f"{'$' + format(seconds * rate, '.4f'):>13}")
    print("\n1st and 2nd burst: time until the burst's last result. worker-s: worker lifetime, idle time included.")


if __name__ == "__main__":
    main()
