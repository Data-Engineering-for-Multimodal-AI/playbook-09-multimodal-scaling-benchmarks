"""Check a folder of your own triage cases, then score them on local elastic workers.

    make check-data DIR=my_cases    # check every file, run nothing
    make score DIR=my_cases         # check, then score every case and write results/

The folder:

    my_cases/
      cases.csv    one row per case: case_id (required), then any of
                   xray        a file name in xrays/ (.jpeg .jpg .png)
                   note        the clinical note text
                   the 14 blood counts in atlas_pipeline/config.py LAB_COLUMNS and
                   "Patient age quantile" (0-19), standardized as in the Einstein dataset
                   xray_label, labs_label, note_label   optional true labels (0 or 1)
      xrays/       the chest X-rays cases.csv names

Each case needs at least one modality. The checks are the things that break the services or
make their scores meaningless: a missing or non-UTF-8 cases.csv, missing or duplicate case IDs,
a case with no modality, X-rays that are missing, not an image type the encoder reads, empty,
unreadable, smaller than 64x64 or Git LFS pointers, empty notes, lab values that are not numbers
or don't look standardized (the lab model was trained on z-scores; raw values such as a
platelet count of 250 are far outside it), an age quantile outside 0-19, labels other than 0
or 1, and lab columns given without the age quantile. Every problem is reported with the file
and the reason; if there is any, nothing runs.

`make score` sends the cases through the same services as the benchmark (one local worker pool
per modality, scale to zero), fuses the available scores into a triage score, and prints the
workers, cold starts and worker-seconds priced at Modal's rates. Needs the trained models
(`make setup`). Nothing is sent to Modal or anywhere else.
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from atlas_pipeline import config  # noqa: E402

AGE = "Patient age quantile"
IMAGE_TYPES = {".jpeg", ".jpg", ".png"}
MIN_SIDE = 64
MAX_Z = 10       # |value| above this is not a z-score of a blood count
LABELS = {"xray_label": "radiology", "labs_label": "labs", "note_label": "notes"}


def _norm(name):
    """Column names compared without case, extra spaces or non-breaking spaces (the dataset has both)."""
    return " ".join(name.replace("\xa0", " ").split()).lower()


CANONICAL = {_norm(c): c for c in ["case_id", "xray", "note", AGE, *config.LAB_COLUMNS, *LABELS]}


def _number(value):
    try:
        x = float(value)
    except ValueError:
        return None
    return x if math.isfinite(x) else None


def check_xray(path):
    if os.path.getsize(path) == 0:
        return "empty file"
    with open(path, "rb") as f:
        head = f.read(40)
    if head.startswith(b"version https://git-lfs"):
        return "Git LFS pointer, not the image (run git lfs pull)"
    from PIL import Image
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            if min(im.size) < MIN_SIDE:
                return f"too small ({im.size[0]}x{im.size[1]}), needs at least {MIN_SIDE}x{MIN_SIDE}"
    except Exception as e:
        return f"can't open as an image ({e.__class__.__name__})"
    return None


def check(folder):
    """Returns (cases, problems). Each case: {case_id, xray (path), note, labs (features), labels}."""
    problems, cases = [], []
    path = os.path.join(folder, "cases.csv")
    if not os.path.isdir(folder):
        return [], [f"{folder}: not a folder"]
    if not os.path.exists(path):
        return [], ["cases.csv: missing (needs a case_id column and at least one of xray, note, lab columns)"]
    try:
        with open(path, encoding="utf-8-sig", newline="") as f:
            text = f.read()
    except UnicodeDecodeError as e:
        return [], [f"cases.csv: not UTF-8 text (byte {e.start})"]
    if text.startswith("version https://git-lfs"):
        return [], ["cases.csv: Git LFS pointer, not the file (run git lfs pull)"]
    reader = csv.DictReader(text.splitlines())
    raw_columns = reader.fieldnames or []
    rename = {c: CANONICAL.get(_norm(c), c) for c in raw_columns}
    columns = [rename[c] for c in raw_columns]
    if "case_id" not in columns:
        return [], ["cases.csv: missing column case_id"]
    lab_columns = [c for c in config.LAB_COLUMNS if c in columns]
    if lab_columns and AGE not in columns:
        problems.append(f"cases.csv: lab columns given without \"{AGE}\"")
    for c in columns:
        if c not in CANONICAL.values():
            problems.append(f"cases.csv: unknown column {c!r} (see config.LAB_COLUMNS for the lab names)")

    seen = set()
    rows = [{rename.get(k, k): v for k, v in row.items()} for row in reader]
    if not rows:
        problems.append("cases.csv: no rows")
    for line, row in enumerate(rows, start=2):
        where = f"cases.csv line {line}"
        if None in row or any(v is None for v in row.values()):
            problems.append(f"{where}: {'more' if None in row else 'fewer'} values than columns")
            continue
        cid = (row.get("case_id") or "").strip()
        if not cid:
            problems.append(f"{where}: empty case_id")
        elif cid in seen:
            problems.append(f"{where}: duplicate case_id {cid}")
        seen.add(cid)
        case = {"case_id": cid, "labels": {}}

        xray = (row.get("xray") or "").strip()
        if xray:
            xpath = os.path.join(folder, "xrays", xray)
            if os.path.splitext(xray)[1].lower() not in IMAGE_TYPES:
                problems.append(f"xrays/{xray}: not an image type the encoder reads (.jpeg .jpg .png)")
            elif not os.path.exists(xpath):
                problems.append(f"xrays/{xray}: named on {where} but not found")
            else:
                reason = check_xray(xpath)
                if reason:
                    problems.append(f"xrays/{xray}: {reason}")
                case["xray"] = os.path.abspath(xpath)
        if "note" in columns and row.get("note") is not None:
            note = row["note"].strip()
            if note:
                case["note"] = note[:1000]   # the services read the first 1,000 characters, as in data.notes()
        values = {}
        for c in lab_columns + ([AGE] if lab_columns and AGE in columns else []):
            raw = (row.get(c) or "").strip()
            if not raw:
                continue
            x = _number(raw)
            if x is None:
                problems.append(f"{where}: {c.strip()} {raw!r} is not a number")
            elif c == AGE and not (0 <= x <= 19 and x == int(x)):
                problems.append(f"{where}: {AGE} {raw} is not a whole number from 0 to 19")
            elif c != AGE and abs(x) > MAX_Z:
                problems.append(f"{where}: {c.strip()} {raw} doesn't look standardized (expected a z-score, "
                                f"about -5 to 5)")
            values[c] = x
        if any(c in values for c in lab_columns):
            if AGE not in values and AGE in columns:
                problems.append(f"{where}: lab values without {AGE}")
            case["labs_values"] = values
        for col in LABELS:
            raw = (row.get(col) or "").strip()
            if raw:
                if raw not in ("0", "1"):
                    problems.append(f"{where}: {col} {raw!r} is not 0 or 1")
                else:
                    case["labels"][LABELS[col]] = int(raw)
        if cid and not any(k in case for k in ("xray", "note", "labs_values")):
            problems.append(f"{where}: case {cid} has no X-ray, note or lab values")
        cases.append(case)
    return cases, problems


def lab_vector(values):
    """The same features as data.lab_features(): values (missing -> 0), missing flags, age / 20."""
    v = [values.get(c) for c in config.LAB_COLUMNS]
    return [x if x is not None else 0.0 for x in v] + [float(x is None) for x in v] + [values[AGE] / 20.0]


def score(cases, batch_size=5):
    from atlas_pipeline import benchmark
    from atlas_pipeline.local_runtime import LocalBackend
    items = {"radiology": [(c, c["xray"]) for c in cases if "xray" in c],
             "labs": [(c, lab_vector(c["labs_values"])) for c in cases if "labs_values" in c],
             "notes": [(c, c["note"]) for c in cases if "note" in c]}
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")   # the workers inherit it
    backend = LocalBackend(scaledown_windows={"radiology": 5, "labs": 5, "notes": 5})
    started = time.time()
    try:
        with ThreadPoolExecutor(max_workers=32) as pool:
            jobs = [(m, chunk, pool.submit(backend.call, m, [x for _, x in chunk]))
                    for m, pairs in items.items()
                    for chunk in (pairs[i:i + batch_size] for i in range(0, len(pairs), batch_size))]
            for modality, chunk, future in jobs:
                for (case, _), p in zip(chunk, future.result()["probabilities"]):
                    case.setdefault("scores", {})[modality] = p
        wall = time.time() - started
        backend.wait_for_scale_down()
        workers = benchmark.worker_report(backend)
    finally:
        backend.close()
    for c in cases:
        c.update(benchmark.triage(c["scores"]))
    return wall, workers


def main():
    parser = argparse.ArgumentParser(description="Check a folder of triage cases and score them locally.")
    parser.add_argument("--dir", required=True)
    parser.add_argument("--check-only", action="store_true", help="check the files, run nothing")
    args = parser.parse_args()
    folder = os.path.abspath(args.dir)

    cases, problems = check(folder)
    if problems:
        print(f"{len(problems)} problem(s) in {args.dir}; nothing was run:")
        for p in problems:
            print(f"  {p}")
        sys.exit(1)
    counts = {m: sum(k in c for c in cases) for m, k in (("X-rays", "xray"), ("lab panels", "labs_values"),
                                                          ("notes", "note"))}
    print(f"{args.dir}: {len(cases)} cases, " + ", ".join(f"{n} {m}" for m, n in counts.items())
          + ". All checks passed.")
    if args.check_only:
        return

    print("Scoring on local elastic workers (cold starts included)...")
    wall, workers = score(cases)
    print(f"\n{'case':<12} {'radiology':>9} {'labs':>6} {'notes':>6} {'triage':>7}")
    for c in cases:
        s = c["scores"]
        cell = lambda m: f"{s[m]:.3f}" if m in s else "-"
        print(f"{c['case_id']:<12} {cell('radiology'):>9} {cell('labs'):>6} {cell('notes'):>6} "
              f"{c['triage_score']:>7.3f}{'  URGENT' if c['urgent'] else ''}")
    labelled = [(m, c["scores"][m], y) for c in cases for m, y in c["labels"].items() if m in c["scores"]]
    for m in ("radiology", "labs", "notes"):
        pairs = [(p, y) for mm, p, y in labelled if mm == m]
        if pairs:
            right = sum((p > 0.5) == bool(y) for p, y in pairs)
            print(f"{m}: {right} of {len(pairs)} labelled cases right")
    total = sum(w["cost_at_modal_rates_usd"] for w in workers.values())
    print(f"\nScored in {wall:.1f} s. Workers: " + ", ".join(
        f"{m} {w['workers_started']} (cold start {w['cold_start_mean_seconds']} s)"
        for m, w in workers.items() if w["workers_started"]) + f". ${total:.4f} at Modal's rates.")
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    out = os.path.join(config.RESULTS_DIR, "own_cases.json")
    with open(out, "w") as f:
        json.dump({"cases": [{k: c[k] for k in ("case_id", "scores", "triage_score", "urgent", "labels")}
                             for c in cases], "workers": workers, "wall_seconds": wall}, f, indent=2)
    print(f"Results saved to {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
