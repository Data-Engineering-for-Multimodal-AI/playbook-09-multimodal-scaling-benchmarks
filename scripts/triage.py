"""Score one case by hand with the three trained models (make triage).

    make triage XRAY=sample-data/chest_xray/test/PNEUMONIA/person100_bacteria_475.jpeg \
                NOTE="Productive cough and fever for three days, crackles at the right base."

Give an X-ray, a note, or both. The models load in this process (no workers, no Modal), so the
time printed for each one is its cold start: what a fresh container pays before its first request.
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--xray", help="a chest X-ray (.jpeg .jpg .png)")
    parser.add_argument("--note", help="a clinical note")
    args = parser.parse_args()
    if not args.xray and not args.note:
        parser.error("give --xray, --note or both")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    from atlas_pipeline import benchmark
    from atlas_pipeline.encoders import ModalityService

    scores = {}
    for modality, item, what in (("radiology", args.xray and os.path.abspath(args.xray), "pneumonia"),
                                 ("notes", args.note, "cardiovascular/pulmonary note")):
        if not item:
            continue
        service = ModalityService(modality, device="cpu")
        out = service.score([item])
        scores[modality] = out["probabilities"][0]
        print(f"{modality:9s} P({what}) = {scores[modality]:.3f}   "
              f"model load {service.load_seconds:.1f} s, scoring {out['compute_seconds']:.2f} s")
    t = benchmark.triage(scores)
    print(f"triage score {t['triage_score']:.3f} (weights {', '.join(f'{m} {benchmark.config.TRIAGE_WEIGHTS[m]}' for m in scores)})"
          f"{'  URGENT' if t['urgent'] else ''}")


if __name__ == "__main__":
    main()
