#!/usr/bin/env python3
"""Train the three modality models on real labels and print their test scores.

    python train_models.py

Writes models/{radiology,labs,notes}.joblib and models/metrics.json. Both backends (local and
Modal) load these files. The first run downloads ResNet-18 (45 MB) and all-MiniLM-L6-v2 (90 MB)
and embeds about 1,600 X-rays and 5,000 notes (a few minutes on a laptop CPU).
"""
from atlas_pipeline.models import train_all

if __name__ == "__main__":
    report = train_all()
    print("\nTest scores (records no model trained on):")
    for modality, r in report.items():
        print(f"  {modality:9s} {r['task']:40s} AUC {r['auc']:.3f}  accuracy {r['accuracy']:.3f}  "
              f"F1 {r['f1']:.3f}  ({r['n']} records, {r['positives']} positive)")
