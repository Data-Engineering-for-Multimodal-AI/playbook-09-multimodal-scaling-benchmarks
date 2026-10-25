"""Train and evaluate one classifier head per modality on real labels (python train_models.py).

- radiology: logistic regression on ResNet-18 embeddings; trained on a balanced sample of the
  X-ray training folder, tested on the whole test folder (624 images)
- labs: logistic regression on the blood counts; 70/30 split of the 603 patients with labs
- notes: logistic regression on MiniLM embeddings; 80/20 split of the 4,966 transcriptions
"""
import json
import os
import time
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from atlas_pipeline import config, data


def split_indices(modality: str) -> Dict[str, np.ndarray]:
    """Deterministic train/test row indices for 'labs' or 'notes'."""
    if modality == "labs":
        y, test_size = data.labs()["admitted"].to_numpy(), 0.3
    else:
        y, test_size = data.notes()["pulmonary"].to_numpy(), 0.2
    train, test = train_test_split(np.arange(len(y)), test_size=test_size, random_state=config.SEED, stratify=y)
    return {"train": np.sort(train), "test": np.sort(test)}


def _metrics(y, p) -> Dict[str, float]:
    return {"auc": float(roc_auc_score(y, p)), "accuracy": float(accuracy_score(y, p > 0.5)),
            "f1": float(f1_score(y, p > 0.5)), "n": int(len(y)), "positives": int(np.sum(y))}


def _cached_embeddings(name: str, compute):
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    path = os.path.join(config.CACHE_DIR, f"{name}.npy")
    if os.path.exists(path):
        return np.load(path)
    values = compute()
    np.save(path, values)
    return values


def train_all(log=print) -> Dict[str, Dict]:
    import joblib
    from atlas_pipeline.encoders import RadiologyEncoder, TextEncoder
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    report = {}

    # Radiology: balanced sample of the training folder
    started = time.time()
    train = data.xray_files("train")
    per_class = config.XRAY_TRAIN_PER_CLASS
    train = pd.concat([group.sample(n=min(per_class, len(group)), random_state=config.SEED)
                       for _, group in train.groupby("label")]).sort_values("path")
    test = data.xray_files("test")
    encoder = RadiologyEncoder()
    X_train = _cached_embeddings(f"xray_train_{per_class}", lambda: encoder.encode(train["path"].tolist()))
    X_test = _cached_embeddings("xray_test", lambda: encoder.encode(test["path"].tolist()))
    head = LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced").fit(X_train, train["label"])
    report["radiology"] = {"task": "pneumonia vs normal chest X-ray", "train_images": len(train),
                           **_metrics(test["label"].to_numpy(), head.predict_proba(X_test)[:, 1]),
                           "seconds": round(time.time() - started, 1)}
    joblib.dump(head, os.path.join(config.MODELS_DIR, "radiology.joblib"))
    log(f"radiology: {report['radiology']}")

    # Labs
    started = time.time()
    lab_rows, split = data.labs(), split_indices("labs")
    X, y = data.lab_features(lab_rows), lab_rows["admitted"].to_numpy()
    head = LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced").fit(X[split["train"]], y[split["train"]])
    report["labs"] = {"task": "hospital admission from blood counts", "train_patients": len(split["train"]),
                      **_metrics(y[split["test"]], head.predict_proba(X[split["test"]])[:, 1]),
                      "seconds": round(time.time() - started, 1)}
    joblib.dump(head, os.path.join(config.MODELS_DIR, "labs.joblib"))
    log(f"labs: {report['labs']}")

    # Notes
    started = time.time()
    note_rows, split = data.notes(), split_indices("notes")
    text_encoder = TextEncoder()
    X = _cached_embeddings("notes_all", lambda: text_encoder.encode(note_rows["text"].tolist()))
    y = note_rows["pulmonary"].to_numpy()
    head = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced").fit(X[split["train"]], y[split["train"]])
    report["notes"] = {"task": "cardiovascular/pulmonary note", "train_notes": len(split["train"]),
                       **_metrics(y[split["test"]], head.predict_proba(X[split["test"]])[:, 1]),
                       "seconds": round(time.time() - started, 1)}
    joblib.dump(head, os.path.join(config.MODELS_DIR, "notes.joblib"))
    log(f"notes: {report['notes']}")

    with open(os.path.join(config.MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2)
    return report
