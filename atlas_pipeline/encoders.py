"""Encoders for the three modalities, each paired with its trained head (see models.py).

Each service is what one serverless container (or local worker) loads once at start-up, which is
the cold start, and then calls for every batch.
"""
import os
import time
from typing import Dict, List

import numpy as np

from atlas_pipeline import config


def _load_head(name: str):
    import joblib
    path = os.path.join(config.MODELS_DIR, f"{name}.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found: run `python train_models.py` first")
    return joblib.load(path)


class RadiologyEncoder:
    """ResNet-18 (ImageNet weights) on a chest X-ray: a 512-number embedding per image."""

    def __init__(self, device: str = None):
        import torch
        import torchvision
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        weights = torchvision.models.ResNet18_Weights.DEFAULT
        model = torchvision.models.resnet18(weights=weights)
        model.fc = torch.nn.Identity()
        self.model = model.eval().to(self.device)
        self.transform = weights.transforms()

    def encode(self, paths: List[str], batch_size: int = 32) -> np.ndarray:
        from PIL import Image
        out = []
        for start in range(0, len(paths), batch_size):
            batch = [self.transform(Image.open(os.path.join(config.XRAY_DIR, p)).convert("RGB"))
                     for p in paths[start:start + batch_size]]
            with self.torch.no_grad():
                out.append(self.model(self.torch.stack(batch).to(self.device)).cpu().numpy())
        return np.vstack(out) if out else np.zeros((0, 512))


class TextEncoder:
    """all-MiniLM-L6-v2 sentence embeddings (384 numbers) of a clinical note."""

    def __init__(self, device: str = None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(config.TEXT_MODEL, device=device)

    def encode(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=32, convert_to_numpy=True, show_progress_bar=False)


class ModalityService:
    """
    An encoder plus its head: score(items) returns the probability per item.

    'radiology' (items: X-ray paths) -> probability of pneumonia
    'labs'      (items: lab feature vectors) -> probability of hospital admission
    'notes'     (items: note texts) -> probability the note is cardiovascular/pulmonary
    """

    def __init__(self, modality: str, device: str = None):
        started = time.time()
        self.modality = modality
        if modality == "radiology":
            self.encoder = RadiologyEncoder(device)
        elif modality == "notes":
            self.encoder = TextEncoder(device)
        elif modality != "labs":
            raise ValueError(modality)
        self.head = _load_head(modality)
        self.load_seconds = time.time() - started

    def score(self, items: List) -> Dict:
        started = time.time()
        if self.modality == "labs":
            features = np.asarray(items, dtype=float)
        else:
            features = self.encoder.encode(items)
        probabilities = self.head.predict_proba(features)[:, 1] if len(items) else np.zeros(0)
        return {"probabilities": probabilities.tolist(), "compute_seconds": time.time() - started}
