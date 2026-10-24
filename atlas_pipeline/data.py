"""Load the three modalities.

- Radiology: chest X-rays (Kermany et al., CC BY 4.0), label NORMAL or PNEUMONIA from the folder.
- Labs: blood counts from the Hospital Israelita Albert Einstein COVID-19 dataset (Kaggle), for the
  603 patients who have them; label: admitted to a regular ward, semi-intensive unit or ICU.
- Notes: medical transcriptions (mtsamples.com via Kaggle); label: Cardiovascular / Pulmonary.

The three datasets describe different patients. A triage "case" pairs one record of each
(see cases()); the pairing is synthetic, the records and labels are real.
"""
import glob
import os
from functools import lru_cache
from typing import Dict, List

import numpy as np
import pandas as pd

from atlas_pipeline import config


def xray_files(split: str) -> pd.DataFrame:
    """X-ray paths (relative to the X-ray folder) and labels for 'train', 'test' or 'val'."""
    rows = []
    for label in ("NORMAL", "PNEUMONIA"):
        for path in sorted(glob.glob(os.path.join(config.XRAY_DIR, split, label, "*.jpeg"))):
            rows.append({"path": os.path.relpath(path, config.XRAY_DIR), "label": int(label == "PNEUMONIA")})
    return pd.DataFrame(rows)


@lru_cache(maxsize=1)
def labs() -> pd.DataFrame:
    """Patients with blood counts: patient_id, the LAB_COLUMNS, age quantile, admitted (0/1)."""
    df = pd.read_excel(config.LABS_FILE)
    df = df[df["Hemoglobin"].notna()].reset_index(drop=True)
    out = df[["Patient ID", "Patient age quantile"] + config.LAB_COLUMNS].rename(columns={"Patient ID": "patient_id"})
    out["admitted"] = df[config.ADMISSION_COLUMNS].max(axis=1).astype(int)
    return out


def lab_features(rows: pd.DataFrame) -> np.ndarray:
    """Lab values (missing -> 0, the dataset mean) plus missing-value flags and age quantile / 20."""
    values = rows[config.LAB_COLUMNS].astype(float)
    return np.hstack([values.fillna(0.0).to_numpy(), values.isna().to_numpy(dtype=float),
                      rows[["Patient age quantile"]].to_numpy(dtype=float) / 20.0])


@lru_cache(maxsize=1)
def notes() -> pd.DataFrame:
    """Transcriptions: note_id, specialty, text (first 1,000 characters), pulmonary (0/1)."""
    df = pd.read_csv(config.NOTES_FILE, index_col=0)
    df = df[df["transcription"].notna()].reset_index(drop=True)
    specialty = df["medical_specialty"].str.strip()
    return pd.DataFrame({"note_id": [f"N{i:04d}" for i in range(len(df))], "specialty": specialty,
                         "text": df["transcription"].str.slice(0, 1000),
                         "pulmonary": (specialty == config.PULMONARY_SPECIALTY).astype(int)})


def radiology_metadata() -> pd.DataFrame:
    """The first edition's 100 synthetic study records (study type, diagnosis, slice count)."""
    return pd.read_csv(config.RADIOLOGY_METADATA_FILE, dtype={"study_id": str})


def cases(n: int, offset: int = 0) -> List[Dict]:
    """
    n triage cases, each pairing one held-out X-ray, lab panel and note (deterministic).

    Only records no model trained on are used: the X-ray test folder and the test splits of the labs
    and notes (see models.split_indices).
    """
    from atlas_pipeline.models import split_indices
    xrays = xray_files("test").sample(frac=1.0, random_state=config.SEED).reset_index(drop=True)
    lab_rows = labs().iloc[split_indices("labs")["test"]].reset_index(drop=True)
    note_rows = notes().iloc[split_indices("notes")["test"]].reset_index(drop=True)
    out = []
    for i in range(offset, offset + n):
        xray, lab, note = xrays.iloc[i % len(xrays)], lab_rows.iloc[i % len(lab_rows)], note_rows.iloc[i % len(note_rows)]
        out.append({
            "case_id": f"C{i:04d}",
            "xray": xray["path"], "xray_label": int(xray["label"]),
            "labs": lab_features(lab_rows.iloc[[i % len(lab_rows)]])[0].tolist(), "labs_label": int(lab["admitted"]),
            "note": note["text"], "note_label": int(note["pulmonary"]),
        })
    return out
