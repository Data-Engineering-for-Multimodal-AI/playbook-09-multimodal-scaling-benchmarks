"""Paths and settings shared by the local and Modal backends."""
import os

ROOT = os.environ.get("ATLAS_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT, "sample-data")
XRAY_DIR = os.path.join(DATA_DIR, "chest_xray")
LABS_FILE = os.path.join(DATA_DIR, "telemetry_1.xlsx")
NOTES_FILE = os.path.join(DATA_DIR, "mtsamples.csv")
RADIOLOGY_METADATA_FILE = os.path.join(DATA_DIR, "radiology_metadata.csv")

MODELS_DIR = os.path.join(ROOT, "models")        # trained heads (python train_models.py)
CACHE_DIR = os.path.join(ROOT, ".cache")         # embeddings cache for training
RESULTS_DIR = os.path.join(ROOT, "results")

SEED = 42

# Pretrained encoders (downloaded on first use)
TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ADVISOR_MODEL = os.environ.get("ATLAS_ADVISOR_MODEL", "Qwen/Qwen3-0.6B")

# Training: X-rays per class from the training folder (all 624 test images are always scored)
XRAY_TRAIN_PER_CLASS = int(os.environ.get("ATLAS_XRAY_TRAIN_PER_CLASS", "500"))

# Blood counts in the Einstein dataset (already standardized by the dataset's authors)
LAB_COLUMNS = [
    "Hematocrit", "Hemoglobin", "Platelets", "Mean platelet volume ", "Red blood Cells", "Lymphocytes",
    "Mean corpuscular hemoglobin concentration\xa0(MCHC)", "Leukocytes", "Basophils",
    "Mean corpuscular hemoglobin (MCH)", "Eosinophils", "Mean corpuscular volume (MCV)", "Monocytes",
    "Red blood cell distribution width (RDW)",
]
ADMISSION_COLUMNS = [
    "Patient addmited to regular ward (1=yes, 0=no)",
    "Patient addmited to semi-intensive unit (1=yes, 0=no)",
    "Patient addmited to intensive care unit (1=yes, 0=no)",
]
PULMONARY_SPECIALTY = "Cardiovascular / Pulmonary"

# Triage: weights of the three modality scores, and the score above which a case is urgent
TRIAGE_WEIGHTS = {"radiology": 0.5, "labs": 0.3, "notes": 0.2}
URGENT_THRESHOLD = 0.6
