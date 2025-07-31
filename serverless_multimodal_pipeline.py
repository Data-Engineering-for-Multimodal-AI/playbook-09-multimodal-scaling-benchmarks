#!/usr/bin/env python3
"""
Serverless Multimodal Medical Inference Pipeline using Modal
Demonstrates elastic GPU scaling with zero idle cost.
"""

import modal
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import json
import time
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
import logging

# Modal app definition
app = modal.App("atlas-health-multimodal")

# Define container images with dependencies and include sample data
gpu_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch", "numpy", "pandas", "scikit-learn"
).add_local_dir("./sample-data", "/root/sample-data")

cpu_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "sentence-transformers", "numpy", "pandas", "scikit-learn", "transformers", "openpyxl"
).add_local_dir("./sample-data", "/root/sample-data")

# Data loading utilities (shared across functions)
@dataclass
class RadiologyRecord:
    study_id: str
    study_type: str  # CT, MRI, XRay
    diagnosis: str
    slice_count: int

@dataclass 
class TelemetryRecord:
    patient_id: str
    age_quantile: int
    vitals: Dict[str, Any]

@dataclass
class ClinicalNote:
    note_id: str
    medical_specialty: str
    note: str

# =============================================================================
# SERVERLESS FUNCTIONS - Each with specific resource allocation
# =============================================================================

@app.function(
    gpu="A100",           # Request A100 GPU
    memory=8192,          # 8GB RAM
    timeout=300,          # 5 minute timeout
    image=gpu_image,
    container_idle_timeout=60  # Tear down after 60s idle
)
def encode_radiology_batch(metadata_batch: List[Dict]) -> np.ndarray:
    """
    GPU-intensive radiology encoder that spins up A100 on demand.
    Processes structured metadata into neural embeddings.
    """
    print(f"GPU Function: Processing {len(metadata_batch)} radiology records on A100")
    start_time = time.time()
    
    # Initialize neural network encoder
    class RadiologyEncoder(nn.Module):
        def __init__(self, embed_dim=64):
            super().__init__()
            # Diagnosis and study type vocabularies
            self.diagnoses = ['Normal', 'Pneumonia', 'COVID-19', 'Tumor', 'Stroke', 
                             'Fracture', 'COPD', 'Fibrosis', 'Hemorrhage', 'MS Lesion']
            self.study_types = ['CT', 'MRI', 'XRay']
            
            # Embedding layers
            self.diagnosis_embed = nn.Embedding(len(self.diagnoses), embed_dim // 2)
            self.study_type_embed = nn.Embedding(len(self.study_types), embed_dim // 4)
            
            # Dense fusion network
            self.fc = nn.Sequential(
                nn.Linear(embed_dim // 2 + embed_dim // 4 + 1, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU()
            )
            
        def encode_diagnosis(self, diagnosis: str) -> int:
            return self.diagnoses.index(diagnosis) if diagnosis in self.diagnoses else 0
        
        def encode_study_type(self, study_type: str) -> int:
            return self.study_types.index(study_type) if study_type in self.study_types else 0
        
        def forward(self, records):
            device = next(self.parameters()).device
            
            diagnoses = torch.tensor([self.encode_diagnosis(r["diagnosis"]) for r in records], device=device)
            study_types = torch.tensor([self.encode_study_type(r["study_type"]) for r in records], device=device)
            slice_counts = torch.tensor([r["slice_count"] / 350.0 for r in records], device=device).unsqueeze(1)
            
            diag_embed = self.diagnosis_embed(diagnoses)
            study_embed = self.study_type_embed(study_types)
            
            combined = torch.cat([diag_embed, study_embed, slice_counts], dim=1)
            return self.fc(combined)
    
    # Move to GPU and encode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = RadiologyEncoder().to(device)
    encoder.eval()
    
    with torch.no_grad():
        embeddings = encoder(metadata_batch).cpu().numpy()
    
    processing_time = time.time() - start_time
    print(f"GPU Function: Completed in {processing_time:.3f}s on {device}")
    print(f"GPU Function: Generated embeddings shape: {embeddings.shape}")
    
    return embeddings


@app.function(
    cpu=2,                # 2 CPU cores
    memory=4096,          # 4GB RAM
    timeout=120,          # 2 minute timeout
    image=cpu_image,
    container_idle_timeout=30
)
def encode_telemetry_batch(vitals_batch: List[Dict]) -> np.ndarray:
    """
    CPU-based telemetry encoder for time series vitals and lab values.
    Processes physiological signals into statistical features.
    """
    print(f"CPU Function: Processing {len(vitals_batch)} telemetry records")
    start_time = time.time()
    
    features = []
    for record in vitals_batch:
        vitals = record["vitals"]
        feature_vector = []
        
        # Time series aggregation
        heart_rate = vitals.get("heart_rate", [75] * 10)
        spo2 = vitals.get("spo2", [98] * 10)
        
        # Statistical features
        feature_vector.extend([
            np.mean(heart_rate), np.std(heart_rate),
            np.mean(spo2), np.std(spo2)
        ])
        
        # Laboratory values
        lab_features = [
            vitals.get("hematocrit", 40.0),
            vitals.get("hemoglobin", 14.0),
            vitals.get("platelets", 250000.0),
            vitals.get("leukocytes", 7000.0),
            vitals.get("lymphocytes", 2500.0),
            vitals.get("monocytes", 500.0)
        ]
        feature_vector.extend(lab_features)
        
        # Demographics
        feature_vector.append(record.get("age_quantile", 10))
        
        features.append(feature_vector)
    
    # Standardization
    features = np.array(features)
    features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
    
    processing_time = time.time() - start_time
    print(f"CPU Function: Completed in {processing_time:.3f}s")
    print(f"CPU Function: Generated features shape: {features.shape}")
    
    return features


@app.function(
    cpu=4,                # 4 CPU cores for transformer inference
    memory=8192,          # 8GB RAM for model weights
    timeout=180,          # 3 minute timeout
    image=cpu_image,
    container_idle_timeout=45
)
def encode_clinical_notes_batch(notes_batch: List[Dict]) -> np.ndarray:
    """
    CPU-based text encoder using sentence transformers.
    Processes clinical narratives into semantic embeddings.
    """
    print(f"CPU Function: Processing {len(notes_batch)} clinical notes")
    start_time = time.time()
    
    from sentence_transformers import SentenceTransformer
    
    # Load pre-trained transformer
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Extract and preprocess text
    texts = []
    for record in notes_batch:
        text = record["note"][:1000]  # Truncate for efficiency
        texts.append(text)
    
    # Generate embeddings
    embeddings = model.encode(texts, show_progress_bar=False, device='cpu')
    
    processing_time = time.time() - start_time
    print(f"CPU Function: Completed in {processing_time:.3f}s")
    print(f"CPU Function: Generated embeddings shape: {embeddings.shape}")
    
    return embeddings


@app.function(
    cpu=1,                # Light compute for fusion
    memory=2048,          # 2GB RAM
    timeout=60,           # 1 minute timeout
    image=cpu_image
)
def fuse_modalities(radiology_embeds: np.ndarray, 
                   telemetry_embeds: np.ndarray, 
                   text_embeds: np.ndarray,
                   patient_ids: List[str]) -> List[Dict]:
    """
    Fusion function that combines all modality embeddings and generates triage scores.
    """
    print(f"Fusion Function: Combining modalities for {len(patient_ids)} patients")
    start_time = time.time()
    
    from sklearn.linear_model import LogisticRegression
    
    # Concatenate all embeddings
    combined_features = np.concatenate([
        radiology_embeds, telemetry_embeds, text_embeds
    ], axis=1)
    
    print(f"Combined feature dimension: {combined_features.shape[1]}")
    
    # Simple logistic regression for triage scoring
    # In practice, this would be a pre-trained model
    synthetic_labels = (combined_features.mean(axis=1) > combined_features.mean()).astype(int)
    classifier = LogisticRegression(random_state=42, max_iter=1000)
    classifier.fit(combined_features, synthetic_labels)
    
    # Generate triage scores
    triage_scores = classifier.predict_proba(combined_features)[:, 1]
    
    # Format results
    results = []
    for i, patient_id in enumerate(patient_ids):
        results.append({
            "patient_id": patient_id,
            "triage_score": float(triage_scores[i]),
            "timestamp": time.time(),
            "modality_dims": {
                "radiology": radiology_embeds.shape[1],
                "telemetry": telemetry_embeds.shape[1], 
                "text": text_embeds.shape[1]
            }
        })
    
    processing_time = time.time() - start_time
    print(f"Fusion Function: Completed in {processing_time:.3f}s")
    
    return results


# =============================================================================
# DAG ORCHESTRATION - Chains functions into inference pipeline
# =============================================================================

@app.local_entrypoint()
def multimodal_inference_dag(radiology_data: List[Dict], 
                           telemetry_data: List[Dict], 
                           clinical_notes: List[Dict]) -> List[Dict]:
    """
    Main inference DAG that orchestrates the multimodal pipeline.
    
    This function demonstrates:
    1. Parallel execution of modality encoders
    2. GPU spinning up only for radiology workload
    3. CPU scaling independently for telemetry and text
    4. Automatic teardown when idle
    """
    print(f"Starting multimodal inference DAG")
    print(f"   Radiology records: {len(radiology_data)}")
    print(f"   Telemetry records: {len(telemetry_data)}")
    print(f"   Clinical notes: {len(clinical_notes)}")
    
    dag_start = time.time()
    
    # Step 1: Launch parallel encoding functions
    print("\nStep 1: Launching parallel modality encoders...")
    
    # These function calls will:
    # - Spin up A100 GPU for radiology (cold start ~10-30s)
    # - Spin up CPU containers for telemetry and text
    # - Execute in parallel once containers are ready
    radiology_future = encode_radiology_batch.remote(radiology_data)
    telemetry_future = encode_telemetry_batch.remote(telemetry_data) 
    text_future = encode_clinical_notes_batch.remote(clinical_notes)
    
    print("   Waiting for modality encoders to complete...")
    
    # Step 2: Collect results (this blocks until all functions complete)
    radiology_embeds = radiology_future
    telemetry_embeds = telemetry_future
    text_embeds = text_future
    
    print("   All modality encoders completed")
    
    # Step 3: Fusion step
    print("\nStep 2: Fusion and scoring...")
    patient_ids = [f"P{i:03d}" for i in range(len(radiology_data))]
    
    results = fuse_modalities.remote(
        radiology_embeds, telemetry_embeds, text_embeds, patient_ids
    )
    
    total_time = time.time() - dag_start
    print(f"\nDAG completed in {total_time:.2f}s")
    print(f"   Average triage score: {np.mean([r['triage_score'] for r in results]):.3f}")
    
    return results


# =============================================================================
# WORKLOAD BURST SIMULATION - Tests elastic scaling behavior
# =============================================================================

@app.function(
    cpu=1,
    memory=2048,  # Increased memory for data loading
    image=cpu_image
)
def load_sample_workload(scenario: str, size: int) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Load real sample data from medical datasets.
    """
    print(f"Loading {scenario} workload with {size} samples from real medical data")
    
    # Load radiology data from CSV
    radiology_df = pd.read_csv("/root/sample-data/radiology_metadata.csv")
    radiology_data = []
    
    # Sample or repeat data to match requested size
    for i in range(min(size, len(radiology_df))):
        row = radiology_df.iloc[i % len(radiology_df)]
        radiology_data.append({
            "study_id": str(row["study_id"]).zfill(3),
            "study_type": row["study_type"],
            "diagnosis": row["diagnosis"],
            "slice_count": int(row["slice_count"])
        })
    
    # Load telemetry data from Excel
    telemetry_df = pd.read_excel("/root/sample-data/telemetry_1.xlsx")
    telemetry_data = []
    
    for i in range(min(size, len(telemetry_df))):
        row = telemetry_df.iloc[i % len(telemetry_df)]
        
        # Extract key vitals and lab values, handling NaN values
        vitals = {}
        
        # Basic vitals (simulated time series from single values)
        if not pd.isna(row.get("Hematocrit", np.nan)):
            vitals["hematocrit"] = float(row["Hematocrit"])
        if not pd.isna(row.get("Hemoglobin", np.nan)):
            vitals["hemoglobin"] = float(row["Hemoglobin"])
        if not pd.isna(row.get("Platelets", np.nan)):
            vitals["platelets"] = float(row["Platelets"])
        if not pd.isna(row.get("Leukocytes", np.nan)):
            vitals["leukocytes"] = float(row["Leukocytes"])
        if not pd.isna(row.get("Lymphocytes", np.nan)):
            vitals["lymphocytes"] = float(row["Lymphocytes"])
        if not pd.isna(row.get("Monocytes", np.nan)):
            vitals["monocytes"] = float(row["Monocytes"])
        
        # Simulate heart rate and spo2 time series (not in dataset)
        vitals["heart_rate"] = [75 + np.random.normal(0, 5) for _ in range(10)]
        vitals["spo2"] = [98 + np.random.normal(0, 1) for _ in range(10)]
        
        telemetry_data.append({
            "patient_id": str(row["Patient ID"])[:10],  # Truncate long IDs
            "age_quantile": int(row["Patient age quantile"]),
            "vitals": vitals
        })
    
    # Load clinical notes from CSV
    notes_df = pd.read_csv("/root/sample-data/mtsamples.csv")
    clinical_notes = []
    
    for i in range(min(size, len(notes_df))):
        row = notes_df.iloc[i % len(notes_df)]
        
        # Extract meaningful note data
        note_text = str(row["transcription"])[:1000]  # Truncate for efficiency
        specialty = str(row["medical_specialty"]) if not pd.isna(row["medical_specialty"]) else "General Medicine"
        
        clinical_notes.append({
            "note_id": f"NOTE_{i:04d}",
            "medical_specialty": specialty,
            "note": note_text
        })
    
    print(f"Loaded {len(radiology_data)} radiology records, {len(telemetry_data)} telemetry records, {len(clinical_notes)} clinical notes")
    return radiology_data, telemetry_data, clinical_notes


@app.local_entrypoint()  
def simulate_clinical_workload_bursts():
    """
    Simulate different clinical workload patterns using real medical data to observe elastic scaling:
    
    Uses real sample data:
    - Radiology metadata (100 real imaging studies)
    - Telemetry data (5,644 COVID-19 patient records)  
    - Clinical notes (4,999 medical transcription samples)
    
    Workload patterns:
    1. Morning imaging surge (GPU burst)
    2. Continuous telemetry stream (CPU steady)
    3. Evening dictation burst (CPU + GPU mixed)
    
    Watch the Modal dashboard to see:
    - GPU nodes spinning up during radiology bursts
    - CPU containers scaling independently
    - Automatic teardown during idle periods
    """
    print("Atlas Health Clinical Workload Simulation")
    print("=" * 60)
    
    scenarios = [
        ("morning_imaging_surge", 50, "Morning imaging surge - GPU intensive"),
        ("telemetry_stream", 30, "Telemetry stream - CPU steady state"),
        ("evening_mixed_burst", 80, "Evening mixed burst - Multi-modal"),
        ("weekend_light_load", 15, "Weekend light load - Minimal scaling")
    ]
    
    all_results = []
    
    for scenario, size, description in scenarios:
        print(f"\n{description}")
        print("-" * 40)
        
        # Load real sample workload
        workload_start = time.time()
        radiology, telemetry, notes = load_sample_workload.remote(scenario, size)
        
        # Process through pipeline
        scenario_start = time.time()
        results = multimodal_inference_dag(radiology, telemetry, notes)
        scenario_time = time.time() - scenario_start
        
        print(f"Scenario completed in {scenario_time:.2f}s")
        print(f"Processed {len(results)} patients")
        print(f"Score range: {min(r['triage_score'] for r in results):.3f} - {max(r['triage_score'] for r in results):.3f}")
        
        # Add scenario metadata
        for result in results:
            result['scenario'] = scenario
            result['scenario_time'] = scenario_time
            
        all_results.extend(results)
        
        # Brief pause between scenarios to observe teardown
        if scenario != scenarios[-1][0]:  # Don't pause after last scenario
            print("Pausing 30s to observe container teardown...")
            time.sleep(30)
    
    print(f"\nAll scenarios completed!")
    print(f"Total patients processed: {len(all_results)}")
    
    # Save comprehensive results
    with open("serverless_scaling_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Results saved to serverless_scaling_results.json")
    
    return all_results


# =============================================================================
# MONITORING AND OBSERVABILITY
# =============================================================================

@app.function(cpu=1, memory=1024, image=cpu_image)
def analyze_scaling_patterns(results: List[Dict]) -> Dict[str, Any]:
    """
    Analyze scaling behavior and cost patterns from inference results.
    """
    import pandas as pd
    
    df = pd.DataFrame(results)
    
    analysis = {
        "total_patients": len(results),
        "scenarios": df['scenario'].value_counts().to_dict(),
        "avg_triage_score": df['triage_score'].mean(),
        "score_distribution": {
            "min": df['triage_score'].min(),
            "max": df['triage_score'].max(),
            "std": df['triage_score'].std()
        },
        "performance_by_scenario": df.groupby('scenario')['scenario_time'].agg(['mean', 'std']).to_dict(),
        "modality_dimensions": {
            "radiology": results[0]['modality_dims']['radiology'],
            "telemetry": results[0]['modality_dims']['telemetry'],
            "text": results[0]['modality_dims']['text']
        }
    }
    
    return analysis


if __name__ == "__main__":
    # This file is designed to be run with Modal CLI using real medical data:
    # modal run serverless_multimodal_pipeline.py::simulate_clinical_workload_bursts
    print("Use Modal CLI to run the serverless pipeline with real medical data:")
    print("   modal run serverless_multimodal_pipeline.py::simulate_clinical_workload_bursts")
    print("   modal run serverless_multimodal_pipeline.py::multimodal_inference_dag --radiology-data=... --telemetry-data=... --clinical-notes=...")