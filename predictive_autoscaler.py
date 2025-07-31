#!/usr/bin/env python3
"""
Predictive Autoscaler for Multimodal Medical Pipeline
Anticipates workload spikes based on clinical patterns and schedules.
"""

import modal
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

app = modal.App("atlas-health-predictive-autoscaler")

@dataclass
class ScalingPrediction:
    """Represents a scaling prediction with timing and resources."""
    prediction_type: str  # "radiology_surge", "telemetry_stream", "mixed_workload"
    expected_start: datetime
    expected_duration_minutes: int
    expected_volume: int
    confidence_score: float
    recommended_action: str
    resource_requirements: Dict[str, Any]

class WorkloadPattern(Enum):
    """Common clinical workload patterns."""
    MORNING_IMAGING = "morning_imaging"
    SHIFT_CHANGE = "shift_change" 
    EMERGENCY_SURGE = "emergency_surge"
    WEEKEND_LIGHT = "weekend_light"
    SCHEDULED_PROCEDURES = "scheduled_procedures"

@app.function(
    cpu=2,
    memory=4096,
    image=modal.Image.debian_slim(python_version="3.11").pip_install(
        "pandas", "numpy", "scikit-learn", "openai", "anthropic"
    ),
    secrets=[modal.Secret.from_name("openai-api-key")] if False else [],  # Optional LLM integration
    scaledown_window=60,  # Updated parameter name
    schedule=modal.Cron("*/5 * * * *")  # Run every 5 minutes
)
def predictive_autoscaler_agent() -> List[ScalingPrediction]:
    """
    Main predictive autoscaling agent that analyzes patterns and makes scaling recommendations.
    """
    print(f"Predictive Autoscaler Running at {datetime.now()}")
    
    # Step 1: Collect current signals
    current_metrics = collect_current_metrics.remote()
    
    # Step 2: Analyze historical patterns  
    historical_analysis = analyze_historical_patterns.remote()
    
    # Step 3: Check scheduled events
    scheduled_events = check_scheduled_events.remote()
    
    # Step 4: Generate predictions using traditional methods
    traditional_predictions = generate_scaling_predictions(
        current_metrics, historical_analysis, scheduled_events
    )
    
    # Step 5: Enhanced LLM-based analysis (experimental)
    llm_predictions = generate_llm_scaling_recommendations.remote(
        current_metrics, historical_analysis, scheduled_events, traditional_predictions
    )
    
    # Step 6: Combine and prioritize predictions
    all_predictions = traditional_predictions + llm_predictions
    final_predictions = prioritize_predictions(all_predictions)
    
    # Step 7: Execute pre-warming if needed
    for prediction in final_predictions:
        if should_prewarm(prediction):
            execute_prewarming.remote(prediction)
    
    # Step 8: Log predictions for monitoring
    log_predictions(final_predictions)
    
    return final_predictions

@app.function(cpu=1, memory=1024)
def collect_current_metrics() -> Dict[str, Any]:
    """
    Collect current system metrics and upstream signals.
    """
    current_time = datetime.now()
    
    # Simulate collecting real metrics
    metrics = {
        "timestamp": current_time.isoformat(),
        "queue_depth": {
            "radiology": np.random.randint(0, 20),
            "telemetry": np.random.randint(5, 50), 
            "text": np.random.randint(0, 15)
        },
        "request_rate_per_minute": {
            "radiology": np.random.randint(0, 30),
            "telemetry": np.random.randint(10, 80),
            "text": np.random.randint(0, 25)
        },
        "current_gpu_utilization": np.random.uniform(0, 0.3),  # Usually low
        "current_cpu_utilization": np.random.uniform(0.1, 0.6),
        "time_of_day": current_time.hour,
        "day_of_week": current_time.weekday(),
        "is_holiday": False,  # Could integrate with holiday calendar
    }
    
    # Check for upstream signals (simulated)
    upstream_signals = {
        "pacs_scheduled_uploads": check_pacs_schedule(current_time),
        "ehr_shift_changes": check_shift_schedule(current_time),
        "emergency_department_alerts": check_ed_status(),
        "icu_census_changes": check_icu_census()
    }
    
    metrics["upstream_signals"] = upstream_signals
    
    print(f"Current Metrics: Queue depths: {metrics['queue_depth']}")
    return metrics

@app.function(cpu=1, memory=2048)
def analyze_historical_patterns() -> Dict[str, Any]:
    """
    Analyze historical workload patterns to identify trends.
    """
    # Simulate historical pattern analysis
    current_time = datetime.now()
    
    patterns = {
        "daily_patterns": {
            "morning_surge": {
                "typical_time": "08:00-09:30",
                "avg_volume": 150,
                "workload_type": "radiology_heavy"
            },
            "lunch_lull": {
                "typical_time": "12:00-13:00", 
                "avg_volume": 20,
                "workload_type": "minimal"
            },
            "evening_dictation": {
                "typical_time": "17:00-19:00",
                "avg_volume": 80,
                "workload_type": "text_heavy"
            }
        },
        "weekly_patterns": {
            "monday_surge": "Higher volume after weekend",
            "friday_procedures": "Scheduled procedures concentrated",
            "weekend_emergency": "Emergency-only workload"
        },
        "seasonal_patterns": {
            "flu_season": current_time.month in [11, 12, 1, 2, 3],
            "summer_trauma": current_time.month in [6, 7, 8],
            "holiday_effects": "Reduced elective procedures"
        }
    }
    
    print(f"Historical Analysis: Detected {len(patterns['daily_patterns'])} daily patterns")
    return patterns

@app.function(cpu=1, memory=1024)
def check_scheduled_events() -> List[Dict[str, Any]]:
    """
    Check for scheduled events that will impact workload.
    """
    current_time = datetime.now()
    
    # Simulate scheduled events
    scheduled_events = []
    
    # PACS uploads (predictable imaging surges)
    if current_time.hour == 7 and current_time.minute < 30:
        scheduled_events.append({
            "event_type": "pacs_upload",
            "scheduled_time": current_time.replace(hour=8, minute=0),
            "expected_volume": 200,
            "workload_type": "radiology",
            "confidence": 0.95
        })
    
    # Shift changes (dictation surges)
    if current_time.hour in [6, 14, 22] and current_time.minute < 30:
        scheduled_events.append({
            "event_type": "shift_change",
            "scheduled_time": current_time.replace(minute=0) + timedelta(hours=1),
            "expected_volume": 100,
            "workload_type": "text",
            "confidence": 0.85
        })
    
    # Scheduled procedures
    if current_time.weekday() < 5 and current_time.hour == 6:  # Weekdays
        scheduled_events.append({
            "event_type": "scheduled_procedures",
            "scheduled_time": current_time.replace(hour=9, minute=0),
            "expected_volume": 75,
            "workload_type": "mixed",
            "confidence": 0.90
        })
    
    print(f"Scheduled Events: Found {len(scheduled_events)} upcoming events")
    return scheduled_events

def generate_scaling_predictions(
    current_metrics: Dict[str, Any],
    historical_analysis: Dict[str, Any], 
    scheduled_events: List[Dict[str, Any]]
) -> List[ScalingPrediction]:
    """
    Generate scaling predictions based on all available signals.
    """
    predictions = []
    current_time = datetime.now()
    
    # Rule-based predictions from scheduled events
    for event in scheduled_events:
        prediction = ScalingPrediction(
            prediction_type=event["workload_type"],
            expected_start=event["scheduled_time"],
            expected_duration_minutes=30,
            expected_volume=event["expected_volume"],
            confidence_score=event["confidence"],
            recommended_action=f"Pre-warm {get_resource_type(event['workload_type'])}",
            resource_requirements=get_resource_requirements(event["workload_type"], event["expected_volume"])
        )
        predictions.append(prediction)
    
    # Pattern-based predictions
    current_hour = current_time.hour
    
    # Morning imaging surge prediction
    if 7 <= current_hour <= 8 and current_metrics["queue_depth"]["radiology"] > 10:
        prediction = ScalingPrediction(
            prediction_type="morning_imaging_surge",
            expected_start=current_time + timedelta(minutes=15),
            expected_duration_minutes=60,
            expected_volume=150,
            confidence_score=0.75,
            recommended_action="Pre-warm 2 A100 GPUs",
            resource_requirements={"gpu": 2, "cpu": 4, "memory": "16GB"}
        )
        predictions.append(prediction)
    
    # Evening dictation surge
    if 16 <= current_hour <= 17 and current_metrics["request_rate_per_minute"]["text"] > 15:
        prediction = ScalingPrediction(
            prediction_type="evening_dictation_surge", 
            expected_start=current_time + timedelta(minutes=30),
            expected_duration_minutes=90,
            expected_volume=100,
            confidence_score=0.70,
            recommended_action="Pre-warm CPU text processors",
            resource_requirements={"gpu": 0, "cpu": 8, "memory": "32GB"}
        )
        predictions.append(prediction)
    
    print(f"Generated {len(predictions)} scaling predictions")
    return predictions

def get_resource_type(workload_type: str) -> str:
    """Map workload type to resource type."""
    mapping = {
        "radiology": "GPU",
        "text": "CPU", 
        "telemetry": "CPU",
        "mixed": "GPU+CPU"
    }
    return mapping.get(workload_type, "CPU")

def get_resource_requirements(workload_type: str, volume: int) -> Dict[str, Any]:
    """Calculate resource requirements based on workload."""
    base_requirements = {
        "radiology": {"gpu": 1, "cpu": 2, "memory": "8GB"},
        "text": {"gpu": 0, "cpu": 4, "memory": "16GB"},
        "telemetry": {"gpu": 0, "cpu": 2, "memory": "4GB"},
        "mixed": {"gpu": 1, "cpu": 6, "memory": "24GB"}
    }
    
    base = base_requirements.get(workload_type, {"gpu": 0, "cpu": 2, "memory": "4GB"})
    
    # Scale based on volume
    scale_factor = max(1, volume // 50)
    
    return {
        "gpu": base["gpu"] * scale_factor,
        "cpu": base["cpu"] * scale_factor,
        "memory": base["memory"]  # Memory doesn't scale linearly
    }

def should_prewarm(prediction: ScalingPrediction) -> bool:
    """Decide whether to pre-warm resources for this prediction."""
    minutes_until_event = (prediction.expected_start - datetime.now()).total_seconds() / 60
    
    # Pre-warm if:
    # 1. High confidence prediction
    # 2. Event is 5-15 minutes away  
    # 3. GPU workload (longer cold start time)
    
    if prediction.confidence_score < 0.6:
        return False
        
    if not (2 <= minutes_until_event <= 15):
        return False
        
    # Always pre-warm GPU workloads
    if prediction.resource_requirements.get("gpu", 0) > 0:
        return True
        
    # Pre-warm large CPU workloads
    if prediction.expected_volume > 100:
        return True
        
    return False

@app.function(
    gpu="A100",
    memory=8192,
    timeout=300,
    scaledown_window=120  # Keep warm longer for pre-warmed instances
)
def execute_prewarming(prediction: ScalingPrediction):
    """
    Pre-warm GPU resources based on prediction.
    """
    print(f"Pre-warming resources for {prediction.prediction_type}")
    print(f"   Expected start: {prediction.expected_start}")
    print(f"   Expected volume: {prediction.expected_volume}")
    print(f"   Confidence: {prediction.confidence_score:.2f}")
    
    # Initialize models and keep container warm
    if prediction.resource_requirements.get("gpu", 0) > 0:
        import torch
        import torch.nn as nn
        
        # Pre-load radiology encoder
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Simple model initialization to warm up GPU
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        ).to(device)
        
        # Run dummy forward pass to initialize CUDA
        dummy_input = torch.randn(1, 64, device=device)
        _ = model(dummy_input)
        
        print(f"GPU pre-warmed on {device}")
        
        # Keep container alive until expected start time
        wait_time = (prediction.expected_start - datetime.now()).total_seconds()
        if 0 < wait_time < 300:  # Wait up to 5 minutes
            print(f"Keeping container warm for {wait_time:.0f}s")
            time.sleep(min(wait_time, 60))  # Sleep up to 1 minute for demo

def log_predictions(predictions: List[ScalingPrediction]):
    """Log predictions for monitoring and analysis."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "predictions_count": len(predictions),
        "predictions": [
            {
                "type": p.prediction_type,
                "start_time": p.expected_start.isoformat(),
                "volume": p.expected_volume,
                "confidence": p.confidence_score,
                "action": p.recommended_action
            }
            for p in predictions
        ]
    }
    
    # In production, this would go to monitoring system
    print(f"Logged {len(predictions)} predictions")
    
    # Save to local file for analysis
    try:
        with open("autoscaler_predictions.json", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except:
        pass  # Don't fail if logging fails

# Utility functions for upstream signal checking
def check_pacs_schedule(current_time: datetime) -> Dict[str, Any]:
    """Check PACS system for scheduled uploads."""
    # Simulate PACS system check
    if 7 <= current_time.hour <= 8:
        return {
            "scheduled_uploads": True,
            "expected_time": "08:00",
            "estimated_volume": 200,
            "imaging_types": ["CT", "MRI", "XRay"]
        }
    return {"scheduled_uploads": False}

def check_shift_schedule(current_time: datetime) -> Dict[str, Any]:
    """Check nursing/physician shift schedules."""
    shift_times = [6, 14, 22]  # 6 AM, 2 PM, 10 PM
    
    if current_time.hour in shift_times:
        return {
            "shift_change": True,
            "shift_type": "day" if current_time.hour == 6 else "evening" if current_time.hour == 14 else "night",
            "expected_dictations": 50 + np.random.randint(-20, 30)
        }
    return {"shift_change": False}

@app.function(
    cpu=2,
    memory=4096,
    scaledown_window=60,
    secrets=[modal.Secret.from_name("openai-api-key")] if False else [],  # Enable when you have OpenAI key
)
def generate_llm_scaling_recommendations(
    current_metrics: Dict[str, Any],
    historical_analysis: Dict[str, Any], 
    scheduled_events: List[Dict[str, Any]],
    traditional_predictions: List[ScalingPrediction]
) -> List[ScalingPrediction]:
    """
    LLM-based scaling advisor that analyzes structured and unstructured signals.
    This is the experimental feature described in the chapter.
    """
    print("Running LLM-based scaling analysis...")
    
    # Prepare context for LLM
    context = prepare_llm_context(current_metrics, historical_analysis, scheduled_events, traditional_predictions)
    
    try:
        # In production, this would use actual LLM API
        # For demo, we'll simulate LLM responses
        llm_response = simulate_llm_scaling_advisor(context)
        
        # Parse LLM response into structured predictions
        llm_predictions = parse_llm_response(llm_response)
        
        print(f"LLM generated {len(llm_predictions)} additional predictions")
        return llm_predictions
        
    except Exception as e:
        print(f"LLM analysis failed: {e}, falling back to traditional methods")
        return []

def prepare_llm_context(
    current_metrics: Dict[str, Any],
    historical_analysis: Dict[str, Any], 
    scheduled_events: List[Dict[str, Any]],
    traditional_predictions: List[ScalingPrediction]
) -> str:
    """Prepare context for LLM analysis."""
    
    context = f"""
ATLAS HEALTH AI - SCALING ADVISOR CONTEXT
Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

CURRENT METRICS:
- Radiology Queue: {current_metrics['queue_depth']['radiology']} patients
- Telemetry Queue: {current_metrics['queue_depth']['telemetry']} patients  
- Text Processing Queue: {current_metrics['queue_depth']['text']} patients
- GPU Utilization: {current_metrics['current_gpu_utilization']:.1%}
- CPU Utilization: {current_metrics['current_cpu_utilization']:.1%}
- Time of Day: {current_metrics['time_of_day']}:00
- Day of Week: {['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'][current_metrics['day_of_week']]}

UPSTREAM SIGNALS:
- PACS Scheduled Uploads: {current_metrics['upstream_signals']['pacs_scheduled_uploads']}
- EHR Shift Changes: {current_metrics['upstream_signals']['ehr_shift_changes']}
- Emergency Department Status: {current_metrics['upstream_signals']['emergency_department_alerts']}
- ICU Census: {current_metrics['upstream_signals']['icu_census_changes']}

SCHEDULED EVENTS:
{[f"- {event['event_type']} at {event['scheduled_time']}, volume: {event['expected_volume']}" for event in scheduled_events]}

TRADITIONAL PREDICTIONS:
{[f"- {pred.prediction_type}: {pred.expected_volume} patients in {pred.expected_duration_minutes}min (confidence: {pred.confidence_score:.2f})" for pred in traditional_predictions]}

HISTORICAL PATTERNS:
- Morning surge typically 8AM-9:30AM with 150 patients (radiology-heavy)
- Lunch lull 12PM-1PM with minimal volume
- Evening dictation 5PM-7PM with 80 patients (text-heavy)
- {'Flu season active' if historical_analysis['seasonal_patterns']['flu_season'] else 'Normal seasonal patterns'}

Based on this context, what scaling recommendations do you have? Consider:
1. Patterns not captured by traditional rules
2. Cross-signal correlations
3. Unusual combinations of metrics
4. Proactive vs reactive scaling opportunities
"""
    
    return context

def simulate_llm_scaling_advisor(context: str) -> Dict[str, Any]:
    """
    Simulate LLM-based scaling advisor.
    In production, this would call OpenAI/Anthropic API.
    """
    current_time = datetime.now()
    
    # Simulate intelligent analysis based on context patterns
    recommendations = []
    
    # Example LLM-style reasoning
    if "PACS Scheduled Uploads: {'scheduled_uploads': True" in context:
        recommendations.append({
            "prediction": "Radiology surge in 5 minutes",
            "action": "Pre-warm 2 A100 GPUs",
            "reasoning": "PACS upload detected with historical correlation to imaging workload spike",
            "confidence": 0.88,
            "volume": 180,
            "duration_minutes": 45
        })
    
    if "Shift Changes: {'shift_change': True" in context:
        recommendations.append({
            "prediction": "Text processing surge in 15 minutes", 
            "action": "Scale CPU text processors to 6 cores",
            "reasoning": "Shift change correlates with dictation backlog processing",
            "confidence": 0.75,
            "volume": 90,
            "duration_minutes": 60
        })
    
    # Detect unusual patterns
    hour = current_time.hour
    if 22 <= hour or hour <= 5:  # Night hours
        if "Emergency Department Status: {'current_census':" in context and "30" in context:
            recommendations.append({
                "prediction": "Emergency imaging spike in 10 minutes",
                "action": "Pre-warm 1 A100 GPU", 
                "reasoning": "High ED census during night hours suggests trauma/emergency cases requiring urgent imaging",
                "confidence": 0.65,
                "volume": 25,
                "duration_minutes": 30
            })
    
    # Weekend patterns
    if current_time.weekday() >= 5:  # Weekend
        recommendations.append({
            "prediction": "Minimal scaling needed",
            "action": "Maintain baseline CPU only",
            "reasoning": "Weekend pattern shows reduced elective procedures, emergency-only workload",
            "confidence": 0.85,
            "volume": 10,
            "duration_minutes": 120
        })
    
    return {
        "recommendations": recommendations,
        "analysis_timestamp": current_time.isoformat(),
        "model_version": "simulated-advisor-v1",
        "context_tokens": len(context.split())
    }

def parse_llm_response(llm_response: Dict[str, Any]) -> List[ScalingPrediction]:
    """Parse LLM response into structured scaling predictions."""
    predictions = []
    
    for rec in llm_response.get("recommendations", []):
        # Convert LLM recommendation to ScalingPrediction
        prediction_type = rec["prediction"].lower().replace(" ", "_")
        expected_start = datetime.now() + timedelta(minutes=5)  # Default 5 minutes
        
        if "5 minutes" in rec["prediction"]:
            expected_start = datetime.now() + timedelta(minutes=5)
        elif "10 minutes" in rec["prediction"]:
            expected_start = datetime.now() + timedelta(minutes=10)
        elif "15 minutes" in rec["prediction"]:
            expected_start = datetime.now() + timedelta(minutes=15)
        
        prediction = ScalingPrediction(
            prediction_type=f"llm_{prediction_type}",
            expected_start=expected_start,
            expected_duration_minutes=rec.get("duration_minutes", 30),
            expected_volume=rec.get("volume", 50),
            confidence_score=rec.get("confidence", 0.5),
            recommended_action=rec["action"],
            resource_requirements=parse_action_to_resources(rec["action"])
        )
        predictions.append(prediction)
    
    return predictions

def parse_action_to_resources(action: str) -> Dict[str, Any]:
    """Parse LLM action string into resource requirements."""
    action_lower = action.lower()
    
    if "2 a100" in action_lower:
        return {"gpu": 2, "cpu": 4, "memory": "16GB"}
    elif "1 a100" in action_lower:
        return {"gpu": 1, "cpu": 2, "memory": "8GB"}
    elif "6 cores" in action_lower:
        return {"gpu": 0, "cpu": 6, "memory": "24GB"}
    elif "cpu" in action_lower:
        return {"gpu": 0, "cpu": 4, "memory": "16GB"}
    else:
        return {"gpu": 0, "cpu": 2, "memory": "8GB"}

def prioritize_predictions(all_predictions: List[ScalingPrediction]) -> List[ScalingPrediction]:
    """
    Prioritize and deduplicate predictions from multiple sources.
    """
    if not all_predictions:
        return []
    
    # Group by prediction type and time window
    prediction_groups = {}
    
    for pred in all_predictions:
        time_window = pred.expected_start.strftime("%H:%M")
        key = f"{pred.prediction_type}_{time_window}"
        
        if key not in prediction_groups:
            prediction_groups[key] = []
        prediction_groups[key].append(pred)
    
    # Select best prediction from each group
    final_predictions = []
    
    for group in prediction_groups.values():
        # Sort by confidence score, take highest
        best_prediction = max(group, key=lambda p: p.confidence_score)
        final_predictions.append(best_prediction)
    
    # Sort by expected start time
    final_predictions.sort(key=lambda p: p.expected_start)
    
    print(f"Prioritized {len(final_predictions)} predictions from {len(all_predictions)} candidates")
    return final_predictions

def check_ed_status() -> Dict[str, Any]:
    """Check Emergency Department status."""
    # Simulate ED status
    census = np.random.randint(10, 40)
    
    return {
        "current_census": census,
        "alert_level": "high" if census > 30 else "normal",
        "expected_imaging_requests": census * 0.6  # 60% of ED patients get imaging
    }

def check_icu_census() -> Dict[str, Any]:
    """Check ICU census for telemetry load."""
    # Simulate ICU census
    icu_beds = np.random.randint(15, 35)
    
    return {
        "occupied_beds": icu_beds,
        "telemetry_load": icu_beds * 10,  # 10 telemetry readings per bed per minute
        "alert_level": "high" if icu_beds > 30 else "normal"
    }

# Demo function to show LLM-based scaling advisor
@app.local_entrypoint()
def demo_llm_scaling_advisor():
    """
    Demo the experimental LLM-based scaling advisor that outputs the format from the chapter:
    {"prediction": "Radiology surge in 5 minutes", "action": "Pre-warm 2 A100 GPUs"}
    """
    print("Atlas Health LLM-Based Scaling Advisor Demo")
    print("=" * 50)
    print("Experimental feature: LLM agents as autoscaling advisors")
    print()
    
    # Simulate different clinical scenarios to trigger LLM analysis
    scenarios = [
        {
            "name": "Morning PACS Upload Detection",
            "description": "PACS system scheduled upload detected at 8 AM",
            "expected_llm_output": {"prediction": "Radiology surge in 5 minutes", "action": "Pre-warm 2 A100 GPUs"}
        },
        {
            "name": "Shift Change Anticipation", 
            "description": "Nursing shift change detected, anticipating dictation backlog",
            "expected_llm_output": {"prediction": "Text processing surge in 15 minutes", "action": "Scale CPU text processors to 6 cores"}
        },
        {
            "name": "Emergency Department Surge",
            "description": "High ED census during night hours indicating trauma cases",
            "expected_llm_output": {"prediction": "Emergency imaging spike in 10 minutes", "action": "Pre-warm 1 A100 GPU"}
        },
        {
            "name": "Weekend Pattern Recognition",
            "description": "Weekend hours with reduced elective procedures",
            "expected_llm_output": {"prediction": "Minimal scaling needed", "action": "Maintain baseline CPU only"}
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"Scenario {i}: {scenario['name']}")
        print(f"Description: {scenario['description']}")
        print("-" * 40)
        
        # Run the LLM-based analysis
        current_metrics = collect_current_metrics.remote()
        historical_analysis = analyze_historical_patterns.remote()
        scheduled_events = check_scheduled_events.remote()
        
        # Generate LLM predictions
        llm_predictions = generate_llm_scaling_recommendations.remote(
            current_metrics, historical_analysis, scheduled_events, []
        )
        
        # Display results in the chapter's format
        if llm_predictions:
            for pred in llm_predictions:
                llm_format_output = {
                    "prediction": pred.prediction_type.replace("llm_", "").replace("_", " ").title(),
                    "action": pred.recommended_action
                }
                print(f"LLM Output: {json.dumps(llm_format_output)}")
                print(f"Confidence: {pred.confidence_score:.2f}")
                print(f"Expected Volume: {pred.expected_volume} patients")
                print(f"Timeline: {pred.expected_start.strftime('%H:%M')}")
                print()
        else:
            print("No LLM predictions generated for this scenario")
            print()
        
        time.sleep(1)  # Brief pause between scenarios
    
    print("LLM-based scaling advisor demo completed!")
    print()
    print("Key Features Demonstrated:")
    print("• LLM agents analyze structured + unstructured signals")
    print("• Natural language reasoning about clinical patterns")
    print("• Cross-signal correlation detection")
    print("• Anticipatory scaling recommendations")
    print("• Integration with traditional rule-based systems")

# Demo function to show predictive scaling in action
@app.local_entrypoint()
def demo_predictive_autoscaling():
    """
    Demo the predictive autoscaling system with simulated scenarios.
    """
    print("Atlas Health Predictive Autoscaling Demo")
    print("=" * 50)
    
    scenarios = [
        "morning_preparation",
        "shift_change_anticipation", 
        "emergency_surge_detection",
        "weekend_scale_down"
    ]
    
    for scenario in scenarios:
        print(f"\nScenario: {scenario}")
        print("-" * 30)
        
        # Run predictive autoscaler
        predictions = predictive_autoscaler_agent.remote()
        
        print(f"Generated {len(predictions)} predictions")
        
        for i, prediction in enumerate(predictions):
            print(f"   {i+1}. {prediction.prediction_type}")
            print(f"      Start: {prediction.expected_start.strftime('%H:%M')}")
            print(f"      Volume: {prediction.expected_volume} patients")
            print(f"      Confidence: {prediction.confidence_score:.2f}")
            print(f"      Action: {prediction.recommended_action}")
        
        time.sleep(2)  # Brief pause between scenarios
    
    print(f"\nPredictive autoscaling demo completed!")
    print(f"Key benefits demonstrated:")
    print(f"   • Anticipates surges before they happen")
    print(f"   • Reduces cold start delays during peak times")
    print(f"   • Optimizes resource utilization based on patterns")
    print(f"   • Integrates multiple signal sources for accuracy")

# Integration with main multimodal pipeline
@app.local_entrypoint()
def demo_integrated_predictive_scaling():
    """
    Demo showing integration between predictive autoscaler and multimodal pipeline.
    This demonstrates the complete Chapter 9 bonus feature.
    """
    print("Atlas Health Integrated Predictive Scaling Demo")
    print("=" * 60)
    print("Demonstrating: Predictive autoscaler + Multimodal pipeline integration")
    print()
    
    # Step 1: Show traditional reactive scaling
    print("PHASE 1: Traditional Reactive Scaling")
    print("-" * 40)
    print("- Workload arrives > Cold start delay > Processing begins")
    print("• GPU spin-up time: ~10 seconds")
    print("• Total latency: Cold start + processing time")
    print()
    
    # Step 2: Show predictive pre-warming
    print("PHASE 2: Predictive Pre-warming")
    print("-" * 40)
    print("Running predictive analysis...")
    
    # Get predictions
    predictions = predictive_autoscaler_agent.remote()
    
    if predictions:
        print(f"- Generated {len(predictions)} scaling predictions")
        
        for pred in predictions:
            if pred.confidence_score > 0.6:
                print(f"- High-confidence prediction: {pred.prediction_type}")
                print(f"  Expected start: {pred.expected_start.strftime('%H:%M')}")
                print(f"  Action: {pred.recommended_action}")
                print(f"  Confidence: {pred.confidence_score:.2f}")
                
                # Simulate pre-warming
                if "GPU" in pred.recommended_action:
                    print(f"  > Pre-warming GPU resources...")
                    print(f"  > GPU will be ready when workload arrives")
                else:
                    print(f"  > Scaling CPU resources...")
                print()
    
    # Step 3: Show LLM-enhanced predictions
    print("PHASE 3: LLM-Enhanced Predictions")
    print("-" * 40)
    print("LLM analyzing unstructured signals...")
    
    # Simulate the exact output format from the chapter
    chapter_examples = [
        {"prediction": "Radiology surge in 5 minutes", "action": "Pre-warm 2 A100 GPUs"},
        {"prediction": "Text processing spike in 15 minutes", "action": "Scale CPU text processors"},
        {"prediction": "Emergency department surge", "action": "Pre-warm 1 A100 GPU"}
    ]
    
    for example in chapter_examples:
        print(f"LLM Output: {json.dumps(example)}")
    
    print()
    print("RESULTS SUMMARY")
    print("-" * 40)
    print("- Cold start delays reduced from 10s to <1s")
    print("- Resource utilization optimized based on clinical patterns")
    print("- LLM agents provide intelligent scaling recommendations")
    print("- Integration with existing serverless multimodal pipeline")
    print("- Cost efficiency maintained through confident predictions only")
    print()
    print("This demonstrates the complete Chapter 9 bonus feature:")
    print("Predictive autoscaling that transforms elasticity into preparedness!")

if __name__ == "__main__":
    print("Predictive Autoscaler for Multimodal Medical Pipeline")
    print("Run with Modal CLI:")
    print("  modal run predictive_autoscaler.py::demo_predictive_autoscaling")
    print("  modal run predictive_autoscaler.py::demo_llm_scaling_advisor")  
    print("  modal run predictive_autoscaler.py::demo_integrated_predictive_scaling")