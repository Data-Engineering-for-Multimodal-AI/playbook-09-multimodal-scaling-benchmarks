#!/usr/bin/env python3
"""
Local demo of LLM-based scaling advisor (Chapter 9 bonus feature)
Runs locally without Modal to demonstrate the concepts quickly.
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
import random

def simulate_current_metrics() -> Dict[str, Any]:
    """Simulate current system metrics."""
    current_time = datetime.now()
    
    return {
        "timestamp": current_time.isoformat(),
        "queue_depth": {
            "radiology": random.randint(0, 20),
            "telemetry": random.randint(5, 50), 
            "text": random.randint(0, 15)
        },
        "request_rate_per_minute": {
            "radiology": random.randint(0, 30),
            "telemetry": random.randint(10, 80),
            "text": random.randint(0, 25)
        },
        "current_gpu_utilization": random.uniform(0, 0.3),
        "current_cpu_utilization": random.uniform(0.1, 0.6),
        "time_of_day": current_time.hour,
        "day_of_week": current_time.weekday(),
        "upstream_signals": {
            "pacs_scheduled_uploads": {"scheduled_uploads": current_time.hour == 8},
            "ehr_shift_changes": {"shift_change": current_time.hour in [6, 14, 22]},
            "emergency_department_alerts": {"current_census": random.randint(10, 40)},
            "icu_census_changes": {"occupied_beds": random.randint(15, 35)}
        }
    }

def simulate_llm_scaling_advisor(metrics: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Simulate the LLM-based scaling advisor from Chapter 9.
    This demonstrates the exact output format mentioned:
    {"prediction": "Radiology surge in 5 minutes", "action": "Pre-warm 2 A100 GPUs"}
    """
    recommendations = []
    current_time = datetime.now()
    
    # PACS upload detection
    if metrics["upstream_signals"]["pacs_scheduled_uploads"]["scheduled_uploads"]:
        recommendations.append({
            "prediction": "Radiology surge in 5 minutes",
            "action": "Pre-warm 2 A100 GPUs",
            "reasoning": "PACS upload detected - historical correlation shows imaging workload spike follows",
            "confidence": "0.88"
        })
    
    # Shift change detection
    if metrics["upstream_signals"]["ehr_shift_changes"]["shift_change"]:
        recommendations.append({
            "prediction": "Text processing surge in 15 minutes", 
            "action": "Scale CPU text processors to 6 cores",
            "reasoning": "Shift change correlates with dictation backlog processing",
            "confidence": "0.75"
        })
    
    # High ED census during night hours
    ed_census = metrics["upstream_signals"]["emergency_department_alerts"]["current_census"]
    if 22 <= current_time.hour or current_time.hour <= 5:
        if ed_census > 30:
            recommendations.append({
                "prediction": "Emergency imaging spike in 10 minutes",
                "action": "Pre-warm 1 A100 GPU",
                "reasoning": "High ED census during night hours suggests trauma cases requiring urgent imaging",
                "confidence": "0.65"
            })
    
    # Weekend scaling down
    if current_time.weekday() >= 5:  # Weekend
        recommendations.append({
            "prediction": "Minimal scaling needed",
            "action": "Maintain baseline CPU only", 
            "reasoning": "Weekend pattern shows reduced elective procedures, emergency-only workload",
            "confidence": "0.85"
        })
    
    # High queue depth detection
    if metrics["queue_depth"]["radiology"] > 15:
        recommendations.append({
            "prediction": "Immediate radiology processing needed",
            "action": "Pre-warm 1 A100 GPU",
            "reasoning": "Current queue depth exceeds normal capacity, immediate scaling required",
            "confidence": "0.92"
        })
    
    return recommendations

def demo_llm_scaling_advisor():
    """Demo the LLM-based scaling advisor locally."""
    print("Atlas Health LLM-Based Scaling Advisor Demo (Local)")
    print("=" * 60)
    print("Chapter 9 Bonus Feature: LLM agents as autoscaling advisors")
    print()
    
    scenarios = [
        "Morning PACS Upload Detection",
        "Nursing Shift Change",
        "Emergency Department Surge", 
        "Weekend Scale-Down",
        "High Queue Depth Alert"
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"Scenario {i}: {scenario}")
        print("-" * 40)
        
        # Simulate current metrics
        metrics = simulate_current_metrics()
        
        # Display key metrics
        print(f"Current Metrics:")
        print(f"  Radiology Queue: {metrics['queue_depth']['radiology']} patients")
        print(f"  Time: {metrics['time_of_day']}:00")
        print(f"  Day: {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][metrics['day_of_week']]}")
        print()
        
        # Run LLM-based analysis
        print("LLM Analysis:")
        recommendations = simulate_llm_scaling_advisor(metrics)
        
        if recommendations:
            for rec in recommendations:
                # Show the exact Chapter 9 format
                chapter_format = {
                    "prediction": rec["prediction"],
                    "action": rec["action"]
                }
                print(f"  {json.dumps(chapter_format)}")
                print(f"  Confidence: {rec['confidence']}")
                print(f"  Reasoning: {rec['reasoning']}")
                print()
        else:
            print("  No scaling recommendations at this time")
            print()
        
        time.sleep(1)  # Brief pause
    
    print("Demo completed!")
    print()
    print("Key Benefits Demonstrated:")
    print("• LLM agents analyze both structured metrics AND unstructured signals")
    print("• Natural language reasoning about clinical workflow patterns") 
    print("• Cross-signal correlation detection (PACS + shift + ED census)")
    print("• Confidence-based scaling decisions")
    print("• Integration with traditional rule-based systems")
    print("- Anticipatory scaling: elasticity > preparedness")

if __name__ == "__main__":
    demo_llm_scaling_advisor()