#!/usr/bin/env python3
"""
LLM-Based Scaling Advisor - Comprehensive Demo: all five scenarios, rules vs language model.

    python comprehensive_llm_demo.py

Scenarios: morning PACS surge, shift-change dictation, night emergency surge, weekend minimal and a
high-volume crisis. For each, the rules' and the model's recommendations are printed, the model's
actions are checked against what the platform can do, and agreement with the rules is reported.
"""
import json
import os

from atlas_pipeline import config
from atlas_pipeline.advisor import compare

if __name__ == "__main__":
    print("Atlas Health LLM-Based Scaling Advisor - Comprehensive Demo")
    print("=" * 60)
    results = compare()
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(config.RESULTS_DIR, "llm_advisor.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to results/llm_advisor.json")
