#!/usr/bin/env python3
"""
Local demo of the LLM-based scaling advisor (Chapter 9 bonus feature).

    python local_llm_scaling_demo.py

Gives the same metrics to the first edition's rules and to a small open language model
(Qwen3-1.7B by default; ATLAS_ADVISOR_MODEL to change) running on this machine, and prints both
recommendations in the chapter's format {"prediction": ..., "action": ...} with how well they agree.
The first run downloads the model (about 4 GB); it needs about 4 GB of free memory. On a Modal GPU: modal run predictive_autoscaler.py::demo_llm_scaling_advisor
"""
from atlas_pipeline.advisor import compare

if __name__ == "__main__":
    print("Atlas Health LLM-Based Scaling Advisor Demo")
    print("=" * 50)
    compare(["morning_pacs_surge", "shift_change_dictation", "weekend_minimal"])
