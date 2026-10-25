"""Scaling advisors: the first edition's rules, and a small language model given the same signals.

The rules are the chapter's hand-written heuristics (PACS uploads, shift changes, night-time ED
census, weekends, several high signals at once). The LLM advisor sends the same metrics to an open
model running on your machine (or in a Modal container), asks for JSON recommendations in the
chapter's format {"prediction": ..., "action": ...}, and checks that the actions are ones the
platform can carry out. Nothing is called over the internet except the one-time model download.
"""
import json
import re
import time
from typing import Any, Dict, List, Optional

from atlas_pipeline import config

# The first edition's scenarios (fixed values, so every run gets the same inputs)
SCENARIOS: Dict[str, Dict[str, Any]] = {
    "morning_pacs_surge": {
        "time_of_day": 8, "day_of_week": 1,
        "queue_depth": {"radiology": 25, "telemetry": 15, "text": 5},
        "upstream_signals": {"pacs_scheduled_uploads": {"scheduled_uploads": True, "expected_volume": 200},
                             "ehr_shift_changes": {"shift_change": False},
                             "emergency_department_alerts": {"current_census": 20},
                             "icu_census_changes": {"occupied_beds": 25}}},
    "shift_change_dictation": {
        "time_of_day": 14, "day_of_week": 2,
        "queue_depth": {"radiology": 8, "telemetry": 30, "text": 18},
        "upstream_signals": {"pacs_scheduled_uploads": {"scheduled_uploads": False},
                             "ehr_shift_changes": {"shift_change": True, "shift_type": "evening"},
                             "emergency_department_alerts": {"current_census": 25},
                             "icu_census_changes": {"occupied_beds": 20}}},
    "night_emergency_surge": {
        "time_of_day": 2, "day_of_week": 0,
        "queue_depth": {"radiology": 12, "telemetry": 40, "text": 3},
        "upstream_signals": {"pacs_scheduled_uploads": {"scheduled_uploads": False},
                             "ehr_shift_changes": {"shift_change": False},
                             "emergency_department_alerts": {"current_census": 35},
                             "icu_census_changes": {"occupied_beds": 32}}},
    "weekend_minimal": {
        "time_of_day": 14, "day_of_week": 6,
        "queue_depth": {"radiology": 3, "telemetry": 12, "text": 2},
        "upstream_signals": {"pacs_scheduled_uploads": {"scheduled_uploads": False},
                             "ehr_shift_changes": {"shift_change": False},
                             "emergency_department_alerts": {"current_census": 15},
                             "icu_census_changes": {"occupied_beds": 18}}},
    "high_volume_crisis": {
        "time_of_day": 16, "day_of_week": 4,
        "queue_depth": {"radiology": 45, "telemetry": 60, "text": 25},
        "upstream_signals": {"pacs_scheduled_uploads": {"scheduled_uploads": True},
                             "ehr_shift_changes": {"shift_change": True},
                             "emergency_department_alerts": {"current_census": 38},
                             "icu_census_changes": {"occupied_beds": 35}}},
}

# Actions the platform can carry out (the 'action' must match one of these)
ACTION_PATTERNS = [
    r"^Pre-warm [1-4] A100 GPUs?$",
    r"^Scale CPU (text|telemetry) (processors|workers) to \d{1,2} (cores|workers)$",
    r"^Maintain baseline CPU only$",
    r"^Emergency scale: [1-4] A100 GPUs \+ \d{1,2} CPU cores$",
    r"^No change$",
]


def rule_advisor(metrics: Dict[str, Any]) -> List[Dict[str, str]]:
    """The first edition's rules (its confidence numbers were fixed constants, so they are left out)."""
    signals = metrics["upstream_signals"]
    out = []
    if signals["pacs_scheduled_uploads"]["scheduled_uploads"]:
        out.append({"prediction": "Radiology surge in 5 minutes", "action": "Pre-warm 2 A100 GPUs",
                    "reasoning": "Scheduled PACS upload; imaging requests follow uploads"})
    if signals["ehr_shift_changes"]["shift_change"]:
        out.append({"prediction": "Text processing surge in 15 minutes", "action": "Scale CPU text processors to 6 cores",
                    "reasoning": "Shift change; dictation backlog follows"})
    if (metrics["time_of_day"] >= 22 or metrics["time_of_day"] <= 5) and \
            signals["emergency_department_alerts"]["current_census"] > 30:
        out.append({"prediction": "Emergency imaging spike in 10 minutes", "action": "Pre-warm 1 A100 GPU",
                    "reasoning": "High ED census at night"})
    if metrics["day_of_week"] >= 5:
        out.append({"prediction": "Minimal scaling needed", "action": "Maintain baseline CPU only",
                    "reasoning": "Weekend: emergency-only workload"})
    high = sum([metrics["queue_depth"]["radiology"] > 30, metrics["queue_depth"]["text"] > 20,
                signals["emergency_department_alerts"]["current_census"] > 35,
                signals["icu_census_changes"]["occupied_beds"] > 32])
    if high >= 3:
        out.append({"prediction": "Multi-modal surge crisis in 2 minutes", "action": "Emergency scale: 3 A100 GPUs + 12 CPU cores",
                    "reasoning": "Three or more high signals at once"})
    elif metrics["queue_depth"]["radiology"] > 15:
        out.append({"prediction": "Immediate radiology processing needed", "action": "Pre-warm 1 A100 GPU",
                    "reasoning": "Radiology queue above normal capacity"})
    return out


PROMPT = """You are the scaling advisor for a hospital's AI inference platform. Radiology runs on A100 GPUs
that take about a minute to start; telemetry and clinical-note models run on CPU workers.

What the signals mean: a scheduled PACS upload brings a burst of X-rays about 5 minutes later; a
shift change brings a backlog of dictated notes; a high emergency-department census at night brings
urgent imaging; weekends are emergency-only; several high queues at once mean a hospital-wide surge.

Current metrics (time_of_day is the hour, day_of_week 0 = Monday, 5 and 6 = weekend):
{metrics}

Recommend at most 2 scaling actions for the next 15 minutes, only ones these metrics call for
("No change" if nothing does). Choose each action from this list, copied exactly:
"Pre-warm 1 A100 GPU", "Pre-warm 2 A100 GPUs", "Pre-warm 3 A100 GPUs",
"Scale CPU text processors to 4 cores", "Scale CPU text processors to 6 cores",
"Scale CPU telemetry workers to 4 workers", "Maintain baseline CPU only",
"Emergency scale: 3 A100 GPUs + 12 CPU cores", "No change"

Answer with a JSON list only; each item has the keys "prediction" (what you expect and when),
"action" (from the list) and "reasoning" (one sentence about these metrics).
"""


def valid_action(action: str) -> bool:
    action = re.sub(r"\b1 A100 GPUs\b", "1 A100 GPU", action.strip())
    return any(re.match(p, action) for p in ACTION_PATTERNS)


class LLMAdvisor:
    """A small open model (config.ADVISOR_MODEL, default Qwen3-1.7B) with greedy decoding."""

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        started = time.time()
        self.model_name = model_name or config.ADVISOR_MODEL
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # Half precision on GPU and CPU: Qwen3-1.7B needs about 3.4 GB of memory instead of 6.8 GB
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, dtype=torch.float16 if self.device == "cuda" else torch.bfloat16).to(self.device).eval()
        self.load_seconds = time.time() - started

    def advise(self, metrics: Dict[str, Any], max_new_tokens: int = 300) -> Dict[str, Any]:
        started = time.time()
        messages = [{"role": "user", "content": PROMPT.format(metrics=json.dumps(metrics, indent=1))}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                    enable_thinking=False)
        prompt += "["           # start the answer as a JSON list
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        text = "[" + self.tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        recommendations, error = parse_recommendations(text)
        return {"recommendations": recommendations, "raw": text, "error": error,
                "seconds": time.time() - started, "model": self.model_name}


def parse_recommendations(text: str):
    """(valid recommendations, error message or None) from the model's answer."""
    match = re.search(r"\[.*\]", text, re.S)
    if not match:
        return [], "no JSON list in the answer"
    try:
        items = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        return [], f"invalid JSON: {e}"
    valid = [i for i in items if isinstance(i, dict) and valid_action(str(i.get("action", "")))]
    dropped = [str(i.get("action")) if isinstance(i, dict) else str(i) for i in items
               if not (isinstance(i, dict) and valid_action(str(i.get("action", ""))))]
    return valid, (f"rejected (not an action the platform can carry out): {'; '.join(dropped)}" if dropped else None)


def compare(scenario_names: Optional[List[str]] = None, advisor: Optional[LLMAdvisor] = None, log=print) -> List[Dict]:
    """Run the rules and the LLM on each scenario and print both."""
    advisor = advisor or LLMAdvisor()
    log(f"LLM advisor: {advisor.model_name} on {advisor.device} (loaded in {advisor.load_seconds:.1f} s)")
    results = []
    for name in scenario_names or list(SCENARIOS):
        metrics = SCENARIOS[name]
        rules = rule_advisor(metrics)
        llm = advisor.advise(metrics)
        log(f"\nScenario: {name}")
        log("  Rules:")
        for r in rules or [{"prediction": "-", "action": "No change"}]:
            log(f"    {json.dumps({'prediction': r['prediction'], 'action': r['action']})}")
        log(f"  LLM ({llm['seconds']:.1f} s):")
        for r in llm["recommendations"]:
            log(f"    {json.dumps({'prediction': r.get('prediction'), 'action': r.get('action')})}")
        if llm["error"]:
            log(f"    note: {llm['error']}")
        if not llm["recommendations"]:
            log(f"    raw answer: {llm['raw'][:200]!r}")
        rule_actions = {normalize(r["action"]) for r in rules} or {"No change"}
        llm_actions = {normalize(str(r.get("action"))) for r in llm["recommendations"]}
        agree = rule_actions & llm_actions
        precision = len(agree) / len(llm_actions) if llm_actions else 0.0
        recall = len(agree) / len(rule_actions)
        log(f"  Agreement with the rules: {len(agree)} of the LLM's {len(llm_actions)} actions "
            f"(precision {precision:.2f}), {len(agree)} of the rules' {len(rule_actions)} (recall {recall:.2f})")
        results.append({"scenario": name, "rules": rules, "llm": llm, "agreeing_actions": sorted(agree),
                        "precision": precision, "recall": recall})
    if results:
        log(f"\nMean precision {sum(r['precision'] for r in results) / len(results):.2f}, "
            f"mean recall {sum(r['recall'] for r in results) / len(results):.2f} over {len(results)} scenarios")
    return results


def normalize(action: str) -> str:
    return re.sub(r"\b1 A100 GPUs\b", "1 A100 GPU", action.strip())
