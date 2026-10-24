"""Predictive autoscaling: forecast surges from the hospital's schedule and pre-warm for them.

The first edition's schedule rules (PACS uploads, shift changes, scheduled procedures) and its
pre-warming policy (confident, 2 to 15 minutes ahead, GPU work or more than 100 expected cases),
evaluated for a given time so the same clock gives the same plan. prewarm_plan() turns the
predictions into a number of warm containers per service; on Modal that becomes min_containers.
"""
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Dict, List

# Cases one warm container (or local worker) keeps up with during a surge, per service
CASES_PER_CONTAINER = {"radiology": 50, "labs": 100, "notes": 50}
MAX_PREWARM = {"radiology": 3, "labs": 2, "notes": 2}


@dataclass
class ScalingPrediction:
    prediction_type: str          # e.g. "pacs_upload"
    workload_type: str            # radiology, text, mixed
    expected_start: datetime
    expected_duration_minutes: int
    expected_volume: int
    confidence_score: float
    recommended_action: str

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["expected_start"] = self.expected_start.isoformat(timespec="minutes")
        return d


def scheduled_events(now: datetime) -> List[Dict]:
    """The hospital calendar: what is scheduled in the next hour."""
    events = []
    if now.weekday() < 5 and now.hour == 7 and now.minute >= 30:
        events.append({"event_type": "pacs_upload", "scheduled_time": now.replace(hour=8, minute=0),
                       "expected_volume": 200, "workload_type": "radiology", "confidence": 0.95})
    if now.hour in (6, 14, 22) and now.minute >= 30:
        events.append({"event_type": "shift_change", "scheduled_time": now.replace(minute=0) + timedelta(hours=1),
                       "expected_volume": 100, "workload_type": "text", "confidence": 0.85})
    if now.weekday() < 5 and now.hour == 8 and now.minute >= 30:
        events.append({"event_type": "scheduled_procedures", "scheduled_time": now.replace(hour=9, minute=0),
                       "expected_volume": 75, "workload_type": "mixed", "confidence": 0.90})
    return events


def predictions(now: datetime) -> List[ScalingPrediction]:
    resource = {"radiology": "GPU", "text": "CPU", "mixed": "GPU+CPU"}
    return [ScalingPrediction(e["event_type"], e["workload_type"], e["scheduled_time"], 30, e["expected_volume"],
                              e["confidence"], f"Pre-warm {resource[e['workload_type']]}")
            for e in scheduled_events(now)]


def should_prewarm(prediction: ScalingPrediction, now: datetime) -> bool:
    minutes = (prediction.expected_start - now).total_seconds() / 60
    if prediction.confidence_score < 0.6 or not 2 <= minutes <= 15:
        return False
    return prediction.workload_type in ("radiology", "mixed") or prediction.expected_volume > 100


def prewarm_plan(now: datetime) -> Dict[str, int]:
    """Warm containers per service for what is predicted in the next 2 to 15 minutes (0 = scale to zero)."""
    services = {"radiology": ("radiology", "mixed"), "labs": ("mixed",), "notes": ("text", "mixed")}
    plan = {m: 0 for m in services}
    for p in predictions(now):
        if not should_prewarm(p, now):
            continue
        for modality, types in services.items():
            if p.workload_type in types:
                n = math.ceil(p.expected_volume / CASES_PER_CONTAINER[modality])
                plan[modality] = max(plan[modality], min(n, MAX_PREWARM[modality]))
    return plan
