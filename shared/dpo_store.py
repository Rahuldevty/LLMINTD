from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4


ROOT_DIR = Path(__file__).resolve().parent.parent
DPO_GENERATIONS_PATH = ROOT_DIR / "logs" / "dpo_generations.jsonl"
DPO_PREFERENCES_PATH = ROOT_DIR / "logs" / "dpo_preferences.jsonl"
PLANNER_CATEGORY_FEEDBACK_PATH = ROOT_DIR / "logs" / "planner_category_feedback.jsonl"


def utc_timestamp() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=True) + "\n")


def load_generation(generation_id: str) -> Optional[Dict[str, Any]]:
    if not DPO_GENERATIONS_PATH.exists():
        return None

    with DPO_GENERATIONS_PATH.open(encoding="utf-8") as file:
        for line in file:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("id") == generation_id:
                return record
    return None


def save_generation(record: Dict[str, Any]) -> Dict[str, Any]:
    generation = {
        "id": record.get("id") or str(uuid4()),
        "created_at": utc_timestamp(),
        **record,
    }
    append_jsonl(DPO_GENERATIONS_PATH, generation)
    return generation


def save_preference(record: Dict[str, Any]) -> Dict[str, Any]:
    preference = {
        "id": record.get("id") or str(uuid4()),
        "created_at": utc_timestamp(),
        **record,
    }
    append_jsonl(DPO_PREFERENCES_PATH, preference)
    return preference


def save_planner_category_feedback(record: Dict[str, Any]) -> Dict[str, Any]:
    feedback = {
        "id": record.get("id") or str(uuid4()),
        "created_at": utc_timestamp(),
        **record,
    }
    append_jsonl(PLANNER_CATEGORY_FEEDBACK_PATH, feedback)
    return feedback
