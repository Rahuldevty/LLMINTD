from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests


ADV_BENCH_URL = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/"
    "main/data/advbench/harmful_behaviors.csv"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build one normalized guardrail evaluation dataset from public safety datasets."
    )
    parser.add_argument("--output", default="eval/combined_safety_prompts.jsonl")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["jailbreakbench", "harmbench", "advbench", "toxicchat", "anthropic_redteam"],
        choices=["jailbreakbench", "harmbench", "advbench", "toxicchat", "anthropic_redteam"],
    )
    parser.add_argument(
        "--limit-per-source",
        type=int,
        default=0,
        help="Maximum examples per source. 0 means use all rows loaded from each source.",
    )
    parser.add_argument(
        "--include-benign-toxicchat",
        action="store_true",
        help="Include non-toxic ToxicChat rows as expected ALLOW examples. This makes the dataset much larger.",
    )
    return parser.parse_args()


def stable_id(source: str, prompt: str) -> str:
    digest = hashlib.sha256(f"{source}:{prompt}".encode("utf-8")).hexdigest()[:16]
    return f"{source}_{digest}"


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        value = " ".join(str(item) for item in value)
    value = str(value)
    return re.sub(r"\s+", " ", value).strip()


def normalize_record(
    *,
    source: str,
    prompt: str,
    expected_decision: str,
    category: str = "",
    source_id: str = "",
    label: str = "",
) -> Optional[Dict[str, Any]]:
    prompt = clean_text(prompt)
    if not prompt:
        return None
    return {
        "id": stable_id(source, prompt),
        "source": source,
        "source_id": source_id,
        "category": category or "uncategorized",
        "prompt": prompt,
        "expected_decision": expected_decision,
        "label": label or ("benign" if expected_decision == "ALLOW" else "harmful"),
    }


def load_hf_dataset(*args, **kwargs):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install datasets first: pip install datasets") from exc
    return load_dataset(*args, **kwargs)


def apply_limit(records: Iterable[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    output = []
    for record in records:
        output.append(record)
        if limit and len(output) >= limit:
            break
    return output


def load_jailbreakbench(limit: int) -> List[Dict[str, Any]]:
    dataset = load_hf_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
    records = []
    for split_name, expected_decision, label in [
        ("harmful", "BLOCK_OR_REWRITE", "harmful"),
        ("benign", "ALLOW", "benign"),
    ]:
        for row in dataset[split_name]:
            record = normalize_record(
                source="jailbreakbench",
                source_id=str(row.get("Index", "")),
                prompt=row.get("Goal", ""),
                expected_decision=expected_decision,
                category=row.get("Category", ""),
                label=label,
            )
            if record:
                records.append(record)
    return apply_limit(records, limit)


def load_harmbench(limit: int) -> List[Dict[str, Any]]:
    train_split = load_hf_dataset("AlignmentResearch/HarmBench", split="train")
    records = []
    for index, row in enumerate(train_split):
        content = row.get("content")
        prompt = content[0] if isinstance(content, list) and content else content
        record = normalize_record(
            source="harmbench",
            source_id=f"train:{index}",
            prompt=prompt or row.get("behavior", "") or row.get("prompt", ""),
            expected_decision="BLOCK_OR_REWRITE",
            category=row.get("FunctionalCategory", "") or row.get("category", "") or row.get("class", ""),
            label="harmful",
        )
        if record:
            records.append(record)
    return apply_limit(records, limit)


def load_advbench(limit: int) -> List[Dict[str, Any]]:
    response = requests.get(ADV_BENCH_URL, timeout=60)
    response.raise_for_status()
    rows = csv.DictReader(response.text.splitlines())
    records = []
    for index, row in enumerate(rows):
        record = normalize_record(
            source="advbench",
            source_id=str(index),
            prompt=row.get("goal") or row.get("Goal") or row.get("behavior") or row.get("Behavior") or "",
            expected_decision="BLOCK_OR_REWRITE",
            category="harmful_behavior",
            label="harmful",
        )
        if record:
            records.append(record)
    return apply_limit(records, limit)


def load_toxicchat(limit: int, include_benign: bool) -> List[Dict[str, Any]]:
    dataset = load_hf_dataset("lmsys/toxic-chat", "toxicchat0124")
    records = []
    for split_name in dataset:
        for index, row in enumerate(dataset[split_name]):
            toxicity = int(row.get("toxicity") or 0)
            jailbreaking = int(row.get("jailbreaking") or 0)
            is_harmful = toxicity == 1 or jailbreaking == 1
            if not is_harmful and not include_benign:
                continue
            category = "jailbreak" if jailbreaking else "toxicity" if toxicity else "benign"
            record = normalize_record(
                source="toxicchat",
                source_id=row.get("conv_id") or f"{split_name}:{index}",
                prompt=row.get("user_input", ""),
                expected_decision="BLOCK_OR_REWRITE" if is_harmful else "ALLOW",
                category=category,
                label="harmful" if is_harmful else "benign",
            )
            if record:
                records.append(record)
    return apply_limit(records, limit)


def extract_first_human_turn(transcript: str) -> str:
    transcript = clean_text(transcript)
    match = re.search(r"Human:\s*(.*?)(?:\s*Assistant:|$)", transcript, flags=re.IGNORECASE)
    return clean_text(match.group(1)) if match else transcript[:1000]


def load_anthropic_redteam(limit: int) -> List[Dict[str, Any]]:
    dataset = load_hf_dataset("Anthropic/hh-rlhf", data_dir="red-team-attempts")
    records = []
    for split_name in dataset:
        for index, row in enumerate(dataset[split_name]):
            prompt = row.get("task_description") or extract_first_human_turn(row.get("transcript", ""))
            tags = row.get("tags")
            category = ",".join(tags) if isinstance(tags, list) and tags else "red_team"
            record = normalize_record(
                source="anthropic_redteam",
                source_id=f"{split_name}:{index}",
                prompt=prompt,
                expected_decision="BLOCK_OR_REWRITE",
                category=category,
                label="harmful",
            )
            if record:
                records.append(record)
    return apply_limit(records, limit)


def deduplicate(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    output = []
    for record in records:
        key = hashlib.sha256(record["prompt"].lower().encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        output.append(record)
    return output


def main() -> None:
    args = parse_args()
    source_loaders = {
        "jailbreakbench": lambda: load_jailbreakbench(args.limit_per_source),
        "harmbench": lambda: load_harmbench(args.limit_per_source),
        "advbench": lambda: load_advbench(args.limit_per_source),
        "toxicchat": lambda: load_toxicchat(args.limit_per_source, args.include_benign_toxicchat),
        "anthropic_redteam": lambda: load_anthropic_redteam(args.limit_per_source),
    }

    all_records = []
    source_counts = {}
    for source in args.sources:
        records = source_loaders[source]()
        source_counts[source] = len(records)
        all_records.extend(records)

    combined = deduplicate(all_records)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in combined:
            file.write(json.dumps(record, ensure_ascii=True) + "\n")

    print(f"Wrote {len(combined)} records to {output_path}")
    print("Loaded records by source:")
    for source, count in source_counts.items():
        print(f"  {source}: {count}")
    print("Note: harmful prompt text is stored in the JSONL file but not printed here.")


if __name__ == "__main__":
    main()
