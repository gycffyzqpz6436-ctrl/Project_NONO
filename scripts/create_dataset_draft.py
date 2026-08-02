from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from nono_lora.dataset_local import load_jsonl_patterns, numeric_ids, render_review_text
from nono_lora.dataset_pipeline import extract_dialogue

PATTERNS = (
    "small_failure",
    "casual_question",
    "minor_success",
    "daily_observation",
    "request_for_advice",
    "shared_hobby",
    "gentle_support",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a local 50-record review template.")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument(
        "--golden", nargs="+", type=Path, default=[Path("dataset/jsonl/*.jsonl")]
    )
    parser.add_argument(
        "--category-plan", type=Path, default=Path("dataset/category_plan.yaml")
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_plan(path: Path) -> dict[str, dict[str, Any]]:
    try:
        import yaml
    except ImportError as exc:
        raise ValueError(
            "PyYAML is required to read category_plan.yaml; "
            "install requirements-generation.txt"
        ) from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    categories = payload.get("categories") if isinstance(payload, dict) else None
    if not isinstance(categories, dict) or not categories:
        raise ValueError("category plan must contain a non-empty categories mapping")
    enabled = {
        str(name): dict(settings)
        for name, settings in categories.items()
        if isinstance(settings, dict) and settings.get("enabled", True)
    }
    if not enabled:
        raise ValueError("category plan has no enabled categories")
    total = sum(float(item.get("target_ratio", 0)) for item in enabled.values())
    if total <= 0:
        raise ValueError("enabled category target ratios must total more than zero")
    return enabled


def plan_categories(
    golden: list[dict[str, Any]], plan: dict[str, dict[str, Any]], count: int
) -> list[str]:
    current = Counter(str(record.get("category", "unknown")) for record in golden)
    total_ratio = sum(float(item.get("target_ratio", 0)) for item in plan.values())
    selected: list[str] = []
    for _ in range(count):
        future_total = len(golden) + len(selected) + 1
        category = max(
            plan,
            key=lambda name: (
                future_total * float(plan[name].get("target_ratio", 0)) / total_ratio
                - current[name]
                - selected.count(name)
                - float(plan[name].get("recently_used_penalty", 0))
            ),
        )
        selected.append(category)
    return selected


def make_draft_records(
    golden: list[dict[str, Any]],
    plan: dict[str, dict[str, Any]],
    *,
    count: int,
) -> list[dict[str, Any]]:
    if count != 50:
        raise ValueError("draft batches must contain exactly 50 records")
    start = max(numeric_ids(golden)) + 1
    categories = plan_categories(golden, plan, count)
    recent_by_category: dict[str, list[str]] = {}
    for category in plan:
        topics = []
        for record in reversed(golden):
            if str(record.get("category", "unknown")) != category:
                continue
            user, _ = extract_dialogue(record)
            if user and user not in topics:
                topics.append(user[:24])
            if len(topics) == 3:
                break
        recent_by_category[category] = topics
    records = []
    for offset, category in enumerate(categories):
        settings = plan[category]
        records.append(
            {
                "id": f"{start + offset:06d}",
                "category": category,
                "conversation_type": PATTERNS[offset % len(PATTERNS)],
                "follow_up_target": offset % 4 == 0,
                "scenario": str(
                    settings.get("notes")
                    or f"{category}でまだ使っていない日常的な場面"
                ),
                "planning": {
                    "used_topics_to_avoid": recent_by_category[category]
                },
                "messages": [
                    {"role": "user", "content": ""},
                    {"role": "assistant", "content": ""},
                ],
            }
        )
    return records


def main() -> int:
    args = parse_args()
    if args.count != 50:
        raise SystemExit("--count must be exactly 50; no partial template was written")
    _, golden = load_jsonl_patterns(args.golden)
    try:
        plan = load_plan(args.category_plan)
        records = make_draft_records(golden, plan, count=args.count)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    start, end = records[0]["id"], records[-1]["id"]
    batch_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    output = args.output or Path(
        f"dataset/candidates/review/nono_draft_{start}_{end}_{batch_id}.txt"
    )
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        render_review_text(records, include_plan=True),
        encoding="utf-8",
        newline="\n",
    )
    print(f"Wrote exactly 50 draft slots to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
