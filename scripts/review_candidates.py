from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from nono_lora.data import LocatedRecord, read_jsonl, validate_record
from nono_lora.dataset_local import (
    ENDING_EXPRESSIONS,
    QUESTION_END,
    basic_structure,
    expression_counts,
    find_similar,
    load_jsonl_patterns,
    opening_key,
)
from nono_lora.dataset_pipeline import extract_dialogue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review local NONO candidate quality.")
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--golden", nargs="+", type=Path, default=[Path("dataset/jsonl/*.jsonl")]
    )
    return parser.parse_args()


def review_records(candidates: list[dict], golden: list[dict]) -> list[dict]:
    categories = Counter(str(record.get("category", "unknown")) for record in golden)
    golden_expressions = expression_counts(golden)
    reports = []
    prior: list[dict] = []
    golden_ids = {str(record.get("id")) for record in golden}
    for index, record in enumerate(candidates, start=1):
        user, assistant = extract_dialogue(record)
        hits = find_similar(record, golden + prior)
        structure_ok, missing = basic_structure(assistant)
        errors = validate_record(LocatedRecord(record, Path("<candidate>"), index))
        status = "pass"
        if errors or str(record.get("id")) in golden_ids or any(
            hit.score >= 0.999 for hit in hits
        ):
            status = "reject"
        elif hits or not structure_ok:
            status = "warning"
        opening = opening_key(assistant)
        phrase_frequency = {
            phrase.replace("～", "〜"): assistant.count(phrase)
            for phrase in ENDING_EXPRESSIONS
            if phrase in assistant
        }
        reports.append(
            {
                "id": str(record.get("id")),
                "result": status,
                "similar": [hit.as_dict() for hit in hits[:5]],
                "past_category_count": categories[str(record.get("category", "unknown"))],
                "opening": opening,
                "past_opening_count": golden_expressions["openings"][opening],
                "ending_phrase_frequency": phrase_frequency,
                "past_ending_counts": {
                    key: golden_expressions["endings"][key]
                    for key in phrase_frequency
                },
                "ends_with_question": bool(QUESTION_END.search(assistant)),
                "characters": {"user": len(user), "assistant": len(assistant)},
                "basic_structure": structure_ok,
                "structure_warnings": missing,
                "schema_errors": errors,
            }
        )
        prior.append(record)
    return reports


def main() -> int:
    args = parse_args()
    candidates = [item.record for item in read_jsonl([args.input])]
    _, golden = load_jsonl_patterns(args.golden)
    reports = review_records(candidates, golden)
    for report in reports:
        print(f"#{report['id']} [{report['result']}]")
        similar = ", ".join(
            f"{item['id']}={item['score']:.2f} ({'; '.join(item['reasons'])})"
            for item in report["similar"]
        )
        print(f"  Similar: {similar or 'none'}")
        print(f"  Past category count: {report['past_category_count']}")
        print(
            f"  Opening: {report['opening']!r} "
            f"(past={report['past_opening_count']})"
        )
        print(
            f"  Teasing/ending: {report['ending_phrase_frequency'] or 'none'}; "
            f"past={report['past_ending_counts'] or 'none'}"
        )
        print(f"  Question ending: {report['ends_with_question']}")
        print(
            f"  Characters: user={report['characters']['user']}, "
            f"NONO={report['characters']['assistant']}"
        )
        print(
            f"  NONO structure: {report['basic_structure']} "
            f"{report['structure_warnings']}"
        )
        if report["schema_errors"]:
            print(f"  Schema errors: {report['schema_errors']}")
    summary = Counter(item["result"] for item in reports)
    print(
        f"Summary: pass={summary['pass']}, warning={summary['warning']}, "
        f"reject={summary['reject']}. Human approval is always required."
    )
    return 1 if summary["reject"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
