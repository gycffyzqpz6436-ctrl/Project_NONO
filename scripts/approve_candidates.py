from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from nono_lora.data import (
    LocatedRecord,
    expand_paths,
    read_jsonl,
    training_record,
    validate_record,
    write_jsonl,
)
from nono_lora.dataset_local import find_similar, normalized_for_similarity
from nono_lora.dataset_pipeline import extract_dialogue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Approve a reviewed 50-record candidate batch."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--database-output", type=Path)
    parser.add_argument("--training-output", type=Path)
    parser.add_argument(
        "--golden",
        nargs="+",
        type=Path,
        default=[Path("dataset/jsonl/*.jsonl")],
        help="Latest complete Golden Dataset used for the final collision check.",
    )
    parser.add_argument("--reviewer", required=True)
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Candidate IDs to reject. Approval requires exactly 50 remaining records.",
    )
    parser.add_argument(
        "--accept-similarity-warnings",
        action="store_true",
        help="Explicitly accept local fuzzy-similarity warnings.",
    )
    return parser.parse_args()


def approve(
    records: list[dict],
    reviewer: str,
    excluded: set[str],
    golden_records: list[dict] | None = None,
    *,
    accept_similarity_warnings: bool = False,
) -> list[dict]:
    golden_records = golden_records or []
    golden_ids = {str(record.get("id")) for record in golden_records}
    golden_users = {extract_dialogue(record)[0] for record in golden_records}
    golden_normalized = {
        normalized_for_similarity(extract_dialogue(record)[0])
        for record in golden_records
    }
    approved = []
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    for record in records:
        if str(record.get("id")) in excluded:
            continue
        record_id = str(record.get("id"))
        if record_id in golden_ids:
            raise ValueError(f"id collision with latest Golden Dataset: {record_id}")
        user, _ = extract_dialogue(record)
        approved_users = {extract_dialogue(item)[0] for item in approved}
        approved_normalized = {
            normalized_for_similarity(extract_dialogue(item)[0]) for item in approved
        }
        if user in golden_users or user in approved_users:
            raise ValueError(f"exact user duplicate: {record_id}")
        normalized = normalized_for_similarity(user)
        if normalized in golden_normalized or normalized in approved_normalized:
            raise ValueError(f"normalized user duplicate: {record_id}")
        if str(record.get("status", "")).lower() != "draft":
            raise ValueError(f"candidate {record_id} must have status=draft")
        schema_errors = validate_record(
            LocatedRecord(record, Path("<approval>"), len(approved) + 1)
        )
        if schema_errors:
            raise ValueError(
                f"invalid messages structure for {record_id}: {'; '.join(schema_errors)}"
            )
        similarities = find_similar(record, golden_records + approved)
        if similarities and not accept_similarity_warnings:
            top = similarities[0]
            raise ValueError(
                f"similarity warning for {record_id}: existing {top.record_id}, "
                f"score={top.score:.2f}, reasons={', '.join(top.reasons)}; "
                "use --accept-similarity-warnings only after human review"
            )
        updated = dict(record)
        updated["status"] = "golden"
        review = dict(updated.get("review", {}))
        review.update(
            {"decision": "approved", "reviewer": reviewer, "reviewed_at": timestamp}
        )
        updated["review"] = review
        approved.append(updated)
    if len(approved) != 50:
        raise ValueError(
            f"approval must produce exactly 50 records; got {len(approved)}"
        )
    return approved


def main() -> int:
    args = parse_args()
    records = [item.record for item in read_jsonl([args.input])]
    try:
        golden_paths = expand_paths(args.golden)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    golden_records = [item.record for item in read_jsonl(golden_paths)]
    ids = [int(str(record["id"])) for record in records if str(record.get("id", "")).isdigit()]
    if len(ids) != len(records):
        raise SystemExit("all candidate IDs must be numeric")
    start, end = min(ids), max(ids)
    database_output = args.database_output or Path(
        f"dataset/database/nono_database_{start:06d}_{end:06d}.jsonl"
    )
    training_output = args.training_output or Path(
        f"dataset/jsonl/nono_dataset_{start:06d}_{end:06d}.jsonl"
    )
    if database_output.exists() or training_output.exists():
        raise SystemExit("approval output already exists; refusing to overwrite it")
    try:
        approved = approve(
            records,
            args.reviewer,
            set(args.exclude),
            golden_records,
            accept_similarity_warnings=args.accept_similarity_warnings,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    write_jsonl(database_output, approved)
    write_jsonl(training_output, (training_record(record) for record in approved))
    print(
        f"Wrote 50 rich records to {database_output} and "
        f"50 training records to {training_output}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
