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
from nono_lora.dataset_semantic import (
    find_semantic_duplicates,
    merge_database_metadata,
    read_database_files,
    read_reference_files,
    style_features,
)


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
        "--database-directory", type=Path, default=Path("dataset/database")
    )
    parser.add_argument(
        "--references-directory", type=Path, default=Path("references")
    )
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
    reference_records: list[dict] | None = None,
    enforce_character_quality: bool = True,
) -> list[dict]:
    golden_records = golden_records or []
    reference_records = reference_records or []
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
        semantic_hits = find_semantic_duplicates(
            record, golden_records + approved, reference_records
        )
        if semantic_hits:
            top = semantic_hits[0]
            raise ValueError(
                f"semantic/reference duplicate for {record_id}: "
                f"{top.source_kind} {top.source_id}, score={top.score:.2f}, "
                f"reasons={', '.join(top.reasons)}"
            )
        if enforce_character_quality:
            quality = style_features(record)
            if (
                quality["soft_ai"]
                or quality["attacking"]
                or quality["mesugaki_strength"] < 7.0
                or quality["teasing_ratio"] < 0.80
                or not quality["answer_or_empathy"]
                or not quality["ending_or_follow_up"]
            ):
                raise ValueError(
                    f"NONO character quality failed for {record_id}: {quality}"
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
    _, database_records = read_database_files(args.database_directory)
    golden_records = merge_database_metadata(golden_records, database_records)
    _, reference_records = read_reference_files(args.references_directory)
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
            reference_records=reference_records,
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
