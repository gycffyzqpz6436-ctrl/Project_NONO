from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nono_lora.data import (
    combine_database_metadata,
    database_record,
    expand_paths,
    read_jsonl,
    sort_key,
    training_record,
    user_assistant_signature,
    write_jsonl,
)
from scripts.validate_jsonl import validate


@dataclass
class MergeResult:
    records: list[dict[str, Any]]
    deduplicated_ids: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge valid NONO JSONL files in numeric ID order."
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Training JSONL containing only id and messages.",
    )
    parser.add_argument(
        "--database-output",
        type=Path,
        help="Optional JSONL retaining unknown metadata fields.",
    )
    parser.add_argument(
        "--deduplicate-identical",
        action="store_true",
        help="Collapse same-ID records only when user/assistant messages match exactly.",
    )
    return parser.parse_args()


def merge_records(
    records: list[dict[str, Any]], *, deduplicate_identical: bool
) -> MergeResult:
    merged: list[dict[str, Any]] = []
    indexes: dict[str, int] = {}
    deduplicated_ids: list[str] = []
    for record in records:
        record_id = str(record["id"])
        if record_id not in indexes:
            indexes[record_id] = len(merged)
            merged.append(record)
            continue
        if not deduplicate_identical:
            raise ValueError(f"duplicate id {record_id}")
        index = indexes[record_id]
        existing = merged[index]
        if user_assistant_signature(existing) != user_assistant_signature(record):
            raise ValueError(
                f"conflicting duplicate id {record_id}: user/assistant content differs"
            )
        merged[index] = combine_database_metadata(existing, record)
        if record_id not in deduplicated_ids:
            deduplicated_ids.append(record_id)
    merged.sort(key=sort_key)
    return MergeResult(merged, deduplicated_ids)


def main() -> int:
    args = parse_args()
    try:
        input_paths = [path.resolve() for path in expand_paths(args.inputs)]
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    output = args.output.resolve()
    database_output = args.database_output.resolve() if args.database_output else None
    targets = {output}
    if database_output:
        targets.add(database_output)
    if len(targets) != (2 if database_output else 1):
        raise SystemExit("Training and database outputs must be different files.")
    if any(target in input_paths for target in targets):
        raise SystemExit("Output must not overwrite an input file.")

    errors = validate(input_paths, None, None)
    if args.deduplicate_identical:
        errors = [error for error in errors if not error.startswith("duplicate id ")]
    if errors:
        raise SystemExit(
            "Inputs are invalid; merge aborted:\n" + "\n".join(f"- {e}" for e in errors)
        )
    source_records = [item.record for item in read_jsonl(input_paths)]
    try:
        result = merge_records(
            source_records, deduplicate_identical=args.deduplicate_identical
        )
    except ValueError as exc:
        raise SystemExit(f"{exc}; merge aborted without output.") from exc

    training_count = write_jsonl(
        output, (training_record(record) for record in result.records)
    )
    if database_output:
        write_jsonl(
            database_output,
            (database_record(record) for record in result.records),
        )
    ids = ", ".join(result.deduplicated_ids) or "none"
    print(
        f"Deduplicated {len(result.deduplicated_ids)} id(s): {ids}. "
        f"Wrote {training_count} training record(s) to {output}"
    )
    if database_output:
        print(f"Wrote {training_count} database record(s) to {database_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
