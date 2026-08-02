from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

from nono_lora.data import LocatedRecord, validate_record, write_jsonl
from nono_lora.dataset_local import (
    find_similar,
    load_jsonl_patterns,
    normalized_for_similarity,
    parse_review_text,
)
from nono_lora.dataset_pipeline import extract_dialogue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import a completed 50-record review TXT.")
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--golden", nargs="+", type=Path, default=[Path("dataset/jsonl/*.jsonl")]
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def validate_import(
    records: list[dict], golden: list[dict], *, source: Path
) -> list[str]:
    errors: list[str] = []
    if len(records) != 50:
        errors.append(f"review text must contain exactly 50 records; got {len(records)}")
        return errors
    ids = [int(record["id"]) for record in records]
    expected = list(range(ids[0], ids[0] + 50))
    if ids != expected:
        errors.append("candidate IDs must be unique, ordered, and consecutive")
    golden_ids = {str(record.get("id")) for record in golden}
    golden_users = {extract_dialogue(record)[0] for record in golden}
    golden_normalized = {
        normalized_for_similarity(extract_dialogue(record)[0]) for record in golden
    }
    candidate_users: set[str] = set()
    candidate_normalized: set[str] = set()
    for line_number, record in enumerate(records, start=1):
        record_id = record["id"]
        user, assistant = extract_dialogue(record)
        if record_id in golden_ids:
            errors.append(f"{record_id}: ID collides with Golden Dataset")
        if not user.strip():
            errors.append(f"{record_id}: User is empty")
        if not assistant.strip():
            errors.append(f"{record_id}: NONO is empty")
        if user in golden_users or user in candidate_users:
            errors.append(f"{record_id}: exact user duplicate")
        normalized = normalized_for_similarity(user)
        if normalized in golden_normalized or normalized in candidate_normalized:
            errors.append(f"{record_id}: normalized user duplicate")
        candidate_users.add(user)
        candidate_normalized.add(normalized)
        errors.extend(
            validate_record(LocatedRecord(record, source, line_number))
        )
    return errors


def import_records(text: str, golden: list[dict], *, source: Path) -> tuple[list[dict], list[str]]:
    records = parse_review_text(text)
    errors = validate_import(records, golden, source=source)
    if errors:
        raise ValueError("\n".join(errors))
    warnings = []
    accepted: list[dict] = []
    for record in records:
        hits = find_similar(record, golden + accepted)
        record["local_similarity_warnings"] = [hit.as_dict() for hit in hits[:5]]
        warnings.extend(
            f"{record['id']} resembles {hit.record_id}: {hit.score:.2f} "
            f"({', '.join(hit.reasons)})"
            for hit in hits[:5]
        )
        accepted.append(record)
    return accepted, warnings


def batch_id_from_name(path: Path) -> str:
    match = re.search(r"_(\d{8}-\d{6})$", path.stem)
    return match.group(1) if match else datetime.now().strftime("%Y%m%d-%H%M%S")


def main() -> int:
    args = parse_args()
    _, golden = load_jsonl_patterns(args.golden)
    try:
        records, warnings = import_records(
            args.input.read_text(encoding="utf-8-sig"), golden, source=args.input
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Import failed:\n{exc}") from exc
    start, end = records[0]["id"], records[-1]["id"]
    output = args.output or Path(
        f"dataset/candidates/nono_candidates_{start}_{end}_{batch_id_from_name(args.input)}.jsonl"
    )
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    write_jsonl(output, records)
    for warning in warnings:
        print(f"WARNING: {warning}")
    print(f"Wrote exactly 50 draft candidates to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
