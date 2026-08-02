from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from nono_lora.data import expand_paths, read_jsonl, validate_record


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate NONO messages JSONL files.")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--expected-start", type=int)
    parser.add_argument("--expected-end", type=int)
    return parser.parse_args()


def analyze(
    paths: list[Path], expected_start: int | None, expected_end: int | None
) -> ValidationReport:
    report = ValidationReport()
    items = []
    try:
        paths = expand_paths(paths)
        items = list(read_jsonl(paths))
    except (OSError, ValueError) as exc:
        report.errors.append(str(exc))
        return report

    locations: dict[str, list[str]] = defaultdict(list)
    contents: dict[str, dict[str, list[str]]] = {
        "user": defaultdict(list),
        "assistant": defaultdict(list),
    }
    for item in items:
        report.errors.extend(validate_record(item))
        locations[item.id].append(f"{item.path}:{item.line_number}")
        for message in item.record.get("messages", []):
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            content = message.get("content")
            if role in contents and isinstance(content, str):
                contents[role][content].append(item.id)

    for record_id, found_at in sorted(locations.items()):
        if len(found_at) > 1:
            report.errors.append(f"duplicate id {record_id}: {', '.join(found_at)}")

    for role, values in contents.items():
        for record_ids in values.values():
            unique_ids = sorted(set(record_ids))
            if len(unique_ids) > 1:
                report.warnings.append(
                    f"duplicate {role} content in ids: {', '.join(unique_ids)}"
                )

    if (expected_start is None) != (expected_end is None):
        report.errors.append("--expected-start and --expected-end must be used together")
    elif expected_start is not None and expected_end is not None:
        if expected_start > expected_end:
            report.errors.append("--expected-start must not exceed --expected-end")
        else:
            present = {int(key) for key in locations if key.isdigit()}
            missing = [
                f"{value:06d}"
                for value in range(expected_start, expected_end + 1)
                if value not in present
            ]
            if missing:
                report.errors.append(f"missing ids: {', '.join(missing)}")
            outside = sorted(
                value
                for value in present
                if value < expected_start or value > expected_end
            )
            if outside:
                report.errors.append(
                    "ids outside expected range: "
                    + ", ".join(f"{value:06d}" for value in outside)
                )
    return report


def validate(
    paths: list[Path], expected_start: int | None, expected_end: int | None
) -> list[str]:
    """Backward-compatible error-only validation used by other commands."""
    return analyze(paths, expected_start, expected_end).errors


def main() -> int:
    args = parse_args()
    try:
        inputs = expand_paths(args.inputs)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    report = analyze(inputs, args.expected_start, args.expected_end)
    for warning in report.warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    if report.errors:
        for error in report.errors:
            print(f"ERROR: {error}", file=sys.stderr)
        print(
            f"Validation failed with {len(report.errors)} error(s) "
            f"and {len(report.warnings)} warning(s).",
            file=sys.stderr,
        )
        return 1
    count = sum(1 for _ in read_jsonl(inputs))
    print(
        f"OK: {count} record(s) across {len(inputs)} file(s); "
        f"{len(report.warnings)} warning(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
