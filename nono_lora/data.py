from __future__ import annotations

import json
import re
from glob import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

ID_PATTERN = re.compile(r"^\d{6}$")
ALLOWED_ROLES = {"system", "user", "assistant"}


@dataclass(frozen=True)
class LocatedRecord:
    record: dict[str, Any]
    path: Path
    line_number: int

    @property
    def id(self) -> str:
        return str(self.record.get("id", ""))


def expand_paths(paths: Iterable[Path]) -> list[Path]:
    """Expand shell-style globs because PowerShell does not do so for Python."""
    expanded: list[Path] = []
    for path in paths:
        raw = str(path)
        if any(character in raw for character in "*?["):
            matches = [Path(match) for match in sorted(glob(raw))]
            if not matches:
                raise ValueError(f"input pattern matched no files: {raw}")
            expanded.extend(matches)
        else:
            expanded.append(path)
    return expanded


def read_jsonl(paths: Iterable[Path]) -> Iterator[LocatedRecord]:
    """Read UTF-8 JSONL without modifying the source files."""
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    raise ValueError(f"{path}:{line_number}: blank line")
                try:
                    value = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{path}:{line_number}: invalid JSON: {exc.msg}"
                    ) from exc
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number}: record must be an object")
                yield LocatedRecord(value, path, line_number)


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def validate_record(item: LocatedRecord) -> list[str]:
    errors: list[str] = []
    prefix = f"{item.path}:{item.line_number}"
    record_id = item.record.get("id")
    if not isinstance(record_id, str) or not ID_PATTERN.fullmatch(record_id):
        errors.append(f"{prefix}: id must be a six-digit string")

    messages = item.record.get("messages")
    if not isinstance(messages, list) or not messages:
        return errors + [f"{prefix}: messages must be a non-empty array"]

    for index, message in enumerate(messages):
        location = f"{prefix}: messages[{index}]"
        if not isinstance(message, dict):
            errors.append(f"{location} must be an object")
            continue
        if message.get("role") not in ALLOWED_ROLES:
            errors.append(
                f"{location}.role must be one of {sorted(ALLOWED_ROLES)}"
            )
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            errors.append(f"{location}.content must be a non-empty string")
        elif content != content.strip():
            errors.append(f"{location}.content has leading or trailing whitespace")

    roles = [
        message.get("role")
        for message in messages
        if isinstance(message, dict)
    ]
    if "assistant" not in roles:
        errors.append(f"{prefix}: messages must contain an assistant message")
    errors.extend(_validate_role_order(roles, prefix))
    return errors


def _validate_role_order(roles: list[Any], prefix: str) -> list[str]:
    if not roles:
        return []
    if roles == ["assistant"]:
        return []
    conversation = list(roles)
    if conversation[0] == "system":
        conversation = conversation[1:]
    if not conversation or conversation[0] != "user":
        return [
            f"{prefix}: messages role order must be optional system, then "
            "alternating user/assistant"
        ]
    expected = ["user" if index % 2 == 0 else "assistant" for index in range(len(conversation))]
    if conversation != expected or conversation[-1] != "assistant":
        return [
            f"{prefix}: messages role order must be optional system, then "
            "alternating user/assistant, ending with assistant"
        ]
    return []


def training_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return only fields consumed by training."""
    return {"id": record["id"], "messages": record["messages"]}


def database_record(
    record: dict[str, Any],
    *,
    default_status: str = "unreviewed",
    default_language: str = "ja",
) -> dict[str, Any]:
    """Preserve unknown metadata and add documented defaults when absent."""
    result = dict(record)
    result.setdefault("status", default_status)
    result.setdefault("language", default_language)
    return result


def user_assistant_signature(record: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    """Compare only ordered user/assistant roles and their exact contents."""
    messages = record.get("messages", [])
    return tuple(
        (message["role"], message["content"])
        for message in messages
        if isinstance(message, dict)
        and message.get("role") in {"user", "assistant"}
        and isinstance(message.get("content"), str)
    )


def combine_database_metadata(
    primary: dict[str, Any], duplicate: dict[str, Any]
) -> dict[str, Any]:
    """Keep all non-conflicting metadata and record conflicting unknown values."""
    result = dict(primary)
    conflicts = dict(result.get("_deduplication_conflicts", {}))
    for key, value in duplicate.items():
        if key in {"id", "messages", "_deduplication_conflicts"}:
            continue
        if key not in result:
            result[key] = value
        elif result[key] != value:
            values = conflicts.setdefault(key, [result[key]])
            if value not in values:
                values.append(value)
    if conflicts:
        result["_deduplication_conflicts"] = conflicts
    return result


def sort_key(record: dict[str, Any]) -> tuple[int, str]:
    raw_id = str(record.get("id", ""))
    return (int(raw_id), raw_id) if raw_id.isdigit() else (2**63 - 1, raw_id)
