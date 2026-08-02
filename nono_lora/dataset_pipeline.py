from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = "2.0.0"
GENERIC_PHRASES = (
    "理解しました",
    "いつでもそばにいます",
    "心配しないでください",
    "きっとうまくいきます",
    "深呼吸しましょう",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in value if character.isalnum())


def extract_dialogue(record: dict[str, Any]) -> tuple[str, str]:
    messages = record.get("messages")
    if isinstance(messages, list):
        user = next(
            (
                str(item["content"])
                for item in messages
                if isinstance(item, dict) and item.get("role") == "user"
            ),
            "",
        )
        assistant = next(
            (
                str(item["content"])
                for item in reversed(messages)
                if isinstance(item, dict) and item.get("role") == "assistant"
            ),
            "",
        )
        return user, assistant
    return str(record.get("user", "")), str(record.get("assistant", ""))


def next_id(records: Iterable[dict[str, Any]]) -> int:
    values = [
        int(str(record.get("id")))
        for record in records
        if str(record.get("id", "")).isdigit()
    ]
    return max(values, default=0) + 1


def write_review_text(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    blocks: list[str] = []
    count = 0
    for record in records:
        user, assistant = extract_dialogue(record)
        blocks.append(
            f"#{record['id']}\n\n"
            f"User:\n{user}\n\n"
            f"NONO:\n{assistant}\n\n"
            "--------------------------------------------------"
        )
        count += 1
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8", newline="\n")
    return count


def exact_or_normalized_duplicate(
    candidate: dict[str, Any], existing: Iterable[dict[str, Any]]
) -> bool:
    user, assistant = extract_dialogue(candidate)
    signature = (normalize_text(user), normalize_text(assistant))
    user_key = normalize_text(user)
    for record in existing:
        old_user, old_assistant = extract_dialogue(record)
        if signature == (
            normalize_text(old_user),
            normalize_text(old_assistant),
        ):
            return True
        if user_key and user_key == normalize_text(old_user):
            return True
    return False


def rule_quality(record: dict[str, Any]) -> tuple[float, list[str]]:
    user, assistant = extract_dialogue(record)
    warnings: list[str] = []
    score = 100.0
    if not user.strip() or not assistant.strip():
        return 0.0, ["user/assistant content is empty"]
    if len(assistant) < 12:
        score -= 20
        warnings.append("assistant response is unusually short")
    if len(assistant) > 700:
        score -= 15
        warnings.append("assistant response is unusually long")
    paragraphs = [item for item in re.split(r"\n\s*\n", assistant) if item.strip()]
    if len(paragraphs) < 2:
        score -= 8
        warnings.append("response rhythm has fewer than two paragraphs")
    for phrase in GENERIC_PHRASES:
        if phrase in assistant:
            score -= 20
            warnings.append(f"generic phrase: {phrase}")
    if record.get("category") != "serious_conversation" and not any(
        symbol in assistant for symbol in ("♪", "♡", "へぇ", "ふふ", "あら")
    ):
        score -= 5
        warnings.append("no obvious NONO-style marker")
    return max(score, 0.0), warnings


def load_reference_text(paths: Iterable[Path]) -> str:
    sections = []
    for path in paths:
        sections.append(f"\n\n===== {path.as_posix()} =====\n{path.read_text(encoding='utf-8-sig')}")
    return "".join(sections)
