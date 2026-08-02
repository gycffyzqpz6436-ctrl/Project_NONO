from __future__ import annotations

import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

from nono_lora.data import expand_paths, read_jsonl
from nono_lora.dataset_pipeline import extract_dialogue, normalize_text

try:
    from rapidfuzz.fuzz import ratio as rapidfuzz_ratio
except ImportError:  # Offline fallback; requirements-generation.txt installs RapidFuzz.
    rapidfuzz_ratio = None

OPENING_EXPRESSIONS = (
    "ぷぷっ♪",
    "へぇ〜♪",
    "へぇ～♪",
    "あ〜あ♪",
    "あ～あ♪",
    "ふふっ♪",
)
ENDING_EXPRESSIONS = (
    "バレバレ",
    "おつかれさま",
    "ざぁこ",
    "ちょろ〜い",
    "ちょろ～い",
    "よわ〜",
    "よわ～",
    "かわい〜",
    "かわい～",
)
SYNTAX_PATTERNS = {
    "type_question": re.compile(r"タイプ[？?]"),
    "did_it_right": re.compile(r"してたでしょ[？?]?"),
    "barebare": re.compile(r"バレバレ"),
    "otsukaresama": re.compile(r"おつかれさま"),
    "zaako": re.compile(r"ざ[ぁあ]こ"),
    "choroi": re.compile(r"ちょろ[〜～ー]い"),
    "yowai": re.compile(r"よわ[〜～ー]"),
    "kawaii": re.compile(r"かわい[〜～ー]"),
}
QUESTION_END = re.compile(r"[？?]\s*(?:[♪♡♥❤])?\s*$")
SEPARATOR = "-" * 50
ID_HEADER = re.compile(r"(?m)^#(\d{6})\s*$")


def resolve_patterns(
    patterns: Iterable[Path], *, allow_unmatched: bool = False
) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        try:
            paths.extend(expand_paths([pattern]))
        except ValueError:
            if not allow_unmatched:
                raise
    return sorted(set(paths))


def load_jsonl_patterns(patterns: Iterable[Path]) -> tuple[list[Path], list[dict[str, Any]]]:
    paths = resolve_patterns(patterns)
    if not paths:
        raise ValueError("no JSONL files found")
    return paths, [item.record for item in read_jsonl(paths)]


def collapse_identical_id_duplicates(
    records: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Collapse only same-ID records whose user/assistant dialogue is identical."""
    unique: dict[str, dict[str, Any]] = {}
    collapsed: list[str] = []
    for record in records:
        record_id = str(record.get("id", ""))
        previous = unique.get(record_id)
        if previous is None:
            unique[record_id] = record
            continue
        if extract_dialogue(previous) != extract_dialogue(record):
            raise ValueError(
                f"conflicting duplicate Golden ID {record_id}: dialogue differs"
            )
        if record_id not in collapsed:
            collapsed.append(record_id)
    return sorted(unique.values(), key=lambda item: int(str(item["id"]))), collapsed


def dataset_state(
    records: list[dict[str, Any]],
    *,
    pending_review_file: str | None = None,
    last_analysis_report: str = "dataset/reports/dataset_analysis.json",
) -> dict[str, Any]:
    unique, collapsed = collapse_identical_id_duplicates(records)
    ids = numeric_ids(unique)
    maximum = max(ids)
    start = maximum + 1
    last_start = max(1, maximum - 49)
    return {
        "golden_record_count": len(unique),
        "maximum_id": f"{maximum:06d}",
        "next_id": f"{start:06d}",
        "next_range": f"{start:06d}-{start + 49:06d}",
        "last_approved_batch": f"{last_start:06d}-{maximum:06d}",
        "last_analysis_report": last_analysis_report,
        "pending_review_file": pending_review_file,
        "collapsed_identical_ids": collapsed,
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def numeric_ids(records: Iterable[dict[str, Any]]) -> list[int]:
    values = [
        int(str(record.get("id")))
        for record in records
        if str(record.get("id", "")).isdigit()
    ]
    if not values:
        raise ValueError("dataset contains no numeric IDs")
    return values


def dry_run_report(
    golden_patterns: Iterable[Path],
    existing_patterns: Iterable[Path],
    reference_patterns: Iterable[Path],
    *,
    count: int,
) -> list[str]:
    golden_paths, golden = load_jsonl_patterns(golden_patterns)
    ids = numeric_ids(golden)
    start = max(ids) + 1
    existing_paths = resolve_patterns(existing_patterns, allow_unmatched=True)
    reference_paths = resolve_patterns(reference_patterns)
    return [
        f"Golden records: {len(golden)}",
        f"Maximum ID: {max(ids):06d}",
        f"Next ID: {start:06d}",
        f"Planned range: {start:06d}-{start + count - 1:06d}",
        "Golden files:",
        *(f"- {path.as_posix()}" for path in golden_paths),
        "Existing files:",
        *(f"- {path.as_posix()}" for path in existing_paths),
        "Reference files:",
        *(f"- {path.as_posix()}" for path in reference_paths),
    ]


def normalized_for_similarity(text: str) -> str:
    return normalize_text(unicodedata.normalize("NFKC", text))


def ngrams(text: str, size: int = 3) -> set[str]:
    value = normalized_for_similarity(text)
    if len(value) < size:
        return {value} if value else set()
    return {value[index : index + size] for index in range(len(value) - size + 1)}


def ngram_similarity(left: str, right: str) -> float:
    a, b = ngrams(left), ngrams(right)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if a | b else 0.0


def fuzzy_similarity(left: str, right: str) -> float:
    a, b = normalized_for_similarity(left), normalized_for_similarity(right)
    if rapidfuzz_ratio is not None:
        return float(rapidfuzz_ratio(a, b)) / 100.0
    return SequenceMatcher(None, a, b).ratio()


def keyword_overlap(left: str, right: str) -> float:
    # Character bigrams work for Japanese text without requiring a tokenizer.
    if min(
        len(normalized_for_similarity(left)),
        len(normalized_for_similarity(right)),
    ) < 8:
        return 0.0
    a, b = ngrams(left, 2), ngrams(right, 2)
    return len(a & b) / min(len(a), len(b)) if a and b else 0.0


def ending_key(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return normalized_for_similarity(lines[-1] if lines else "")[-16:]


def opening_key(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    first = lines[0] if lines else ""
    for expression in OPENING_EXPRESSIONS:
        if expression in first:
            return expression.replace("～", "〜")
    return first[:12]


@dataclass(frozen=True)
class SimilarityHit:
    record_id: str
    score: float
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.record_id,
            "score": round(self.score, 4),
            "reasons": list(self.reasons),
        }


def find_similar(
    candidate: dict[str, Any],
    existing: Iterable[dict[str, Any]],
    *,
    warning_threshold: float = 0.72,
) -> list[SimilarityHit]:
    user, assistant = extract_dialogue(candidate)
    category = str(candidate.get("category", "unknown"))
    situation = normalized_for_similarity(str(candidate.get("scenario", "")))
    finish = ending_key(assistant)
    hits: list[SimilarityHit] = []
    for record in existing:
        old_user, old_assistant = extract_dialogue(record)
        exact = user == old_user
        normalized = normalized_for_similarity(user) == normalized_for_similarity(old_user)
        ng = ngram_similarity(user, old_user)
        fuzzy = fuzzy_similarity(user, old_user)
        keywords = keyword_overlap(user, old_user)
        metadata_matches = 0
        reasons: list[str] = []
        if exact:
            reasons.append("exact user match")
        elif normalized:
            reasons.append("normalized user match")
        if ng >= 0.55:
            reasons.append(f"character n-gram={ng:.2f}")
        if fuzzy >= 0.72:
            reasons.append(f"RapidFuzz={fuzzy:.2f}")
        if keywords >= 0.65:
            reasons.append(f"keyword overlap={keywords:.2f}")
        if category != "unknown" and category == str(record.get("category", "unknown")):
            metadata_matches += 1
            reasons.append("same category")
        old_situation = normalized_for_similarity(str(record.get("scenario", "")))
        if situation and old_situation and situation == old_situation:
            metadata_matches += 1
            reasons.append("same situation")
        if finish and finish == ending_key(old_assistant):
            metadata_matches += 1
            reasons.append("same ending")
        score = max(1.0 if exact or normalized else 0.0, ng, fuzzy, keywords)
        score = min(1.0, score + metadata_matches * 0.03)
        if metadata_matches >= 3 and score >= 0.55:
            score = max(score, 0.78)
            reasons.append("category/situation/ending combination match")
        if score >= warning_threshold:
            hits.append(SimilarityHit(str(record.get("id", "")), score, tuple(reasons)))
    return sorted(hits, key=lambda hit: (-hit.score, hit.record_id))


def expression_counts(records: Iterable[dict[str, Any]]) -> dict[str, Counter[str]]:
    openings: Counter[str] = Counter()
    endings: Counter[str] = Counter()
    patterns: Counter[str] = Counter()
    for record in records:
        _, assistant = extract_dialogue(record)
        openings[opening_key(assistant)] += 1
        ending = ending_key(assistant)
        for expression in ENDING_EXPRESSIONS:
            if normalized_for_similarity(expression) in ending:
                endings[expression.replace("～", "〜")] += 1
        for name, pattern in SYNTAX_PATTERNS.items():
            patterns[name] += len(pattern.findall(assistant))
    return {"openings": openings, "endings": endings, "patterns": patterns}


def analyze_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    id_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    exact_users: dict[str, list[str]] = defaultdict(list)
    normalized_users: dict[str, list[str]] = defaultdict(list)
    categories: Counter[str] = Counter()
    user_lengths: list[int] = []
    assistant_lengths: list[int] = []
    questions = 0
    for record in records:
        record_id = str(record.get("id", ""))
        id_groups[record_id].append(record)
        user, assistant = extract_dialogue(record)
        exact_users[user].append(record_id)
        normalized_users[normalized_for_similarity(user)].append(record_id)
        categories[str(record.get("category", "unknown"))] += 1
        user_lengths.append(len(user))
        assistant_lengths.append(len(assistant))
        questions += bool(QUESTION_END.search(assistant))
    identical_ids, conflicting_ids = [], []
    for record_id, values in id_groups.items():
        if len(values) < 2:
            continue
        signatures = {extract_dialogue(value) for value in values}
        target = identical_ids if len(signatures) == 1 else conflicting_ids
        target.append(record_id)
    expressions = expression_counts(records)
    recent = expression_counts(records[-50:])
    recent_warnings: list[str] = []
    for group in ("openings", "endings", "patterns"):
        for value, count in recent[group].items():
            if count >= 8:
                recent_warnings.append(
                    f"recent 50 {group}: '{value}' appears {count} times"
                )
    ids = numeric_ids(records)
    start = max(ids) + 1
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "record_count": len(records),
        "id": {
            "minimum": f"{min(ids):06d}",
            "maximum": f"{max(ids):06d}",
            "next": f"{start:06d}",
            "planned_range": f"{start:06d}-{start + 49:06d}",
            "identical_duplicates": sorted(identical_ids),
            "conflicting_duplicates": sorted(conflicting_ids),
        },
        "user_duplicates": {
            "exact": {
                text: ids for text, ids in exact_users.items() if len(set(ids)) > 1
            },
            "normalized": {
                text: ids
                for text, ids in normalized_users.items()
                if len(set(ids)) > 1
            },
        },
        "categories": dict(categories.most_common()),
        "openings": dict(expressions["openings"].most_common()),
        "endings": dict(expressions["endings"].most_common()),
        "syntax_patterns": dict(expressions["patterns"].most_common()),
        "follow_up_rate": questions / len(records) if records else 0.0,
        "average_characters": {
            "user": sum(user_lengths) / len(user_lengths) if user_lengths else 0.0,
            "assistant": (
                sum(assistant_lengths) / len(assistant_lengths)
                if assistant_lengths
                else 0.0
            ),
        },
        "recent_50_bias_warnings": recent_warnings,
    }


def render_analysis_markdown(report: dict[str, Any]) -> str:
    def table(values: dict[str, Any]) -> str:
        rows = ["| Value | Count |", "|---|---:|"]
        rows.extend(f"| {key or '(empty)'} | {value} |" for key, value in values.items())
        return "\n".join(rows)

    return f"""# NONO Dataset Analysis

- Records: {report['record_count']}
- Maximum ID: {report['id']['maximum']}
- Next ID: {report['id']['next']}
- Planned range: {report['id']['planned_range']}
- Follow-up rate: {report['follow_up_rate']:.1%}
- Average user characters: {report['average_characters']['user']:.1f}
- Average assistant characters: {report['average_characters']['assistant']:.1f}

## ID problems

- Identical duplicate IDs: {', '.join(report['id']['identical_duplicates']) or 'none'}
- Conflicting duplicate IDs: {', '.join(report['id']['conflicting_duplicates']) or 'none'}

## Categories

{table(report['categories'])}

## Opening expressions

{table(report['openings'])}

## Ending expressions

{table(report['endings'])}

## Syntax patterns

{table(report['syntax_patterns'])}

## Recent 50 warnings

{chr(10).join(f'- {item}' for item in report['recent_50_bias_warnings']) or '- none'}
"""


def split_review_blocks(text: str) -> list[tuple[str, str]]:
    matches = list(ID_HEADER.finditer(text))
    if not matches:
        raise ValueError("no #6-digit ID headers found")
    blocks = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        blocks.append((match.group(1), text[match.end() : end]))
    return blocks


def field_value(block: str, label: str, next_labels: Iterable[str]) -> str:
    alternatives = "|".join(re.escape(item) for item in next_labels)
    pattern = re.compile(
        rf"(?ms)^\s*{re.escape(label)}:\s*\n?(.*?)(?=^\s*(?:{alternatives}):\s*$|^\s*{re.escape(SEPARATOR)}\s*$|\Z)"
    )
    match = pattern.search(block)
    return match.group(1).strip() if match else ""


def parse_review_text(text: str) -> list[dict[str, Any]]:
    records = []
    labels = ("Category", "Pattern", "Follow-up", "Used topics to avoid",
              "Suggested direction", "User", "NONO")
    for record_id, block in split_review_blocks(text):
        values = {
            label: field_value(block, label, [item for item in labels if item != label])
            for label in labels
        }
        records.append(
            {
                "schema_version": "2.1.0",
                "id": record_id,
                "character": "NONO",
                "status": "draft",
                "language": "ja",
                "category": values["Category"] or "unknown",
                "conversation_type": values["Pattern"] or "unspecified",
                "follow_up_target": values["Follow-up"].lower() in {"yes", "true", "1"},
                "scenario": values["Suggested direction"],
                "planning": {
                    "used_topics_to_avoid": [
                        item.strip()
                        for item in re.split(r"[,、]", values["Used topics to avoid"])
                        if item.strip()
                    ]
                },
                "messages": [
                    {"role": "user", "content": values["User"]},
                    {"role": "assistant", "content": values["NONO"]},
                ],
                "review": {
                    "decision": "pending",
                    "reviewer": None,
                    "reviewed_at": None,
                    "notes": "",
                },
            }
        )
    return records


def render_review_text(records: Iterable[dict[str, Any]], *, include_plan: bool) -> str:
    blocks = []
    for record in records:
        user, assistant = extract_dialogue(record)
        lines = [f"#{record['id']}", ""]
        if include_plan:
            planning = record.get("planning", {})
            topics = planning.get("used_topics_to_avoid", [])
            lines.extend(
                [
                    "Category:",
                    str(record.get("category", "unknown")),
                    "",
                    "Pattern:",
                    str(record.get("conversation_type", "unspecified")),
                    "",
                    "Follow-up:",
                    "yes" if record.get("follow_up_target") else "no",
                    "",
                    "Used topics to avoid:",
                    "、".join(topics),
                    "",
                    "Suggested direction:",
                    str(record.get("scenario", "")),
                    "",
                ]
            )
        lines.extend(["User:", user, "", "NONO:", assistant, "", SEPARATOR])
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"


def basic_structure(assistant: str) -> tuple[bool, list[str]]:
    paragraphs = [item.strip() for item in re.split(r"\n\s*\n", assistant) if item.strip()]
    missing = []
    if len(paragraphs) < 3:
        missing.append("reaction/insight/answer rhythm is too short")
    if not any(token in assistant for token in ("♪", "へぇ", "ふふ", "あ〜", "あ～")):
        missing.append("reaction marker")
    if not any(token in assistant for token in ("でしょ", "バレ", "タイプ", "ってこと", "なんだ")):
        missing.append("insight")
    has_explicit_tease = any(
        token in assistant
        for token in (
            "ざぁこ", "ちょろ", "よわ", "かわい", "おつかれ", "じゃん",
            "だね", "すぎ", "甘い", "惜し", "上手", "えら", "ちゃん",
        )
    )
    has_playful_mind_reading = "でしょ" in assistant and any(
        marker in assistant for marker in ("♪", "♡", "〜")
    )
    if not has_explicit_tease and not has_playful_mind_reading:
        missing.append("teasing/closing")
    return not missing, missing
