from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from nono_lora.data import LocatedRecord, read_jsonl, validate_record
from nono_lora.dataset_local import (
    QUESTION_END,
    basic_structure,
    expression_counts,
    find_similar,
    load_jsonl_patterns,
    opening_key,
)
from nono_lora.dataset_pipeline import extract_dialogue

TEASES = ("ざぁこ", "ちょろ〜い", "よわ〜", "かわい〜", "バレバレ", "おつかれさま")
MIND_READING = ("でしょ", "バレ", "思って", "つもり", "また", "どうせ", "見えて")
CARE = ("大丈夫", "でき", "いい", "えら", "休", "試して", "わかる", "好き")
ATTACK = ("死ね", "消えろ", "ゴミ", "クズ", "きもい")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review local NONO candidate quality.")
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--golden", nargs="+", type=Path, default=[Path("dataset/jsonl/*.jsonl")]
    )
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    return parser.parse_args()


def _natural_user(user: str) -> bool:
    return 2 <= len(user) <= 160 and not re.search(r"(です。){3,}|(?:。){3,}", user)


def review_records(candidates: list[dict], golden: list[dict]) -> list[dict[str, Any]]:
    categories = Counter(str(item.get("category", "unknown")) for item in golden)
    expressions = expression_counts(golden)
    golden_ids = {str(item.get("id")) for item in golden}
    reports: list[dict[str, Any]] = []
    prior: list[dict] = []
    for index, record in enumerate(candidates, 1):
        record_id = str(record.get("id", ""))
        user, assistant = extract_dialogue(record)
        hits = find_similar(record, golden + prior)
        structure_ok, missing = basic_structure(assistant)
        schema_errors = validate_record(LocatedRecord(record, Path("<candidate>"), index))
        opening = opening_key(assistant)
        teasing = {word: assistant.count(word) for word in TEASES if word in assistant}
        mind_reading = any(word in assistant for word in MIND_READING)
        helpful = any(word in assistant for word in CARE)
        non_attack = not any(word in assistant for word in ATTACK)
        follow_up = bool(QUESTION_END.search(assistant))
        follow_up_target = bool(record.get("follow_up_target"))
        natural_follow_up = follow_up == follow_up_target or not follow_up_target
        problems = list(schema_errors) + list(missing)
        if not _natural_user(user):
            problems.append("User文を高校生・若者の自然な短文へ調整")
        if not non_attack:
            problems.append("本気の攻撃表現を除去")
        if not helpful:
            problems.append("短い回答または共感を追加")
        if not mind_reading:
            problems.append("一文目付近に見透かし表現を追加")
        if not natural_follow_up:
            problems.append("Follow-up目標に合う自然な問い返しへ修正")
        exact_hit = any(hit.score >= 0.999 for hit in hits)
        result = "reject" if schema_errors or record_id in golden_ids or exact_hit else (
            "warning" if hits or problems else "pass"
        )
        score = max(0, 100 - (35 if result == "reject" else 0) - 8 * len(problems)
                    - min(25, 5 * len(hits)))
        reports.append(
            {
                "id": record_id,
                "result": result,
                "overall_score": score,
                "similar": [hit.as_dict() for hit in hits[:5]],
                "user_topic_duplicate": any(
                    any("user match" in reason or "keyword" in reason for reason in hit.reasons)
                    for hit in hits
                ),
                "situation_duplicate": any(
                    "same situation" in hit.reasons for hit in hits
                ),
                "ending_duplicate": any("same ending" in hit.reasons for hit in hits),
                "past_category_count": categories[str(record.get("category", "unknown"))],
                "opening": opening,
                "past_opening_count": expressions["openings"][opening],
                "teasing_expressions": teasing,
                "teasing_past_counts": {
                    word: sum(extract_dialogue(item)[1].count(word) for item in golden)
                    for word in teasing
                },
                "follow_up": follow_up,
                "follow_up_natural": natural_follow_up,
                "natural_young_user": _natural_user(user),
                "non_attacking": non_attack,
                "answer_or_empathy": helpful,
                "mind_reading": mind_reading,
                "assistant_characters": len(assistant),
                "basic_structure": structure_ok,
                "suggestions": problems or ["修正不要"],
                "schema_errors": schema_errors,
            }
        )
        prior.append(record)
    return reports


def batch_report(records: list[dict], reports: list[dict]) -> dict[str, Any]:
    results = Counter(item["result"] for item in reports)
    categories = Counter(str(item.get("category", "unknown")) for item in records)
    openings = Counter(item["opening"] for item in reports)
    teasing = Counter(
        word for item in reports for word, count in item["teasing_expressions"].items()
        for _ in range(count)
    )
    structures = [str(item.get("conversation_type", "unspecified")) for item in records]
    consecutive = [
        {"index": index + 1, "pattern": structures[index]}
        for index in range(1, len(structures))
        if structures[index] == structures[index - 1]
    ]
    return {
        "records": reports,
        "summary": {
            "pass": results["pass"],
            "warning": results["warning"],
            "reject": results["reject"],
            "follow_up_rate": (
                sum(item["follow_up"] for item in reports) / len(reports) if reports else 0
            ),
            "category_distribution": dict(categories),
            "opening_ranking": dict(openings.most_common()),
            "teasing_ranking": dict(teasing.most_common()),
            "consecutive_syntax": consecutive,
            "similarity_count": sum(bool(item["similar"]) for item in reports),
            "needs_fix": [
                item["id"] for item in reports if item["result"] != "pass"
            ],
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# NONO Candidate Review", "",
        f"- Pass: {summary['pass']}",
        f"- Warning: {summary['warning']}",
        f"- Reject: {summary['reject']}",
        f"- Follow-up rate: {summary['follow_up_rate']:.1%}",
        f"- Similarity warnings: {summary['similarity_count']}",
        f"- Needs fix: {', '.join(summary['needs_fix']) or 'none'}", "",
        "## Batch distribution", "",
        f"- Categories: {summary['category_distribution']}",
        f"- Openings: {summary['opening_ranking']}",
        f"- Teasing: {summary['teasing_ranking']}",
        f"- Consecutive syntax: {summary['consecutive_syntax'] or 'none'}", "",
    ]
    for item in report["records"]:
        similar = ", ".join(
            f"{hit['id']}={hit['score']:.2f} ({'; '.join(hit['reasons'])})"
            for hit in item["similar"]
        ) or "none"
        lines += [
            f"## #{item['id']} — {item['result']} ({item['overall_score']}/100)", "",
            f"- Similar: {similar}",
            f"- Topic/situation/ending duplicate: {item['user_topic_duplicate']} / "
            f"{item['situation_duplicate']} / {item['ending_duplicate']}",
            f"- Opening: {item['opening']} (past {item['past_opening_count']})",
            f"- Teasing: {item['teasing_expressions'] or 'none'}",
            f"- Follow-up / natural: {item['follow_up']} / {item['follow_up_natural']}",
            f"- Natural young User: {item['natural_young_user']}",
            f"- Non-attacking / helpful / mind-reading: {item['non_attacking']} / "
            f"{item['answer_or_empathy']} / {item['mind_reading']}",
            f"- NONO characters: {item['assistant_characters']}",
            f"- Suggestions: {'; '.join(item['suggestions'])}", "",
        ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    candidates = [item.record for item in read_jsonl([args.input])]
    _, golden = load_jsonl_patterns(args.golden)
    report = batch_report(candidates, review_records(candidates, golden))
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report), end="")
    return 1 if report["summary"]["reject"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
