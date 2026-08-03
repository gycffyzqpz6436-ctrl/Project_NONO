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
from nono_lora.dataset_semantic import (
    batch_style_warnings,
    find_semantic_duplicates,
    read_database_files,
    read_reference_files,
    style_features,
)

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
    parser.add_argument("--database-directory", type=Path, default=Path("dataset/database"))
    parser.add_argument("--references-directory", type=Path, default=Path("references"))
    return parser.parse_args()


def _natural_user(user: str) -> bool:
    return 2 <= len(user) <= 160 and not re.search(r"(です。){3,}|(?:。){3,}", user)


def review_records(
    candidates: list[dict],
    golden: list[dict],
    references: list[dict] | None = None,
) -> list[dict[str, Any]]:
    references = references or []
    categories = Counter(str(item.get("category", "unknown")) for item in golden)
    expressions = expression_counts(golden)
    golden_ids = {str(item.get("id")) for item in golden}
    reports: list[dict[str, Any]] = []
    prior: list[dict] = []
    for index, record in enumerate(candidates, 1):
        record_id = str(record.get("id", ""))
        user, assistant = extract_dialogue(record)
        hits = find_similar(record, golden + prior)
        semantic_hits = find_semantic_duplicates(
            record, golden + prior, references
        )
        structure_ok, missing = basic_structure(assistant)
        schema_errors = validate_record(LocatedRecord(record, Path("<candidate>"), index))
        opening = opening_key(assistant)
        teasing = {word: assistant.count(word) for word in TEASES if word in assistant}
        mind_reading = any(word in assistant for word in MIND_READING)
        paragraphs = [
            value.strip() for value in re.split(r"\n\s*\n", assistant) if value.strip()
        ]
        helpful = any(word in assistant for word in CARE) or (
            len(paragraphs) >= 2 and len(paragraphs[1]) >= 12
        )
        non_attack = not any(word in assistant for word in ATTACK)
        follow_up = bool(QUESTION_END.search(assistant))
        follow_up_target = bool(record.get("follow_up_target"))
        natural_follow_up = follow_up == follow_up_target or not follow_up_target
        style = style_features(record)
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
        if style["soft_ai"]:
            problems.append("優しいAI定型句を煽り中心へ変更")
        if style["attacking"]:
            problems.append("本気の攻撃表現を除去")
        if style["mesugaki_strength"] < 7.0:
            problems.append("メスガキ強度を7以上へ上げる")
        if style["teasing_ratio"] < 0.80:
            problems.append("煽り比率を80%以上へ上げる")
        if not style["ending_or_follow_up"]:
            problems.append("最後を煽りまたは自然な問い返しで締める")
        exact_hit = any(
            any(
                reason in {"exact user match", "normalized user match"}
                for reason in hit.reasons
            )
            for hit in hits
        )
        semantic_reject = bool(semantic_hits)
        style_reject = (
            style["soft_ai"] or style["attacking"]
            or style["mesugaki_strength"] < 5.5
            or not style["answer_or_empathy"]
            or not style["ending_or_follow_up"]
        )
        result = (
            "reject"
            if schema_errors or record_id in golden_ids or exact_hit
            or semantic_reject or style_reject
            else ("warning" if hits or problems else "pass")
        )
        score = max(0, 100 - (35 if result == "reject" else 0) - 8 * len(problems)
                    - min(25, 5 * len(hits)))
        reports.append(
            {
                "id": record_id,
                "result": result,
                "overall_score": score,
                "similar": [hit.as_dict() for hit in hits[:5]],
                "semantic_similar": [
                    hit.as_dict() for hit in semantic_hits[:8]
                ],
                "similar_golden_ids": [
                    hit.source_id for hit in semantic_hits
                    if hit.source_kind == "golden"
                ][:8],
                "similar_references": [
                    hit.source_id for hit in semantic_hits
                    if hit.source_kind == "reference"
                ][:8],
                "semantic_reasons": sorted(
                    {reason for hit in semantic_hits for reason in hit.reasons}
                ),
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
                "nono_score": style["nono_score"],
                "mesugaki_strength": style["mesugaki_strength"],
                "teasing_ratio": style["teasing_ratio"],
                "soft_ai": style["soft_ai"],
                "ending_or_follow_up": style["ending_or_follow_up"],
                "ending_pattern": style["ending"],
                "paragraph_count": style["paragraph_count"],
                "basic_structure": structure_ok,
                "suggestions": problems or ["修正不要"],
                "schema_errors": schema_errors,
            }
        )
        prior.append(record)
    return reports


def batch_report(records: list[dict], reports: list[dict]) -> dict[str, Any]:
    style_warnings = batch_style_warnings(records)
    opening_counts = Counter(item["opening"] for item in reports)
    ending_counts = Counter(item["ending_pattern"] for item in reports)
    for index, item in enumerate(reports):
        repeated = []
        if item["opening"] and opening_counts[item["opening"]] > 3:
            repeated.append("冒頭表現がバッチ内で多すぎる")
        if item["ending_pattern"] and ending_counts[item["ending_pattern"]] > 2:
            repeated.append("オチ表現がバッチ内で多すぎる")
        if index and (
            item["opening"] == reports[index - 1]["opening"]
            or item["ending_pattern"] == reports[index - 1]["ending_pattern"]
        ):
            repeated.append("直前候補と冒頭またはオチが同じ")
        if repeated:
            item["suggestions"].extend(repeated)
            if item["result"] == "pass":
                item["result"] = "warning"
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
            "semantic_similarity_count": sum(
                bool(item["semantic_similar"]) for item in reports
            ),
            "style_warnings": style_warnings,
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
        f"- Semantic/reference warnings: {summary['semantic_similarity_count']}",
        f"- Batch style warnings: {summary['style_warnings'] or 'none'}",
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
            f"- Semantic Golden IDs: {', '.join(item['similar_golden_ids']) or 'none'}",
            f"- Similar references: {', '.join(item['similar_references']) or 'none'}",
            f"- Semantic reasons: {'; '.join(item['semantic_reasons']) or 'none'}",
            f"- Topic/situation/ending duplicate: {item['user_topic_duplicate']} / "
            f"{item['situation_duplicate']} / {item['ending_duplicate']}",
            f"- Opening: {item['opening']} (past {item['past_opening_count']})",
            f"- Teasing: {item['teasing_expressions'] or 'none'}",
            f"- Follow-up / natural: {item['follow_up']} / {item['follow_up_natural']}",
            f"- Natural young User: {item['natural_young_user']}",
            f"- Non-attacking / helpful / mind-reading: {item['non_attacking']} / "
            f"{item['answer_or_empathy']} / {item['mind_reading']}",
            f"- NONO characters: {item['assistant_characters']}",
            f"- NONO score / mesugaki / teasing ratio: {item['nono_score']} / "
            f"{item['mesugaki_strength']} / {item['teasing_ratio']:.1%}",
            f"- Soft AI: {item['soft_ai']}",
            f"- Suggestions: {'; '.join(item['suggestions'])}", "",
        ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    candidates = [item.record for item in read_jsonl([args.input])]
    _, golden = load_jsonl_patterns(args.golden)
    _, database = read_database_files(args.database_directory)
    from nono_lora.dataset_semantic import merge_database_metadata
    golden = merge_database_metadata(golden, database)
    _, references = read_reference_files(args.references_directory)
    report = batch_report(
        candidates, review_records(candidates, golden, references)
    )
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
