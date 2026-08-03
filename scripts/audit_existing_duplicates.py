from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from nono_lora.data import read_jsonl
from nono_lora.dataset_local import (
    fuzzy_similarity,
    load_jsonl_patterns,
    ngram_similarity,
    normalized_for_similarity,
    write_json_atomic,
)
from nono_lora.dataset_pipeline import extract_dialogue
from nono_lora.dataset_semantic import (
    merge_database_metadata,
    read_database_files,
    read_reference_files,
    semantic_similarity,
    style_features,
)

ALLOWED_ACTIONS = {
    "keep_both",
    "rewrite_user",
    "rewrite_assistant",
    "rewrite_both",
    "delete_duplicate",
    "merge_metadata",
    "manual_review",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the existing Golden Dataset without modifying it."
    )
    parser.add_argument(
        "--golden", nargs="+", type=Path, default=[Path("dataset/jsonl/*.jsonl")]
    )
    parser.add_argument(
        "--database-directory", type=Path, default=Path("dataset/database")
    )
    parser.add_argument(
        "--references-directory", type=Path, default=Path("references")
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("dataset/reports/existing_duplicate_audit.json"),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("dataset/reports/existing_duplicate_audit.md"),
    )
    parser.add_argument(
        "--repairs-output",
        type=Path,
        default=Path("dataset/candidates/review/existing_duplicate_repairs.md"),
    )
    return parser.parse_args()


def _dialogue_payload(record: dict[str, Any]) -> dict[str, str]:
    user, assistant = extract_dialogue(record)
    return {"user": user, "assistant": assistant}


def _assistant_overlap(left: str, right: str) -> dict[str, Any]:
    left_style = style_features(
        {"messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": left}]}
    )
    right_style = style_features(
        {"messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": right}]}
    )
    shared_teases = sorted(
        set(left_style["tease_words"]) & set(right_style["tease_words"])
    )
    return {
        "fuzzy": round(fuzzy_similarity(left, right), 4),
        "n_gram": round(ngram_similarity(left, right), 4),
        "same_opening": (
            left_style["opening"] == right_style["opening"]
            and bool(left_style["opening"])
        ),
        "same_ending": (
            left_style["ending"] == right_style["ending"]
            and bool(left_style["ending"])
        ),
        "shared_teasing": shared_teases,
    }


def _item(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    detection: str,
    score: float,
    reasons: list[str],
    recommendation: str,
    source_a: str = "golden",
    source_b: str = "golden",
) -> dict[str, Any]:
    if recommendation not in ALLOWED_ACTIONS:
        raise ValueError(f"invalid recommendation: {recommendation}")
    left_dialogue = _dialogue_payload(left)
    right_dialogue = _dialogue_payload(right)
    overlap = _assistant_overlap(
        left_dialogue["assistant"], right_dialogue["assistant"]
    )
    return {
        "ids": [str(left.get("id", "")), str(right.get("id", ""))],
        "sources": [source_a, source_b],
        "detection": detection,
        "score": round(score, 4),
        "users": [left_dialogue["user"], right_dialogue["user"]],
        "assistants": [left_dialogue["assistant"], right_dialogue["assistant"]],
        "assistant_overlap": overlap,
        "reasons": reasons,
        "problem": _problem_text(detection, reasons),
        "recommendation": recommendation,
        "repair_suggestion": _repair_suggestion(recommendation, detection, reasons),
    }


def _problem_text(detection: str, reasons: list[str]) -> str:
    labels = {
        "identical_id_duplicate": "同一ID・同一会話が複数ファイルに存在する",
        "conflicting_id_duplicate": "同一IDなのに会話内容が異なる",
        "exact_user_duplicate": "User本文が完全一致する",
        "normalized_user_duplicate": "表記揺れを除くとUser本文が一致する",
        "semantic_duplicate": "話題・状況・悩み・回答方針が意味的に重なる",
        "assistant_structure_duplicate": "NONO回答の構成・オチ・煽りが強く似ている",
        "reference_reuse": "参考会話を流用した可能性がある",
        "database_id_duplicate": "管理用DBで同一IDが複数存在する",
        "database_id_conflict": "管理用DBの同一IDで会話内容が異なる",
    }
    suffix = f": {'; '.join(reasons)}" if reasons else ""
    return labels.get(detection, detection) + suffix


def _repair_suggestion(
    recommendation: str, detection: str, reasons: list[str]
) -> str:
    if recommendation == "delete_duplicate":
        return "完全一致する物理重複のみを、人間承認後に片方のファイルから除外する。IDは変更しない。"
    if recommendation == "merge_metadata":
        return "会話本文は一つに保ち、欠けている管理メタデータだけを統合する。"
    if recommendation == "rewrite_user":
        return "IDを維持し、場所・出来事・感情・結末が異なる新しいUser状況へ書き直す。"
    if recommendation == "rewrite_assistant":
        return "Userは維持し、見透かし、回答方針、煽り、オチ、問い返しを別構成へ変更する。"
    if recommendation == "rewrite_both":
        return "IDは維持し、既存概念と重ならない状況へUser/NONOを一組で全面的に書き直す。"
    if recommendation == "keep_both":
        return "会話タイプと用途の差をメタデータへ明記し、両方を維持する。"
    return "人間が両会話と学習上の役割を確認し、維持または書き直しを決定する。"


def audit_records(
    located_golden: list[Any],
    database: list[dict[str, Any]],
    references: list[dict[str, Any]],
) -> dict[str, Any]:
    raw = [item.record for item in located_golden]
    canonical_by_id: dict[str, dict[str, Any]] = {}
    collapsed: list[str] = []
    for record in raw:
        record_id = str(record.get("id", ""))
        if record_id in canonical_by_id:
            if extract_dialogue(canonical_by_id[record_id]) == extract_dialogue(record):
                collapsed.append(record_id)
            continue
        canonical_by_id[record_id] = record
    canonical = sorted(
        canonical_by_id.values(), key=lambda item: int(str(item["id"]))
    )
    canonical = merge_database_metadata(canonical, database)
    by_id: dict[str, list[Any]] = defaultdict(list)
    for item in located_golden:
        by_id[str(item.record.get("id", ""))].append(item)
    findings: list[dict[str, Any]] = []

    for record_id, entries in sorted(by_id.items()):
        if len(entries) < 2:
            continue
        first = entries[0].record
        same = all(
            extract_dialogue(first) == extract_dialogue(item.record)
            for item in entries[1:]
        )
        for other in entries[1:]:
            findings.append(
                _item(
                    first,
                    other.record,
                    detection=(
                        "identical_id_duplicate"
                        if same
                        else "conflicting_id_duplicate"
                    ),
                    score=1.0,
                    reasons=[
                        f"{entries[0].path}:{entries[0].line_number}",
                        f"{other.path}:{other.line_number}",
                    ],
                    recommendation="merge_metadata" if same else "manual_review",
                )
            )

    database_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in database:
        database_by_id[str(record.get("id", ""))].append(record)
    for record_id, records in sorted(database_by_id.items()):
        if len(records) < 2:
            continue
        first = records[0]
        same = all(
            extract_dialogue(first) == extract_dialogue(item) for item in records[1:]
        )
        for other in records[1:]:
            findings.append(
                _item(
                    first,
                    other,
                    detection="database_id_duplicate" if same else "database_id_conflict",
                    score=1.0,
                    reasons=["same management DB ID"],
                    recommendation="merge_metadata" if same else "manual_review",
                    source_a="database",
                    source_b="database",
                )
            )

    exact_users: dict[str, list[dict[str, Any]]] = defaultdict(list)
    normalized_users: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in canonical:
        user, _ = extract_dialogue(record)
        exact_users[user].append(record)
        normalized_users[normalized_for_similarity(user)].append(record)
    exact_pairs: set[tuple[str, str]] = set()
    for user, records in exact_users.items():
        if not user or len(records) < 2:
            continue
        for index, left in enumerate(records):
            for right in records[index + 1 :]:
                pair = tuple(sorted((str(left["id"]), str(right["id"]))))
                exact_pairs.add(pair)
                left_type = str(left.get("conversation_type", ""))
                right_type = str(right.get("conversation_type", ""))
                assistant_score = fuzzy_similarity(
                    extract_dialogue(left)[1], extract_dialogue(right)[1]
                )
                findings.append(
                    _item(
                        left,
                        right,
                        detection="exact_user_duplicate",
                        score=1.0,
                        reasons=[f"assistant similarity={assistant_score:.2f}"],
                        recommendation=(
                            "keep_both"
                            if left_type and right_type and left_type != right_type
                            and assistant_score < 0.60
                            else "rewrite_both"
                        ),
                    )
                )
    for user, records in normalized_users.items():
        if not user or len(records) < 2:
            continue
        for index, left in enumerate(records):
            for right in records[index + 1 :]:
                pair = tuple(sorted((str(left["id"]), str(right["id"]))))
                if pair in exact_pairs:
                    continue
                findings.append(
                    _item(
                        left,
                        right,
                        detection="normalized_user_duplicate",
                        score=1.0,
                        reasons=["Unicode・空白・句読点・全半角の正規化後に一致"],
                        recommendation="rewrite_user",
                    )
                )

    known_pairs = {
        tuple(sorted(item["ids"]))
        for item in findings
        if item["sources"] == ["golden", "golden"]
    }
    for index, left in enumerate(canonical):
        for right in canonical[index + 1 :]:
            pair = tuple(sorted((str(left["id"]), str(right["id"]))))
            if pair in known_pairs:
                continue
            hit = semantic_similarity(left, right, source_kind="golden")
            if hit:
                findings.append(
                    _item(
                        left,
                        right,
                        detection="semantic_duplicate",
                        score=hit.score,
                        reasons=list(hit.reasons),
                        recommendation=(
                            "rewrite_both"
                            if any("same event" in reason or "same problem" in reason
                                   for reason in hit.reasons)
                            else "manual_review"
                        ),
                    )
                )
                known_pairs.add(pair)
                continue
            left_assistant = extract_dialogue(left)[1]
            right_assistant = extract_dialogue(right)[1]
            fuzzy = fuzzy_similarity(left_assistant, right_assistant)
            ngram = ngram_similarity(left_assistant, right_assistant)
            if max(fuzzy, ngram) >= 0.82:
                findings.append(
                    _item(
                        left,
                        right,
                        detection="assistant_structure_duplicate",
                        score=max(fuzzy, ngram),
                        reasons=[
                            f"assistant fuzzy={fuzzy:.2f}",
                            f"assistant n-gram={ngram:.2f}",
                        ],
                        recommendation="rewrite_assistant",
                    )
                )

    for record in canonical:
        for reference in references:
            hit = semantic_similarity(record, reference, source_kind="reference")
            if hit:
                findings.append(
                    _item(
                        record,
                        reference,
                        detection="reference_reuse",
                        score=hit.score,
                        reasons=list(hit.reasons),
                        recommendation="manual_review",
                        source_b="reference",
                    )
                )

    counts = Counter(item["detection"] for item in findings)
    recommendations = Counter(item["recommendation"] for item in findings)
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "scope": {
            "golden_physical_records": len(raw),
            "golden_unique_ids": len(canonical),
            "database_records": len(database),
            "reference_records": len(references),
            "collapsed_identical_ids_for_analysis_only": collapsed,
        },
        "summary": {
            "finding_count": len(findings),
            "by_detection": dict(counts),
            "by_recommendation": dict(recommendations),
        },
        "findings": sorted(
            findings,
            key=lambda item: (
                -item["score"],
                item["detection"],
                item["ids"][0],
                item["ids"][1],
            ),
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    scope, summary = report["scope"], report["summary"]
    lines = [
        "# Existing Golden Duplicate Audit", "",
        "This report is read-only. No Golden or database record was modified.", "",
        f"- Golden physical records: {scope['golden_physical_records']}",
        f"- Golden unique IDs: {scope['golden_unique_ids']}",
        f"- Database records: {scope['database_records']}",
        f"- Reference records: {scope['reference_records']}",
        f"- Findings: {summary['finding_count']}",
        f"- Detection counts: {summary['by_detection']}",
        f"- Recommendations: {summary['by_recommendation']}", "",
    ]
    for index, item in enumerate(report["findings"], 1):
        lines += [
            f"## {index}. {' / '.join(item['ids'])}", "",
            f"- Sources: {' / '.join(item['sources'])}",
            f"- Detection: `{item['detection']}`",
            f"- Score: {item['score']:.4f}",
            f"- Reasons: {'; '.join(item['reasons']) or 'none'}",
            f"- Problem: {item['problem']}",
            f"- Recommendation: `{item['recommendation']}`",
            f"- Repair: {item['repair_suggestion']}", "",
            "**User A**", "", item["users"][0], "",
            "**User B**", "", item["users"][1], "",
            "**NONO overlap**", "",
            f"- Fuzzy: {item['assistant_overlap']['fuzzy']:.4f}",
            f"- n-gram: {item['assistant_overlap']['n_gram']:.4f}",
            f"- Same opening: {item['assistant_overlap']['same_opening']}",
            f"- Same ending: {item['assistant_overlap']['same_ending']}",
            f"- Shared teasing: {item['assistant_overlap']['shared_teasing']}", "",
        ]
    return "\n".join(lines) + "\n"


def render_repairs(report: dict[str, Any]) -> str:
    lines = [
        "# Existing Duplicate Repair Review", "",
        "Golden/DBを自動変更しない。各項目を人間が承認してから個別に修正する。", "",
    ]
    for index, item in enumerate(report["findings"], 1):
        lines += [
            f"## {index}. {' / '.join(item['ids'])}", "",
            f"Detection: {item['detection']}  ",
            f"Score: {item['score']:.4f}  ",
            f"Recommended action: `{item['recommendation']}`", "",
            "### 修正前 A", "",
            f"User:\n{item['users'][0]}", "",
            f"NONO:\n{item['assistants'][0]}", "",
            "### 修正前 B", "",
            f"User:\n{item['users'][1]}", "",
            f"NONO:\n{item['assistants'][1]}", "",
            "### 修正案", "",
            item["repair_suggestion"], "",
            f"重複理由: {'; '.join(item['reasons'])}", "",
            "Human decision: [ ] keep  [ ] repair  [ ] merge  [ ] delete exact duplicate", "",
            "---", "",
        ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    golden_paths, _ = load_jsonl_patterns(args.golden)
    located = list(read_jsonl(golden_paths))
    _, database = read_database_files(args.database_directory)
    _, references = read_reference_files(args.references_directory)
    report = audit_records(located, database, references)
    write_json_atomic(args.json_output, report)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(
        render_markdown(report), encoding="utf-8", newline="\n"
    )
    args.repairs_output.parent.mkdir(parents=True, exist_ok=True)
    args.repairs_output.write_text(
        render_repairs(report), encoding="utf-8", newline="\n"
    )
    print(
        f"Audited {report['scope']['golden_physical_records']} Golden records "
        f"({report['scope']['golden_unique_ids']} unique IDs); "
        f"found {report['summary']['finding_count']} candidate issue(s)."
    )
    print(f"JSON: {args.json_output}")
    print(f"Markdown: {args.markdown_output}")
    print(f"Repairs: {args.repairs_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
