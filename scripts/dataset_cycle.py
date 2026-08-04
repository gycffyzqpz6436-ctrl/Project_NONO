from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from nono_lora.data import database_record, read_jsonl, training_record, write_jsonl
from nono_lora.dataset_local import (
    analyze_records,
    collapse_identical_id_duplicates,
    dataset_state,
    expression_counts,
    load_jsonl_patterns,
    render_analysis_markdown,
    write_json_atomic,
)
from nono_lora.dataset_pipeline import extract_dialogue
from scripts.approve_candidates import approve
from scripts.create_dataset_draft import load_plan, make_draft_records
from scripts.import_review_text import batch_id_from_name, import_records
from scripts.review_candidates import batch_report, render_markdown, review_records
from nono_lora.dataset_semantic import (
    merge_database_metadata,
    read_database_files,
    read_reference_files,
)

GOLDEN = [Path("dataset/jsonl/*.jsonl")]
STATE = Path("dataset/state/dataset_state.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a complete local NONO dataset cycle.")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare", help="Analyze Golden and create 50 draft slots.")
    prepare_parser.add_argument("--count", type=int, default=50)
    prepare_parser.add_argument("--golden", nargs="+", type=Path, default=GOLDEN)
    prepare_parser.add_argument(
        "--category-plan", type=Path, default=Path("dataset/category_plan.yaml")
    )
    prepare_parser.add_argument(
        "--review-directory", type=Path, default=Path("dataset/candidates/review")
    )
    prepare_parser.add_argument(
        "--database-directory", type=Path, default=Path("dataset/database")
    )
    prepare_parser.add_argument(
        "--references-directory", type=Path, default=Path("references")
    )
    for name in ("review", "repair"):
        child = sub.add_parser(name, help=f"{name.title()} a completed review TXT.")
        child.add_argument("review_file", type=Path)
        child.add_argument("--golden", nargs="+", type=Path, default=GOLDEN)
        child.add_argument(
            "--database-directory", type=Path, default=Path("dataset/database")
        )
        child.add_argument(
            "--references-directory", type=Path, default=Path("references")
        )
        if name == "review":
            child.add_argument(
                "--replace-results",
                action="store_true",
                help="Replace only the derived candidate and review reports for this TXT.",
            )
    approval = sub.add_parser("approve", help="Approve, optionally commit and push.")
    approval.add_argument("review_file", type=Path)
    approval.add_argument("--reviewer", required=True)
    approval.add_argument("--golden", nargs="+", type=Path, default=GOLDEN)
    approval.add_argument("--commit", action="store_true")
    approval.add_argument("--push", action="store_true")
    approval.add_argument(
        "--database-directory", type=Path, default=Path("dataset/database")
    )
    approval.add_argument(
        "--references-directory", type=Path, default=Path("references")
    )
    return parser.parse_args()


def _load_golden(patterns: list[Path]) -> tuple[list[Path], list[dict], list[str]]:
    paths, raw = load_jsonl_patterns(patterns)
    unique, collapsed = collapse_identical_id_duplicates(raw)
    return paths, unique, collapsed


def _load_context(
    patterns: list[Path],
    database_directory: Path,
    references_directory: Path,
) -> tuple[list[Path], list[dict], list[str], list[Path], list[dict], list[Path], list[dict]]:
    paths, golden, collapsed = _load_golden(patterns)
    database_paths, database = read_database_files(database_directory)
    reference_paths, references = read_reference_files(references_directory)
    return (
        paths,
        merge_database_metadata(golden, database),
        collapsed,
        database_paths,
        database,
        reference_paths,
        references,
    )


def _write_analysis(records: list[dict]) -> dict[str, Any]:
    report = analyze_records(records)
    json_path = Path("dataset/reports/dataset_analysis.json")
    md_path = Path("dataset/reports/dataset_analysis.md")
    write_json_atomic(json_path, report)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_analysis_markdown(report), encoding="utf-8", newline="\n")
    return report


def _write_state(records: list[dict], pending: Path | None) -> None:
    write_json_atomic(
        STATE,
        dataset_state(
            records,
            pending_review_file=pending.as_posix() if pending else None,
        ),
    )


def _instructions(
    review_file: Path,
    golden_paths: list[Path],
    records: list[dict],
    draft: list[dict],
    database_paths: list[Path],
    reference_paths: list[Path],
) -> str:
    analysis = analyze_records(records)
    planned = {}
    for item in draft:
        category = str(item["category"])
        planned[category] = planned.get(category, 0) + 1
    recent = records[-100:]
    expressions = expression_counts(recent)
    openings = ", ".join(
        f"{key}({count})" for key, count in expressions["openings"].most_common(5)
    )
    teasing = ", ".join(
        f"{key}({count})" for key, count in expressions["patterns"].most_common(5)
    )
    endings = ", ".join(
        f"{key}({count})" for key, count in expressions["endings"].most_common(5)
    )
    recent_categories = Counter(
        str(item.get("category", "unknown")) for item in recent
    )
    category_ranking = ", ".join(
        f"{key}({count})" for key, count in recent_categories.most_common(8)
    )
    recent_follow_ups = sum(
        bool(re.search(r"[？?]\s*[♡♪〜～]*\s*$", extract_dialogue(item)[1]))
        for item in recent
    )
    recent_follow_up_rate = recent_follow_ups / len(recent) if recent else 0.0
    follow_up_target = sum(bool(item["follow_up_target"]) for item in draft) / len(draft)
    return f"""# Codex writing instructions

対象TXT: `{review_file.as_posix()}`
既存Golden: {', '.join(f'`{path.as_posix()}`' for path in golden_paths)}
管理用DB: {', '.join(f'`{path.as_posix()}`' for path in database_paths) or 'なし'}
参考会話: {', '.join(f'`{path.as_posix()}`' for path in reference_paths) or 'なし（referencesへ配置すること）'}
分析レポート: `dataset/reports/dataset_analysis.json`, `dataset/reports/dataset_analysis.md`

## 実施内容

- 執筆前に全Golden {len(records)}件、管理用DB、reference、dataset_analysis、直近100件を読む
- 執筆前に「今回避ける話題」を内部で決め、50件すべての照合基準にする
- `User:` と `NONO:` の欄だけを50件すべて埋める
- ID、Category、Pattern、Follow-up、Used topics to avoid、Suggested directionを変更しない
- 既存Goldenと同じ話題、状況、オチ、言い換えだけの会話を作らない
- 参考会話はUser文体の参考だけにし、質問・出来事・回答方針を流用しない
- 煽りを中心（目標80〜90%）にし、優しいAI定型句を中心にしない
- 今回のカテゴリ計画: {planned}
- 直近100件で多いカテゴリ: {category_ranking or 'なし'}
- 直近100件で多い冒頭: {openings or 'なし'}
- 直近100件で多い煽り・構文: {teasing or 'なし'}
- 直近100件で多い問い返し・オチ: {endings or 'なし'}
- 直近100件の問い返し率: {recent_follow_up_rate:.0%}
- 問い返し目標: {follow_up_target:.0%}（各枠のFollow-up指定を優先）
- 単語だけを変えた言い換え、同じ会話構造、同じ回答方針、同じオチは禁止
- 新規50件で「でしょ」を使うレコードは40%以下
- 同一構文を3件以上連続させず、同じ語尾を5件以上連続させない
- 「ちゃんと」「次は」を避け、文脈に合う別表現へ置き換える
- reviewの「最近100件との構文類似」を確認し、強い類似は構成から変更する

## NONOキャラクタールール

- 最初から相手の行動や内心を見透かし、断定調で捕まえる
- 軽い煽りを説明より先に置き、説明は短く話し言葉で
- からかいを約40%にし、本気の攻撃や先生・相談員口調を避ける
- 「あ〜あ♪」「へぇ〜？」「ぷっ♡」等でリズムを崩し、会話的な語尾を使う
- 回答または共感を必ず含め、最後は軽い甘やかしや追い煽り

50件を書いたら `dataset_cycle review` を実行し、warning/rejectを本文修正して再reviewする。
pass 50・warning 0・reject 0まで繰り返し、人間へレビューTXT全文とレビュー結果を提示する。
この段階では承認、Golden追加、commit、pushを行わない。

Golden解析: {analysis['record_count']}件、次回範囲 {analysis['id']['planned_range']}
"""


def prepare_cycle(
    *,
    count: int,
    golden_patterns: list[Path],
    category_plan: Path,
    review_directory: Path,
    database_directory: Path = Path("dataset/database"),
    references_directory: Path = Path("references"),
    batch_id: str | None = None,
) -> dict[str, Any]:
    if count != 50:
        raise ValueError("prepare requires exactly 50 records")
    (
        paths, golden, collapsed, database_paths, _, reference_paths, _
    ) = _load_context(
        golden_patterns, database_directory, references_directory
    )
    _write_analysis(golden)
    draft = make_draft_records(golden, load_plan(category_plan), count=count)
    batch = batch_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    start, end = draft[0]["id"], draft[-1]["id"]
    review_file = review_directory / f"nono_draft_{start}_{end}_{batch}.txt"
    instructions = review_file.with_suffix(".instructions.md")
    if review_file.exists() or instructions.exists():
        raise ValueError("prepare output already exists; refusing to overwrite")
    from nono_lora.dataset_local import render_review_text
    review_file.parent.mkdir(parents=True, exist_ok=True)
    review_file.write_text(
        render_review_text(draft, include_plan=True), encoding="utf-8", newline="\n"
    )
    instructions.write_text(
        _instructions(
            review_file, paths, golden, draft, database_paths, reference_paths
        ),
        encoding="utf-8",
        newline="\n",
    )
    _write_state(golden, review_file)
    return {
        "golden_records": len(golden),
        "collapsed_ids": collapsed,
        "range": f"{start}-{end}",
        "review_file": review_file,
        "instructions_file": instructions,
    }


def review_cycle(
    review_file: Path,
    golden_patterns: list[Path],
    database_directory: Path = Path("dataset/database"),
    references_directory: Path = Path("references"),
    *,
    replace_results: bool = False,
) -> tuple[Path, Path, Path, dict[str, Any]]:
    _, golden, _, _, _, _, references = _load_context(
        golden_patterns, database_directory, references_directory
    )
    records, _ = import_records(
        review_file.read_text(encoding="utf-8-sig"), golden, source=review_file
    )
    start, end = records[0]["id"], records[-1]["id"]
    batch = batch_id_from_name(review_file)
    candidate = Path(
        f"dataset/candidates/nono_candidates_{start}_{end}_{batch}.jsonl"
    )
    json_output = Path(f"dataset/reports/review_{start}_{end}_{batch}.json")
    md_output = json_output.with_suffix(".md")
    if not replace_results and any(
        path.exists() for path in (candidate, json_output, md_output)
    ):
        raise ValueError("review output already exists; refusing to overwrite")
    write_jsonl(candidate, records)
    report = batch_report(records, review_records(records, golden, references))
    write_json_atomic(json_output, report)
    md_output.write_text(render_markdown(report), encoding="utf-8", newline="\n")
    _write_state(golden, review_file)
    return candidate, json_output, md_output, report


def _review_report_path(review_file: Path) -> Path:
    match = re.search(r"_(\d{6})_(\d{6})_(\d{8}-\d{6})$", review_file.stem)
    if not match:
        raise ValueError("review filename must contain range and batch ID")
    return Path(f"dataset/reports/review_{match.group(1)}_{match.group(2)}_{match.group(3)}.json")


def repair_cycle(review_file: Path) -> Path:
    report_path = _review_report_path(review_file)
    if not report_path.exists():
        raise ValueError("review report not found; run dataset_cycle review first")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    records = {
        item["id"]: item
        for item in __import__("nono_lora.dataset_local", fromlist=["parse_review_text"])
        .parse_review_text(review_file.read_text(encoding="utf-8-sig"))
    }
    targets = [item for item in report["records"] if item["result"] != "pass"]
    match = re.search(r"_(\d{6})_(\d{6})_(\d{8}-\d{6})$", review_file.stem)
    assert match
    output = review_file.parent / (
        f"repair_{match.group(1)}_{match.group(2)}_{match.group(3)}.md"
    )
    lines = ["# NONO repair instructions", ""]
    for item in targets:
        record = records[item["id"]]
        user, assistant = extract_dialogue(record)
        planning = record.get("planning", {})
        lines += [
            f"## #{item['id']}", "",
            f"**Current User**\n\n{user}", "",
            f"**Current NONO**\n\n{assistant}", "",
            f"- Problems: {'; '.join(item['suggestions'])}",
            f"- Similar Golden IDs: {', '.join(hit['id'] for hit in item['similar']) or 'none'}",
            f"- Avoid topics: {', '.join(planning.get('used_topics_to_avoid', [])) or 'none'}",
            f"- Avoid opening: {item['opening']}",
            f"- Avoid ending: {'same ending' if item['ending_duplicate'] else 'none identified'}",
            f"- New direction: {record.get('scenario', '別の未使用の日常場面')}", "",
        ]
    output.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return output


def approve_cycle(
    review_file: Path,
    *,
    reviewer: str,
    golden_patterns: list[Path],
    commit: bool,
    push: bool,
    database_directory: Path = Path("dataset/database"),
    references_directory: Path = Path("references"),
    input_fn: Callable[[str], str] = input,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> dict[str, Any]:
    _, golden, _, _, _, _, references = _load_context(
        golden_patterns, database_directory, references_directory
    )
    records, warnings = import_records(
        review_file.read_text(encoding="utf-8-sig"), golden, source=review_file
    )
    report = batch_report(records, review_records(records, golden, references))
    summary = report["summary"]
    start, end = records[0]["id"], records[-1]["id"]
    database_output = Path(f"dataset/database/nono_database_{start}_{end}.jsonl")
    training_output = Path(f"dataset/jsonl/nono_dataset_{start}_{end}.jsonl")
    prompt = (
        "About to approve:\n"
        f"  Range: {start}-{end}\n  Records: {len(records)}\n"
        f"  Warnings: {summary['warning']}\n  Rejects: {summary['reject']}\n"
        f"  Database output: {database_output}\n  Training output: {training_output}\n"
        f"  Git commit: {'enabled' if commit else 'disabled'}\n"
        f"  Git push: {'enabled' if push else 'disabled'}\n\nType APPROVE to continue: "
    )
    if input_fn(prompt) != "APPROVE":
        return {"cancelled": True}
    if (
        warnings
        or summary["warning"]
        or summary["reject"]
        or summary.get("semantic_similarity_count", 0)
        or summary.get("style_warnings", [])
    ):
        raise ValueError("approval requires pass=50, warning=0, reject=0")
    if database_output.exists() or training_output.exists():
        raise ValueError("approval output already exists; refusing to overwrite")
    approved = approve(
        records,
        reviewer,
        set(),
        golden,
        reference_records=references,
    )
    write_jsonl(database_output, (database_record(item) for item in approved))
    write_jsonl(training_output, (training_record(item) for item in approved))
    for output in (database_output, training_output):
        loaded = list(read_jsonl([output]))
        if len(loaded) != 50:
            raise ValueError(f"post-write validation failed: {output}")
    _, refreshed, _ = _load_golden(golden_patterns)
    _write_analysis(refreshed)
    _write_state(refreshed, None)
    staged = [database_output, training_output, STATE]
    if commit:
        runner(["git", "status", "--short"], check=True, text=True, capture_output=True)
        runner(["git", "add", "--", *(str(path) for path in staged)], check=True)
        runner(
            ["git", "commit", "-m", f"dataset: add NONO conversations {start}-{end}"],
            check=True,
        )
    if push:
        if not commit:
            raise ValueError("--push requires --commit")
        try:
            runner(["git", "push"], check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "push failed; outputs and commit were kept. Retry with: git push"
            ) from exc
    return {
        "cancelled": False,
        "database_output": database_output,
        "training_output": training_output,
        "staged": staged,
    }


def main() -> int:
    args = parse_args()
    try:
        if args.command == "prepare":
            result = prepare_cycle(
                count=args.count,
                golden_patterns=args.golden,
                category_plan=args.category_plan,
                review_directory=args.review_directory,
                database_directory=args.database_directory,
                references_directory=args.references_directory,
            )
            print(
                "Prepared dataset cycle:\n"
                f"  Golden records: {result['golden_records']}\n"
                f"  Planned range: {result['range']}\n"
                f"  Review file: {result['review_file']}\n"
                f"  Instructions: {result['instructions_file']}"
            )
        elif args.command == "review":
            candidate, json_path, md_path, report = review_cycle(
                args.review_file,
                args.golden,
                args.database_directory,
                args.references_directory,
                replace_results=args.replace_results,
            )
            print(f"Candidate: {candidate}\nReports: {json_path}\n         {md_path}")
            print(f"Summary: {report['summary']}")
        elif args.command == "repair":
            print(f"Repair instructions: {repair_cycle(args.review_file)}")
        else:
            result = approve_cycle(
                args.review_file,
                reviewer=args.reviewer,
                golden_patterns=args.golden,
                commit=args.commit,
                push=args.push,
                database_directory=args.database_directory,
                references_directory=args.references_directory,
            )
            print("Approval cancelled." if result["cancelled"] else f"Approved: {result}")
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
