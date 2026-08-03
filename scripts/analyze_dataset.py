from __future__ import annotations

import argparse
import json
from pathlib import Path

from nono_lora.dataset_local import (
    analyze_records,
    collapse_identical_id_duplicates,
    dataset_state,
    load_jsonl_patterns,
    render_analysis_markdown,
    write_json_atomic,
)
from nono_lora.dataset_semantic import (
    merge_database_metadata,
    read_database_files,
    read_reference_files,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze local NONO Golden JSONL.")
    parser.add_argument(
        "inputs", nargs="*", type=Path, default=[Path("dataset/jsonl/*.jsonl")]
    )
    parser.add_argument(
        "--json-output", type=Path, default=Path("dataset/reports/dataset_analysis.json")
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("dataset/reports/dataset_analysis.md"),
    )
    parser.add_argument(
        "--state-output", type=Path, default=Path("dataset/state/dataset_state.json")
    )
    parser.add_argument(
        "--database-directory", type=Path, default=Path("dataset/database")
    )
    parser.add_argument(
        "--references-directory", type=Path, default=Path("references")
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    golden_paths, raw_records = load_jsonl_patterns(args.inputs)
    records, _collapsed = collapse_identical_id_duplicates(raw_records)
    database_paths, database = read_database_files(args.database_directory)
    reference_paths, references = read_reference_files(args.references_directory)
    records = merge_database_metadata(records, database)
    report = analyze_records(records)
    report["sources"] = {
        "golden_files": [path.as_posix() for path in golden_paths],
        "database_files": [path.as_posix() for path in database_paths],
        "reference_files": [path.as_posix() for path in reference_paths],
        "reference_records": len(references),
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    args.markdown_output.write_text(
        render_analysis_markdown(report), encoding="utf-8", newline="\n"
    )
    write_json_atomic(
        args.state_output,
        dataset_state(records, last_analysis_report=args.json_output.as_posix()),
    )
    print(f"Wrote {args.json_output} and {args.markdown_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
