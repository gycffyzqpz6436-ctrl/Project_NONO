from __future__ import annotations

import argparse
import json
from pathlib import Path

from nono_lora.dataset_local import (
    analyze_records,
    load_jsonl_patterns,
    render_analysis_markdown,
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _, records = load_jsonl_patterns(args.inputs)
    report = analyze_records(records)
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
    print(f"Wrote {args.json_output} and {args.markdown_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
