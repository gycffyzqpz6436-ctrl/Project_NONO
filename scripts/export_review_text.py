from __future__ import annotations

import argparse
from pathlib import Path

from nono_lora.data import read_jsonl
from nono_lora.dataset_local import render_review_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export candidate JSONL to editable TXT.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--include-plan",
        action="store_true",
        help="Include category and planning fields for lossless local re-import.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = [item.record for item in read_jsonl([args.input])]
    if len(records) != 50:
        raise SystemExit(f"candidate JSONL must contain exactly 50 records; got {len(records)}")
    output = args.output or args.input.with_name(args.input.stem + "_review.txt")
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.write_text(
        render_review_text(records, include_plan=args.include_plan),
        encoding="utf-8",
        newline="\n",
    )
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
