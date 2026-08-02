from __future__ import annotations

"""Backward-compatible local inspection command.

API-based generation was removed in schema version 2.1. Use
``scripts.create_dataset_draft`` to create a writable 50-record template.
"""

import argparse
from pathlib import Path

from nono_lora.dataset_local import dry_run_report, resolve_patterns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect local NONO dataset inputs without network access."
    )
    parser.add_argument(
        "--golden", nargs="+", type=Path, default=[Path("dataset/jsonl/*.jsonl")]
    )
    parser.add_argument(
        "--existing", nargs="+", type=Path, default=[Path("dataset/jsonl/*.jsonl")]
    )
    parser.add_argument(
        "--references",
        nargs="+",
        type=Path,
        default=[
            Path("docs/character/*.md"),
            Path("docs/implementation/examples.md"),
        ],
    )
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.dry_run:
        raise SystemExit(
            "API generation was removed. Use `python -m scripts.create_dataset_draft "
            "--count 50`, or add --dry-run to inspect inputs."
        )
    if args.count != 50:
        raise SystemExit("--count must be 50")
    try:
        lines = dry_run_report(
            args.golden,
            args.existing,
            args.references,
            count=args.count,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Dry run failed: {exc}") from exc
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
