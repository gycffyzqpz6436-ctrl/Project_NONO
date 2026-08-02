from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any

from nono_lora.data import read_jsonl, sort_key, write_jsonl
from scripts.validate_jsonl import validate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproducibly split NONO JSONL into train/validation/test."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--train-output", type=Path, default=Path("data/processed/train.jsonl")
    )
    parser.add_argument(
        "--validation-output",
        type=Path,
        default=Path("data/processed/validation.jsonl"),
    )
    parser.add_argument(
        "--test-output", type=Path, default=Path("data/processed/test.jsonl")
    )
    parser.add_argument("--train-ratio", type=float, default=0.90)
    parser.add_argument("--validation-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def split_counts(size: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    if size < 3:
        raise ValueError("at least three records are required")
    if any(ratio <= 0.0 for ratio in ratios):
        raise ValueError("all split ratios must be greater than zero")
    if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("train, validation, and test ratios must sum to 1.0")
    exact = [size * ratio for ratio in ratios]
    counts = [math.floor(value) for value in exact]
    remainder = size - sum(counts)
    order = sorted(
        range(3), key=lambda index: (exact[index] - counts[index], -index), reverse=True
    )
    for index in order[:remainder]:
        counts[index] += 1
    if any(count == 0 for count in counts):
        raise ValueError("dataset is too small for the requested non-empty splits")
    return counts[0], counts[1], counts[2]


def split_records(
    records: list[dict[str, Any]],
    train_ratio: float = 0.90,
    validation_ratio: float = 0.05,
    test_ratio: float = 0.05,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    train_count, validation_count, test_count = split_counts(
        len(records), (train_ratio, validation_ratio, test_ratio)
    )
    positions = list(range(len(records)))
    random.Random(seed).shuffle(positions)
    validation_positions = set(positions[:validation_count])
    test_positions = set(
        positions[validation_count : validation_count + test_count]
    )
    train = [
        record
        for index, record in enumerate(records)
        if index not in validation_positions and index not in test_positions
    ]
    validation = [
        record for index, record in enumerate(records) if index in validation_positions
    ]
    test = [record for index, record in enumerate(records) if index in test_positions]
    if len(train) != train_count:
        raise AssertionError("internal split count mismatch")
    return (
        sorted(train, key=sort_key),
        sorted(validation, key=sort_key),
        sorted(test, key=sort_key),
    )


def main() -> int:
    args = parse_args()
    source = args.input.resolve()
    output_paths = [
        args.train_output.resolve(),
        args.validation_output.resolve(),
        args.test_output.resolve(),
    ]
    if len(set(output_paths)) != 3 or source in output_paths:
        raise SystemExit("Input and all three output paths must be different files.")
    errors = validate([source], None, None)
    if errors:
        raise SystemExit(
            "Input is invalid; split aborted:\n" + "\n".join(f"- {e}" for e in errors)
        )
    records = [item.record for item in read_jsonl([source])]
    train, validation, test = split_records(
        records,
        args.train_ratio,
        args.validation_ratio,
        args.test_ratio,
        args.seed,
    )
    train_count = write_jsonl(output_paths[0], train)
    validation_count = write_jsonl(output_paths[1], validation)
    test_count = write_jsonl(output_paths[2], test)
    print(
        f"Wrote train={train_count}, validation={validation_count}, "
        f"test={test_count} (seed={args.seed})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
