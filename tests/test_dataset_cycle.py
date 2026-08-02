import json
import os
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from nono_lora.data import write_jsonl
from nono_lora.dataset_local import (
    collapse_identical_id_duplicates,
    dataset_state,
    render_review_text,
)
from scripts.dataset_cycle import approve_cycle, prepare_cycle, repair_cycle


def record(number: int, *, user: str | None = None, assistant: str | None = None):
    return {
        "id": f"{number:06d}",
        "status": "draft",
        "category": "daily_life",
        "conversation_type": f"pattern_{number}",
        "follow_up_target": False,
        "scenario": f"未使用場面{number}",
        "planning": {"used_topics_to_avoid": ["既存話題"]},
        "messages": [
            {"role": "user", "content": user or f"固有の相談内容{number}番"},
            {
                "role": "assistant",
                "content": assistant
                or f"また先回りして悩んでたでしょ♡\n\nぷっ♡ でも一つずつ試せばいい〜♪\n\nできたら褒めてあげる、かわい〜♡{number}",
            },
        ],
    }


class WorkingDirectory:
    def __init__(self, path: Path):
        self.path = path
        self.previous = Path.cwd()

    def __enter__(self):
        os.chdir(self.path)

    def __exit__(self, *_):
        os.chdir(self.previous)


class DatasetCycleTests(unittest.TestCase):
    def _workspace(self, root: Path):
        (root / "dataset/jsonl").mkdir(parents=True)
        (root / "dataset").mkdir(exist_ok=True)
        write_jsonl(root / "dataset/jsonl/golden.jsonl", (record(i) for i in range(1, 301)))
        (root / "dataset/category_plan.yaml").write_text(
            "categories:\n"
            "  daily_life:\n"
            "    target_ratio: 1.0\n"
            "    enabled: true\n"
            "    notes: 未使用の日常場面\n",
            encoding="utf-8",
        )

    def test_prepare_creates_301_to_350_and_instructions(self):
        with TemporaryDirectory() as directory, WorkingDirectory(Path(directory)):
            root = Path(directory)
            self._workspace(root)
            result = prepare_cycle(
                count=50,
                golden_patterns=[Path("dataset/jsonl/*.jsonl")],
                category_plan=Path("dataset/category_plan.yaml"),
                review_directory=Path("dataset/candidates/review"),
                batch_id="20260802-120000",
            )
            self.assertEqual(result["range"], "000301-000350")
            self.assertEqual(
                result["review_file"].read_text(encoding="utf-8").count("\n#"), 49
            )
            instructions = result["instructions_file"].read_text(encoding="utf-8")
            self.assertIn("User:", instructions)
            self.assertIn("JSONL化、承認、commit、pushを行わない", instructions)
            state = json.loads(Path("dataset/state/dataset_state.json").read_text("utf-8"))
            self.assertEqual(state["golden_record_count"], 300)
            self.assertEqual(state["next_id"], "000301")

    def test_prepare_rejects_partial_batch(self):
        with TemporaryDirectory() as directory, WorkingDirectory(Path(directory)):
            root = Path(directory)
            self._workspace(root)
            with self.assertRaisesRegex(ValueError, "exactly 50"):
                prepare_cycle(
                    count=49,
                    golden_patterns=[Path("dataset/jsonl/*.jsonl")],
                    category_plan=Path("dataset/category_plan.yaml"),
                    review_directory=Path("dataset/candidates/review"),
                )

    def test_identical_000150_is_collapsed_but_conflict_stops(self):
        records = [record(i) for i in range(1, 301)] + [record(150)]
        unique, ids = collapse_identical_id_duplicates(records)
        self.assertEqual(len(unique), 300)
        self.assertEqual(ids, ["000150"])
        conflicting = records + [record(150, user="異なる会話")]
        with self.assertRaisesRegex(ValueError, "conflicting duplicate"):
            collapse_identical_id_duplicates(conflicting)

    def test_repair_contains_only_warning_and_reject(self):
        with TemporaryDirectory() as directory, WorkingDirectory(Path(directory)):
            root = Path(directory)
            review = root / "dataset/candidates/review/nono_draft_000301_000350_20260802-120000.txt"
            review.parent.mkdir(parents=True)
            records = [record(i) for i in range(301, 351)]
            review.write_text(render_review_text(records, include_plan=True), encoding="utf-8")
            report_path = root / "dataset/reports/review_000301_000350_20260802-120000.json"
            report_path.parent.mkdir(parents=True)
            report_path.write_text(
                json.dumps(
                    {
                        "records": [
                            {
                                "id": "000301", "result": "warning",
                                "suggestions": ["見透かしを追加"], "similar": [{"id": "000010"}],
                                "opening": "ぷっ♡", "ending_duplicate": True,
                            },
                            {
                                "id": "000302", "result": "pass",
                                "suggestions": [], "similar": [], "opening": "",
                                "ending_duplicate": False,
                            },
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            output = repair_cycle(review)
            text = output.read_text(encoding="utf-8")
            self.assertIn("#000301", text)
            self.assertNotIn("#000302", text)

    def test_approve_requires_exact_confirmation(self):
        with TemporaryDirectory() as directory, WorkingDirectory(Path(directory)):
            root = Path(directory)
            self._workspace(root)
            review = root / "review.txt"
            review.write_text(
                render_review_text([record(i) for i in range(301, 351)], include_plan=True),
                encoding="utf-8",
            )
            result = approve_cycle(
                review,
                reviewer="yuto",
                golden_patterns=[Path("dataset/jsonl/*.jsonl")],
                commit=False,
                push=False,
                input_fn=lambda _: "yes",
            )
            self.assertTrue(result["cancelled"])
            self.assertFalse(Path("dataset/jsonl/nono_dataset_000301_000350.jsonl").exists())

    def test_approve_outputs_and_git_scope_and_push_failure_keeps_files(self):
        with TemporaryDirectory() as directory, WorkingDirectory(Path(directory)):
            root = Path(directory)
            self._workspace(root)
            review = root / "review.txt"
            review.write_text(
                render_review_text([record(i) for i in range(301, 351)], include_plan=True),
                encoding="utf-8",
            )
            calls = []

            def runner(args, **kwargs):
                calls.append(args)
                if args[:2] == ["git", "push"]:
                    raise subprocess.CalledProcessError(1, args)
                return subprocess.CompletedProcess(args, 0, "", "")

            clean = {
                "summary": {"pass": 50, "warning": 0, "reject": 0},
                "records": [],
            }
            candidates = [record(i) for i in range(301, 351)]
            with patch(
                "scripts.dataset_cycle.import_records",
                return_value=(candidates, []),
            ), patch(
                "scripts.dataset_cycle.batch_report", return_value=clean
            ), patch(
                "scripts.dataset_cycle.approve", return_value=candidates
            ):
                with self.assertRaisesRegex(RuntimeError, "outputs and commit were kept"):
                    approve_cycle(
                        review,
                        reviewer="yuto",
                        golden_patterns=[Path("dataset/jsonl/*.jsonl")],
                        commit=True,
                        push=True,
                        input_fn=lambda _: "APPROVE",
                        runner=runner,
                    )
            db = Path("dataset/database/nono_database_000301_000350.jsonl")
            training = Path("dataset/jsonl/nono_dataset_000301_000350.jsonl")
            self.assertTrue(db.exists())
            self.assertTrue(training.exists())
            add = next(item for item in calls if item[:2] == ["git", "add"])
            self.assertEqual(
                {Path(item).as_posix() for item in add[3:]},
                {db.as_posix(), training.as_posix(), "dataset/state/dataset_state.json"},
            )
            state = dataset_state([record(i) for i in range(1, 351)])
            self.assertEqual(state["next_range"], "000351-000400")


if __name__ == "__main__":
    unittest.main()
