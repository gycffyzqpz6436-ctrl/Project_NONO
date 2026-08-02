import builtins
import importlib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from nono_lora.data import database_record, write_jsonl
from nono_lora.dataset_local import (
    analyze_records,
    dry_run_report,
    find_similar,
    normalized_for_similarity,
    parse_review_text,
    render_review_text,
)
from nono_lora.dataset_pipeline import extract_dialogue, next_id
from scripts.approve_candidates import approve
from scripts.create_dataset_draft import make_draft_records
from scripts.import_review_text import import_records, validate_import


def record(record_id: int, user: str | None = None, assistant: str | None = None,
           **metadata):
    value = {
        "schema_version": "2.1.0",
        "id": f"{record_id:06d}",
        "status": "draft",
        "category": metadata.pop("category", "daily_life"),
        "messages": [
            {"role": "user", "content": user or f"固有の話題その{record_id}"},
            {
                "role": "assistant",
                "content": assistant
                or f"へぇ〜♪\n\nその{record_id}番のこと、気にしてたでしょ？"
                f"\n\n少しずつ試せばいいよ。\n\nかわい〜♪",
            },
        ],
    }
    value.update(metadata)
    return value


def fifty_review_records(start: int = 301):
    return [record(start + index) for index in range(50)]


class LocalPipelineTests(unittest.TestCase):
    def test_modules_import_without_openai_installed_or_api_key(self):
        real_import = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name == "openai" or name.startswith("openai."):
                raise AssertionError("OpenAI SDK must not be imported")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded):
            for name in (
                "scripts.generate_dataset",
                "scripts.analyze_dataset",
                "scripts.create_dataset_draft",
                "scripts.import_review_text",
                "scripts.review_candidates",
                "scripts.export_review_text",
                "scripts.approve_candidates",
            ):
                importlib.reload(importlib.import_module(name))

    def test_dry_run_is_local_and_reports_300_to_350(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            golden = root / "golden"
            references = root / "references"
            golden.mkdir()
            references.mkdir()
            for start in (1, 101, 201):
                write_jsonl(
                    golden / f"{start:06d}.jsonl",
                    (record(index) for index in range(start, start + 100)),
                )
            reference = references / "character.md"
            reference.write_text("# Character", encoding="utf-8")
            with patch("urllib.request.urlopen", side_effect=AssertionError("network")):
                lines = dry_run_report(
                    [golden / "*.jsonl"],
                    [golden / "*.jsonl"],
                    [references / "*.md"],
                    count=50,
                )
            self.assertEqual(
                lines[:4],
                [
                    "Golden records: 300",
                    "Maximum ID: 000300",
                    "Next ID: 000301",
                    "Planned range: 000301-000350",
                ],
            )

    def test_next_id_after_000300(self):
        self.assertEqual(next_id([{"id": f"{i:06d}"} for i in range(1, 301)]), 301)

    def test_create_exactly_fifty_template_slots(self):
        golden = [record(index) for index in range(1, 301)]
        plan = {
            "school": {"target_ratio": 0.7, "notes": "学校内の未使用の日常場面"},
            "daily_life": {"target_ratio": 0.3, "notes": "未使用の日常場面"},
        }
        draft = make_draft_records(golden, plan, count=50)
        self.assertEqual(len(draft), 50)
        self.assertEqual((draft[0]["id"], draft[-1]["id"]), ("000301", "000350"))
        self.assertEqual(extract_dialogue(draft[0]), ("", ""))
        with self.assertRaisesRegex(ValueError, "exactly 50"):
            make_draft_records(golden, plan, count=49)

    def test_import_rejects_49_and_51_records(self):
        for size in (49, 51):
            with self.subTest(size=size):
                errors = validate_import(
                    [record(301 + index) for index in range(size)],
                    [],
                    source=Path("review.txt"),
                )
                self.assertTrue(any("exactly 50" in error for error in errors))

    def test_exact_and_normalized_duplicate_rejected(self):
        golden = [record(1, user="ＡＩって、すごい！")]
        exact = fifty_review_records()
        exact[0]["messages"][0]["content"] = "ＡＩって、すごい！"
        self.assertTrue(any("exact user duplicate" in item for item in
                            validate_import(exact, golden, source=Path("x.txt"))))
        normalized = fifty_review_records()
        normalized[0]["messages"][0]["content"] = "aiってすごい"
        self.assertEqual(
            normalized_for_similarity("ＡＩって、すごい！"),
            normalized_for_similarity("aiってすごい"),
        )
        self.assertTrue(any("normalized user duplicate" in item for item in
                            validate_import(normalized, golden, source=Path("x.txt"))))

    def test_similar_sentence_warning_contains_id_score_and_reason(self):
        golden = [record(10, user="明日のテスト勉強が全然終わらない")]
        candidate = record(301, user="明日のテスト勉強がぜんぜん終わってない")
        hits = find_similar(candidate, golden)
        self.assertTrue(hits)
        self.assertEqual(hits[0].record_id, "000010")
        self.assertGreaterEqual(hits[0].score, 0.72)
        self.assertTrue(hits[0].reasons)

    def test_id_collision_blocks_approval(self):
        with self.assertRaisesRegex(ValueError, "id collision"):
            approve(
                fifty_review_records(),
                "reviewer",
                set(),
                [record(301, user="別の既存会話")],
                accept_similarity_warnings=True,
            )

    def test_invalid_review_format(self):
        with self.assertRaisesRegex(ValueError, "no #6-digit"):
            parse_review_text("User:\n本文\n\nNONO:\n回答")

    def test_export_import_round_trip(self):
        original = fifty_review_records()
        text = render_review_text(original, include_plan=True)
        imported, _ = import_records(text, [], source=Path("review.txt"))
        self.assertEqual(
            [(item["id"], extract_dialogue(item)) for item in imported],
            [(item["id"], extract_dialogue(item)) for item in original],
        )
        self.assertTrue(all(item["status"] == "draft" for item in imported))

    def test_question_rate_and_expression_bias(self):
        records = [
            record(
                index,
                assistant=(
                    "ぷぷっ♪\n\nまた隠してたでしょ？\n\n確認してみなよ。\n\nざぁこ♪"
                    if index <= 10
                    else "反応\n\n見えてるよ。\n\n少し休もう。"
                ),
            )
            for index in range(1, 21)
        ]
        report = analyze_records(records)
        self.assertAlmostEqual(report["follow_up_rate"], 0.0)
        self.assertEqual(report["openings"]["ぷぷっ♪"], 10)
        self.assertEqual(report["endings"]["ざぁこ"], 10)
        self.assertTrue(any("openings" in item for item in report["recent_50_bias_warnings"]))
        question_records = [
            record(index, assistant="へぇ〜♪\n\n最後はどうする？")
            for index in range(1, 11)
        ]
        self.assertEqual(analyze_records(question_records)["follow_up_rate"], 1.0)

    def test_unknown_database_metadata_is_preserved(self):
        source = record(301, custom={"nested": ["keep"]}, local_similarity_warnings=[])
        converted = database_record(source)
        self.assertEqual(converted["custom"], {"nested": ["keep"]})
        self.assertIn("local_similarity_warnings", converted)


if __name__ == "__main__":
    unittest.main()
