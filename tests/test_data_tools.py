import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from nono_lora.data import (
    database_record,
    read_jsonl,
    training_record,
    write_jsonl,
)
from nono_lora.training import (
    to_prompt_completion,
    training_data_files,
    validate_precision_settings,
)
from scripts.merge_jsonl import merge_records
from scripts.split_jsonl import split_records
from scripts.validate_jsonl import analyze, validate


def write_lines(path: Path, records) -> None:
    write_jsonl(path, records)


def record(
    record_id: str,
    user: str = "こんにちは",
    assistant: str = "へぇ〜？",
    **extra,
):
    result = {
        "id": record_id,
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
    }
    result.update(extra)
    return result


class FakeChatTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        text = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )
        if add_generation_prompt:
            text += "<assistant>"
        return text


class ValidationTests(unittest.TestCase):
    def test_normal_data(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "valid.jsonl"
            write_lines(
                source,
                [
                    record("000001"),
                    record("000002", user="こんばんは", assistant="おそ〜い♪"),
                ],
            )
            report = analyze([source], 1, 2)
            self.assertEqual(report.errors, [])
            self.assertEqual(report.warnings, [])

    def test_exact_user_and_assistant_duplicates_are_reported(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "duplicates.jsonl"
            write_lines(
                source,
                [
                    record("000001", "同じ質問", "同じ返事"),
                    record("000002", "同じ質問", "同じ返事"),
                ],
            )
            report = analyze([source], 1, 2)
            self.assertEqual(report.errors, [])
            self.assertTrue(
                any("duplicate user content" in item for item in report.warnings)
            )
            self.assertTrue(
                any("duplicate assistant content" in item for item in report.warnings)
            )

    def test_missing_id_is_an_error(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "missing.jsonl"
            write_lines(source, [record("000001"), record("000003")])
            errors = validate([source], 1, 3)
            self.assertTrue(any("missing ids: 000002" in item for item in errors))

    def test_invalid_messages_are_errors(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "invalid.jsonl"
            invalid = [
                {"id": "000001", "messages": []},
                {
                    "id": "000002",
                    "messages": [
                        {"role": "assistant", "content": "先行"},
                        {"role": "user", "content": "逆順"},
                    ],
                },
                record("000003", user=" "),
                record("000004", assistant="\t"),
                record("000005", user=" 先頭空白"),
                record("000006", assistant="末尾空白 "),
            ]
            write_lines(source, invalid)
            errors = validate([source], 1, 6)
            self.assertTrue(any("non-empty array" in item for item in errors))
            self.assertTrue(any("role order" in item for item in errors))
            self.assertGreaterEqual(
                sum("content must be a non-empty string" in item for item in errors), 2
            )
            self.assertGreaterEqual(
                sum("leading or trailing whitespace" in item for item in errors), 2
            )

    def test_invalid_json_line_is_an_error(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "invalid-json.jsonl"
            source.write_text('{"id":\n', encoding="utf-8")
            errors = validate([source], None, None)
            self.assertTrue(any("invalid JSON" in item for item in errors))

    def test_assistant_only_greeting_is_valid(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "greeting.jsonl"
            write_lines(
                source,
                [
                    {
                        "id": "000001",
                        "messages": [{"role": "assistant", "content": "おかえり♪"}],
                    }
                ],
            )
            self.assertEqual(validate([source], None, None), [])


class MergeTests(unittest.TestCase):
    def test_duplicate_id_is_an_error_by_default(self):
        records = [record("000001"), record("000001")]
        with self.assertRaisesRegex(ValueError, "duplicate id 000001"):
            merge_records(records, deduplicate_identical=False)

    def test_identical_duplicate_is_merged_and_metadata_is_preserved(self):
        first = record("000001", category="daily", source="first")
        second = record("000001", tags=["greeting"], source="second")
        result = merge_records([first, second], deduplicate_identical=True)
        self.assertEqual(result.deduplicated_ids, ["000001"])
        self.assertEqual(len(result.records), 1)
        merged = result.records[0]
        self.assertEqual(merged["category"], "daily")
        self.assertEqual(merged["tags"], ["greeting"])
        self.assertEqual(
            merged["_deduplication_conflicts"]["source"], ["first", "second"]
        )

    def test_same_id_with_different_user_or_assistant_is_always_an_error(self):
        for conflicting in (
            record("000001", user="別の質問"),
            record("000001", assistant="別の返事"),
        ):
            with self.subTest(conflicting=conflicting):
                with self.assertRaisesRegex(ValueError, "content differs"):
                    merge_records(
                        [record("000001"), conflicting],
                        deduplicate_identical=True,
                    )

    def test_training_and_database_output_fields(self):
        source = record("000001", custom={"nested": ["保持"]})
        self.assertEqual(set(training_record(source)), {"id", "messages"})
        database = database_record(source)
        self.assertEqual(database["custom"], {"nested": ["保持"]})
        self.assertEqual(database["status"], "unreviewed")
        self.assertEqual(database["language"], "ja")


class EncodingAndSplitTests(unittest.TestCase):
    def test_utf8_special_characters_and_newlines_round_trip(self):
        special = "へぇ〜？♡♪（笑）\n次の行も保持"
        source_record = record("000001", assistant=special, unknown="未知♡")
        with TemporaryDirectory() as directory:
            source = Path(directory) / "utf8.jsonl"
            write_lines(source, [source_record])
            loaded = list(read_jsonl([source]))[0].record
            self.assertEqual(loaded, source_record)
            raw = source.read_text(encoding="utf-8")
            self.assertIn("♡", raw)
            self.assertIn("♪", raw)
            self.assertIn("〜", raw)
            self.assertIn("（笑）", raw)
            self.assertIn("\\n", raw)

    def test_three_way_split_is_reproducible_and_200_is_180_10_10(self):
        records = [record(f"{value:06d}") for value in range(1, 201)]
        first = split_records(records, seed=123)
        second = split_records(records, seed=123)
        self.assertEqual(
            [[item["id"] for item in split] for split in first],
            [[item["id"] for item in split] for split in second],
        )
        self.assertEqual([len(split) for split in first], [180, 10, 10])
        all_ids = {item["id"] for split in first for item in split}
        self.assertEqual(all_ids, {item["id"] for item in records})

    def test_chat_template_produces_assistant_only_completion(self):
        messages = record("000001")["messages"]
        formatted = to_prompt_completion(messages, FakeChatTokenizer())
        self.assertEqual(formatted["prompt"], "<user>こんにちは</user><assistant>")
        self.assertEqual(formatted["completion"], "へぇ〜？</assistant>")

    def test_training_data_files_excludes_held_out_test(self):
        files = training_data_files(
            {
                "train_file": "train.jsonl",
                "validation_file": "validation.jsonl",
                "test_file": "test.jsonl",
            }
        )
        self.assertEqual(
            files,
            {"train": "train.jsonl", "validation": "validation.jsonl"},
        )
        self.assertNotIn("test", files)

    def test_bf16_and_fp16_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "cannot both be true"):
            validate_precision_settings(
                bf16=True,
                fp16=True,
                cuda_available=True,
                bf16_supported=True,
            )

    def test_bf16_requires_supported_cuda_device(self):
        for cuda_available, bf16_supported in ((False, False), (True, False)):
            with self.subTest(
                cuda_available=cuda_available,
                bf16_supported=bf16_supported,
            ):
                with self.assertRaisesRegex(RuntimeError, "does not support BF16"):
                    validate_precision_settings(
                        bf16=True,
                        fp16=False,
                        cuda_available=cuda_available,
                        bf16_supported=bf16_supported,
                    )

    def test_supported_bf16_configuration_is_valid(self):
        validate_precision_settings(
            bf16=True,
            fp16=False,
            cuda_available=True,
            bf16_supported=True,
        )


if __name__ == "__main__":
    unittest.main()
