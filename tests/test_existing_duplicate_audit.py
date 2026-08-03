import unittest
from pathlib import Path

from nono_lora.data import LocatedRecord
from scripts.audit_existing_duplicates import (
    ALLOWED_ACTIONS,
    audit_records,
    render_markdown,
    render_repairs,
)


def record(record_id: str, user: str, assistant: str = "ぷっ♡\n\n確認すればいい〜\n\n単純だね♡"):
    return {
        "id": record_id,
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
    }


class ExistingDuplicateAuditTests(unittest.TestCase):
    def test_reports_identical_and_conflicting_same_ids_without_mutation(self):
        first = record("000001", "同じ質問")
        identical = record("000001", "同じ質問")
        conflict = record("000002", "質問A")
        conflict_other = record("000002", "質問B")
        original = [dict(first), dict(identical), dict(conflict), dict(conflict_other)]
        located = [
            LocatedRecord(value, Path(f"file{index}.jsonl"), 1)
            for index, value in enumerate((first, identical, conflict, conflict_other))
        ]
        report = audit_records(located, [], [])
        detections = {item["detection"] for item in report["findings"]}
        self.assertIn("identical_id_duplicate", detections)
        self.assertIn("conflicting_id_duplicate", detections)
        self.assertEqual((first, identical, conflict, conflict_other), tuple(original))

    def test_reports_exact_normalized_semantic_and_reference_reuse(self):
        located = [
            LocatedRecord(record("000001", "ただいま。"), Path("a.jsonl"), 1),
            LocatedRecord(record("000002", "ただいま。"), Path("a.jsonl"), 2),
            LocatedRecord(record("000003", "ありがとう。"), Path("a.jsonl"), 3),
            LocatedRecord(record("000004", "ありがとう"), Path("a.jsonl"), 4),
            LocatedRecord(
                record("000005", "ゲームで大事な装備を間違えて売った"),
                Path("a.jsonl"),
                5,
            ),
            LocatedRecord(
                record("000006", "貴重なゲーム素材を誤って売却した"),
                Path("a.jsonl"),
                6,
            ),
        ]
        references = [
            record(
                "reference:samples#1",
                "大切なゲームアイテムをうっかり売った",
            )
        ]
        report = audit_records(located, [], references)
        detections = {item["detection"] for item in report["findings"]}
        self.assertIn("exact_user_duplicate", detections)
        self.assertIn("normalized_user_duplicate", detections)
        self.assertIn("semantic_duplicate", detections)
        self.assertIn("reference_reuse", detections)

    def test_recommendations_and_human_review_outputs(self):
        located = [
            LocatedRecord(record("000001", "同じ質問"), Path("a.jsonl"), 1),
            LocatedRecord(record("000002", "同じ質問"), Path("a.jsonl"), 2),
        ]
        report = audit_records(located, [], [])
        self.assertTrue(report["findings"])
        self.assertTrue(
            all(
                item["recommendation"] in ALLOWED_ACTIONS
                for item in report["findings"]
            )
        )
        markdown = render_markdown(report)
        repairs = render_repairs(report)
        self.assertIn("No Golden or database record was modified", markdown)
        self.assertIn("修正前 A", repairs)
        self.assertIn("Human decision", repairs)


if __name__ == "__main__":
    unittest.main()
