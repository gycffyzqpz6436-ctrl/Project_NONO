import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from nono_lora.dataset_semantic import (
    batch_style_warnings,
    find_semantic_duplicates,
    merge_database_metadata,
    read_reference_files,
    style_features,
)
from scripts.review_candidates import review_records
from scripts.approve_candidates import approve


def record(
    record_id: str,
    user: str,
    assistant: str = (
        "ぷっ♡　また余裕ぶって確認を飛ばしたでしょ♪\n\n"
        "一度一覧を見直せば答えは出る〜\n\n"
        "焦って自分で罠へ入るの、ほんと単純でかわい〜♡"
    ),
    **metadata,
):
    value = {
        "id": record_id,
        "status": "draft",
        "category": metadata.pop("category", "gaming"),
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
    }
    value.update(metadata)
    return value


class SemanticDuplicateTests(unittest.TestCase):
    def test_same_event_with_reworded_game_item_is_rejected(self):
        golden = [record("000372", "ゲームで大事な装備を間違えて売っちゃった")]
        candidate = record("000529", "貴重そうな素材をうっかり売却した")
        hits = find_semantic_duplicates(candidate, golden)
        self.assertTrue(hits)
        self.assertEqual(hits[0].source_id, "000372")
        self.assertTrue(any("same event/situation" in reason for reason in hits[0].reasons))
        with self.assertRaisesRegex(ValueError, "semantic/reference duplicate"):
            approve(
                [candidate],
                "reviewer",
                set(),
                golden,
                enforce_character_quality=False,
            )

    def test_same_clay_crack_situation_is_rejected(self):
        golden = [record("000322", "粘土で作ったもの、乾かしたらひび割れた")]
        candidate = record("000547", "乾燥させた陶土に細い亀裂が入った")
        self.assertTrue(find_semantic_duplicates(candidate, golden))

    def test_reference_paraphrase_is_reported_with_reference_source(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "samples.txt"
            source.write_text(
                "User:\nゲームの大事な装備を間違えて売った\n\n"
                "Assistant:\n確認して買い戻す\n",
                encoding="utf-8",
            )
            _, references = read_reference_files(root)
            candidate = record("000600", "貴重なゲーム素材を誤って売却した")
            hits = find_semantic_duplicates(candidate, [], references)
            self.assertTrue(hits)
            self.assertEqual(hits[0].source_kind, "reference")
            self.assertIn("samples.txt", hits[0].source_id)

    def test_same_ending_question_flow_is_detected(self):
        ending = "で、次は何から確認してみる〜？♡"
        golden = [
            record(
                "000100",
                "学校の棚を整理したい",
                f"ぷっ♡　また後回しにしたでしょ♪\n\n種類で分ければいい〜\n\n{ending}",
            )
        ]
        candidate = record(
            "000600",
            "台所の引き出しを片付けたい",
            f"あは♪　全部押し込んでたでしょ♡\n\n用途で分ければいい〜\n\n{ending}",
        )
        hits = find_semantic_duplicates(candidate, golden)
        self.assertTrue(hits)
        self.assertTrue(
            any("same ending/question flow" in reason for reason in hits[0].reasons)
        )

    def test_database_metadata_is_joined_by_id(self):
        golden = [record("000001", "質問")]
        database = [
            {
                "id": "000001",
                "category": "school",
                "scenario": "図書室での小さな失敗",
                "conversation_type": "small_failure",
            }
        ]
        merged = merge_database_metadata(golden, database)
        self.assertEqual(merged[0]["category"], "school")
        self.assertEqual(merged[0]["scenario"], "図書室での小さな失敗")


class CharacterQualityTests(unittest.TestCase):
    def test_kind_assistant_template_is_flagged(self):
        gentle = record(
            "000001",
            "疲れた",
            "今日はよく頑張ったね。\n\n無理しないでゆっくり休んでね。",
        )
        quality = style_features(gentle)
        self.assertTrue(quality["soft_ai"])
        self.assertLess(quality["mesugaki_strength"], 5.5)
        report = review_records([gentle], [], [])[0]
        self.assertEqual(report["result"], "reject")

    def test_strong_nono_has_mind_reading_answer_and_teasing(self):
        quality = style_features(record("000001", "困った"))
        self.assertTrue(quality["mind_reading"])
        self.assertTrue(quality["answer_or_empathy"])
        self.assertTrue(quality["ending_or_follow_up"])
        self.assertGreaterEqual(quality["mesugaki_strength"], 7.0)

    def test_batch_warns_about_follow_up_and_repeated_style(self):
        records = [record(f"{index:06d}", f"固有話題{index}") for index in range(1, 6)]
        warnings = batch_style_warnings(records)
        self.assertTrue(any("follow-up rate" in item for item in warnings))
        self.assertTrue(any("opening" in item or "ending" in item for item in warnings))

    def test_review_exposes_required_semantic_and_character_fields(self):
        golden = [record("000001", "ゲームで大事な装備を間違えて売っちゃった")]
        candidate = record("000501", "貴重なゲーム素材を誤って売却した")
        report = review_records([candidate], golden, [])[0]
        self.assertEqual(report["result"], "reject")
        for key in (
            "similar_golden_ids",
            "similar_references",
            "semantic_reasons",
            "nono_score",
            "mesugaki_strength",
            "teasing_ratio",
            "soft_ai",
            "mind_reading",
            "answer_or_empathy",
            "ending_or_follow_up",
            "suggestions",
        ):
            self.assertIn(key, report)


if __name__ == "__main__":
    unittest.main()
