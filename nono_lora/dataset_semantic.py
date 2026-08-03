from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from nono_lora.data import read_jsonl
from nono_lora.dataset_local import (
    ending_key,
    fuzzy_similarity,
    ngram_similarity,
    normalized_for_similarity,
    opening_key,
)
from nono_lora.dataset_pipeline import extract_dialogue

CONCEPTS: dict[str, tuple[str, ...]] = {
    "place_school": ("学校", "教室", "校内", "体育館", "図書室", "理科室", "美術室", "音楽室"),
    "place_home": ("家", "部屋", "玄関", "台所", "冷蔵庫", "浴室", "洗濯"),
    "place_game": ("ゲーム", "クエスト", "ボス", "ランク", "コントローラー", "素材", "装備"),
    "object_paper": ("紙", "プリント", "ノート", "教科書", "新聞", "ラベル", "付箋"),
    "object_map": ("地図", "案内図", "路線図"),
    "object_wearable": ("靴", "イヤホン", "校章", "バッジ", "服", "ボタン", "靴下", "白衣"),
    "object_craft": ("糸", "刺しゅう", "編み物", "ビーズ", "粘土", "模型", "絵の具", "水彩"),
    "object_clay": ("粘土", "陶土"),
    "object_game_item": ("素材", "装備", "アイテム", "武器", "防具"),
    "object_ui_text": ("字幕", "文字サイズ", "文字の大きさ", "UI", "表示サイズ"),
    "object_container": ("瓶", "ボトル", "容器", "コップ", "箱", "袋", "缶"),
    "event_forget": ("忘れ", "持ってくるの忘", "置いてき"),
    "event_reverse": ("逆", "上下", "左右", "裏返し"),
    "event_sell": ("売っ", "売却", "手放し"),
    "event_crack": ("ひび", "割れ", "亀裂"),
    "event_align": ("そろえ", "揃え", "同じ位置", "同じ高さ", "まっすぐ"),
    "event_sort": ("色順", "色ごと", "並べ", "分類", "分け"),
    "event_view": ("見るのが好き", "眺め", "読むの面白", "観察"),
    "event_dry": ("乾か", "乾燥", "干し"),
    "event_size_adjust": ("大きく", "小さく", "サイズ", "大きめ"),
    "event_misplace": ("迷子", "見失", "場所を間違", "違う階", "どこ"),
    "event_spill": ("こぼ", "飛び散", "漏れ"),
    "event_mistake": ("間違", "失敗", "ミス", "やっちゃ", "しちゃ"),
    "emotion_anxiety": ("不安", "落ち着か", "焦", "気になる", "心配"),
    "emotion_tired": ("疲れ", "眠", "気力", "へとへと"),
    "emotion_proud": ("できた", "成功", "直った", "終えた", "そろえられ"),
    "social_friend": ("友達", "友人", "みんな", "グループ"),
}

RESPONSE_CONCEPTS: dict[str, tuple[str, ...]] = {
    "response_check": ("確認", "見直", "比べ", "チェック"),
    "response_retry": ("もう一度", "次は", "やり直", "再開", "試し"),
    "response_organize": ("分け", "そろえ", "並べ", "整理"),
    "response_rest": ("休", "離れ", "区切"),
    "response_ask": ("先生へ", "聞けば", "相談", "伝え"),
    "response_settings": ("設定", "サイズ", "感度", "ミュート"),
}

TEASE_WORDS = (
    "ざぁこ", "よわ", "ちょろ", "かわい", "ばればれ", "バレバレ",
    "単純", "欲張り", "せっかち", "見栄", "間抜け", "詰めが甘",
    "お子さま", "お人よし", "自信家", "びびり", "雑すぎ",
)
SOFT_AI = ("大丈夫だよ", "無理しないで", "十分頑張", "少しずつでいいよ")
MIND_READING = ("でしょ", "バレ", "顔して", "つもり", "どうせ", "気づいて", "思って")
ANSWER_MARKERS = ("〜", "から", "すれば", "すると", "なら", "でいい", "ことある")
QUESTION_END = re.compile(r"[？?]\s*[♡♪〜～]*\s*$")


@dataclass(frozen=True)
class SemanticHit:
    source_id: str
    score: float
    reasons: tuple[str, ...]
    source_kind: str = "golden"

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.source_id,
            "source": self.source_kind,
            "score": round(self.score, 4),
            "reasons": list(self.reasons),
        }


def concept_set(text: str, mapping: dict[str, tuple[str, ...]] = CONCEPTS) -> set[str]:
    value = normalized_for_similarity(text)
    return {
        name
        for name, terms in mapping.items()
        if any(normalized_for_similarity(term) in value for term in terms)
    }


def merge_database_metadata(
    golden: list[dict[str, Any]], database: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    metadata = {str(item.get("id")): item for item in database}
    merged = []
    for record in golden:
        rich = metadata.get(str(record.get("id")), {})
        value = dict(record)
        for key, item in rich.items():
            if key not in {"id", "messages"}:
                value[key] = item
        merged.append(value)
    return merged


def read_database_files(directory: Path) -> tuple[list[Path], list[dict[str, Any]]]:
    if not directory.exists():
        return [], []
    paths = sorted(
        path for path in directory.iterdir() if path.suffix.lower() in {".json", ".jsonl"}
    )
    records: list[dict[str, Any]] = []
    for path in paths:
        if path.suffix.lower() == ".jsonl":
            records.extend(item.record for item in read_jsonl([path]))
            continue
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        values = payload if isinstance(payload, list) else payload.get("records", [])
        records.extend(item for item in values if isinstance(item, dict))
    return paths, records


def _records_from_text(path: Path, text: str) -> list[dict[str, Any]]:
    blocks = re.split(r"(?m)^(?:-{5,}|#{1,3}\s*)", text)
    records = []
    for index, block in enumerate(blocks, 1):
        user_match = re.search(
            r"(?ms)^\s*(?:User|ユーザー|質問)\s*[:：]\s*(.+?)(?=^\s*(?:NONO|Assistant|回答)\s*[:：]|\Z)",
            block,
        )
        assistant_match = re.search(
            r"(?ms)^\s*(?:NONO|Assistant|回答)\s*[:：]\s*(.+?)\s*\Z", block
        )
        if user_match:
            messages = [{"role": "user", "content": user_match.group(1).strip()}]
            if assistant_match:
                messages.append(
                    {"role": "assistant", "content": assistant_match.group(1).strip()}
                )
            records.append(
                {
                    "id": f"reference:{path.as_posix()}#{index}",
                    "messages": messages,
                    "_reference_path": path.as_posix(),
                }
            )
    return records


def read_reference_files(directory: Path) -> tuple[list[Path], list[dict[str, Any]]]:
    if not directory.exists():
        return [], []
    paths = sorted(path for path in directory.rglob("*") if path.is_file())
    records: list[dict[str, Any]] = []
    for path in paths:
        if path.name.lower() == "readme.md":
            continue
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            for item in read_jsonl([path]):
                value = dict(item.record)
                value["id"] = f"reference:{path.as_posix()}#{item.line_number}"
                value["_reference_path"] = path.as_posix()
                records.append(value)
        elif suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            values = payload if isinstance(payload, list) else payload.get("records", [])
            for index, item in enumerate(values, 1):
                if isinstance(item, dict):
                    value = dict(item)
                    value["id"] = f"reference:{path.as_posix()}#{index}"
                    value["_reference_path"] = path.as_posix()
                    records.append(value)
        elif suffix in {".txt", ".md"}:
            records.extend(_records_from_text(path, path.read_text(encoding="utf-8-sig")))
    return paths, records


def semantic_similarity(
    candidate: dict[str, Any],
    source: dict[str, Any],
    *,
    source_kind: str,
) -> SemanticHit | None:
    user, assistant = extract_dialogue(candidate)
    old_user, old_assistant = extract_dialogue(source)
    normalized_equal = normalized_for_similarity(user) == normalized_for_similarity(old_user)
    fuzzy = fuzzy_similarity(user, old_user)
    ngram = ngram_similarity(user, old_user)
    current = concept_set(user)
    previous = concept_set(old_user)
    shared = current & previous
    shared_events = {item for item in shared if item.startswith("event_")}
    shared_objects = {
        item for item in shared
        if item.startswith("object_") or item.startswith("place_") or item.startswith("social_")
    }
    reasons: list[str] = []
    score = max(fuzzy, ngram)
    if normalized_equal:
        reasons.append("normalized User match")
        score = 1.0
    if fuzzy >= 0.72:
        reasons.append(f"User wording similarity={fuzzy:.2f}")
    if ngram >= 0.55:
        reasons.append(f"character n-gram={ngram:.2f}")
    if shared_events and shared_objects:
        semantic_score = min(0.96, 0.76 + 0.05 * (len(shared_events) + len(shared_objects) - 2))
        score = max(score, semantic_score)
        reasons.append(
            "same event/situation: " + ", ".join(sorted(shared_events | shared_objects))
        )
    elif len(shared_events) >= 2:
        score = max(score, 0.76)
        reasons.append("same problem flow: " + ", ".join(sorted(shared_events)))
    candidate_response = concept_set(assistant, RESPONSE_CONCEPTS)
    old_response = concept_set(old_assistant, RESPONSE_CONCEPTS)
    shared_response = candidate_response & old_response
    if score >= 0.62 and shared_response:
        score = min(1.0, score + 0.05)
        reasons.append("same answer policy: " + ", ".join(sorted(shared_response)))
    current_paragraphs = [
        value.strip() for value in re.split(r"\n\s*\n", assistant) if value.strip()
    ]
    old_paragraphs = [
        value.strip() for value in re.split(r"\n\s*\n", old_assistant) if value.strip()
    ]
    current_ending = current_paragraphs[-1] if current_paragraphs else ""
    old_ending = old_paragraphs[-1] if old_paragraphs else ""
    ending_similarity = (
        fuzzy_similarity(current_ending, old_ending)
        if current_ending and old_ending
        else 0.0
    )
    if ending_similarity >= 0.82:
        score = max(score, 0.78)
        reasons.append(f"same ending/question flow={ending_similarity:.2f}")
    if (
        len(shared_response) >= 2
        and bool(current & previous)
        and len(current_paragraphs) == len(old_paragraphs)
    ):
        score = max(score, 0.76)
        reasons.append(
            "same answer/paragraph flow: " + ", ".join(sorted(shared_response))
        )
    shared_teases = {
        word for word in TEASE_WORDS if word in assistant and word in old_assistant
    }
    if score >= 0.62 and shared_teases:
        score = min(1.0, score + 0.03)
        reasons.append("same teasing vocabulary: " + ", ".join(sorted(shared_teases)))
    if score < 0.72:
        return None
    return SemanticHit(
        str(source.get("id", "")), score, tuple(reasons), source_kind=source_kind
    )


def find_semantic_duplicates(
    candidate: dict[str, Any],
    golden: Iterable[dict[str, Any]],
    references: Iterable[dict[str, Any]] = (),
) -> list[SemanticHit]:
    hits = []
    for source_kind, records in (("golden", golden), ("reference", references)):
        for source in records:
            hit = semantic_similarity(candidate, source, source_kind=source_kind)
            if hit:
                hits.append(hit)
    return sorted(hits, key=lambda item: (-item.score, item.source_kind, item.source_id))


def style_features(record: dict[str, Any]) -> dict[str, Any]:
    _, assistant = extract_dialogue(record)
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", assistant) if part.strip()]
    tease_hits = sum(assistant.count(word) for word in TEASE_WORDS)
    mind_reading = any(word in assistant for word in MIND_READING)
    answer = any(word in assistant for word in ANSWER_MARKERS) and len(paragraphs) >= 2
    soft_hits = [word for word in SOFT_AI if word in assistant]
    attack = any(word in assistant for word in ("死ね", "消えろ", "ゴミ", "クズ", "きもい"))
    ending = paragraphs[-1] if paragraphs else ""
    follow_up = bool(QUESTION_END.search(ending))
    tease_chars = sum(
        len(paragraph)
        for paragraph in paragraphs
        if any(word in paragraph for word in TEASE_WORDS)
        or any(word in paragraph for word in MIND_READING)
    )
    ratio = tease_chars / len(assistant) if assistant else 0.0
    mesugaki = min(
        10.0,
        2.5 * mind_reading + 1.5 * bool(tease_hits) + 4.0 * min(1.0, ratio / 0.55)
        + 1.0 * bool(ending and any(mark in ending for mark in ("♡", "♪", "〜"))),
    )
    nono_score = round(
        max(0.0, min(100.0, mesugaki * 10 + 10 * answer - 20 * bool(soft_hits))), 1
    )
    return {
        "nono_score": nono_score,
        "mesugaki_strength": round(mesugaki, 1),
        "teasing_ratio": round(ratio, 3),
        "soft_ai": bool(soft_hits),
        "soft_ai_phrases": soft_hits,
        "mind_reading": mind_reading,
        "answer_or_empathy": answer,
        "attacking": attack,
        "ending_or_follow_up": bool(ending) and (
            follow_up or any(mark in ending for mark in ("♡", "♪", "〜"))
        ),
        "follow_up": follow_up,
        "opening": opening_key(assistant),
        "ending": ending_key(assistant),
        "tease_words": {word: assistant.count(word) for word in TEASE_WORDS if word in assistant},
        "paragraph_count": len(paragraphs),
    }


def batch_style_warnings(records: list[dict[str, Any]]) -> list[str]:
    features = [style_features(record) for record in records]
    openings = Counter(item["opening"] for item in features)
    endings = Counter(item["ending"] for item in features)
    warnings = []
    for label, counter, limit in (("opening", openings, 3), ("ending", endings, 2)):
        for value, count in counter.items():
            if value and count > limit:
                warnings.append(f"{label} '{value}' repeated {count} times")
    consecutive = 0
    for left, right in zip(features, features[1:]):
        if left["opening"] == right["opening"] or left["ending"] == right["ending"]:
            consecutive += 1
    if consecutive:
        warnings.append(f"same opening/ending used consecutively {consecutive} time(s)")
    rate = sum(item["follow_up"] for item in features) / len(features) if features else 0
    if not 0.30 <= rate <= 0.50:
        warnings.append(f"follow-up rate {rate:.1%} is outside 30%-50%")
    return warnings
