"""병원 특화 표현 CSV의 정규화와 검증."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
import unicodedata

import pytest

from src.recommendation.hospital import (
    DEFAULT_HOSPITAL_DICTIONARY_PATH,
    HospitalDictionaryError,
    load_hospital_phrases,
)


FIELDS = ("text", "priority", "category", "context")


def _write_csv(path: Path, rows, fieldnames=FIELDS) -> None:
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def _row(
    text="물 주세요",
    priority="400",
    category="basic_care",
    context="patient_staff",
):
    return {
        "text": text,
        "priority": priority,
        "category": category,
        "context": context,
    }


def test_nfc_normalization_and_natural_spacing_are_preserved(tmp_path):
    path = tmp_path / "hospital.csv"
    _write_csv(path, [_row(text=unicodedata.normalize("NFD", "물 주세요"))])

    phrases = load_hospital_phrases(path)

    assert len(phrases) == 1
    assert phrases[0].word == "물 주세요"
    assert phrases[0].chosung == "ㅁㅈㅅㅇ"
    assert phrases[0].source == "hospital"
    assert phrases[0].priority == 400


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "non-empty"),
        (" 물 주세요", "spacing"),
        ("물 주세요 ", "spacing"),
        ("물  주세요", "spacing"),
        ("물2주세요", "unsupported"),
        ("water", "unsupported"),
        ("물!", "unsupported"),
    ],
)
def test_invalid_text_is_rejected(tmp_path, text, message):
    path = tmp_path / "hospital.csv"
    _write_csv(path, [_row(text=text)])

    with pytest.raises(HospitalDictionaryError, match=message):
        load_hospital_phrases(path)


def test_duplicate_after_normalization_is_rejected(tmp_path):
    path = tmp_path / "hospital.csv"
    _write_csv(
        path,
        [
            _row(text="가"),
            _row(text=unicodedata.normalize("NFD", "가")),
        ],
    )

    with pytest.raises(HospitalDictionaryError, match="duplicates"):
        load_hospital_phrases(path)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"priority": "high"}, "integer"),
        ({"priority": "-1"}, "non-negative"),
        ({"category": ""}, "category"),
        ({"context": "unknown"}, "context"),
    ],
)
def test_invalid_metadata_is_rejected(tmp_path, overrides, message):
    path = tmp_path / "hospital.csv"
    row = _row()
    row.update(overrides)
    _write_csv(path, [row])

    with pytest.raises(HospitalDictionaryError, match=message):
        load_hospital_phrases(path)


def test_header_and_missing_file_are_reported(tmp_path):
    invalid_header = tmp_path / "invalid.csv"
    _write_csv(invalid_header, [_row()], fieldnames=("text", "priority"))

    with pytest.raises(HospitalDictionaryError, match="header"):
        load_hospital_phrases(invalid_header)
    with pytest.raises(HospitalDictionaryError, match="file not found"):
        load_hospital_phrases(tmp_path / "missing.csv")


def test_actual_hospital_dictionary_contract():
    phrases = load_hospital_phrases(DEFAULT_HOSPITAL_DICTIONARY_PATH)
    counts = Counter(phrase.category for phrase in phrases)

    assert len(phrases) == 79
    assert len({phrase.word for phrase in phrases}) == 79
    assert counts == {
        "emergency_safety": 11,
        "basic_care": 18,
        "symptom_discomfort": 21,
        "communication": 11,
        "daily_conversation": 18,
    }
