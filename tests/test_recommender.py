"""병원 표현과 일반 사전을 병합한 공개 추천 API."""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
import subprocess
import sys

import pytest

from src.recommendation.recommender import (
    build_recommender,
    get_suggestions,
    initialize_recommender,
)


def _write_wordfreq(path: Path, entries) -> None:
    payload = {
        "schema_version": 1,
        "language": "ko",
        "word_count": len(entries),
        "words": entries,
    }
    with gzip.open(path, "wt", encoding="utf-8") as target:
        json.dump(payload, target, ensure_ascii=False)


def _write_hospital(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=("text", "priority", "category", "context"),
        )
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def dictionaries(tmp_path):
    wordfreq_path = tmp_path / "words.json.gz"
    hospital_path = tmp_path / "hospital.csv"
    _write_wordfreq(
        wordfreq_path,
        [
            {"word": "감사", "chosung": "ㄱㅅ", "freq": 0.9, "rank": 1},
            {"word": "감성", "chosung": "ㄱㅅ", "freq": 0.8, "rank": 2},
            {"word": "안녕하세요", "chosung": "ㅇㄴㅎㅅㅇ", "freq": 0.7, "rank": 3},
            {"word": "가수", "chosung": "ㄱㅅ", "freq": 0.6, "rank": 4},
        ],
    )
    _write_hospital(
        hospital_path,
        [
            {
                "text": "감사합니다",
                "priority": "200",
                "category": "daily_conversation",
                "context": "patient_patient",
            },
            {
                "text": "안녕하세요",
                "priority": "200",
                "category": "daily_conversation",
                "context": "patient_patient",
            },
        ],
    )
    return wordfreq_path, hospital_path


def test_hospital_priority_and_duplicate_override(dictionaries):
    engine = build_recommender(*dictionaries)

    assert engine.get_suggestions("ㄱㅅ") == ["감사합니다", "감사", "감성"]
    candidates = engine.trie.query("ㅇㄴㅎㅅㅇ")
    assert len(candidates) == 1
    assert candidates[0].word == "안녕하세요"
    assert candidates[0].source == "hospital"


def test_public_api_is_explicitly_initialized_and_capped_at_three(dictionaries):
    engine = initialize_recommender(*dictionaries)

    assert engine.available
    assert get_suggestions("ㄱㅅ", limit=20) == ["감사합니다", "감사", "감성"]
    assert get_suggestions("ㄱㅅ", limit=1) == ["감사합니다"]


@pytest.mark.parametrize(
    ("prefix", "limit"),
    [
        ("", 3),
        (None, 3),
        ("abc", 3),
        ("ㄱㅏ", 3),
        ("ㄱㅅ", 0),
        ("ㄱㅅ", True),
    ],
)
def test_invalid_public_query_returns_empty(dictionaries, prefix, limit):
    engine = build_recommender(*dictionaries)

    assert engine.get_suggestions(prefix, limit) == []


def test_dictionary_error_warns_and_disables_only_autocomplete(
    tmp_path,
    dictionaries,
):
    _, hospital_path = dictionaries

    with pytest.warns(RuntimeWarning, match="autocomplete disabled"):
        engine = initialize_recommender(
            tmp_path / "missing.json.gz",
            hospital_path,
        )

    assert not engine.available
    assert engine.error_message is not None
    assert get_suggestions("ㄱㅅ") == []


def test_module_import_does_not_load_data_or_wordfreq():
    code = """
import gzip
from pathlib import Path
import sys

def fail_if_opened(*args, **kwargs):
    raise AssertionError("dictionary opened during import")

gzip.open = fail_if_opened
Path.open = fail_if_opened
import src.recommendation.recommender
assert "wordfreq" not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.fixture(scope="module")
def actual_engine():
    return build_recommender()


@pytest.mark.parametrize(
    ("prefix", "expected"),
    [
        ("ㄱㅅㅎㄴㄷ", "감사합니다"),
        ("ㅇㄴㅎㅅㅇ", "안녕하세요"),
        ("ㄷㅇㅈㅅㅇ", "도와주세요"),
        ("ㅅㅅㄱㅎㄷㅇ", "숨쉬기 힘들어요"),
        ("ㅁㅈㅅㅇ", "물 주세요"),
        ("ㄱㅎㅅㅂㄹㅈㅅㅇ", "간호사 불러주세요"),
    ],
)
def test_required_hospital_expression_is_returned(actual_engine, prefix, expected):
    assert actual_engine.get_suggestions(prefix)[0] == expected


def test_initialized_query_is_under_100ms(actual_engine):
    elapsed = []
    for _ in range(1000):
        from time import perf_counter

        started_at = perf_counter()
        actual_engine.get_suggestions("ㄱㅅ", 3)
        elapsed.append((perf_counter() - started_at) * 1000.0)

    assert max(elapsed) < 100.0
