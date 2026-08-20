"""초성 사전 입력 데이터 생성기의 필터링과 결정성을 검증한다."""

import gzip
import hashlib
import json
import unicodedata

import pytest

from scripts.build_chosung_dictionary import (
    ATTRIBUTION,
    FILTERING_RULES,
    LICENSE,
    SCHEMA_VERSION,
    SORT_ORDER,
    build_dictionary,
    parse_args,
    write_dictionary,
)


@pytest.fixture
def source_fixture():
    decomposed_ga = unicodedata.normalize("NFD", "가")
    words = [
        "한국",
        decomposed_ga,
        "english",
        "123",
        "한 국",
        "한A국",
        "나",
        "가",
        "삶",
    ]
    frequency_values = [
        0.1,
        0.09,
        0.08,
        0.07,
        0.06,
        0.05,
        0.04,
        0.03,
        0.02,
    ]
    frequencies = dict(zip(words, frequency_values, strict=True))
    return words, frequencies


def test_build_dictionary_filters_normalizes_and_preserves_order(
    source_fixture,
):
    words, frequencies = source_fixture

    payload = build_dictionary(words, frequencies)

    assert payload["input_word_count"] == 9
    assert payload["word_count"] == 4
    assert payload["excluded_word_count"] == 5
    assert payload["words"] == [
        {"word": "한국", "chosung": "ㅎㄱ", "freq": 0.1, "rank": 1},
        {"word": "가", "chosung": "ㄱ", "freq": 0.09, "rank": 2},
        {
            "word": "나",
            "chosung": "ㄴ",
            "freq": pytest.approx(0.04),
            "rank": 3,
        },
        {
            "word": "삶",
            "chosung": "ㅅ",
            "freq": pytest.approx(0.02),
            "rank": 4,
        },
    ]


def test_max_words_limits_source_before_filtering(source_fixture):
    words, frequencies = source_fixture

    payload = build_dictionary(words, frequencies, max_words=3)

    assert payload["input_word_count"] == 3
    assert payload["word_count"] == 2
    assert payload["excluded_word_count"] == 1
    assert [entry["word"] for entry in payload["words"]] == ["한국", "가"]


def test_max_words_cli_argument():
    args = parse_args(["--max-words", "25"])

    assert args.max_words == 25


@pytest.mark.parametrize("value", [0, -1])
def test_build_dictionary_rejects_non_positive_max_words(
    source_fixture,
    value,
):
    words, frequencies = source_fixture

    with pytest.raises(ValueError, match="positive integer"):
        build_dictionary(words, frequencies, max_words=value)


def test_metadata_schema(source_fixture):
    words, frequencies = source_fixture

    payload = build_dictionary(words, frequencies)

    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["language"] == "ko"
    assert payload["source"] == "wordfreq"
    assert payload["wordfreq_version"] == "3.1.1"
    assert payload["wordlist"] == "best"
    assert payload["filtering_rules"] == list(FILTERING_RULES)
    assert payload["sort_order"] == SORT_ORDER
    assert payload["license"] == LICENSE
    assert payload["attribution"] == ATTRIBUTION


def test_deterministic_gzip_output(source_fixture, tmp_path):
    words, frequencies = source_fixture
    payload = build_dictionary(words, frequencies)
    first_path = tmp_path / "first.json.gz"
    second_path = tmp_path / "second.json.gz"

    first_stats = write_dictionary(payload, first_path)
    second_stats = write_dictionary(payload, second_path)

    first_bytes = first_path.read_bytes()
    second_bytes = second_path.read_bytes()

    assert first_bytes == second_bytes
    assert first_stats == second_stats
    assert first_stats["sha256"] == hashlib.sha256(first_bytes).hexdigest()

    with gzip.open(first_path, "rt", encoding="utf-8") as source:
        restored = json.load(source)

    assert restored == payload
