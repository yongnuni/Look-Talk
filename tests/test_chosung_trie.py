"""초성 Trie의 캐시 조회와 사전 데이터 검증."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
import subprocess
import sys

import pytest

from src.recommendation.models import ChosungWord
from src.recommendation.trie import (
    DEFAULT_DICTIONARY_PATH,
    MAX_CACHED_CANDIDATES,
    ChosungDictionaryError,
    ChosungTrie,
    load_chosung_words,
)


def _word(word, chosung, freq, rank):
    return ChosungWord(
        word=word,
        chosung=chosung,
        freq=freq,
        rank=rank,
    )


def _payload(words):
    return {
        "schema_version": 1,
        "language": "ko",
        "word_count": len(words),
        "words": words,
    }


def _entry(word="한국", chosung="ㅎㄱ", freq=0.1, rank=1):
    return {
        "word": word,
        "chosung": chosung,
        "freq": freq,
        "rank": rank,
    }


def _write_payload(path: Path, payload) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as target:
        json.dump(payload, target, ensure_ascii=False)


@pytest.fixture
def sample_words():
    return [
        _word("학교", "ㅎㄱ", 0.7, 3),
        _word("한국", "ㅎㄱ", 0.9, 1),
        _word("한강", "ㅎㄱ", 0.8, 2),
        _word("한글", "ㅎㄱ", 0.6, 4),
        _word("감사", "ㄱㅅ", 0.5, 5),
    ]


def test_one_and_multiple_edge_prefix_queries(sample_words):
    trie = ChosungTrie.from_words(sample_words)

    assert [item.word for item in trie.query("ㅎ")] == [
        "한국",
        "한강",
        "학교",
        "한글",
    ]
    assert [item.word for item in trie.query("ㅎㄱ")] == [
        "한국",
        "한강",
        "학교",
        "한글",
    ]
    assert [item.word for item in trie.query("ㄱㅅ")] == ["감사"]


def test_unordered_insertion_is_sorted_by_rank(sample_words):
    trie = ChosungTrie()
    for candidate in sample_words:
        trie.insert(candidate)

    assert [item.rank for item in trie.query("ㅎㄱ")] == [1, 2, 3, 4]


def test_duplicate_word_is_not_cached_twice():
    trie = ChosungTrie()
    candidate = _word("한국", "ㅎㄱ", 0.9, 1)

    trie.insert(candidate)
    trie.insert(candidate)

    assert trie.query("ㅎㄱ") == [candidate]


def test_better_duplicate_replaces_existing_candidate():
    trie = ChosungTrie()
    lower_ranked = _word("한국", "ㅎㄱ", 0.5, 5)
    higher_ranked = _word("한국", "ㅎㄱ", 0.9, 1)

    trie.insert(lower_ranked)
    trie.insert(higher_ranked)

    assert trie.query("ㅎㄱ") == [higher_ranked]


def test_node_cache_and_query_limit_are_capped_at_twenty():
    candidates = [
        _word(chr(0xAC00 + index), "ㄱ", 1.0 / rank, rank)
        for index, rank in enumerate(range(30, 0, -1))
    ]
    trie = ChosungTrie.from_words(candidates)

    assert len(trie.root.children["ㄱ"].cached_candidates) == 20
    assert [item.rank for item in trie.query("ㄱ", limit=20)] == list(
        range(1, 21)
    )
    assert len(trie.query("ㄱ", limit=1)) == 1
    assert len(trie.query("ㄱ", limit=3)) == 3
    assert len(trie.query("ㄱ", limit=200)) == 20
    assert trie.max_cached_candidates == 20


@pytest.mark.parametrize(
    "prefix",
    ["", None, 123, "abc", "123", " ", "ㄱㅏ", "ㄳ", "ㄱA"],
)
def test_invalid_prefix_returns_empty(sample_words, prefix):
    trie = ChosungTrie.from_words(sample_words)

    assert trie.query(prefix) == []


@pytest.mark.parametrize("limit", [0, -1, None, "3", True])
def test_invalid_limit_returns_empty(sample_words, limit):
    trie = ChosungTrie.from_words(sample_words)

    assert trie.query("ㅎ", limit=limit) == []


def test_missing_prefix_returns_empty(sample_words):
    trie = ChosungTrie.from_words(sample_words)

    assert trie.query("ㅃㅃ") == []


def test_root_does_not_cache_or_return_candidates(sample_words):
    trie = ChosungTrie.from_words(sample_words)

    assert trie.root.cached_candidates == []
    assert trie.query("") == []


def test_query_uses_node_cache_without_descendant_traversal(sample_words):
    class ExplodingDict(dict):
        def __getitem__(self, key):
            raise AssertionError("descendant access occurred")

        def get(self, key, default=None):
            raise AssertionError("descendant access occurred")

        def __iter__(self):
            raise AssertionError("descendant traversal occurred")

        def items(self):
            raise AssertionError("descendant traversal occurred")

        def values(self):
            raise AssertionError("descendant traversal occurred")

    trie = ChosungTrie.from_words(sample_words)
    prefix_node = trie.root.children["ㅎ"]
    prefix_node.children = ExplodingDict(prefix_node.children)

    assert [item.word for item in trie.query("ㅎ", limit=3)] == [
        "한국",
        "한강",
        "학교",
    ]


def test_missing_dictionary_file(tmp_path):
    missing_path = tmp_path / "missing.json.gz"

    with pytest.raises(ChosungDictionaryError, match="file not found"):
        load_chosung_words(missing_path)


def test_corrupted_gzip(tmp_path):
    path = tmp_path / "corrupted.json.gz"
    path.write_bytes(b"not a gzip file")

    with pytest.raises(ChosungDictionaryError, match="invalid gzip"):
        load_chosung_words(path)


def test_invalid_json(tmp_path):
    path = tmp_path / "invalid.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as target:
        target.write("{")

    with pytest.raises(ChosungDictionaryError, match="invalid JSON"):
        load_chosung_words(path)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"schema_version": 2, "language": "ko", "words": []}, "schema"),
        ({"schema_version": 1, "language": "en", "words": []}, "language"),
        ({"schema_version": 1, "language": "ko"}, "words"),
        (
            _payload([{"word": "한국", "chosung": "ㅎㄱ", "rank": 1}]),
            "missing fields",
        ),
        (_payload([_entry(chosung="ㅎA")]), "unsupported"),
        (_payload([_entry(rank=2)]), "ranks"),
        (
            _payload(
                [
                    _entry(word="한국", chosung="ㅎㄱ", freq=0.1, rank=1),
                    _entry(word="학교", chosung="ㅎㄱ", freq=0.2, rank=2),
                ]
            ),
            "non-increasing",
        ),
    ],
)
def test_invalid_dictionary_schema(tmp_path, payload, message):
    path = tmp_path / "invalid-schema.json.gz"
    _write_payload(path, payload)

    with pytest.raises(ChosungDictionaryError, match=message):
        load_chosung_words(path)


def test_module_import_does_not_load_dictionary_or_wordfreq():
    code = """
import gzip
import sys

def fail_if_opened(*args, **kwargs):
    raise AssertionError("dictionary opened during import")

gzip.open = fail_if_opened
import src.recommendation.trie
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
def actual_dictionary_and_trie():
    words = load_chosung_words(DEFAULT_DICTIONARY_PATH)
    trie = ChosungTrie.from_words(words)
    return words, trie


def test_actual_dictionary_load_and_query_smoke(actual_dictionary_and_trie):
    words, trie = actual_dictionary_and_trie

    assert len(words) == 26_795
    assert trie.node_count > 1
    assert trie.max_cached_candidates == MAX_CACHED_CANDIDATES

    for prefix in ("ㅇ", "ㅇㄴ", "ㄱㅅ", "ㅎㄱ", "ㄱㅅㅎㄴㄷ"):
        results = trie.query(prefix, limit=10)
        assert all(item.chosung.startswith(prefix) for item in results)
