"""검증된 초성 단어 데이터로 구축하는 메모리 Trie."""

from __future__ import annotations

from dataclasses import dataclass, field
import gzip
import json
from pathlib import Path
from typing import Any, Iterable

from src.recommendation.chosung import CHOSUNG
from src.recommendation.models import ChosungWord


DICTIONARY_SCHEMA_VERSION = 1
DICTIONARY_LANGUAGE = "ko"
MAX_CACHED_CANDIDATES = 20
DEFAULT_DICTIONARY_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "chosung_words.json.gz"
)

_CHOSUNG_SET = frozenset(CHOSUNG)


class ChosungDictionaryError(RuntimeError):
    """초성 사전 파일을 읽거나 검증할 수 없을 때 발생한다."""


@dataclass(slots=True)
class TrieNode:
    """초성 한 글자 간선의 자식과 접두사 후보 캐시를 보관한다."""

    children: dict[str, TrieNode] = field(default_factory=dict)
    cached_candidates: list[ChosungWord] = field(default_factory=list)


def _dictionary_error(path: Path, detail: str) -> ChosungDictionaryError:
    return ChosungDictionaryError(f"invalid Chosung dictionary {path}: {detail}")


def _validate_payload(
    payload: Any,
    path: Path,
) -> tuple[ChosungWord, ...]:
    if not isinstance(payload, dict):
        raise _dictionary_error(path, "top-level JSON value must be an object")

    schema_version = payload.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != DICTIONARY_SCHEMA_VERSION
    ):
        raise _dictionary_error(
            path,
            f"schema_version must be {DICTIONARY_SCHEMA_VERSION}",
        )

    if payload.get("language") != DICTIONARY_LANGUAGE:
        raise _dictionary_error(
            path,
            f"language must be {DICTIONARY_LANGUAGE!r}",
        )

    raw_words = payload.get("words")
    if not isinstance(raw_words, list):
        raise _dictionary_error(path, "words must be an array")

    declared_word_count = payload.get("word_count")
    if declared_word_count is not None and (
        isinstance(declared_word_count, bool)
        or not isinstance(declared_word_count, int)
        or declared_word_count != len(raw_words)
    ):
        raise _dictionary_error(path, "word_count does not match words array")

    words: list[ChosungWord] = []
    seen_words: set[str] = set()
    previous_frequency: float | None = None

    for index, raw_entry in enumerate(raw_words, start=1):
        if not isinstance(raw_entry, dict):
            raise _dictionary_error(path, f"words[{index - 1}] must be an object")

        missing_fields = {
            field_name
            for field_name in ("word", "chosung", "freq", "rank")
            if field_name not in raw_entry
        }
        if missing_fields:
            fields = ", ".join(sorted(missing_fields))
            raise _dictionary_error(
                path,
                f"words[{index - 1}] missing fields: {fields}",
            )

        try:
            word = ChosungWord(
                word=raw_entry["word"],
                chosung=raw_entry["chosung"],
                freq=raw_entry["freq"],
                rank=raw_entry["rank"],
            )
        except (TypeError, ValueError) as exc:
            raise _dictionary_error(
                path,
                f"words[{index - 1}] {exc}",
            ) from exc

        if word.rank != index:
            raise _dictionary_error(
                path,
                "ranks must be contiguous, one-based, and match array order",
            )

        if previous_frequency is not None and word.freq > previous_frequency:
            raise _dictionary_error(
                path,
                "frequencies must be in non-increasing order",
            )

        if word.word in seen_words:
            raise _dictionary_error(path, f"duplicate word: {word.word!r}")

        words.append(word)
        seen_words.add(word.word)
        previous_frequency = float(word.freq)

    return tuple(words)


def load_chosung_words(
    path: str | Path | None = None,
) -> tuple[ChosungWord, ...]:
    """gzip JSON 사전을 읽고 데이터 계약을 검증한다.

    기본 경로는 이 모듈 위치를 기준으로 계산하며 현재 작업 디렉터리에
    의존하지 않는다. 파일·압축·JSON·스키마 오류는 모두 원인을 포함한
    ``ChosungDictionaryError``로 전달한다.
    """

    dictionary_path = (
        DEFAULT_DICTIONARY_PATH
        if path is None
        else Path(path).expanduser().resolve()
    )

    try:
        with gzip.open(dictionary_path, "rt", encoding="utf-8") as source:
            payload = json.load(source)
    except FileNotFoundError as exc:
        raise ChosungDictionaryError(
            f"Chosung dictionary file not found: {dictionary_path}"
        ) from exc
    except gzip.BadGzipFile as exc:
        raise ChosungDictionaryError(
            f"invalid gzip Chosung dictionary: {dictionary_path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ChosungDictionaryError(
            f"invalid JSON in Chosung dictionary {dictionary_path}: {exc.msg}"
        ) from exc
    except UnicodeDecodeError as exc:
        raise ChosungDictionaryError(
            f"Chosung dictionary is not valid UTF-8: {dictionary_path}"
        ) from exc
    except (EOFError, OSError) as exc:
        raise ChosungDictionaryError(
            f"failed to read gzip Chosung dictionary {dictionary_path}: {exc}"
        ) from exc

    return _validate_payload(payload, dictionary_path)


def _candidate_order(candidate: ChosungWord) -> tuple[int, float, str]:
    return (candidate.rank, -float(candidate.freq), candidate.word)


class ChosungTrie:
    """각 노드에 상위 20개 후보를 캐싱하는 초성 Trie."""

    def __init__(self) -> None:
        self.root = TrieNode()
        self.node_count = 1
        self.max_cached_candidates = 0

    def insert(self, candidate: ChosungWord) -> None:
        """단어를 초성 경로에 삽입하고 지나간 노드 캐시를 갱신한다."""

        if not isinstance(candidate, ChosungWord):
            raise TypeError("candidate must be a ChosungWord")

        node = self.root
        for edge in candidate.chosung:
            if edge not in _CHOSUNG_SET:
                raise ValueError(f"unsupported Chosung edge: {edge!r}")

            child = node.children.get(edge)
            if child is None:
                child = TrieNode()
                node.children[edge] = child
                self.node_count += 1

            node = child
            self._cache_candidate(node, candidate)

    def _cache_candidate(
        self,
        node: TrieNode,
        candidate: ChosungWord,
    ) -> None:
        existing_index = next(
            (
                index
                for index, existing in enumerate(node.cached_candidates)
                if existing.word == candidate.word
            ),
            None,
        )

        if existing_index is not None:
            existing = node.cached_candidates[existing_index]
            if _candidate_order(existing) <= _candidate_order(candidate):
                return
            node.cached_candidates[existing_index] = candidate
        else:
            node.cached_candidates.append(candidate)

        node.cached_candidates.sort(key=_candidate_order)
        del node.cached_candidates[MAX_CACHED_CANDIDATES:]
        self.max_cached_candidates = max(
            self.max_cached_candidates,
            len(node.cached_candidates),
        )

    @classmethod
    def from_words(cls, words: Iterable[ChosungWord]) -> ChosungTrie:
        """검증된 단어 iterable을 삽입해 Trie를 구축한다."""

        trie = cls()
        for word in words:
            trie.insert(word)

        return trie

    @classmethod
    def from_file(
        cls,
        path: str | Path | None = None,
    ) -> ChosungTrie:
        """명시적으로 사전을 로드하고 Trie를 한 번 구축한다."""

        return cls.from_words(load_chosung_words(path))

    def query(
        self,
        prefix: str,
        limit: int = MAX_CACHED_CANDIDATES,
    ) -> list[ChosungWord]:
        """접두사 노드 캐시에서 최대 20개 후보만 반환한다.

        경로 탐색 뒤 하위 노드를 순회하지 않으므로 복잡도는
        ``O(len(prefix) + returned candidates)``이다.
        """

        if not isinstance(prefix, str) or not prefix:
            return []
        if any(edge not in _CHOSUNG_SET for edge in prefix):
            return []
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or limit <= 0
        ):
            return []

        node = self.root
        for edge in prefix:
            node = node.children.get(edge)
            if node is None:
                return []

        capped_limit = min(limit, MAX_CACHED_CANDIDATES)
        return list(node.cached_candidates[:capped_limit])
