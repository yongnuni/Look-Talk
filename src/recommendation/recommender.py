"""wordfreq 파생 데이터와 병원 표현을 결합하는 공개 추천 API."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from time import perf_counter
import warnings

from src.recommendation.hospital import (
    HospitalDictionaryError,
    load_hospital_phrases,
)
from src.recommendation.models import ChosungWord
from src.recommendation.trie import (
    ChosungDictionaryError,
    ChosungTrie,
    load_chosung_words,
)


PUBLIC_SUGGESTION_LIMIT = 3


@dataclass(slots=True)
class SuggestionEngine:
    """명시적으로 한 번 구축한 Trie와 초기화 결과를 보관한다."""

    trie: ChosungTrie | None
    initialization_ms: float
    wordfreq_count: int = 0
    hospital_count: int = 0
    error_message: str | None = None

    @property
    def available(self) -> bool:
        return self.trie is not None

    def get_suggestions(
        self,
        chosung_input: str,
        limit: int = PUBLIC_SUGGESTION_LIMIT,
    ) -> list[str]:
        """검증된 초성 접두사의 표시 문자열을 최대 3개 반환한다."""

        if self.trie is None:
            return []
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or limit <= 0
        ):
            return []

        capped_limit = min(limit, PUBLIC_SUGGESTION_LIMIT)
        return [
            candidate.word
            for candidate in self.trie.query(chosung_input, capped_limit)
        ]


def build_recommender(
    wordfreq_path: str | Path | None = None,
    hospital_path: str | Path | None = None,
) -> SuggestionEngine:
    """두 사전을 로드해 추천 엔진을 구축하며 오류는 호출자에게 전달한다."""

    started_at = perf_counter()
    wordfreq_words = load_chosung_words(wordfreq_path)
    hospital_phrases = load_hospital_phrases(hospital_path)
    trie = ChosungTrie.from_words(chain(wordfreq_words, hospital_phrases))

    return SuggestionEngine(
        trie=trie,
        initialization_ms=(perf_counter() - started_at) * 1000.0,
        wordfreq_count=len(wordfreq_words),
        hospital_count=len(hospital_phrases),
    )


_default_engine: SuggestionEngine | None = None


def initialize_recommender(
    wordfreq_path: str | Path | None = None,
    hospital_path: str | Path | None = None,
) -> SuggestionEngine:
    """기본 엔진을 한 번 초기화하고 사전 오류 시 빈 엔진으로 대체한다."""

    global _default_engine

    try:
        engine = build_recommender(wordfreq_path, hospital_path)
    except (ChosungDictionaryError, HospitalDictionaryError) as exc:
        message = str(exc)
        warnings.warn(
            f"autocomplete disabled: {message}",
            RuntimeWarning,
            stacklevel=2,
        )
        engine = SuggestionEngine(
            trie=None,
            initialization_ms=0.0,
            error_message=message,
        )

    _default_engine = engine
    return engine


def get_suggestions(
    chosung_input: str,
    limit: int = PUBLIC_SUGGESTION_LIMIT,
) -> list[str]:
    """초기화된 기본 엔진에서 공개 후보를 최대 3개 반환한다."""

    if _default_engine is None:
        return []
    return _default_engine.get_suggestions(chosung_input, limit)
