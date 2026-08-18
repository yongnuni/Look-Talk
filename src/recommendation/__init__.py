"""단어 추천 기능의 공개 인터페이스."""

from src.recommendation.chosung import (
    decompose_hangul_syllable,
    extract_chosung,
)
from src.recommendation.models import ChosungWord
from src.recommendation.trie import (
    ChosungDictionaryError,
    ChosungTrie,
    TrieNode,
    load_chosung_words,
)

__all__ = [
    "ChosungDictionaryError",
    "ChosungTrie",
    "ChosungWord",
    "TrieNode",
    "decompose_hangul_syllable",
    "extract_chosung",
    "load_chosung_words",
]
