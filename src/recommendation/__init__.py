"""단어 추천 기능의 공개 인터페이스."""

from src.recommendation.chosung import (
    decompose_hangul_syllable,
    extract_chosung,
)
from src.recommendation.models import ChosungWord
from src.recommendation.hospital import (
    HospitalDictionaryError,
    load_hospital_phrases,
)
from src.recommendation.input_state import (
    AUTOCOMPLETE_MODE,
    FAVORITES_MODE,
    SuggestionStateController,
    SuggestionUpdate,
    format_suggestion_update,
)
from src.recommendation.recommender import (
    SuggestionEngine,
    build_recommender,
    get_suggestions,
    initialize_recommender,
)
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
    "HospitalDictionaryError",
    "SuggestionEngine",
    "SuggestionStateController",
    "SuggestionUpdate",
    "TrieNode",
    "AUTOCOMPLETE_MODE",
    "FAVORITES_MODE",
    "build_recommender",
    "decompose_hangul_syllable",
    "extract_chosung",
    "format_suggestion_update",
    "get_suggestions",
    "initialize_recommender",
    "load_chosung_words",
    "load_hospital_phrases",
]
