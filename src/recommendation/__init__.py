"""단어 추천 기능의 공개 인터페이스."""

from src.recommendation.chosung import (
    decompose_hangul_syllable,
    extract_chosung,
)

__all__ = [
    "decompose_hangul_syllable",
    "extract_chosung",
]
