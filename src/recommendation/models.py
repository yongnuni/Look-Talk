"""초성 자동완성에서 공유하는 불변 데이터 모델."""

from dataclasses import dataclass
import math
from numbers import Real

from src.recommendation.chosung import CHOSUNG, extract_chosung


_CHOSUNG_SET = frozenset(CHOSUNG)


@dataclass(frozen=True, slots=True)
class ChosungWord:
    """빈도 사전의 단어 한 항목."""

    word: str
    chosung: str
    freq: float
    rank: int

    def __post_init__(self) -> None:
        if not isinstance(self.word, str) or not self.word:
            raise ValueError("word must be a non-empty string")
        if not all("가" <= char <= "힣" for char in self.word):
            raise ValueError("word must contain only precomposed Hangul syllables")

        if not isinstance(self.chosung, str) or not self.chosung:
            raise ValueError("chosung must be a non-empty string")
        if any(char not in _CHOSUNG_SET for char in self.chosung):
            raise ValueError("chosung contains an unsupported character")
        if extract_chosung(self.word) != self.chosung:
            raise ValueError("chosung does not match word")

        if (
            isinstance(self.freq, bool)
            or not isinstance(self.freq, Real)
            or not math.isfinite(float(self.freq))
            or self.freq < 0
        ):
            raise ValueError("freq must be a finite non-negative number")

        if (
            isinstance(self.rank, bool)
            or not isinstance(self.rank, int)
            or self.rank <= 0
        ):
            raise ValueError("rank must be a positive integer")
