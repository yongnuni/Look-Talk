"""초성 자동완성에서 공유하는 불변 데이터 모델."""

from dataclasses import dataclass
import math
from numbers import Real
import unicodedata

from src.recommendation.chosung import CHOSUNG, extract_chosung


_CHOSUNG_SET = frozenset(CHOSUNG)
WORD_SOURCE = "wordfreq"
HOSPITAL_SOURCE = "hospital"
HOSPITAL_CONTEXTS = frozenset({
    "patient_staff",
    "patient_patient",
    "both",
})


@dataclass(frozen=True, slots=True)
class ChosungWord:
    """일반 단어 또는 병원 표현의 불변 Trie 후보."""

    word: str
    chosung: str
    freq: float
    rank: int
    source: str = WORD_SOURCE
    priority: int = 0
    category: str | None = None
    context: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.word, str) or not self.word:
            raise ValueError("word must be a non-empty string")
        if not unicodedata.is_normalized("NFC", self.word):
            raise ValueError("word must be NFC-normalized")

        if self.source == WORD_SOURCE:
            if not all("가" <= char <= "힣" for char in self.word):
                raise ValueError(
                    "wordfreq word must contain only precomposed Hangul syllables"
                )
        elif self.source == HOSPITAL_SOURCE:
            if (
                self.word != self.word.strip()
                or "  " in self.word
                or not all(char == " " or "가" <= char <= "힣" for char in self.word)
            ):
                raise ValueError(
                    "hospital text must contain precomposed Hangul and single spaces"
                )
        else:
            raise ValueError(f"unsupported source: {self.source!r}")

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

        if (
            isinstance(self.priority, bool)
            or not isinstance(self.priority, int)
            or self.priority < 0
        ):
            raise ValueError("priority must be a non-negative integer")

        if self.source == WORD_SOURCE:
            if self.category is not None or self.context is not None:
                raise ValueError(
                    "wordfreq candidates cannot have category or context"
                )
        else:
            if not isinstance(self.category, str) or not self.category:
                raise ValueError("hospital category must be a non-empty string")
            if self.context not in HOSPITAL_CONTEXTS:
                raise ValueError("hospital context is unsupported")
