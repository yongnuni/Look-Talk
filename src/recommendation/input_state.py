"""favorites와 입력 중 자동완성을 분리하는 표시 상태 제어기."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Sequence
import warnings

from src.recommendation.chosung import extract_chosung
from src.recommendation.recommender import get_suggestions


FAVORITES_MODE = "favorites"
AUTOCOMPLETE_MODE = "autocomplete"
SUGGESTION_SLOT_COUNT = 3

SuggestionProvider = Callable[[str, int], list[str]]


@dataclass(frozen=True, slots=True)
class SuggestionUpdate:
    """화면이나 터미널에 전달할 변경 시점의 불변 스냅샷."""

    mode: str
    chosung_input: str
    items: tuple[str, ...]
    slots: tuple[str, str, str]
    elapsed_ms: float


def format_suggestion_update(update: SuggestionUpdate) -> str:
    """중복 출력 판단과 분리된 터미널 진단 문자열을 만든다."""

    return (
        f"[suggestions] mode={update.mode} "
        f"input={update.chosung_input!r} "
        f"items={list(update.items)!r} "
        f"elapsed_ms={update.elapsed_ms:.2f}"
    )


def _normalize_favorites(sentences: Sequence[str]) -> tuple[str, ...]:
    if isinstance(sentences, str):
        sentences = (sentences,)
    return tuple(
        sentence
        for sentence in sentences
        if isinstance(sentence, str) and sentence
    )[:SUGGESTION_SLOT_COUNT]


def _fill_slots(items: Sequence[str]) -> tuple[str, str, str]:
    normalized = list(items[:SUGGESTION_SLOT_COUNT])
    normalized.extend([""] * (SUGGESTION_SLOT_COUNT - len(normalized)))
    return normalized[0], normalized[1], normalized[2]


def _current_word(current_text: str) -> str:
    if not current_text or current_text.endswith(" "):
        return ""
    return current_text.rsplit(" ", 1)[-1]


class SuggestionStateController:
    """합성 입력 문자열이 바뀔 때만 추천 모드와 슬롯을 갱신한다."""

    def __init__(
        self,
        favorite_sentences: Sequence[str] | None = None,
        suggestion_provider: SuggestionProvider = get_suggestions,
    ) -> None:
        # 프론트엔드 phrase API/브리지가 준비되면 favorite_sentences로
        # 실제 설정값을 주입한다. Python 앱 자체에는 기본 문장을 두지 않는다.
        favorites = () if favorite_sentences is None else favorite_sentences
        self.favorite_sentences = _normalize_favorites(favorites)
        self.autocomplete_suggestions: tuple[str, ...] = ()
        self.mode = FAVORITES_MODE
        self.slots = _fill_slots(self.favorite_sentences)
        self.last_current_text: str | None = None
        self.current_chosung = ""
        self.last_lookup_ms = 0.0
        self._suggestion_provider = suggestion_provider

    @property
    def visible_items(self) -> tuple[str, ...]:
        if self.mode == FAVORITES_MODE:
            return self.favorite_sentences
        return self.autocomplete_suggestions

    def update(self, current_text: str) -> SuggestionUpdate | None:
        """입력이 바뀌고 표시 후보 또는 모드가 달라지면 스냅샷을 반환한다."""

        if not isinstance(current_text, str):
            current_text = ""
        if self.last_current_text == current_text:
            return None

        first_update = self.last_current_text is None
        previous_mode = self.mode
        previous_slots = self.slots
        self.last_current_text = current_text

        if not current_text:
            self.mode = FAVORITES_MODE
            self.current_chosung = ""
            self.autocomplete_suggestions = ()
            self.last_lookup_ms = 0.0
            self.slots = _fill_slots(self.favorite_sentences)
        else:
            self.mode = AUTOCOMPLETE_MODE
            self.current_chosung = extract_chosung(_current_word(current_text))
            self.autocomplete_suggestions = ()
            self.last_lookup_ms = 0.0

            if self.current_chosung:
                started_at = perf_counter()
                try:
                    suggestions = self._suggestion_provider(
                        self.current_chosung,
                        SUGGESTION_SLOT_COUNT,
                    )
                except Exception as exc:  # 키보드는 추천 기능 오류와 격리한다.
                    warnings.warn(
                        f"autocomplete query failed: {exc}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    suggestions = []
                self.last_lookup_ms = (perf_counter() - started_at) * 1000.0
                self.autocomplete_suggestions = tuple(
                    item
                    for item in suggestions
                    if isinstance(item, str) and item
                )[:SUGGESTION_SLOT_COUNT]

            self.slots = _fill_slots(self.autocomplete_suggestions)

        if (
            not first_update
            and self.mode == previous_mode
            and self.slots == previous_slots
        ):
            return None

        return SuggestionUpdate(
            mode=self.mode,
            chosung_input=self.current_chosung,
            items=self.visible_items,
            slots=self.slots,
            elapsed_ms=self.last_lookup_ms,
        )
