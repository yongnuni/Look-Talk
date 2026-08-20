"""독립 추천 hitbox와 기존 키 중 프레임당 하나의 입력 대상을 결정합니다."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.keyboard import hit_test_buttons, hit_test_suggestions


@dataclass(frozen=True, slots=True)
class SuggestionTarget:
    """선택 시점의 슬롯과 후보 문장을 함께 고정한 추천 대상입니다."""

    index: int
    text: str

    @property
    def key_id(self) -> str:
        """기존 input-events CSV의 key_id에 기록할 안정적인 식별자입니다."""

        return f"suggestion_{self.index + 1}"


def hit_test_suggestion_target(
    suggestion_rects,
    suggestions: Sequence[str],
    point_x,
    point_y,
) -> SuggestionTarget | None:
    """내용이 있는 추천 슬롯만 선택 가능한 대상으로 반환합니다."""

    index = hit_test_suggestions(
        suggestion_rects,
        point_x,
        point_y,
    )
    if index is None or index >= len(suggestions):
        return None

    text = suggestions[index]
    if not isinstance(text, str) or not text:
        return None

    return SuggestionTarget(index=index, text=text)


def resolve_input_target(
    suggestion_rects,
    suggestions: Sequence[str],
    button_list,
    point_x,
    point_y,
):
    """추천과 키를 중복시키지 않고 현재 프레임의 대상 하나를 반환합니다.

    추천 영역이 키와 실수로 겹치는 레이아웃에서도 내용이 있는 추천을 먼저
    반환합니다. 빈 추천 슬롯은 입력을 가로채지 않습니다.
    """

    suggestion_target = hit_test_suggestion_target(
        suggestion_rects,
        suggestions,
        point_x,
        point_y,
    )
    if suggestion_target is not None:
        return suggestion_target

    button = hit_test_buttons(button_list, point_x, point_y)
    return button.text if button is not None else None


def target_log_id(target):
    """세션 로그의 기존 hovered_key/clicked_key 필드용 식별자입니다."""

    if isinstance(target, SuggestionTarget):
        return target.key_id
    return target
