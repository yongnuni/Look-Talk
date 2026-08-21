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


@dataclass(frozen=True, slots=True)
class SuggestionSlot:
    """고정 감지 레이어에 넘기는 추천 슬롯.

    fixation 레이어는 rect만 요구하므로, 키캡 버튼과 똑같은 규칙으로 확장·
    hit-test된다. 선택 결과로는 기존 SuggestionTarget을 그대로 돌려준다.
    """

    index: int
    text: str
    rect: object

    @property
    def target(self) -> SuggestionTarget:
        return SuggestionTarget(index=self.index, text=self.text)


def build_fixation_targets(
    suggestion_rects,
    suggestions: Sequence[str],
    button_list,
):
    """이번 프레임의 선택 대상 전체를 고정 레이어용 목록으로 만듭니다.

    추천을 앞에 둬서, 확장이 걸리지 않은 평상시 판정 순서가 기존
    resolve_input_target(추천 우선)과 같게 유지됩니다. 빈 추천 슬롯은 아예
    목록에 넣지 않습니다 — 선택 대상이 아니고, 이웃 확장을 가로막지도
    않아야 하기 때문입니다.
    """

    targets = []

    for index, rect in enumerate(suggestion_rects):
        text = suggestions[index] if index < len(suggestions) else ""

        if not isinstance(text, str) or not text:
            continue

        targets.append(
            SuggestionSlot(index=index, text=text, rect=rect)
        )

    targets.extend(button_list)

    return targets


def to_input_target(hit):
    """고정 레이어가 돌려준 대상을 기존 입력 대상 표현으로 바꿉니다."""

    if hit is None:
        return None

    if isinstance(hit, SuggestionSlot):
        return hit.target

    return hit.text


def resolve_input_target(
    suggestion_rects,
    suggestions: Sequence[str],
    button_list,
    point_x,
    point_y,
    fixation_hitbox=None,
    fixation_targets=None,
):
    """추천과 키를 중복시키지 않고 현재 프레임의 대상 하나를 반환합니다.

    추천 영역이 키와 실수로 겹치는 레이아웃에서도 내용이 있는 추천을 먼저
    반환합니다. 빈 추천 슬롯은 입력을 가로채지 않습니다.

    fixation_hitbox를 넘기면 추천 슬롯과 키 양쪽에 고정 감지형 확장이
    적용됩니다(src/tracking/fixation.py). 이때 확장이 걸린 대상이 겹침 구간의
    우선권을 갖고, 확장이 없을 때의 판정 순서는 추천 우선 그대로입니다.
    fixation_targets를 함께 넘기면 그 목록을 재사용합니다(프레임당 1회 생성).
    """

    if fixation_hitbox is not None:
        targets = (
            build_fixation_targets(
                suggestion_rects,
                suggestions,
                button_list,
            )
            if fixation_targets is None
            else fixation_targets
        )

        return to_input_target(
            fixation_hitbox.hit_test(targets, point_x, point_y)
        )

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
