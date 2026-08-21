"""추천(자동완성) 슬롯에 적용되는 고정 감지형 확장 검증.

키캡과 추천 슬롯은 같은 레이어(src/tracking/fixation.py)를 쓴다. 여기서는
"둘이 서로를 1/3까지만 침범한다", "확장된 쪽이 겹침 구간을 가져간다",
"빈 슬롯은 선택 대상도 확장 대상도 아니다" 세 가지를 고정한다.
"""

import pytest

from src.keyboard import (
    calculate_keyboard_layout,
    create_buttons,
    keys_kor_normal,
)
from src.recommendation.selection import (
    SuggestionSlot,
    SuggestionTarget,
    build_fixation_targets,
    resolve_input_target,
)
from src.tracking.fixation import FixationDetector, FixationHitbox, expanded_bounds

SCREEN_W, SCREEN_H = 1920, 1080

SUGGESTIONS = ["병원에 가고 싶어요", "물 주세요", "아파요"]


def _layout():
    return calculate_keyboard_layout(SCREEN_W, SCREEN_H)


def _setup(suggestions=None):
    layout = _layout()
    buttons = create_buttons(keys_kor_normal, layout)
    rects = layout["suggestion_rects"]
    slots = SUGGESTIONS if suggestions is None else suggestions
    targets = build_fixation_targets(rects, slots, buttons)
    return layout, buttons, rects, slots, targets


def _hitbox(expand_ratio=0.5):
    return FixationHitbox(
        detector=FixationDetector(
            px_per_deg=40.0,
            velocity_deg_per_sec=30.0,
            dispersion_deg=1.0,
            release_deg=1.5,
            min_duration_sec=0.15,
            max_gap_sec=0.3,
        ),
        expand_ratio=expand_ratio,
    )


def _hold(hitbox, targets, point, start=0.0, duration=0.4, step=0.05):
    now = start
    while now <= start + duration + 1e-9:
        hitbox.update(point[0], point[1], targets, now=now)
        now += step
    return now


# =========================================================
# 대상 목록 구성
# =========================================================

def test_only_filled_slots_become_targets():
    _, buttons, rects, _, targets = _setup(["물 주세요", "", None])

    slots = [t for t in targets if isinstance(t, SuggestionSlot)]

    assert [slot.index for slot in slots] == [0]
    # 추천이 키보다 앞 — 확장이 없을 때의 판정 순서(추천 우선)를 유지한다.
    assert targets[0] is slots[0]
    assert targets[1:] == buttons


# =========================================================
# 추천 슬롯의 확장
# =========================================================

def test_fixated_suggestion_expands_and_is_selected_outside_its_box():
    _, buttons, rects, slots, targets = _setup()
    hitbox = _hitbox()

    first = rects[0]
    _hold(hitbox, targets, first.center)

    assert isinstance(hitbox.anchor_target, SuggestionSlot)
    assert hitbox.anchor_target.index == 0

    # 슬롯 바로 아래(원래는 아무 대상도 없는 여백)가 확장으로 흡수된다.
    below = (first.center[0], first.bottom + 2)

    assert resolve_input_target(
        rects, slots, buttons, *below,
        fixation_hitbox=hitbox, fixation_targets=targets,
    ) == SuggestionTarget(index=0, text=slots[0])


def test_expanded_suggestion_wins_the_overlap_with_the_neighbour_slot():
    _, buttons, rects, slots, targets = _setup()
    hitbox = _hitbox()

    first, second = rects[0], rects[1]
    _hold(hitbox, targets, first.center)

    third = second.width / 3
    y = second.center[1]

    # 이웃 슬롯 앞쪽 1/3은 확장된 슬롯이 가져간다.
    assert hitbox.hit_test(targets, second.x + 2, y) is hitbox.anchor_target

    # 1/3을 넘으면 이웃 슬롯이 자기 영역을 지킨다.
    kept = hitbox.hit_test(targets, second.x + third + 5, y)
    assert isinstance(kept, SuggestionSlot) and kept.index == 1


def test_suggestion_and_keys_cap_each_other_at_a_third():
    """추천 슬롯과 키캡이 서로를 1/3보다 깊게 먹지 않는다."""
    _, _, _, _, targets = _setup()

    for target in targets:
        left, top, right, bottom = expanded_bounds(target, targets, 0.5)

        for other in targets:
            if other is target:
                continue

            overlap_w = min(right, other.rect.right) - max(left, other.rect.x)
            overlap_h = min(bottom, other.rect.bottom) - max(top, other.rect.y)

            if overlap_w <= 0 or overlap_h <= 0:
                continue

            other_area = other.rect.width * other.rect.height

            assert overlap_w * overlap_h <= other_area / 3 + 1


def test_key_expansion_does_not_swallow_a_filled_suggestion():
    """키를 응시 중일 때 추천 슬롯 중앙은 여전히 추천으로 잡힌다."""
    layout, buttons, rects, slots, targets = _setup()
    hitbox = _hitbox()

    # 추천 바로 아래 줄(숫자 행) 키를 응시
    number_key = min(buttons, key=lambda b: (b.rect.y, b.rect.x))
    _hold(hitbox, targets, number_key.rect.center)

    assert hitbox.anchor_target is number_key

    hit = hitbox.hit_test(targets, *rects[0].center)

    assert isinstance(hit, SuggestionSlot) and hit.index == 0


def test_empty_slot_never_anchors_the_expansion():
    _, buttons, rects, slots, targets = _setup(["", "", ""])
    hitbox = _hitbox()

    _hold(hitbox, targets, rects[1].center)

    # 빈 슬롯은 대상이 아니므로 그 자리에서는 아무것도 확장되지 않는다.
    assert not any(isinstance(t, SuggestionSlot) for t in targets)
    assert not isinstance(hitbox.anchor_target, SuggestionSlot)


# =========================================================
# 확장을 끈 경로
# =========================================================

def test_without_fixation_the_original_priority_is_unchanged():
    _, buttons, rects, slots, _ = _setup()

    assert resolve_input_target(
        rects, slots, buttons, *rects[0].center
    ) == SuggestionTarget(index=0, text=slots[0])

    below = (rects[0].center[0], rects[0].bottom + 2)
    assert resolve_input_target(rects, slots, buttons, *below) is None
