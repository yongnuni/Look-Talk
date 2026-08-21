"""고정(fixation) 감지 기반 히트박스 확장 검증.

핵심 불변식 세 가지를 고정한다.
1. 고정이 성립하지 않으면 기존 판정과 결과가 완전히 같다.
2. 확장은 인접 키를 최대 1/3까지만 덮고, 겹치는 구간에서는 확장된 쪽이 이긴다.
3. 확장은 판정 영역에만 존재한다 — 키캡의 위치·크기는 정적으로 유지된다.
"""

import pytest

from src.keyboard import (
    KEYBOARD_LAYOUT_CHEONJIIN,
    KEYBOARD_LAYOUT_QWERTY,
    calculate_cheonjiin_layout,
    calculate_keyboard_layout,
    create_buttons,
    create_cheonjiin_buttons,
    hit_test_buttons,
    keys_kor_normal,
)
from src.tracking.dwell import DwellController
from src.tracking.fixation import (
    FixationDetector,
    FixationHitbox,
    expanded_bounds,
)

# 1° = 40px 인 가상의 화면. 임계값이 px로 딱 떨어져 검증이 명확해진다.
PX_PER_DEG = 40.0

DISPERSION_PX = PX_PER_DEG * 1.0     # 고정 성립 반경
RELEASE_PX = PX_PER_DEG * 1.5        # 고정 해제 반경
MIN_DURATION_SEC = 0.15


def _detector():
    return FixationDetector(
        px_per_deg=PX_PER_DEG,
        velocity_deg_per_sec=30.0,     # = 1200 px/s
        dispersion_deg=1.0,
        release_deg=1.5,
        min_duration_sec=MIN_DURATION_SEC,
        max_gap_sec=0.3,
    )


def _hitbox(expand_ratio=0.5, enabled=True):
    return FixationHitbox(
        enabled=enabled,
        detector=_detector(),
        expand_ratio=expand_ratio,
    )


def _buttons(screen_w=1920, screen_h=1080):
    return create_buttons(
        keys_kor_normal,
        calculate_keyboard_layout(screen_w, screen_h),
    )


def _layout_buttons(layout_name, screen_w=1920, screen_h=1080):
    if layout_name == KEYBOARD_LAYOUT_CHEONJIIN:
        return create_cheonjiin_buttons(
            calculate_cheonjiin_layout(screen_w, screen_h),
        )
    return _buttons(screen_w, screen_h)


def _adjacent_pair(buttons):
    """같은 행에서 좌우로 맞닿은 두 키와 그 사이 여백 좌표."""
    row_y = max(
        {button.rect.y for button in buttons},
        key=lambda y: sum(button.rect.y == y for button in buttons),
    )
    row = sorted(
        (button for button in buttons if button.rect.y == row_y),
        key=lambda button: button.rect.x,
    )
    left, right = row[0], row[1]
    gap_point = (
        (left.rect.right + right.rect.x) / 2,
        left.rect.center[1],
    )
    return left, right, gap_point


def _hold(hitbox, buttons, point, start=0.0, duration=0.4, step=0.05):
    """한 점을 계속 응시시켜 고정을 성립시킨다."""
    now = start
    end = start + duration
    state = None

    while now <= end + 1e-9:
        state = hitbox.update(point[0], point[1], buttons, now=now)
        now += step

    return state, now


# =========================================================
# FixationDetector — I-VT / I-DT
# =========================================================

def test_stationary_gaze_becomes_fixation_after_min_duration():
    detector = _detector()

    assert not detector.update(500, 500, now=0.0).active

    # 최소 지속 시간 직전까지는 아직 고정이 아니다.
    assert not detector.update(500, 500, now=MIN_DURATION_SEC - 0.01).active

    assert detector.update(500, 500, now=MIN_DURATION_SEC).active


def test_saccade_velocity_breaks_fixation():
    detector = _detector()

    detector.update(500, 500, now=0.0)
    assert detector.update(500, 500, now=0.2).active

    # 0.05초에 100px 이동 = 2000px/s = 50°/s > 30°/s 임계값
    assert not detector.update(600, 500, now=0.25).active


def test_slow_drift_beyond_release_radius_breaks_fixation():
    detector = _detector()

    detector.update(500, 500, now=0.0)
    assert detector.update(500, 500, now=0.2).active

    # 속도는 임계값 아래지만(20px/0.05s = 400px/s = 10°/s)
    # 해제 반경을 넘길 때까지 표류하면 고정이 끊긴다.
    x = 500
    now = 0.2
    broke = False

    while x < 500 + RELEASE_PX * 2:
        x += 20
        now += 0.05
        if not detector.update(x, 500, now=now).active:
            broke = True
            break

    assert broke


def test_microtremor_inside_dispersion_keeps_fixation():
    detector = _detector()

    now = 0.0
    detector.update(500, 500, now=now)

    for offset in [3, -4, 2, -3, 5, -2, 4, -5]:
        now += 0.05
        state = detector.update(500 + offset, 500 + offset, now=now)

    assert state.active
    assert state.duration_sec >= MIN_DURATION_SEC


def test_frame_gap_restarts_instead_of_faking_a_slow_move():
    detector = _detector()

    detector.update(500, 500, now=0.0)
    assert detector.update(500, 500, now=0.2).active

    # 1초 공백 뒤 멀리 떨어진 좌표 — 큰 dt로 나누면 느린 이동처럼 보이지만
    # 연속된 시선으로 이어 붙이지 않고 새로 시작해야 한다.
    assert not detector.update(1200, 900, now=1.2).active


def test_invalid_tracking_clears_fixation():
    detector = _detector()

    detector.update(500, 500, now=0.0)
    assert detector.update(500, 500, now=0.2).active

    assert not detector.update(500, 500, valid=False, now=0.25).active
    assert not detector.update(-1, -1, now=0.3).active


# =========================================================
# FixationHitbox — 확장 판정
# =========================================================

def test_gap_between_keys_is_dead_until_fixation_starts():
    buttons = _buttons()
    _, _, gap_point = _adjacent_pair(buttons)
    hitbox = _hitbox()

    # 탐색(도약) 중에는 확장이 걸리지 않는다 → 기존 판정과 동일
    hitbox.update(gap_point[0], gap_point[1], buttons, now=0.0)

    assert hitbox.hit_test(buttons, *gap_point) is None
    assert hit_test_buttons(buttons, *gap_point) is None


def test_fixated_key_absorbs_the_neighbouring_gap():
    buttons = _buttons()
    left, _, gap_point = _adjacent_pair(buttons)
    hitbox = _hitbox()

    _hold(hitbox, buttons, left.rect.center)

    assert hitbox.active
    assert hitbox.anchor_target is left
    # 확장 전에는 어떤 키도 아니던 여백이 고정된 키로 흡수된다.
    assert hit_test_buttons(buttons, *gap_point) is None
    assert hitbox.hit_test(buttons, *gap_point) is left


def test_expansion_wins_inside_the_overlap_and_stops_at_a_third():
    buttons = _buttons()
    left, right, _ = _adjacent_pair(buttons)
    hitbox = _hitbox()

    _hold(hitbox, buttons, left.rect.center)
    assert hitbox.anchor_target is left

    third = right.rect.width / 3
    y = right.rect.center[1]

    # 겹치는 구간(인접 키의 앞쪽 1/3)은 확장된 키가 가져간다.
    assert hitbox.hit_test(buttons, right.rect.x, y) is left
    assert hitbox.hit_test(buttons, right.rect.x + third - 2, y) is left

    # 1/3을 넘어서면 인접 키가 자기 영역을 그대로 지킨다.
    assert hitbox.hit_test(buttons, right.rect.x + third + 2, y) is right
    assert hitbox.hit_test(buttons, *right.rect.center) is right


@pytest.mark.parametrize("layout_name", [KEYBOARD_LAYOUT_QWERTY, KEYBOARD_LAYOUT_CHEONJIIN])
def test_no_key_is_covered_more_than_a_third(layout_name):
    """어떤 키도 자기 면적의 1/3 넘게 다른 키의 확장에 먹히지 않는다."""
    buttons = _layout_buttons(layout_name)

    for button in buttons:
        left, top, right, bottom = expanded_bounds(button, buttons, 0.5)

        for other in buttons:
            if other is button:
                continue

            overlap_w = min(right, other.rect.right) - max(left, other.rect.x)
            overlap_h = min(bottom, other.rect.bottom) - max(top, other.rect.y)

            if overlap_w <= 0 or overlap_h <= 0:
                continue

            other_area = other.rect.width * other.rect.height

            assert overlap_w * overlap_h <= other_area / 3 + 1


def test_new_fixation_inside_the_overlap_moves_the_expansion():
    """겹침 구간이라도 그쪽에 새 고정이 서면 확장이 넘어간다(고착 방지)."""
    buttons = _buttons()
    left, right, _ = _adjacent_pair(buttons)
    hitbox = _hitbox()

    _, now = _hold(hitbox, buttons, left.rect.center)
    assert hitbox.anchor_target is left

    overlap_point = (
        right.rect.x + right.rect.width / 6,
        right.rect.center[1],
    )

    # 옮겨간 직후에는 아직 이전 확장이 이긴다.
    hitbox.update(overlap_point[0], overlap_point[1], buttons, now=now + 0.05)
    assert hitbox.hit_test(buttons, *overlap_point) is left

    # 그 자리에 고정이 성립하면 확장이 그 키로 넘어간다.
    _hold(hitbox, buttons, overlap_point, start=now + 0.10)

    assert hitbox.anchor_target is right
    assert hitbox.hit_test(buttons, *overlap_point) is right


def test_fixation_centre_in_a_gap_anchors_to_the_nearest_key():
    buttons = _buttons()
    left, _, gap_point = _adjacent_pair(buttons)
    hitbox = _hitbox()

    # 시선이 키 사이 여백에 멈춘 경우 — 이 기능이 실제로 필요한 상황
    gap_point_near_left = (left.rect.right + 1, left.rect.center[1])
    _hold(hitbox, buttons, gap_point_near_left)

    assert hitbox.active
    assert hitbox.anchor_target is left
    assert hitbox.hit_test(buttons, *gap_point) is left


def test_expansion_survives_gaze_drifting_to_the_key_edge():
    """각도 기준 고정 반경보다 키 반폭이 커도 확장이 풀리지 않아야 한다.

    가장자리로 흐르는 순간이 확장이 가장 필요한 순간이다.
    """
    buttons = _buttons()
    left, _, gap_point = _adjacent_pair(buttons)
    hitbox = _hitbox()

    _, now = _hold(hitbox, buttons, left.rect.center)
    assert (left.rect.width / 2) > RELEASE_PX   # 반폭 > 고정 해제 반경

    hitbox.update(gap_point[0], gap_point[1], buttons, now=now)

    assert hitbox.anchor_target is left
    assert hitbox.hit_test(buttons, *gap_point) is left


def test_looking_at_another_key_moves_the_expansion_there():
    buttons = _buttons()
    left, right, gap_point = _adjacent_pair(buttons)
    hitbox = _hitbox()

    _, now = _hold(hitbox, buttons, left.rect.center)
    assert hitbox.anchor_target is left

    # 다른 키의 실제 사각형을 보기 시작하면 이전 확장은 즉시 내려간다.
    hitbox.update(*right.rect.center, buttons, now=now)
    assert hitbox.anchor_target is None

    # 그 키에 고정이 성립하면 확장이 그쪽으로 옮겨 간다.
    _hold(hitbox, buttons, right.rect.center, start=now + 0.05)

    assert hitbox.anchor_target is right
    assert hitbox.hit_test(buttons, *gap_point) is right


def test_personal_radius_sets_a_floor_for_the_expansion():
    """캘리브레이션에서 얻은 개인별 오차 반경을 최소 확장량으로 쓸 수 있다."""
    buttons = _buttons()
    top_left = buttons[0]      # 숫자 행 왼쪽 끝 — 위쪽은 빈 공간

    hitbox = FixationHitbox(
        detector=_detector(),
        expand_ratio=0.0,          # 키 크기 기반 확장을 끈 상태
        personal_radius_px=30,
    )

    _hold(hitbox, buttons, top_left.rect.center)
    assert hitbox.anchor_target is top_left

    center_x = top_left.rect.center[0]

    assert hitbox.hit_test(buttons, center_x, top_left.rect.y - 20) is top_left
    assert hitbox.hit_test(buttons, center_x, top_left.rect.y - 45) is None


def test_far_outside_the_expansion_stays_unselected():
    buttons = _buttons()
    left, _, _ = _adjacent_pair(buttons)
    hitbox = _hitbox()

    _hold(hitbox, buttons, left.rect.center)

    far_point = (
        left.rect.center[0],
        left.rect.y - left.rect.height * 2,
    )

    assert hitbox.hit_test(buttons, *far_point) is None


def test_expansion_scales_with_the_key_size():
    """확장량은 고정 px가 아니라 키 크기에서 나온다."""
    small_buttons = _buttons(1280, 720)
    large_buttons = _buttons(1920, 1080)

    def _margin(buttons):
        hitbox = _hitbox()
        button = buttons[0]
        _hold(hitbox, buttons, button.rect.center)
        rect = hitbox.anchor_debug_rect()
        return button.rect.x - rect[0]

    small_margin = _margin(small_buttons)
    large_margin = _margin(large_buttons)

    assert small_margin == pytest.approx(
        small_buttons[0].rect.width * 0.5, abs=1
    )
    assert large_margin == pytest.approx(
        large_buttons[0].rect.width * 0.5, abs=1
    )
    assert large_margin > small_margin


def test_anchor_survives_target_list_rebuild():
    """Shift/한영 전환으로 buttonList가 새로 생성돼도 확장이 유지된다."""
    buttons = _buttons()
    left, _, gap_point = _adjacent_pair(buttons)
    hitbox = _hitbox()

    state, now = _hold(hitbox, buttons, left.rect.center)
    assert hitbox.active

    rebuilt = _buttons()
    assert all(a is not b for a, b in zip(buttons, rebuilt))

    hitbox.update(
        left.rect.center[0],
        left.rect.center[1],
        rebuilt,
        now=now,
    )

    assert hitbox.anchor_target in rebuilt
    assert hitbox.hit_test(rebuilt, *gap_point) is rebuilt[
        buttons.index(left)
    ]


def test_disabled_hitbox_behaves_exactly_like_before():
    buttons = _buttons()
    left, _, gap_point = _adjacent_pair(buttons)
    hitbox = _hitbox(enabled=False)

    _hold(hitbox, buttons, left.rect.center)

    assert not hitbox.active
    assert hitbox.hit_test(buttons, *gap_point) is None
    assert hitbox.hit_test(buttons, *left.rect.center) is left


# =========================================================
# 시각 확대 — 표시 전용
# =========================================================

def test_visual_rect_starts_at_the_real_keycap_and_grows():
    """확대는 실제 키캡 크기에서 시작해 짧은 ease-out으로 커진다."""
    buttons = _buttons()
    left, _, _ = _adjacent_pair(buttons)
    hitbox = _hitbox()

    now = 0.0
    while hitbox.anchor_target is None and now < 1.0:
        hitbox.update(*left.rect.center, buttons, now=now)
        now += 0.05

    rect = left.rect
    assert hitbox.anchor_visual_rect() == (
        rect.x,
        rect.y,
        rect.right,
        rect.bottom,
    )

    hitbox.update(*left.rect.center, buttons, now=now + 0.5)
    grown = hitbox.anchor_visual_rect()

    assert grown[0] < rect.x and grown[2] > rect.right
    assert grown[1] < rect.y and grown[3] > rect.bottom


def test_visual_expansion_stays_inside_the_hitbox():
    """보이는 것보다 판정이 조금 더 너그럽다(암묵 확장)."""
    buttons = _buttons()
    left, _, _ = _adjacent_pair(buttons)
    hitbox = _hitbox()

    _hold(hitbox, buttons, left.rect.center)

    visual = hitbox.anchor_visual_rect()
    hit = hitbox.anchor_debug_rect()

    assert hit[0] <= visual[0] and hit[1] <= visual[1]
    assert visual[2] <= hit[2] and visual[3] <= hit[3]


def test_visual_expansion_can_be_disabled_without_touching_the_hitbox():
    buttons = _buttons()
    left, _, gap_point = _adjacent_pair(buttons)

    hitbox = FixationHitbox(
        detector=_detector(),
        expand_ratio=0.5,
        visual_enabled=False,
    )

    _hold(hitbox, buttons, left.rect.center)

    assert hitbox.anchor_visual_rect() is None
    assert hitbox.hit_test(buttons, *gap_point) is left


# =========================================================
# 공용 경로 — 세 트리거가 함께 쓰는 DwellController
# =========================================================

def test_dwell_controller_shares_the_expanded_hitbox():
    buttons = _buttons()
    left, _, gap_point = _adjacent_pair(buttons)

    controller = DwellController(fixation_hitbox=_hitbox())

    center_x, center_y = left.rect.center
    now = 0.0

    for _ in range(10):
        hovered_key, _, _ = controller.update(
            center_x, center_y, buttons, now=now
        )
        now += 0.05

    assert hovered_key == left.text
    assert controller.fixation_state.active

    # 시선이 키 사이 여백으로 흘러도 hover가 끊기지 않는다.
    hovered_key, _, _ = controller.update(*gap_point, buttons, now=now)

    assert hovered_key == left.text


def test_dwell_controller_without_fixation_keeps_gap_dead():
    buttons = _buttons()
    _, _, gap_point = _adjacent_pair(buttons)

    controller = DwellController(fixation_hitbox=_hitbox(enabled=False))

    now = 0.0
    for _ in range(10):
        hovered_key, _, _ = controller.update(*gap_point, buttons, now=now)
        now += 0.05

    assert hovered_key is None
