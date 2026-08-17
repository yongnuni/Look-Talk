"""키 렌더링 사각형과 시선 hit-test 경계의 일치 검증."""

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


def _button_sets(screen_w=1920, screen_h=1080):
    qwerty_layout = calculate_keyboard_layout(screen_w, screen_h)
    cheonjiin_layout = calculate_cheonjiin_layout(screen_w, screen_h)
    return {
        KEYBOARD_LAYOUT_QWERTY: create_buttons(
            keys_kor_normal,
            qwerty_layout,
        ),
        KEYBOARD_LAYOUT_CHEONJIIN: create_cheonjiin_buttons(
            cheonjiin_layout,
        ),
    }


@pytest.mark.parametrize(
    "layout_name",
    [KEYBOARD_LAYOUT_QWERTY, KEYBOARD_LAYOUT_CHEONJIIN],
)
def test_every_key_center_and_inside_corners_hit_the_rendered_key(layout_name):
    """문자 키와 모든 특수 키의 중앙·네 모서리 안쪽을 검사한다."""
    buttons = _button_sets()[layout_name]

    for button in buttons:
        rect = button.rect
        points = [
            rect.center,
            (rect.x, rect.y),
            (rect.right - 1, rect.y),
            (rect.x, rect.bottom - 1),
            (rect.right - 1, rect.bottom - 1),
        ]

        for point_x, point_y in points:
            assert hit_test_buttons(buttons, point_x, point_y) is button


@pytest.mark.parametrize(
    "layout_name,expected_keys",
    [
        (
            KEYBOARD_LAYOUT_QWERTY,
            {"Shift", "한/영", " ", "Del", "확인"},
        ),
        (
            KEYBOARD_LAYOUT_CHEONJIIN,
            {"한/영", " ", "Del", "확인"},
        ),
    ],
)
def test_required_special_keys_use_the_same_hitbox(layout_name, expected_keys):
    buttons = _button_sets()[layout_name]
    buttons_by_key = {button.text: button for button in buttons}

    assert expected_keys <= buttons_by_key.keys()
    for key in expected_keys:
        button = buttons_by_key[key]
        assert hit_test_buttons(buttons, *button.rect.center) is button


@pytest.mark.parametrize(
    "layout_name",
    [KEYBOARD_LAYOUT_QWERTY, KEYBOARD_LAYOUT_CHEONJIIN],
)
def test_every_key_boundary_just_outside_does_not_hit(layout_name):
    buttons = _button_sets()[layout_name]

    for button in buttons:
        rect = button.rect
        center_x, center_y = rect.center
        outside_points = [
            (rect.x - 1, center_y),
            (rect.right, center_y),
            (center_x, rect.y - 1),
            (center_x, rect.bottom),
        ]

        for point_x, point_y in outside_points:
            assert hit_test_buttons(buttons, point_x, point_y) is None


@pytest.mark.parametrize(
    "layout_name",
    [KEYBOARD_LAYOUT_QWERTY, KEYBOARD_LAYOUT_CHEONJIIN],
)
def test_gap_between_adjacent_keys_does_not_hit(layout_name):
    buttons = _button_sets()[layout_name]
    row_counts = {
        y: sum(button.rect.y == y for button in buttons)
        for y in {button.rect.y for button in buttons}
    }
    adjacent_row_y = max(row_counts, key=row_counts.get)
    adjacent_row = sorted(
        (button for button in buttons if button.rect.y == adjacent_row_y),
        key=lambda button: button.rect.x,
    )
    left, right = adjacent_row[:2]

    gap_x = (left.rect.right + right.rect.x) / 2
    gap_y = left.rect.center[1]

    assert left.rect.right < right.rect.x
    assert hit_test_buttons(buttons, gap_x, gap_y) is None


@pytest.mark.parametrize("screen_w,screen_h", [(1280, 720), (1920, 1080)])
@pytest.mark.parametrize(
    "layout_name",
    [KEYBOARD_LAYOUT_QWERTY, KEYBOARD_LAYOUT_CHEONJIIN],
)
def test_scaled_layout_keeps_hit_test_attached_to_button_rect(
    layout_name,
    screen_w,
    screen_h,
):
    buttons = _button_sets(screen_w, screen_h)[layout_name]

    for button in buttons:
        rect = button.rect
        assert 0 <= rect.x < rect.right <= screen_w
        assert 0 <= rect.y < rect.bottom <= screen_h
        assert hit_test_buttons(buttons, *rect.center) is button


@pytest.mark.parametrize(
    "layout_name",
    [KEYBOARD_LAYOUT_QWERTY, KEYBOARD_LAYOUT_CHEONJIIN],
)
def test_dwell_uses_exact_shared_hitbox(layout_name):
    buttons = _button_sets()[layout_name]
    button = buttons[0]
    controller = DwellController()

    hovered_key, dwell_ratio, clicked_key = controller.update(
        *button.rect.center,
        buttons,
    )
    assert (hovered_key, dwell_ratio, clicked_key) == (
        button.text,
        0.0,
        None,
    )

    hovered_key, dwell_ratio, clicked_key = controller.update(
        button.rect.right,
        button.rect.center[1],
        buttons,
    )
    assert (hovered_key, dwell_ratio, clicked_key) == (None, 0.0, None)
