"""추천 박스와 긴 후보 텍스트의 안전한 렌더링."""

import numpy as np
import pytest

from src.keyboard import calculate_keyboard_layout
from src.ui import (
    IDLE_BG,
    _fit_suggestion_text,
    draw_suggestion_boxes,
)


@pytest.mark.parametrize("candidate_count", [0, 1, 2, 3])
def test_zero_to_three_candidates_render_three_boxes(candidate_count):
    layout = calculate_keyboard_layout(1280, 720)
    rects = layout["suggestion_rects"]
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    suggestions = ("감사합니다", "안녕하세요", "도와주세요")[:candidate_count]

    rendered = draw_suggestion_boxes(canvas, suggestions, rects)

    assert rendered.shape == canvas.shape
    for rect in rects:
        sample = rendered[
            rect.y + rect.height // 3,
            rect.x + rect.width // 2,
        ]
        assert np.any(sample != 0)

    if candidate_count == 0:
        first = rects[0]
        center = rendered[int(first.center[1]), int(first.center[0])]
        assert tuple(center) == IDLE_BG


def test_short_text_uses_one_line_without_mutating_source():
    source = "감사합니다"

    display, text_font, _, width, height, _ = _fit_suggestion_text(
        source,
        388,
        79,
    )

    assert display == source
    assert "\n" not in display
    assert source == "감사합니다"
    assert text_font.size <= int(79 * 0.30)
    assert width < 388
    assert height < 79


def test_long_text_uses_at_most_two_lines_or_readable_ellipsis():
    source = "의사 선생님 불러주세요 " * 20

    display, text_font, _, width, height, _ = _fit_suggestion_text(
        source,
        388,
        79,
    )

    assert display.count("\n") <= 1
    assert display.endswith("…") or "\n" in display
    assert text_font.size >= max(12, int(79 * 0.14))
    assert width <= 388 - 2 * max(8, int(388 * 0.04))
    assert height <= 79 - 2 * max(4, int(79 * 0.10))
    assert source == "의사 선생님 불러주세요 " * 20


def test_text_layout_scales_with_resolution():
    text = "잠시만 기다려주세요"
    small = _fit_suggestion_text(text, 388, 79)
    large = _fit_suggestion_text(text, 582, 118)

    assert large[1].size >= small[1].size
