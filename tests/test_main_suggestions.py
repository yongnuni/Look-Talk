"""메인 루프의 추천 상태 동기화와 렌더 순서 회귀 검사."""

from pathlib import Path


MAIN_PATH = Path(__file__).resolve().parents[1] / "main.py"


def test_recommender_is_initialized_once_and_render_order_is_preserved():
    source = MAIN_PATH.read_text(encoding="utf-8")

    assert source.count("suggestion_engine = initialize_recommender()") == 1

    composite_at = source.index("current_text = (")
    update_at = source.index("suggestion_state.update(current_text)")
    text_area_at = source.index("draw_text_area(kbd_bg, current_text, target)")
    suggestions_at = source.index("draw_suggestion_boxes(", text_area_at)
    keyboard_at = source.index("kbd_bg = drawAll(", suggestions_at)

    assert composite_at < update_at < text_area_at < suggestions_at < keyboard_at


def test_suggestion_state_is_not_duplicated_at_process_key_calls():
    source = MAIN_PATH.read_text(encoding="utf-8")

    assert source.count("suggestion_state.update(current_text)") == 1


def test_main_does_not_inject_default_favorite_sentences():
    source = MAIN_PATH.read_text(encoding="utf-8")

    assert "suggestion_state = SuggestionStateController()" in source
    assert "SuggestionStateController(favorite_sentences=" not in source
