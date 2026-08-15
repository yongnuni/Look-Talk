"""process_key()의 확장 반환값(deleted_count, inserted_text) 검증.

핵심 목적: 이 코드베이스의 한글/천지인 조합은 "확정처럼 보이던 문자가
다음 입력에서 소급 재해석/치환된다"(Phase A 조사 결과). tap마다 남기는
꼬리 diff(deleted_count/inserted_text)가 이런 소급 변경까지 포함해
정확히 재구성 가능한지가 이번 검증의 핵심이다 - 잘못되면 CPM/오타율이
처음부터 오염된다.
"""

import pytest

import src.hangul as hangul
from src.cheonjiin import cheonjiin_composer
from src.keyboard import (
    process_key,
    _diff_tail,
    KEYBOARD_LAYOUT_QWERTY,
    KEYBOARD_LAYOUT_CHEONJIIN,
    keys_kor_normal,
    create_buttons,
)


@pytest.fixture(autouse=True)
def _reset_composition_state():
    hangul.finalText = ""
    hangul.jamo_buffer[:] = ['', '', '']
    cheonjiin_composer.reset()
    yield
    hangul.finalText = ""
    hangul.jamo_buffer[:] = ['', '', '']
    cheonjiin_composer.reset()


def _composite():
    return hangul.finalText + hangul.compose_jamo_buffer()


def _cheonjiin_composite():
    return (
        hangul.finalText
        + hangul.compose_jamo_buffer()
        + cheonjiin_composer.get_pending_preview()
    )


def _replay(diffs):
    """tap_commit 행들의 (deleted_count, inserted_text) 시퀀스를 재생해
    최종 문자열을 복원한다. InputEventLogger가 기록하는 형식 그대로다.
    """
    composite = ""
    for deleted_count, inserted_text in diffs:
        if deleted_count > 0:
            composite = (
                composite[:-deleted_count]
                if deleted_count <= len(composite)
                else ""
            )
        composite += inserted_text
    return composite


# ── _diff_tail 자체 (순수 함수) ──────────────────────────────────


def test_diff_tail_pure_append():
    assert _diff_tail("가", "가나") == (0, "나")


def test_diff_tail_pure_full_replace():
    assert _diff_tail("간", "가나") == (1, "가나")


def test_diff_tail_pure_no_change():
    assert _diff_tail("가나", "가나") == (0, "")


# ── QWERTY: 종성 소급 재해석 (hangul.py:151-182) ──────────────────


def test_qwerty_diff_basic_append():
    button_list = create_buttons(keys_kor_normal)

    _, _, _, deleted, inserted = process_key(
        "ㄱ", True, False, button_list, KEYBOARD_LAYOUT_QWERTY
    )

    assert (deleted, inserted) == (0, "ㄱ")
    assert _composite() == "ㄱ"


def test_qwerty_retroactive_jongsung_reinterpretation():
    """"ㄱㅏㄴㅏ" 입력: 3번째 탭까지는 "간"(종성 확정처럼 보임)이지만,
    4번째 탭(ㅏ)이 오는 순간 "이 ㄴ은 종성이 아니라 다음 음절의 초성"으로
    재해석되어 "가"+"나"로 갈라진다. diff가 이를 정확히 표현하고, replay가
    최종 문자열("가나")을 정확히 복원해야 한다.
    """
    button_list = create_buttons(keys_kor_normal)
    is_korean, is_shift = True, False
    diffs = []

    for key in ["ㄱ", "ㅏ", "ㄴ", "ㅏ"]:
        is_korean, is_shift, button_list, deleted, inserted = process_key(
            key, is_korean, is_shift, button_list, KEYBOARD_LAYOUT_QWERTY
        )
        diffs.append((deleted, inserted))

    assert _composite() == "가나"
    assert diffs[2] == (1, "간")   # "가" -> "간" (아직 확정 아님)
    assert diffs[3] == (1, "가나")  # "간" -> "가"+"나" 로 소급 재해석
    assert _replay(diffs) == _composite() == "가나"


def test_qwerty_backspace_diff_unwinds_one_jamo_at_a_time():
    button_list = create_buttons(keys_kor_normal)
    is_korean, is_shift = True, False
    diffs = []

    for key in ["ㄱ", "ㅏ", "Del", "Del"]:
        is_korean, is_shift, button_list, deleted, inserted = process_key(
            key, is_korean, is_shift, button_list, KEYBOARD_LAYOUT_QWERTY
        )
        diffs.append((deleted, inserted))

    assert _composite() == ""
    assert diffs == [
        (0, "ㄱ"),
        (1, "가"),
        (1, "ㄱ"),
        (1, ""),
    ]
    assert _replay(diffs) == _composite() == ""


# ── 천지인: 모음 소급 치환 (cheonjiin.py:_replace_last_vowel) ─────


def test_cheonjiin_retroactive_vowel_replacement_in_final_text():
    """'ㅣ' -> 'ㅡ' -> 'ㆍ' 순서로 입력.

    2번째 탭에서 hangul.flush_buffer()가 (jamo_buffer[0]이 비어있어)
    독립 모음 'ㅣ'를 finalText로 옮기지 못하고 버리는 채로 'ㅡ'가
    finalText에 들어간다(hangul.py flush_buffer의 비대칭 동작 - 실제
    코드 그대로 관찰된 현상이다). 3번째 탭('ㆍ')은 'ㅡ'+'ㆍ' 조합이
    'ㅜ'로 완성되는 순간, 이미 finalText에 들어간 'ㅡ'를 직접 잘라내고
    'ㅜ'로 치환한다(_replace_last_vowel의 finalText.endswith 분기).
    diff/replay가 이 finalText 자체의 소급 변경까지 정확히 잡아내야 한다.
    """
    button_list = create_buttons(keys_kor_normal)
    is_korean, is_shift = True, False
    diffs = []

    for key in ["ㅣ", "ㅡ", "ㆍ"]:
        is_korean, is_shift, button_list, deleted, inserted = process_key(
            key, is_korean, is_shift, button_list, KEYBOARD_LAYOUT_CHEONJIIN
        )
        diffs.append((deleted, inserted))

    assert hangul.finalText == "ㅜ"
    assert diffs[2] == (1, "ㅜ")  # finalText 끝의 'ㅡ'가 'ㅜ'로 소급 치환
    assert _replay(diffs) == _cheonjiin_composite() == "ㅜ"


def test_cheonjiin_pending_vowel_preview_included_in_diff():
    """완성되지 않은 모음 요소(ㆍ 단독)는 pending preview로만 표시된다.
    composite에 preview를 포함하지 않으면 이 탭의 diff가 빈 값이 되어
    탭 자체가 기록에서 유실된다 - 그래서 process_key의 _composite_text가
    get_pending_preview()를 포함해야 한다.
    """
    button_list = create_buttons(keys_kor_normal)

    _, _, _, deleted, inserted = process_key(
        "ㆍ", True, False, button_list, KEYBOARD_LAYOUT_CHEONJIIN
    )

    assert (deleted, inserted) == (0, "ㆍ")
    assert _cheonjiin_composite() == "ㆍ"
