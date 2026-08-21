"""완성형 한글 분해와 초성 추출 유틸리티 검증."""

import pytest

from src.recommendation import (
    decompose_hangul_syllable,
    extract_chosung,
)
from src.recommendation.chosung import CHOSUNG, JONGSUNG, JUNGSUNG


@pytest.mark.parametrize(
    ("char", "expected"),
    [
        ("가", ("ㄱ", "ㅏ", None)),
        ("각", ("ㄱ", "ㅏ", "ㄱ")),
        ("까", ("ㄲ", "ㅏ", None)),
        ("힣", ("ㅎ", "ㅣ", "ㅎ")),
    ],
)
def test_decompose_hangul_syllable(char, expected):
    assert decompose_hangul_syllable(char) == expected


def test_hangul_component_tables_are_complete():
    assert CHOSUNG == tuple("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
    assert JUNGSUNG == tuple("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
    assert JONGSUNG == (
        None,
        "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ",
        "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ",
        "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
    )
    assert (len(CHOSUNG), len(JUNGSUNG), len(JONGSUNG)) == (19, 21, 28)


@pytest.mark.parametrize(
    "invalid_value",
    ["", "가나", "A", "1", " ", "ㄱ", None, 123],
)
def test_decompose_hangul_syllable_returns_none_for_invalid_input(
    invalid_value,
):
    assert decompose_hangul_syllable(invalid_value) is None


def test_extract_chosung_from_precomposed_hangul():
    assert extract_chosung("한국") == "ㅎㄱ"


def test_extract_chosung_preserves_all_independent_initials():
    independent_initials = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"

    assert extract_chosung(independent_initials) == independent_initials


@pytest.mark.parametrize(
    "text",
    ["", "english", "123456", "   ", "english 123"],
)
def test_extract_chosung_returns_empty_for_unsupported_input(text):
    assert extract_chosung(text) == ""


def test_extract_chosung_skips_unsupported_characters_in_mixed_text():
    assert extract_chosung("A한1 국!ㄱㅏ") == "ㅎㄱㄱ"


@pytest.mark.parametrize("invalid_value", [None, 123])
def test_extract_chosung_returns_empty_for_non_string_input(invalid_value):
    assert extract_chosung(invalid_value) == ""


def test_hangul_boundary_values_are_inclusive():
    assert decompose_hangul_syllable("가") == ("ㄱ", "ㅏ", None)
    assert decompose_hangul_syllable("힣") == ("ㅎ", "ㅣ", "ㅎ")
