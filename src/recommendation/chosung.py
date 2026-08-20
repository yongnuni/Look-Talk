"""완성형 한글 분해와 초성 추출을 위한 순수 유틸리티."""

CHOSUNG = (
    "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
)

JUNGSUNG = (
    "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ",
    "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ",
    "ㅣ",
)

# 완성형 한글의 종성 인덱스 0은 종성이 없음을 뜻한다.
JONGSUNG = (
    None,
    "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ",
    "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ",
    "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
)

HANGUL_SYLLABLE_BASE = 0xAC00
HANGUL_SYLLABLE_END = 0xD7A3
JUNGSUNG_COUNT = 21
JONGSUNG_COUNT = 28
SYLLABLES_PER_CHOSUNG = JUNGSUNG_COUNT * JONGSUNG_COUNT

_CHOSUNG_SET = frozenset(CHOSUNG)


def decompose_hangul_syllable(
    char: str,
) -> tuple[str, str, str | None] | None:
    """완성형 한글 한 글자를 ``(초성, 중성, 종성)``으로 분해한다.

    종성이 없으면 세 번째 값으로 ``None``을 반환한다. 문자열이 아니거나,
    길이가 1이 아니거나, ``가``~``힣`` 범위 밖이면 예외 대신 ``None``을
    반환한다.
    """

    if not isinstance(char, str) or len(char) != 1:
        return None

    code_point = ord(char)
    if not HANGUL_SYLLABLE_BASE <= code_point <= HANGUL_SYLLABLE_END:
        return None

    syllable_index = code_point - HANGUL_SYLLABLE_BASE
    chosung_index = syllable_index // SYLLABLES_PER_CHOSUNG
    jungsung_index = (
        syllable_index % SYLLABLES_PER_CHOSUNG
    ) // JONGSUNG_COUNT
    jongsung_index = syllable_index % JONGSUNG_COUNT

    return (
        CHOSUNG[chosung_index],
        JUNGSUNG[jungsung_index],
        JONGSUNG[jongsung_index],
    )


def extract_chosung(text: str) -> str:
    """문자열의 완성형 한글과 독립 초성에서 초성 시퀀스를 추출한다.

    19개 독립 초성은 그대로 유지한다. 영문, 숫자, 공백, 모음 및 지원하지
    않는 문자는 건너뛴다. 문자열이 아니거나 빈 문자열이면 ``""``을
    반환한다.
    """

    if not isinstance(text, str) or not text:
        return ""

    extracted = []

    for char in text:
        if char in _CHOSUNG_SET:
            extracted.append(char)
            continue

        decomposed = decompose_hangul_syllable(char)
        if decomposed is not None:
            extracted.append(decomposed[0])

    return "".join(extracted)
