import time
from typing import Optional

import src.hangul as hangul


# ── 천지인 자음 연타 순환 ──────────────────────────────────────
#
# 초성에서는 된소리까지 순환한다.
# 종성에서는 실제 종성으로 사용할 수 없는 ㄸ, ㅃ, ㅉ을 제외한다.

INITIAL_CONSONANT_CYCLES = {
    "ㄱㅋ": ("ㄱ", "ㅋ", "ㄲ"),
    "ㄴㄹ": ("ㄴ", "ㄹ"),
    "ㄷㅌ": ("ㄷ", "ㅌ", "ㄸ"),
    "ㅂㅍ": ("ㅂ", "ㅍ", "ㅃ"),
    "ㅅㅎ": ("ㅅ", "ㅎ", "ㅆ"),
    "ㅈㅊ": ("ㅈ", "ㅊ", "ㅉ"),
    "ㅇㅁ": ("ㅇ", "ㅁ"),
}

FINAL_CONSONANT_CYCLES = {
    "ㄱㅋ": ("ㄱ", "ㅋ", "ㄲ"),
    "ㄴㄹ": ("ㄴ", "ㄹ"),
    "ㄷㅌ": ("ㄷ", "ㅌ"),
    "ㅂㅍ": ("ㅂ", "ㅍ"),
    "ㅅㅎ": ("ㅅ", "ㅎ", "ㅆ"),
    "ㅈㅊ": ("ㅈ", "ㅊ"),
    "ㅇㅁ": ("ㅇ", "ㅁ"),
}


# ── 천지인 모음 조합 ───────────────────────────────────────────
#
# ㅏ = ㅣ + ㆍ
# ㅓ = ㆍ + ㅣ
# ㅗ = ㆍ + ㅡ
# ㅜ = ㅡ + ㆍ

VOWEL_SEQUENCE_MAP = {
    # 기본 모음
    ("ㅣ",): "ㅣ",
    ("ㅡ",): "ㅡ",

    ("ㅣ", "ㆍ"): "ㅏ",
    ("ㅣ", "ㆍ", "ㆍ"): "ㅑ",

    ("ㆍ", "ㅣ"): "ㅓ",
    ("ㆍ", "ㆍ", "ㅣ"): "ㅕ",

    ("ㆍ", "ㅡ"): "ㅗ",
    ("ㆍ", "ㆍ", "ㅡ"): "ㅛ",

    ("ㅡ", "ㆍ"): "ㅜ",
    ("ㅡ", "ㆍ", "ㆍ"): "ㅠ",

    # ㅐ·ㅔ 계열
    ("ㅣ", "ㆍ", "ㅣ"): "ㅐ",
    ("ㅣ", "ㆍ", "ㆍ", "ㅣ"): "ㅒ",

    ("ㆍ", "ㅣ", "ㅣ"): "ㅔ",
    ("ㆍ", "ㆍ", "ㅣ", "ㅣ"): "ㅖ",

    # 복합 모음
    ("ㆍ", "ㅡ", "ㅣ"): "ㅚ",
    ("ㆍ", "ㅡ", "ㅣ", "ㆍ"): "ㅘ",
    ("ㆍ", "ㅡ", "ㅣ", "ㆍ", "ㅣ"): "ㅙ",

    ("ㅡ", "ㆍ", "ㅣ"): "ㅟ",
    ("ㅡ", "ㆍ", "ㆍ", "ㅣ"): "ㅝ",
    ("ㅡ", "ㆍ", "ㆍ", "ㅣ", "ㅣ"): "ㅞ",

    ("ㅡ", "ㅣ"): "ㅢ",
}


VOWEL_PREFIXES = {
    sequence[:index]
    for sequence in VOWEL_SEQUENCE_MAP
    for index in range(1, len(sequence) + 1)
}


class CheonjiinComposer:
    """
    천지인 물리 키를 실제 한글 자모로 변환한다.

    - 같은 자음 키를 제한시간 내 다시 선택하면 다음 자음으로 변경
    - ㅣ, ㆍ, ㅡ 입력 순서를 실제 모음으로 변환
    - 최종 음절 조합은 기존 src.hangul.add_jamo()에 위임
    """

    def __init__(self, repeat_timeout_sec: float = 2.5):
        self.repeat_timeout_sec = repeat_timeout_sec

        self.last_consonant_key: Optional[str] = None
        self.last_consonant: Optional[str] = None
        self.last_consonant_position: Optional[str] = None
        self.last_cycle_index = 0
        self.last_input_time = 0.0

        self.vowel_tokens: list[str] = []
        self.last_emitted_vowel: Optional[str] = None

    def reset(self) -> None:
        """천지인 입력 상태를 모두 초기화한다."""

        self._reset_consonant_state()
        self._reset_vowel_state()

    def cancel_uncommitted_vowel(self) -> bool:
        """
        아직 실제 모음으로 변환되지 않은 ㆍ 입력 등을 취소한다.

        취소한 입력이 있으면 True를 반환한다.
        """

        if self.vowel_tokens and self.last_emitted_vowel is None:
            self._reset_vowel_state()
            return True

        return False
    def get_pending_preview(self) -> str:
        """
        아직 완성된 모음으로 변환되지 않은 천지인 요소를
        입력창에 임시 표시하기 위한 문자열을 반환한다.

        예:
        ㆍ 입력     → "ㆍ"
        ㆍㆍ 입력   → "ㆍㆍ"
        ㆍㅡ 입력   → 이미 ㅗ로 조합되므로 ""
        """

        if (
            self.vowel_tokens
            and self.last_emitted_vowel is None
        ):
            return "".join(self.vowel_tokens)

        return ""

    def input_key(self, key: str) -> Optional[str]:
        """
        천지인 키 하나를 처리한다.

        실제로 적용된 자모를 반환하고,
        아직 모음 조합 대기 중이면 None을 반환한다.
        """

        if key in INITIAL_CONSONANT_CYCLES:
            return self._input_consonant(key)

        if key in {"ㅣ", "ㆍ", "ㅡ"}:
            return self._input_vowel_element(key)

        return None

    def _input_consonant(self, key: str) -> str:
        now = time.monotonic()

        # 자음을 선택하면 진행 중이던 모음 요소 조합은 종료한다.
        self._reset_vowel_state()

        is_repeat = (
            key == self.last_consonant_key
            and self.last_consonant is not None
            and now - self.last_input_time <= self.repeat_timeout_sec
        )

        if is_repeat:
            cycle = self._get_current_consonant_cycle(key)

            next_index = (
                self.last_cycle_index + 1
            ) % len(cycle)

            previous = self.last_consonant
            replacement = cycle[next_index]

            if self._replace_last_consonant(
                previous,
                replacement,
                self.last_consonant_position
            ):
                self.last_cycle_index = next_index
                self.last_consonant = replacement
                self.last_input_time = now

                return replacement

        # 첫 선택 또는 제한시간이 지난 뒤의 새 입력
        consonant = INITIAL_CONSONANT_CYCLES[key][0]

        hangul.add_jamo(consonant)

        position = self._find_inserted_consonant_position(
            consonant
        )

        self.last_consonant_key = key
        self.last_consonant = consonant
        self.last_consonant_position = position
        self.last_cycle_index = 0
        self.last_input_time = now

        return consonant

    def _input_vowel_element(
        self,
        element: str
    ) -> Optional[str]:
        # 모음 요소를 선택하면 자음 연타 상태를 종료한다.
        self._reset_consonant_state()

        candidate_tokens = tuple(
            self.vowel_tokens + [element]
        )

        # 현재 입력이 어떤 모음 조합의 접두사가 아니라면,
        # 새 모음 입력으로 다시 시작한다.
        if candidate_tokens not in VOWEL_PREFIXES:
            self._reset_vowel_state()
            candidate_tokens = (element,)

        self.vowel_tokens = list(candidate_tokens)

        vowel = VOWEL_SEQUENCE_MAP.get(
            candidate_tokens
        )

        # ㆍ 또는 ㆍㆍ처럼 아직 모음이 완성되지 않은 상태
        if vowel is None:
            return None

        if self.last_emitted_vowel is None:
            hangul.add_jamo(vowel)
        else:
            replaced = self._replace_last_vowel(
                self.last_emitted_vowel,
                vowel
            )

            if not replaced:
                hangul.add_jamo(vowel)

        self.last_emitted_vowel = vowel

        return vowel

    def _get_current_consonant_cycle(
        self,
        key: str
    ) -> tuple[str, ...]:
        if self.last_consonant_position == "jong":
            return FINAL_CONSONANT_CYCLES[key]

        return INITIAL_CONSONANT_CYCLES[key]

    @staticmethod
    def _find_inserted_consonant_position(
        consonant: str
    ) -> Optional[str]:
        if hangul.jamo_buffer[2] == consonant:
            return "jong"

        if hangul.jamo_buffer[0] == consonant:
            return "cho"

        return None

    @staticmethod
    def _replace_last_consonant(
        previous: str,
        replacement: str,
        position: Optional[str]
    ) -> bool:
        if (
            position == "jong"
            and hangul.jamo_buffer[2] == previous
            and replacement in hangul.JONGSUNG
        ):
            hangul.jamo_buffer[2] = replacement
            return True

        if (
            position == "cho"
            and hangul.jamo_buffer[0] == previous
            and replacement in hangul.CHOSUNG
        ):
            hangul.jamo_buffer[0] = replacement
            return True

        return False

    @staticmethod
    def _replace_last_vowel(
        previous: str,
        replacement: str
    ) -> bool:
        # 현재 조합 중인 음절의 중성 교체
        if hangul.jamo_buffer[1] == previous:
            hangul.jamo_buffer[1] = replacement
            return True

        # 독립 모음이 finalText에 들어간 경우 교체
        if hangul.finalText.endswith(previous):
            hangul.finalText = (
                hangul.finalText[:-len(previous)]
                + replacement
            )
            return True

        return False

    def _reset_consonant_state(self) -> None:
        self.last_consonant_key = None
        self.last_consonant = None
        self.last_consonant_position = None
        self.last_cycle_index = 0
        self.last_input_time = 0.0

    def _reset_vowel_state(self) -> None:
        self.vowel_tokens = []
        self.last_emitted_vowel = None


cheonjiin_composer = CheonjiinComposer()