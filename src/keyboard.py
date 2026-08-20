import webbrowser
from dataclasses import dataclass

from src.config import SCREEN_W, SCREEN_H
from src.cheonjiin import cheonjiin_composer

from src.hangul import (
    add_jamo,
    flush_buffer,
    double_consonants
)

# ── 키보드 레이아웃 (문자 전용 — 기능키는 별도 데이터로 분리 관리) ──

keys_kor_normal = [
    ["1","2","3","4","5","6","7","8","9","0"],
    ["ㅂ","ㅈ","ㄷ","ㄱ","ㅅ","ㅛ","ㅕ","ㅑ","ㅐ","ㅔ"],
    ["ㅁ","ㄴ","ㅇ","ㄹ","ㅎ","ㅗ","ㅓ","ㅏ","ㅣ"],
    ["ㅋ","ㅌ","ㅊ","ㅍ","ㅠ","ㅜ","ㅡ",",","."]
]

keys_kor_shift = [
    ["!","@","#","$","%","^","&","*","(",")"],
    ["ㅃ","ㅉ","ㄸ","ㄲ","ㅆ","ㅛ","ㅕ","ㅑ","ㅒ","ㅖ"],
    ["ㅁ","ㄴ","ㅇ","ㄹ","ㅎ","ㅗ","ㅓ","ㅏ","ㅣ"],
    ["ㅋ","ㅌ","ㅊ","ㅍ","ㅠ","ㅜ","ㅡ","<",">"]
]

keys_eng_normal = [
    ["1","2","3","4","5","6","7","8","9","0"],
    ["q","w","e","r","t","y","u","i","o","p"],
    ["a","s","d","f","g","h","j","k","l",";"],
    ["z","x","c","v","b","n","m",",","."]
]

keys_eng_shift = [
    ["!","@","#","$","%","^","&","*","(",")"],
    ["Q","W","E","R","T","Y","U","I","O","P"],
    ["A","S","D","F","G","H","J","K","L",":"],
    ["Z","X","C","V","B","N","M","<",">"]
]

# ── 키보드 레이아웃 모드 ────────────────────────────────────────

KEYBOARD_LAYOUT_QWERTY = "qwerty"
KEYBOARD_LAYOUT_CHEONJIIN = "cheonjiin"


# 표준 천지인 문자 코어
# 빈 칸은 배열 위치만 유지하고 실제 버튼은 생성하지 않는다.
CHEONJIIN_CHARACTER_ROWS = [
    ["ㅣ", "ㆍ", "ㅡ"],
    ["ㄱㅋ", "ㄴㄹ", "ㄷㅌ"],
    ["ㅂㅍ", "ㅅㅎ", "ㅈㅊ"],
    [None, "ㅇㅁ", None],
]
CHEONJIIN_CHARACTER_KEYS = {
    key
    for row in CHEONJIIN_CHARACTER_ROWS
    for key in row
    if key is not None
}

# 하단 기능키 행 — 문자 배열과 완전히 분리된 데이터
FUNCTION_ROW_LEFT = ["Shift", "한/영"]
FUNCTION_ROW_CENTER = " "
FUNCTION_ROW_RIGHT = ["Del"]

# 렌더링 전용 표시 이름. 내부 판정값(button.text)은 그대로 유지되므로
# dwell/mouth/process_key는 이 매핑을 몰라도 된다.
DISPLAY_LABELS = {
    "Shift": "shift",
    "Del": "되돌리기",
    " ": "스페이스",
}


@dataclass(frozen=True)
class KeyRect:
    """렌더링과 시선 hit-test가 함께 사용하는 키의 실제 사각형."""

    x: int
    y: int
    width: int
    height: int

    @property
    def right(self):
        return self.x + self.width

    @property
    def bottom(self):
        return self.y + self.height

    @property
    def center(self):
        return (
            self.x + self.width / 2,
            self.y + self.height / 2,
        )

    def contains(self, point_x, point_y):
        """서로 맞닿은 사각형도 중복 판정하지 않는 반열린 경계 검사."""
        return (
            self.x <= point_x < self.right
            and self.y <= point_y < self.bottom
        )

    def pillow_bbox(self):
        """Pillow의 양끝 포함 좌표계에서 정확히 width × height를 그린다."""
        return [
            self.x,
            self.y,
            self.right - 1,
            self.bottom - 1,
        ]


class Button:
    def __init__(
        self,
        pos,
        text,
        size=None,
        font_role="default",
        display_label=None
    ):
        width, height = size or [85, 85]
        self.rect = KeyRect(
            x=int(pos[0]),
            y=int(pos[1]),
            width=int(width),
            height=int(height),
        )
        self.text = text
        self.font_role = font_role
        self.display_label = display_label

    @property
    def pos(self):
        """기존 호출부 호환용. 좌표의 원본은 rect 하나뿐이다."""
        return [self.rect.x, self.rect.y]

    @property
    def size(self):
        """기존 호출부 호환용. 크기의 원본은 rect 하나뿐이다."""
        return [self.rect.width, self.rect.height]


def hit_test_buttons(button_list, point_x, point_y):
    """실제로 렌더링되는 사각형 안에 있는 버튼 하나를 반환한다."""
    for button in button_list:
        if button.rect.contains(point_x, point_y):
            return button
    return None


def calculate_suggestion_rects(suggestion_rect, key_gap_x):
    """예약된 추천 영역을 동일 계열 너비의 독립 슬롯 3개로 나눈다."""

    x, y, width, height = suggestion_rect
    usable_width = width - 2 * key_gap_x
    base_width, remainder = divmod(usable_width, 3)
    rects = []
    current_x = x

    for index in range(3):
        rect_width = base_width + (1 if index < remainder else 0)
        rects.append(KeyRect(current_x, y, rect_width, height))
        current_x += rect_width
        if index < 2:
            current_x += key_gap_x

    return tuple(rects)


def hit_test_suggestions(suggestion_rects, point_x, point_y):
    """독립 추천 hitbox 중 점을 포함하는 슬롯 인덱스를 반환한다."""

    for index, rect in enumerate(suggestion_rects):
        if rect.contains(point_x, point_y):
            return index
    return None


def _clamp(value, lo, hi):
    return max(lo, min(value, hi))


# ── bottom-anchored 레이아웃 계산 ────────────────────────────────

ROW_H_MIN, ROW_H_MAX = 60, 130
ROW_GAP_MIN, ROW_GAP_MAX = 3, 20
SECTION_GAP_MIN, SECTION_GAP_MAX = 8, 18
SUGGESTION_H_MIN = 40
OUTER_MARGIN_RATIO = 0.04
KEY_GAP_X_RATIO = 0.006

N_CHAR_ROWS = 4            # 숫자행 + 자모 3행
N_ROWS = N_CHAR_ROWS + 1   # + 기능키 행
N_GAPS = N_ROWS - 1


def calculate_keyboard_layout(screen_w, screen_h):
    """
    입력창 아래에 "추천 단어 한 줄" 정도의 공간만 남기고, 그 아래부터
    숫자 행~기능키 행을 하나의 블록으로 이어 붙이는 레이아웃 계산.

    기능키 행을 화면 하단에 먼저 고정(anchor)하지 않는다 — 그러면 입력창과
    키보드 사이 공백을 정확히 조절할 수 없기 때문이다. 대신 입력창 아래
    공간을 먼저 정하고, 그 아래에 키보드 블록을 이어 붙인 뒤, 화면 하단을
    벗어나는지만 검사한다.

    section_gap/suggestion_h/row_gap/row_h를 먼저 최종 확정한 뒤에만
    y 좌표를 계산한다 (좌표를 만들고 값만 줄이는 방식은 쓰지 않는다).
    """

    minimum_bottom_margin = _clamp(int(screen_h * 0.02), 12, 32)

    input_y = int(screen_h * 0.03)
    input_h = _clamp(int(screen_h * 0.08), 48, 72)

    row_h_ideal = _clamp(int(screen_h * 0.11), ROW_H_MIN, ROW_H_MAX)
    row_gap_ideal = _clamp(int(screen_h * 0.015), ROW_GAP_MIN, ROW_GAP_MAX)
    section_gap_ideal = _clamp(int(screen_h * 0.015), SECTION_GAP_MIN, SECTION_GAP_MAX)
    suggestion_h_ideal = row_h_ideal   # 키캡 한 줄 정도의 빈 공간

    row_h = row_h_ideal
    row_gap = row_gap_ideal
    section_gap = section_gap_ideal
    suggestion_h = suggestion_h_ideal

    input_bottom = input_y + input_h

    def _block_height(section_gap, suggestion_h, row_gap, row_h):
        # 추천 영역(위+아래 section_gap 포함) + 숫자행~기능키행(5행, 4간격)
        return (
            2 * section_gap + suggestion_h
            + N_ROWS * row_h + N_GAPS * row_gap
        )

    ideal_block_h = _block_height(section_gap_ideal, suggestion_h_ideal, row_gap_ideal, row_h_ideal)
    function_row_bottom_ideal = input_bottom + ideal_block_h
    max_keyboard_bottom = screen_h - minimum_bottom_margin

    if function_row_bottom_ideal > max_keyboard_bottom:

        deficit = function_row_bottom_ideal - max_keyboard_bottom

        # 1순위: 추천 영역 위/아래 간격 축소
        gap_slack = 2 * (section_gap_ideal - SECTION_GAP_MIN)
        reduction = min(deficit, gap_slack)
        if reduction > 0:
            section_gap = section_gap_ideal - (reduction // 2)
        deficit -= reduction

        # 2순위: 추천 영역 높이를 row_h보다 조금 축소
        if deficit > 0:
            slack = suggestion_h_ideal - SUGGESTION_H_MIN
            reduction = min(deficit, slack)
            suggestion_h = suggestion_h_ideal - reduction
            deficit -= reduction

        # 3순위: 문자 행 사이 간격 축소
        if deficit > 0:
            slack = N_GAPS * (row_gap_ideal - ROW_GAP_MIN)
            reduction = min(deficit, slack)
            if N_GAPS > 0 and reduction > 0:
                row_gap = row_gap_ideal - (reduction // N_GAPS)
            deficit -= reduction

        # 4순위: 행 높이를 dwell 선택 가능한 최소치까지 축소
        if deficit > 0:
            slack = N_ROWS * (row_h_ideal - ROW_H_MIN)
            reduction = min(deficit, slack)
            if N_ROWS > 0 and reduction > 0:
                row_h = row_h_ideal - (reduction // N_ROWS)
            deficit -= reduction

    # ── section_gap/suggestion_h/row_gap/row_h 확정 완료. 이제 y 좌표만 계산 ──

    suggestion_y = input_bottom + section_gap
    number_y = suggestion_y + suggestion_h + section_gap

    row1_y = number_y + row_h + row_gap
    row2_y = row1_y + row_h + row_gap
    row3_y = row2_y + row_h + row_gap
    function_row_y = row3_y + row_h + row_gap

    keyboard_top = number_y
    function_row_h = row_h

    outer_margin_x = int(screen_w * OUTER_MARGIN_RATIO)
    key_gap_x = max(4, int(screen_w * KEY_GAP_X_RATIO))

    # 문자 키: 가장 넓은 행(10키) 기준 공통 폭 산출 → 모든 문자 행이 동일 key_w 사용
    available_w = screen_w - 2 * outer_margin_x
    char_key_w = (available_w - key_gap_x * (10 - 1)) // 10
    char_key_w = max(40, char_key_w)

    confirm_w = int(screen_w * 0.09)
    confirm_gap = int(screen_w * 0.015)
    confirm_x = screen_w - outer_margin_x - confirm_w
    input_x = outer_margin_x
    input_w = confirm_x - confirm_gap - input_x

    suggestion_rect = (
        outer_margin_x,
        suggestion_y,
        screen_w - 2 * outer_margin_x,
        suggestion_h,
    )

    return {
        "screen_w": screen_w,
        "screen_h": screen_h,
        "outer_margin_x": outer_margin_x,
        "key_gap_x": key_gap_x,
        "char_key_w": char_key_w,
        "row_h": row_h,
        "row_gap": row_gap,

        "number_y": number_y,
        "row1_y": row1_y,
        "row2_y": row2_y,
        "row3_y": row3_y,
        "function_row_y": function_row_y,
        "function_row_h": function_row_h,

        "input_rect": (input_x, input_y, input_w, input_h),
        "confirm_rect": (confirm_x, input_y, confirm_w, input_h),
        "suggestion_rect": suggestion_rect,
        "suggestion_rects": calculate_suggestion_rects(
            suggestion_rect,
            key_gap_x,
        ),

        "keyboard_top": keyboard_top,
    }


LAYOUT = calculate_keyboard_layout(SCREEN_W, SCREEN_H)
def calculate_cheonjiin_layout(screen_w, screen_h):
    """
    기존 입력창·자동완성·확인 버튼 위치를 유지하면서
    천지인 3열 × 4행 문자 코어를 화면 중앙에 크게 배치한다.
    """

    base_layout = calculate_keyboard_layout(screen_w, screen_h)

    key_gap_x = max(8, int(screen_w * 0.010))

    # 쿼티보다 열 수가 적으므로 키 폭을 크게 설정한다.
    key_w = _clamp(
        int(screen_w * 0.13),
        110,
        180
    )

    key_h = base_layout["row_h"]

    total_w = (
        3 * key_w
        + 2 * key_gap_x
    )

    start_x = (
        screen_w - total_w
    ) // 2

    return {
        **base_layout,

        "cheonjiin_start_x": start_x,
        "cheonjiin_key_w": key_w,
        "cheonjiin_key_h": key_h,
        "cheonjiin_key_gap_x": key_gap_x,
        "cheonjiin_total_w": total_w,

        # 기존 쿼티 문자 4행의 세로 위치를 그대로 재사용한다.
        "cheonjiin_row_ys": [
            base_layout["number_y"],
            base_layout["row1_y"],
            base_layout["row2_y"],
            base_layout["row3_y"],
        ],
    }


CHEONJIIN_LAYOUT = calculate_cheonjiin_layout(
    SCREEN_W,
    SCREEN_H
)

def _create_row_buttons(row, y, layout):

    key_w = layout["char_key_w"]
    key_gap = layout["key_gap_x"]
    row_h = layout["row_h"]

    row_width = len(row) * key_w + (len(row) - 1) * key_gap
    start_x = (layout["screen_w"] - row_width) // 2

    buttons = []
    x = start_x

    for key in row:
        buttons.append(Button([x, y], key, size=[key_w, row_h]))
        x += key_w + key_gap

    return buttons


def create_character_buttons(rows, layout=None):

    layout = layout or LAYOUT

    row_ys = [
        layout["number_y"],
        layout["row1_y"],
        layout["row2_y"],
        layout["row3_y"],
    ]

    buttons = []

    for row, y in zip(rows, row_ys):
        buttons.extend(_create_row_buttons(row, y, layout))

    return buttons


def create_function_buttons(layout=None):

    layout = layout or LAYOUT

    outer_margin_x = layout["outer_margin_x"]
    key_gap = layout["key_gap_x"]
    y = layout["function_row_y"]
    row_h = layout["function_row_h"]

    area_x = outer_margin_x
    area_w = layout["screen_w"] - 2 * outer_margin_x

    space_w = int(area_w * 0.42)
    space_x = area_x + (area_w - space_w) // 2

    side_w = int(area_w * 0.13)

    buttons = []

    # 왼쪽: 스페이스 바로 왼쪽부터 바깥 방향으로 배치
    cursor_x = space_x - key_gap
    for value in reversed(FUNCTION_ROW_LEFT):
        cursor_x -= side_w
        buttons.append(Button([cursor_x, y], value, size=[side_w, row_h]))
        cursor_x -= key_gap

    buttons.append(
        Button([space_x, y], FUNCTION_ROW_CENTER, size=[space_w, row_h])
    )

    # 오른쪽: 스페이스 오른쪽 끝부터 바깥 방향으로 배치
    cursor_x = space_x + space_w + key_gap
    for value in FUNCTION_ROW_RIGHT:
        buttons.append(Button([cursor_x, y], value, size=[side_w, row_h]))
        cursor_x += side_w + key_gap

    return buttons


def create_confirm_button(layout=None):

    layout = layout or LAYOUT

    x, y, w, h = layout["confirm_rect"]

    return Button([x, y], "확인", size=[w, h])

def create_cheonjiin_character_buttons(layout=None):
    """
    천지인 문자 코어 버튼을 생성한다.
    None으로 지정된 위치는 빈 슬롯으로 남긴다.
    """

    layout = layout or CHEONJIIN_LAYOUT

    start_x = layout["cheonjiin_start_x"]
    key_w = layout["cheonjiin_key_w"]
    key_h = layout["cheonjiin_key_h"]
    key_gap_x = layout["cheonjiin_key_gap_x"]
    row_ys = layout["cheonjiin_row_ys"]

    buttons = []

    for row_index, row in enumerate(
        CHEONJIIN_CHARACTER_ROWS
    ):
        y = row_ys[row_index]

        for column_index, key in enumerate(row):
            if key is None:
                continue

            x = (
                start_x
                + column_index * (
                    key_w + key_gap_x
                )
            )

            buttons.append(
                Button(
                    [x, y],
                    key,
                    size=[key_w, key_h]
                )
            )

    return buttons
def create_cheonjiin_function_buttons(layout=None):
    """
    천지인 하단 기능키를 생성한다.

    배치 순서:
    한/영 → 스페이스 → 되돌리기

    내부 판정값은 기존 process_key와의 호환을 위해
    '한/영', ' ', 'Del'을 그대로 사용한다.
    """

    layout = layout or CHEONJIIN_LAYOUT

    start_x = layout["cheonjiin_start_x"]
    total_w = layout["cheonjiin_total_w"]

    gap = layout["cheonjiin_key_gap_x"]
    y = layout["function_row_y"]
    row_h = layout["function_row_h"]

    usable_w = total_w - 2 * gap

    # 스페이스를 가장 넓게 배치
    language_w = int(usable_w * 0.23)
    space_w = int(usable_w * 0.54)
    delete_w = (
        usable_w
        - language_w
        - space_w
    )

    language_x = start_x
    space_x = (
        language_x
        + language_w
        + gap
    )
    delete_x = (
        space_x
        + space_w
        + gap
    )

    return [
        Button(
            [language_x, y],
            "한/영",
            size=[language_w, row_h],
            font_role="function_small"
        ),
        Button(
            [space_x, y],
            " ",
            size=[space_w, row_h],
            font_role="function_small"
        ),
        Button(
            [delete_x, y],
            "Del",
            size=[delete_w, row_h],
            font_role="function_small",
            display_label="되돌리기"
        ),
    ]

def create_cheonjiin_buttons(layout=None):
    """
    천지인 문자키, 하단 기능키, 입력창 오른쪽 확인 버튼을
    하나의 Button 목록으로 반환한다.
    """

    layout = layout or CHEONJIIN_LAYOUT

    character_buttons = (
        create_cheonjiin_character_buttons(layout)
    )

    function_buttons = (
        create_cheonjiin_function_buttons(layout)
    )

    # 확인 버튼은 하단에 넣지 않고
    # 기존처럼 입력창 오른쪽에 유지한다.
    confirm_button = create_confirm_button(layout)

    return (
        character_buttons
        + function_buttons
        + [confirm_button]
    )

def create_buttons(keys, layout=None):

    layout = layout or LAYOUT

    character_buttons = create_character_buttons(keys, layout)
    function_buttons = create_function_buttons(layout)
    confirm_button = create_confirm_button(layout)

    return character_buttons + function_buttons + [confirm_button]


def _composite_text(is_korean, keyboard_layout):
    """finalText + compose_jamo_buffer() + (해당 시) 천지인 pending preview.

    main.py가 화면에 그리는 current_text와 동일한 조합이다. process_key()
    호출 전/후로 이 값을 스냅샷해 tap 1회의 변화(diff)를 계산하는 기준으로
    쓴다. pending preview를 빼면 ㆍ 단독 탭처럼 아직 완성된 모음이 안 된
    입력이 빈 diff로 유실된다.
    """
    from src import hangul

    text = hangul.finalText + hangul.compose_jamo_buffer()

    if (
        keyboard_layout == KEYBOARD_LAYOUT_CHEONJIIN
        and is_korean
    ):
        text += cheonjiin_composer.get_pending_preview()

    return text


def _diff_tail(before, after):
    """before -> after 변화를 "꼬리에서 지운 개수 + 꼬리에 붙인 문자열"로 표현한다.

    공통 접두사 뒤부터를 diff로 잡는다. 이 코드베이스의 소급 변경(hangul
    flush_buffer, 천지인 _replace_last_vowel, 백스페이스)은 전부 문자열
    꼬리에서만 일어나므로(조사 결과 확인됨), 그 전제 하에서 deleted_count/
    inserted_text만으로 after를 정확히 재구성할 수 있다:
    after == before[:len(before) - deleted_count] + inserted_text
    """
    common_len = 0
    max_common = min(len(before), len(after))

    while (
        common_len < max_common
        and before[common_len] == after[common_len]
    ):
        common_len += 1

    deleted_count = len(before) - common_len
    inserted_text = after[common_len:]

    return deleted_count, inserted_text


def apply_suggestion(
    suggestion,
    is_korean,
    keyboard_layout=KEYBOARD_LAYOUT_QWERTY,
):
    """화면에 보이는 마지막 입력 단어를 추천 문장으로 교체합니다.

    조합 중인 자모와 천지인 pending preview까지 포함한 문자열을 기준으로
    교체하므로, 선택 전 화면과 선택 후 ``finalText`` 사이의 꼬리 diff를
    정확히 반환합니다. 선택 결과는 확정 문자열로 두고 모든 조합 상태를
    한 곳에서 초기화합니다.
    """

    if not isinstance(suggestion, str) or not suggestion:
        return 0, ""

    from src import hangul

    composite_before = _composite_text(is_korean, keyboard_layout)
    last_space_index = composite_before.rfind(" ")
    preserved_prefix = composite_before[:last_space_index + 1]
    composite_after = preserved_prefix + suggestion

    hangul.finalText = composite_after
    hangul.jamo_buffer[:] = ['', '', '']
    cheonjiin_composer.reset()

    return _diff_tail(composite_before, composite_after)


class PendingWordBoundaryState:
    """추천 선택 뒤 다음 단어의 공백을 실제 문자 입력까지 지연합니다."""

    _CLEARING_KEYS = {"Del", "확인", "Enter"}
    _PRESERVING_FUNCTION_KEYS = {"Shift", "한/영"}

    def __init__(self):
        self.pending_word_boundary = False

    def mark_pending(self):
        self.pending_word_boundary = True

    def clear(self):
        self.pending_word_boundary = False

    @staticmethod
    def _is_word_forming_key(key):
        return (
            key in CHEONJIIN_CHARACTER_KEYS
            or (isinstance(key, str) and key.isalnum())
        )

    @staticmethod
    def _commit_visible_input(is_korean, keyboard_layout):
        """문장부호 등 화면상의 조합 문자열을 공백 앞에 확정합니다."""

        if (
            keyboard_layout == KEYBOARD_LAYOUT_CHEONJIIN
            and is_korean
        ):
            cheonjiin_composer.commit()
        elif is_korean:
            flush_buffer()

    def handle_key(
        self,
        key,
        is_korean,
        is_shift,
        button_list,
        keyboard_layout=KEYBOARD_LAYOUT_QWERTY,
    ):
        """실제 키 한 건과 필요한 자동 공백을 하나의 diff로 처리합니다."""

        if not self.pending_word_boundary:
            return process_key(
                key,
                is_korean,
                is_shift,
                button_list,
                keyboard_layout,
            )

        from src import hangul

        composite_before = _composite_text(is_korean, keyboard_layout)

        if key == " ":
            self._commit_visible_input(is_korean, keyboard_layout)
            if not hangul.finalText.endswith(" "):
                hangul.finalText += " "
            self.clear()
            deleted_count, inserted_text = _diff_tail(
                composite_before,
                _composite_text(is_korean, keyboard_layout),
            )
            return (
                is_korean,
                is_shift,
                button_list,
                deleted_count,
                inserted_text,
            )

        if key in self._CLEARING_KEYS:
            self.clear()
            return process_key(
                key,
                is_korean,
                is_shift,
                button_list,
                keyboard_layout,
            )

        if key in self._PRESERVING_FUNCTION_KEYS:
            return process_key(
                key,
                is_korean,
                is_shift,
                button_list,
                keyboard_layout,
            )

        if self._is_word_forming_key(key):
            self._commit_visible_input(is_korean, keyboard_layout)
            if not hangul.finalText.endswith(" "):
                hangul.finalText += " "
            self.clear()

            result = process_key(
                key,
                is_korean,
                is_shift,
                button_list,
                keyboard_layout,
            )
            next_is_korean, next_is_shift, next_button_list = result[:3]
            deleted_count, inserted_text = _diff_tail(
                composite_before,
                _composite_text(next_is_korean, keyboard_layout),
            )
            return (
                next_is_korean,
                next_is_shift,
                next_button_list,
                deleted_count,
                inserted_text,
            )

        # 문장부호는 공백 없이 처리하고 다음 일반 문자까지 pending을 유지한다.
        return process_key(
            key,
            is_korean,
            is_shift,
            button_list,
            keyboard_layout,
        )


def process_key(key,is_korean,is_shift,buttonList,keyboard_layout=KEYBOARD_LAYOUT_QWERTY):
    from src import hangul
    is_cheonjiin = (
        keyboard_layout == KEYBOARD_LAYOUT_CHEONJIIN
    )

    # QWERTY로 레이아웃이 바뀐 뒤 천지인 순환 후보가 남지 않게 한다.
    # QWERTY의 실제 입력 분기는 아래 기존 로직을 그대로 사용한다.
    if not is_cheonjiin:
        cheonjiin_composer.reset()

    composite_before = _composite_text(is_korean, keyboard_layout)

    if (
        is_cheonjiin
        and is_korean
        and key in CHEONJIIN_CHARACTER_KEYS
    ):
        emitted_jamo = cheonjiin_composer.input_key(key)

        if emitted_jamo is None:
            print(
                f"[천지인] {key} → 모음 조합 대기"
            )
        else:
            print(
                f"[천지인] {key} → {emitted_jamo}"
            )

        deleted_count, inserted_text = _diff_tail(
            composite_before,
            _composite_text(is_korean, keyboard_layout)
        )

        return (
            is_korean,
            is_shift,
            buttonList,
            deleted_count,
            inserted_text
        )
    # 아직 화면에 나타나지 않은 ㆍ 입력은
    # 되돌리기를 한 번 눌러 취소할 수 있다.
    if (
        is_cheonjiin
        and is_korean
        and key == "Del"
        and cheonjiin_composer.cancel_uncommitted_vowel()
        ):
        print("[천지인] 미완성 모음 입력 취소")

        deleted_count, inserted_text = _diff_tail(
            composite_before,
            _composite_text(is_korean, keyboard_layout)
        )

        return (
            is_korean,
            is_shift,
            buttonList,
            deleted_count,
            inserted_text
            )

    # 천지인 조합/후보가 있으면 첫 스페이스는 확정 키로만 사용한다.
    # commit()이 상태와 한글 버퍼를 비우므로 다음 스페이스는 아래 기존
    # 공백 입력 분기로 내려간다.
    if (
        is_cheonjiin
        and is_korean
        and key == " "
        and cheonjiin_composer.commit()
    ):
        deleted_count, inserted_text = _diff_tail(
            composite_before,
            _composite_text(is_korean, keyboard_layout)
        )

        return (
            is_korean,
            is_shift,
            buttonList,
            deleted_count,
            inserted_text
        )

    # 스페이스, 한/영, 확인, 되돌리기 등 기능키를 선택하면
    # 자음 연타 및 모음 요소 입력 상태를 종료한다.
    if is_cheonjiin:
        cheonjiin_composer.reset()

    if key == "확인":
        # 추후 메시지 전송(제출) 기능 연결 예정 — 현재는 placeholder.
        # 입력 문장을 건드리지 않고, 기존 Enter의 검색 실행 기능과도 무관하다.
        deleted_count, inserted_text = _diff_tail(
            composite_before,
            _composite_text(is_korean, keyboard_layout)
        )
        return (is_korean, is_shift, buttonList, deleted_count, inserted_text)

    if is_korean:

        if key == "Del":

            if hangul.jamo_buffer[2]:
                hangul.jamo_buffer[2] = ''

            elif hangul.jamo_buffer[1]:
                hangul.jamo_buffer[1] = ''

            elif hangul.jamo_buffer[0]:
                hangul.jamo_buffer[0] = ''

            else:
                hangul.finalText = hangul.finalText[:-1]

        elif key == "한/영":

            flush_buffer()

            is_korean = False
            is_shift = False

            buttonList = create_buttons(
                keys_eng_normal
            )

        elif key == " ":

            flush_buffer()

            hangul.finalText += " "

        elif key == "Shift":

            is_shift = not is_shift

            buttonList = create_buttons(
                keys_kor_shift
                if is_shift
                else keys_kor_normal
            )

        elif key == "Enter":

            flush_buffer()

            query = hangul.finalText.strip()

            if query:

                webbrowser.open(
                    f"https://www.google.com/search?q={query}"
                )

                hangul.finalText = ""

        else:

            if is_shift and key in double_consonants:

                add_jamo(
                    double_consonants[key]
                )

            else:

                add_jamo(key)

            is_shift = False

            buttonList = create_buttons(
                keys_kor_normal
            )

    else:

        if key == "Del":

            hangul.finalText = (
                hangul.finalText[:-1]
            )

        elif key == "한/영":

            is_korean = True
            is_shift = False

            if keyboard_layout == KEYBOARD_LAYOUT_CHEONJIIN:
                buttonList = create_cheonjiin_buttons()
            else:
                buttonList = create_buttons(
                    keys_kor_normal
                )

        elif key == " ":

            hangul.finalText += " "

        elif key == "Shift":

            is_shift = not is_shift

            buttonList = create_buttons(
                keys_eng_shift
                if is_shift
                else keys_eng_normal
            )

        elif key == "Enter":

            query = hangul.finalText.strip()

            if query:

                webbrowser.open(
                    f"https://www.google.com/search?q={query}"
                )

                hangul.finalText = ""

        else:

            hangul.finalText += key

            is_shift = False

            buttonList = create_buttons(
                keys_eng_normal
            )

    deleted_count, inserted_text = _diff_tail(
        composite_before,
        _composite_text(is_korean, keyboard_layout)
    )

    return (
        is_korean,
        is_shift,
        buttonList,
        deleted_count,
        inserted_text
    )

def get_button_center(buttonList, key_name):
    for btn in buttonList:
        if btn.text == key_name:
            return btn.rect.center
    return None
