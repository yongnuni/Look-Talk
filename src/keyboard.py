import webbrowser

from src.config import SCREEN_W, SCREEN_H
from src.cheonjiin import cheonjiin_composer

from src.hangul import (
    add_jamo,
    flush_buffer,
    finalText,
    jamo_buffer,
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
    "Del": "뒤돌리기",
    " ": "스페이스",
}


class Button:
    def __init__(
        self,
        pos,
        text,
        size=None,
        font_role="default"
    ):
        self.pos = pos
        self.size = size or [85, 85]
        self.text = text
        self.font_role = font_role


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

    return {
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
        "suggestion_rect": (
            outer_margin_x,
            suggestion_y,
            screen_w - 2 * outer_margin_x,
            suggestion_h
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
    start_x = (SCREEN_W - row_width) // 2

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
    area_w = SCREEN_W - 2 * outer_margin_x

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
    한/영 → 스페이스 → 뒤돌리기

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
            font_role="function_small"
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

def create_buttons(keys):

    character_buttons = create_character_buttons(keys)
    function_buttons = create_function_buttons()
    confirm_button = create_confirm_button()

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


def process_key(key,is_korean,is_shift,buttonList,keyboard_layout=KEYBOARD_LAYOUT_QWERTY):
    from src import hangul
    is_cheonjiin = (
        keyboard_layout == KEYBOARD_LAYOUT_CHEONJIIN
    )

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
            cx = btn.pos[0] + btn.size[0] / 2
            cy = btn.pos[1] + btn.size[1] / 2
            return (cx, cy)
    return None
