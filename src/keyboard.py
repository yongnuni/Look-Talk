import webbrowser

from src.config import SCREEN_W, SCREEN_H
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
    def __init__(self, pos, text, size=[85, 85]):
        self.pos = pos
        self.size = size
        self.text = text


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


def create_buttons(keys):

    character_buttons = create_character_buttons(keys)
    function_buttons = create_function_buttons()
    confirm_button = create_confirm_button()

    return character_buttons + function_buttons + [confirm_button]


def process_key(key, is_korean, is_shift, buttonList):

    from src import hangul

    if key == "확인":
        # 추후 메시지 전송(제출) 기능 연결 예정 — 현재는 placeholder.
        # 입력 문장을 건드리지 않고, 기존 Enter의 검색 실행 기능과도 무관하다.
        return (is_korean, is_shift, buttonList)

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

    return (
        is_korean,
        is_shift,
        buttonList
    )

def get_button_center(buttonList, key_name):
    for btn in buttonList:
        if btn.text == key_name:
            cx = btn.pos[0] + btn.size[0] / 2
            cy = btn.pos[1] + btn.size[1] / 2
            return (cx, cy)
    return None
