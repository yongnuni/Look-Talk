import webbrowser

from src.config import SCREEN_W, SCREEN_H
from src.hangul import (
    add_jamo,
    flush_buffer,
    finalText,
    jamo_buffer,
    double_consonants
)

# ── 키보드 레이아웃 ───────────────────────────────────────────

keys_kor_normal = [
    ["1","2","3","4","5","6","7","8","9","0","Del"],
    ["ㅂ","ㅈ","ㄷ","ㄱ","ㅅ","ㅛ","ㅕ","ㅑ","ㅐ","ㅔ"],
    ["ㅁ","ㄴ","ㅇ","ㄹ","ㅎ","ㅗ","ㅓ","ㅏ","ㅣ","Shift"],
    ["ㅋ","ㅌ","ㅊ","ㅍ","ㅠ","ㅜ","ㅡ",",",".","Enter","한/영"],
    [" "]
]

keys_kor_shift = [
    ["!","@","#","$","%","^","&","*","(",")","Del"],
    ["ㅃ","ㅉ","ㄸ","ㄲ","ㅆ","ㅛ","ㅕ","ㅑ","ㅒ","ㅖ"],
    ["ㅁ","ㄴ","ㅇ","ㄹ","ㅎ","ㅗ","ㅓ","ㅏ","ㅣ","Shift"],
    ["ㅋ","ㅌ","ㅊ","ㅍ","ㅠ","ㅜ","ㅡ","<",">","Enter","한/영"],
    [" "]
]

keys_eng_normal = [
    ["1","2","3","4","5","6","7","8","9","0","Del"],
    ["q","w","e","r","t","y","u","i","o","p"],
    ["a","s","d","f","g","h","j","k","l",";","Shift"],
    ["z","x","c","v","b","n","m",",",".","Enter","한/영"],
    [" "]
]

keys_eng_shift = [
    ["!","@","#","$","%","^","&","*","(",")","Del"],
    ["Q","W","E","R","T","Y","U","I","O","P"],
    ["A","S","D","F","G","H","J","K","L",":","Shift"],
    ["Z","X","C","V","B","N","M","<",">","Enter","한/영"],
    [" "]
]


class Button:
    def __init__(self, pos, text, size=[85, 85]):
        self.pos = pos
        self.size = size
        self.text = text


def create_buttons(keys):

    buttonList = []

    button_w = 75
    button_h = 65

    top_margin = 180
    vertical_gap = 78

    special_widths = {
        "Enter":1.5,
        "Shift":1.5,
        "한/영":1.5,
        "Del":1.5,
        " ":5
    }

    for i, row in enumerate(keys):

        widths = [
            int(button_w * special_widths.get(k, 1))
            for k in row
        ]

        total_w = sum(widths)

        margin = (SCREEN_W - total_w) // (len(row) + 1)

        x = margin
        y = top_margin + i * vertical_gap

        for key, w in zip(row, widths):

            buttonList.append(
                Button(
                    [x, y],
                    key,
                    size=[w, button_h]
                )
            )

            x += w + margin

    return buttonList


def process_key(key, is_korean, is_shift, buttonList):

    from src import hangul

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