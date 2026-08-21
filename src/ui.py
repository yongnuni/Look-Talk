import cv2
import time
import numpy as np
import os
from functools import lru_cache

from PIL import ImageFont, ImageDraw, Image
from src.config import SCREEN_W, SCREEN_H, FIXATION_FRAMES

cursor_img = cv2.imread(
    os.path.join("assets", "cursor.png"),
    cv2.IMREAD_UNCHANGED
)

from src.config import (
    FONT_PATH,
    FONT_SIZE,
    SCREEN_W,
    SCREEN_H,
    COUNTDOWN_SEC,
    CALIB_POINTS
)
from src.keyboard import LAYOUT, DISPLAY_LABELS
from src.tracking.mouth import draw_mouth
from src.tracking.eye_tracking import (
    LEFT_EYE,
    RIGHT_EYE,
    LEFT_IRIS,
    RIGHT_IRIS,
    LEFT_IRIS_RING,
    RIGHT_IRIS_RING,
    iris_confidence,
    draw_eye_contour,
    draw_iris_ring
)

font = ImageFont.truetype(
    FONT_PATH,
    FONT_SIZE
)

small_font = ImageFont.truetype(
    FONT_PATH,
    20
)

# 키캡 텍스트는 최종 확정된 row_h에 비례해 크기를 맞춘다(해상도별 자동 조정).
KEY_FONT_SIZE = max(18, int(LAYOUT["row_h"] * 0.42))
FUNCTION_KEY_FONT_SIZE = max(20, int(LAYOUT["row_h"] * 0.30))

key_font = ImageFont.truetype(FONT_PATH, KEY_FONT_SIZE)
function_key_font = ImageFont.truetype(FONT_PATH, FUNCTION_KEY_FONT_SIZE)

# 고정 감지로 확대된 키캡은 글자도 같은 비율로 커진다. 매 프레임 폰트를
# 새로 여는 비용을 피하려고 크기별로 한 번만 만들어 캐시한다.
_scaled_key_fonts = {}


def _key_font_for(base_size, scale):
    size = max(12, int(round(base_size * scale)))

    if size not in _scaled_key_fonts:
        _scaled_key_fonts[size] = ImageFont.truetype(FONT_PATH, size)

    return _scaled_key_fonts[size]


# ── 카운트다운 ────────────────────────────────────────────────

def show_countdown(cap, face_mesh):

    start = time.time()

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]

        remaining = COUNTDOWN_SEC - (
            time.time() - start
        )

        if remaining <= 0:
            break

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        rgb.flags.writeable = False

        results = face_mesh.process(rgb)

        rgb.flags.writeable = True

        face_found = False
        conf = 0

        if results.multi_face_landmarks:

            lms = results.multi_face_landmarks[0]

            draw_eye_contour(
                frame,
                lms,
                LEFT_EYE,
                w,
                h
            )

            draw_eye_contour(
                frame,
                lms,
                RIGHT_EYE,
                w,
                h
            )

            draw_iris_ring(
                frame,
                lms,
                LEFT_IRIS,
                LEFT_IRIS_RING,
                w,
                h,
                (0, 200, 255)
            )

            draw_iris_ring(
                frame,
                lms,
                RIGHT_IRIS,
                RIGHT_IRIS_RING,
                w,
                h,
                (0, 200, 255)
            )
            draw_mouth(
                frame,
                lms,
                w,
                h
            )

            conf = iris_confidence(lms)

            face_found = True

            bw = int(
                (w - 40) * conf
            )

            cv2.rectangle(
                frame,
                (20, h - 50),
                (w - 20, h - 36),
                (50, 50, 50),
                -1
            )

            qcol = (
                (0, 200, 80)
                if conf > 0.5
                else (0, 140, 255)
            )

            cv2.rectangle(
                frame,
                (20, h - 50),
                (20 + bw, h - 36),
                qcol,
                -1
            )

        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (0, 0),
            (w, 70),
            (10, 10, 10),
            -1
        )

        frame = cv2.addWeighted(
            overlay,
            0.7,
            frame,
            0.3,
            0
        )

        # ===== PIL 한글 출력 =====

        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)

        message = "카메라를 정면으로 바라봐 주세요"

        bbox = draw.textbbox(
            (0, 0),
            message,
            font=font
        )

        text_w = bbox[2] - bbox[0]

        draw.text(
            (
                (w - text_w) // 2,
                10
            ),
            message,
            font=font,
            fill=(255, 255, 255)
        )

        draw.text(
            (20, h - 80),
            "얼굴 감지됨" if face_found else "얼굴을 찾는 중...",
            font=small_font,
            fill=(100, 255, 100) if face_found else (255, 180, 80)
        )

        frame = np.array(img_pil)

        # ===== 카운트다운 =====

        cv2.putText(
            frame,
            str(int(remaining) + 1),
            (w // 2 - 50, h // 2 + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            5,
            (0, 220, 255),
            8
        )

        display = cv2.resize(
            frame,
            (SCREEN_W, SCREEN_H)
        )

        cv2.imshow(
            "Eye Keyboard",
            display
        )

        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False

    return True


# ── 캘리브레이션 화면 ─────────────────────────────────────────

def draw_calib_screen(
    canvas,
    calib,
    elapsed_ratio
):

    canvas[:] = (15,15,15)

    sw = canvas.shape[1]
    sh = canvas.shape[0]

    for i in range(calib.idx):

        px = int(
            calib.calib_points[i][0] * sw
        )

        py = int(
            calib.calib_points[i][1] * sh
        )

        cv2.circle(
            canvas,
            (px,py),
            10,
            (60,180,60),
            -1
        )

        cv2.putText(
            canvas,
            str(i+1),
            (px-4, py+5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (15,15,15),
            1
        )

    if calib.idx < len(calib.calib_points):

        tx = int(
            calib.calib_points[calib.idx][0] * sw
        )

        ty = int(
            calib.calib_points[calib.idx][1] * sh
        )

        cv2.circle(
            canvas,
            (tx,ty),
            36,
            (50,50,50),
            -1
        )

        cv2.ellipse(
            canvas,
            (tx,ty),
            (36,36),
            -90,
            0,
            int(360 * elapsed_ratio),
            (0,220,255),
            4
        )

        cv2.circle(
            canvas,
            (tx,ty),
            14,
            (0,220,255),
            -1
        )

        cv2.circle(
            canvas,
            (tx,ty),
            5,
            (15,15,15),
            -1
        )

    img_pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img_pil)

    if (
        hasattr(calib, "warning")
        and calib.warning
        and calib.warning_start
        and time.time() - calib.warning_start < 1.0
    ):

        text = "시선이 불안정합니다"

        bbox = draw.textbbox(
            (0, 0),
            text,
            font=font
        )

        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        draw.text(
            (
                (sw - tw) // 2,
                (sh - th) // 2
            ),
            text,
            font=font,
            fill=(255, 80, 80)
        )

        canvas[:] = np.array(img_pil)

        return

    draw.text(
        (20, sh - 35),
        "r: 재시작   q: 종료",
        font=small_font,
        fill=(120, 120, 120)
    )

    canvas[:] = np.array(img_pil)


# ── 키보드 그리기 ─────────────────────────────────────────────

IDLE_BG = (235, 236, 240)
IDLE_BORDER = (205, 206, 210)
HOVER_BG = (219, 234, 254)
HOVER_BORDER = (147, 197, 253)
DWELL_BG_END = (37, 99, 235)
DWELL_BORDER_END = (29, 78, 216)
PROGRESS_BAR_COLOR = (34, 197, 94)
KEY_TEXT_COLOR = (30, 41, 59)
KEY_TEXT_COLOR_DWELL = (255, 255, 255)


def _draw_key(
    draw,
    button,
    box,
    dwell_key,
    dwell_ratio,
    show_cursor,
    locked_key=None,
    font_scale=1.0,
):
    """키캡 하나를 그린다.

    box는 실제로 그릴 사각형 (x, y, right, bottom)이다. 고정 감지로 확대된
    키만 자기 rect보다 큰 box를 받고, 나머지 키는 rect 그대로를 받는다.
    """

    key = button.text

    nx = int(box[0])
    ny = int(box[1])
    nw = int(box[2]) - nx
    nh = int(box[3]) - ny

    # hover 여부는 좌표를 다시 검사하지 않고 판정 결과(dwell_key)를 그대로
    # 따른다. 고정 확장으로 실제 사각형 밖에서 hover가 잡힌 경우에도
    # 화면 표시와 판정이 어긋나지 않는다.
    on_key = dwell_key is not None and key == dwell_key

    text_color = KEY_TEXT_COLOR

    # 입벌림 모드에서 잠긴 키
    if locked_key is not None and key == locked_key:

        bg_color = DWELL_BG_END
        border_color = DWELL_BORDER_END
        text_color = KEY_TEXT_COLOR_DWELL

    # 바라보고 있는 키 (dwell_ratio가 0이면 hover 색 그대로)
    elif on_key and show_cursor:

        t = dwell_ratio

        bg_color = tuple(
            int(
                HOVER_BG[i]
                + (DWELL_BG_END[i] - HOVER_BG[i]) * t
            )
            for i in range(3)
        )

        border_color = tuple(
            int(
                HOVER_BORDER[i]
                + (DWELL_BORDER_END[i] - HOVER_BORDER[i]) * t
            )
            for i in range(3)
        )

        if t > 0.5:
            text_color = KEY_TEXT_COLOR_DWELL

    # 커서 숨김 모드
    elif on_key and not show_cursor:

        bg_color = (255, 235, 0)
        border_color = (255, 120, 0)

    else:

        bg_color = IDLE_BG
        border_color = IDLE_BORDER

    radius = int(min(nw, nh) * 0.18)

    if on_key and not show_cursor:

        draw.rounded_rectangle(
            [nx - 5, ny - 5, nx + nw + 5, ny + nh + 5],
            radius=radius + 5,
            outline=(255, 255, 0),
            width=6
        )

    draw.rounded_rectangle(
        [nx, ny, nx + nw - 1, ny + nh - 1],
        radius=radius,
        fill=bg_color,
        outline=border_color,
        width=2
    )

    if on_key and dwell_ratio > 0:

        bar_w = int(nw * dwell_ratio)
        bar_right = nx + max(1, min(nw, bar_w)) - 1

        draw.rounded_rectangle(
            [
                nx,
                ny + nh - 6,
                bar_right,
                ny + nh - 1
            ],
            radius=3,
            fill=PROGRESS_BAR_COLOR
        )

    label = (
        button.display_label
        or DISPLAY_LABELS.get(key, key)
    )

    base_font_size = (
        FUNCTION_KEY_FONT_SIZE
        if getattr(button, "font_role", "default") == "function_small"
        else KEY_FONT_SIZE
    )

    current_key_font = _key_font_for(base_font_size, font_scale)

    bbox = draw.textbbox(
        (0, 0),
        label,
        font=current_key_font
    )

    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    text_x = nx + (nw - text_w) // 2 - bbox[0]
    text_y = ny + (nh - text_h) // 2 - bbox[1]

    draw.text(
        (text_x, text_y),
        label,
        font=current_key_font,
        fill=text_color
    )


_suggestion_measure_image = Image.new("RGB", (1, 1))
_suggestion_measure_draw = ImageDraw.Draw(_suggestion_measure_image)


@lru_cache(maxsize=64)
def _suggestion_font(font_size):
    return ImageFont.truetype(FONT_PATH, font_size)


def _suggestion_text_size(text, text_font, spacing=0):
    bbox = _suggestion_measure_draw.multiline_textbbox(
        (0, 0),
        text,
        font=text_font,
        spacing=spacing,
        align="center",
    )
    return bbox, bbox[2] - bbox[0], bbox[3] - bbox[1]


def _two_line_suggestion_variants(text):
    variants = []
    for split_at in range(1, len(text)):
        first = text[:split_at].rstrip()
        second = text[split_at:].lstrip()
        if first and second:
            variants.append(f"{first}\n{second}")
    return variants


def _ellipsize_suggestion(text, text_font, max_width):
    ellipsis = "…"
    low = 0
    high = len(text)

    while low < high:
        middle = (low + high + 1) // 2
        candidate = text[:middle].rstrip() + ellipsis
        _, text_width, _ = _suggestion_text_size(candidate, text_font)
        if text_width <= max_width:
            low = middle
        else:
            high = middle - 1

    return text[:low].rstrip() + ellipsis


@lru_cache(maxsize=256)
def _fit_suggestion_text(text, rect_width, rect_height):
    """박스에 맞는 표시 텍스트·폰트와 중앙 정렬용 측정값을 반환한다."""

    padding_x = max(8, int(rect_width * 0.04))
    padding_y = max(4, int(rect_height * 0.10))
    max_width = max(1, rect_width - 2 * padding_x)
    max_height = max(1, rect_height - 2 * padding_y)
    preferred_size = max(18, int(rect_height * 0.30))
    minimum_size = max(12, int(rect_height * 0.14))

    # 한 줄 표시를 우선하며 최소 가독 크기까지 단계적으로 축소한다.
    for font_size in range(preferred_size, minimum_size - 1, -1):
        text_font = _suggestion_font(font_size)
        bbox, text_width, text_height = _suggestion_text_size(text, text_font)
        if text_width <= max_width and text_height <= max_height:
            return text, text_font, bbox, text_width, text_height, 0

    # 한 줄로 들어가지 않을 때만 최대 두 줄을 허용한다.
    variants = _two_line_suggestion_variants(text)
    for font_size in range(preferred_size, minimum_size - 1, -1):
        text_font = _suggestion_font(font_size)
        spacing = max(2, int(font_size * 0.15))
        fitting_variants = []
        for variant in variants:
            bbox, text_width, text_height = _suggestion_text_size(
                variant,
                text_font,
                spacing,
            )
            if text_width <= max_width and text_height <= max_height:
                line_widths = [
                    _suggestion_text_size(line, text_font)[1]
                    for line in variant.splitlines()
                ]
                fitting_variants.append(
                    (
                        max(line_widths),
                        abs(line_widths[0] - line_widths[1]),
                        variant,
                        bbox,
                        text_width,
                        text_height,
                    )
                )

        if fitting_variants:
            (
                _,
                _,
                fitted_text,
                bbox,
                text_width,
                text_height,
            ) = min(fitting_variants)
            return (
                fitted_text,
                text_font,
                bbox,
                text_width,
                text_height,
                spacing,
            )

    # 두 줄도 최소 크기에 들어가지 않을 때만 표시 문자열을 말줄임한다.
    text_font = _suggestion_font(minimum_size)
    fitted_text = _ellipsize_suggestion(text, text_font, max_width)
    bbox, text_width, text_height = _suggestion_text_size(
        fitted_text,
        text_font,
    )
    return fitted_text, text_font, bbox, text_width, text_height, 0


def _draw_suggestion(
    draw,
    box,
    text,
    index,
    hovered_index,
    dwell_index,
    dwell_ratio,
    show_cursor,
    locked_index,
):
    """추천 슬롯 하나를 그린다.

    box는 실제로 그릴 사각형 (x, y, right, bottom)이다. 고정 감지로 확대된
    슬롯만 자기 rect보다 큰 box를 받는다 — 나머지 슬롯의 위치·크기는 그대로다.
    """

    box_x = int(box[0])
    box_y = int(box[1])
    box_width = int(box[2]) - box_x
    box_height = int(box[3]) - box_y

    on_suggestion = bool(text) and hovered_index == index
    text_color = KEY_TEXT_COLOR

    if text and locked_index == index:
        bg_color = DWELL_BG_END
        border_color = DWELL_BORDER_END
        text_color = KEY_TEXT_COLOR_DWELL
    elif (
        on_suggestion
        and dwell_index == index
        and show_cursor
    ):
        ratio = max(0.0, min(1.0, dwell_ratio))
        bg_color = tuple(
            int(HOVER_BG[i] + (DWELL_BG_END[i] - HOVER_BG[i]) * ratio)
            for i in range(3)
        )
        border_color = tuple(
            int(
                HOVER_BORDER[i]
                + (DWELL_BORDER_END[i] - HOVER_BORDER[i]) * ratio
            )
            for i in range(3)
        )
        if ratio > 0.5:
            text_color = KEY_TEXT_COLOR_DWELL
    elif on_suggestion and show_cursor:
        bg_color = HOVER_BG
        border_color = HOVER_BORDER
    elif on_suggestion and not show_cursor:
        bg_color = (255, 235, 0)
        border_color = (255, 120, 0)
    else:
        bg_color = IDLE_BG
        border_color = IDLE_BORDER

    radius = int(min(box_width, box_height) * 0.18)
    draw.rounded_rectangle(
        [box_x, box_y, box_x + box_width - 1, box_y + box_height - 1],
        radius=radius,
        fill=bg_color,
        outline=border_color,
        width=2,
    )

    if not text:
        return

    if (
        on_suggestion
        and dwell_index == index
        and dwell_ratio > 0
    ):
        bar_width = int(box_width * min(1.0, dwell_ratio))
        bar_right = box_x + max(1, min(box_width, bar_width)) - 1
        draw.rounded_rectangle(
            [
                box_x,
                box_y + box_height - 6,
                bar_right,
                box_y + box_height - 1,
            ],
            radius=3,
            fill=PROGRESS_BAR_COLOR,
        )

    (
        display_text,
        text_font,
        bbox,
        text_width,
        text_height,
        spacing,
    ) = _fit_suggestion_text(text, box_width, box_height)
    text_x = box_x + (box_width - text_width) // 2 - bbox[0]
    text_y = box_y + (box_height - text_height) // 2 - bbox[1]
    draw.multiline_text(
        (text_x, text_y),
        display_text,
        font=text_font,
        fill=text_color,
        spacing=spacing,
        align="center",
    )


def draw_suggestion_boxes(
    img,
    suggestions,
    suggestion_rects=None,
    hovered_index=None,
    dwell_index=None,
    dwell_ratio=0.0,
    show_cursor=True,
    locked_index=None,
    expanded_index=None,
    expanded_rect=None,
):
    """세 추천 박스와 내용이 있는 후보의 hover/선택 진행을 표시한다.

    expanded_index/expanded_rect가 주어지면(고정 감지로 확대된 추천) 그 슬롯만
    나머지를 그린 뒤 맨 위에 덮어 그린다. 다른 슬롯의 위치·크기는 그대로다 —
    키캡과 같은 규칙이다.
    """

    rects = (
        LAYOUT["suggestion_rects"]
        if suggestion_rects is None
        else suggestion_rects
    )
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)

    expand_index = (
        expanded_index
        if expanded_rect is not None
        else None
    )

    for index, rect in enumerate(rects):

        if index == expand_index:
            continue

        _draw_suggestion(
            draw,
            (rect.x, rect.y, rect.right, rect.bottom),
            suggestions[index] if index < len(suggestions) else "",
            index,
            hovered_index,
            dwell_index,
            dwell_ratio,
            show_cursor,
            locked_index,
        )

    if expand_index is not None and expand_index < len(rects):

        _draw_suggestion(
            draw,
            expanded_rect,
            (
                suggestions[expand_index]
                if expand_index < len(suggestions)
                else ""
            ),
            expand_index,
            hovered_index,
            dwell_index,
            dwell_ratio,
            show_cursor,
            locked_index,
        )

    return np.array(img_pil)


def drawAll(
    img,
    buttonList,
    gaze_x,
    gaze_y,
    dwell_key,
    dwell_ratio,
    show_cursor,
    locked_key=None,
    expanded_button=None,
    expanded_rect=None
):
    """키보드 렌더링.

    expanded_button/expanded_rect가 주어지면(고정 감지로 확대된 키) 그 키만
    나머지를 전부 그린 뒤 맨 위에 덮어 그린다. 다른 키의 위치와 크기는 전혀
    바뀌지 않는다 — 레이아웃은 정적이고, 확대된 키가 이웃 위에 겹쳐 보일
    뿐이다.

    gaze_x/gaze_y는 호출부 호환을 위해 남겨 둔 인자다. hover 표시는 좌표를
    다시 검사하지 않고 판정 결과(dwell_key)를 그대로 따른다 — 확장된 히트박스
    로 잡힌 hover도 화면에 그대로 반영되어야 하기 때문이다.
    """

    img_pil = Image.fromarray(img)

    draw = ImageDraw.Draw(img_pil)

    expand_target = (
        expanded_button
        if expanded_rect is not None
        else None
    )

    expand_target_drawn = False

    for button in buttonList:

        if button is expand_target:
            expand_target_drawn = True
            continue

        rect = button.rect

        _draw_key(
            draw,
            button,
            (rect.x, rect.y, rect.right, rect.bottom),
            dwell_key,
            dwell_ratio,
            show_cursor,
            locked_key,
        )

    if expand_target_drawn:

        rect = expand_target.rect

        # 글자도 키캡과 같은 비율로 키운다. 가로/세로 확대율이 다르면
        # 작은 쪽을 따라가 글자가 키 밖으로 넘치지 않게 한다.
        font_scale = min(
            (expanded_rect[2] - expanded_rect[0]) / max(1, rect.width),
            (expanded_rect[3] - expanded_rect[1]) / max(1, rect.height),
        )

        _draw_key(
            draw,
            expand_target,
            expanded_rect,
            dwell_key,
            dwell_ratio,
            show_cursor,
            locked_key,
            font_scale=max(1.0, font_scale),
        )

    return np.array(img_pil)


def draw_gaze_cursor(
    img,
    gaze_x,
    gaze_y,
    fixation_count
):
    """PNG 커서 렌더링"""

    if gaze_x < 0:
        return img

    if cursor_img is None:
        return img

    cursor_size = 48

    cursor = cv2.resize(
        cursor_img,
        (cursor_size, cursor_size)
    )

    h, w = cursor.shape[:2]

    x = gaze_x
    y = gaze_y

    if (
        x + w > img.shape[1]
        or
        y + h > img.shape[0]
        or
        x < 0
        or
        y < 0
    ):
        return img

    roi = img[y:y+h, x:x+w]

    # PNG 투명도 처리

    if cursor.shape[2] == 4:

        alpha = cursor[:, :, 3] / 255.0

        for c in range(3):

            roi[:, :, c] = (
                (1 - alpha) * roi[:, :, c]
                +
                alpha * cursor[:, :, c]
            )

    else:

        roi[:] = cursor

    # 좌표 출력

    cv2.putText(
        img,
        f"({gaze_x}, {gaze_y})",
        (gaze_x + 55, gaze_y + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1
    )

    # 응시 고정 표시

    if fixation_count >= FIXATION_FRAMES:

        cv2.circle(
            img,
            (gaze_x + 15, gaze_y + 15),
            5,
            (0, 255, 120),
            -1
        )

    return img
 
 
def draw_status_bar(img, is_korean, fixation_count):
    """하단 상태 바 렌더링."""

    from src.config import DWELL_SEC

    status = (
        f"     Dwell: {DWELL_SEC}s"
        f"  |  Fixation: {fixation_count}f"
    )

    # 영어/숫자는 OpenCV 사용 가능
    cv2.putText(
        img,
        status,
        (SCREEN_W - 340, SCREEN_H - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (150, 150, 150),
        1
    )

    # 한글은 PIL 사용
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)

    draw.text(
        (20, SCREEN_H - 60),
        "r : 재캘리브레이션   t : 시선정확도테스트   m : 입벌림 입력 방식 변경\n"
        "k : 커서 표시/숨기기   o : Raw   p : Pose   h : SQPnP   g : Ridge   b : 전체마커표시   q : 종료",
        font=small_font,
        fill=(150, 150, 150)
    )

    return np.array(img_pil)
 
 
def draw_test_complete_overlay(img):
    """테스트 완료 팝업 오버레이 렌더링."""
 
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
 
    draw.rectangle([250, 220, 1030, 430], fill=(0, 0, 0))
 
    draw.text(
        (470, 270),
        "테스트 완료!",
        font=font,
        fill=(0, 255, 0)
    )
 
    draw.text(
        (320, 340),
        "일반 키보드 모드로 전환됩니다.",
        font=font,
        fill=(255, 255, 255)
    )
 
    return np.array(img_pil)
 
 
def draw_text_area(img, current_text, target_text=None):
    """
    상단 입력 문장 영역 렌더링.

    src.keyboard.LAYOUT["input_rect"]를 그대로 사용한다 — 이 값은
    확인 버튼(confirm_rect) 폭만큼 이미 오른쪽 공간을 제외하고 계산되어 있다.
    """

    x, y, w, h = LAYOUT["input_rect"]

    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)

    radius = int(h * 0.22)

    draw.rounded_rectangle(
        [x, y, x + w, y + h],
        radius=radius,
        fill=(255, 255, 255),
        outline=(210, 213, 219),
        width=2
    )

    text_x = x + int(h * 0.35)

    if target_text is not None:
        draw.text(
            (x, max(0, y - 26)),
            f"목표 문장 : {target_text}",
            font=small_font,
            fill=(22, 163, 74)
        )

    bbox = draw.textbbox((0, 0), current_text, font=font)
    text_h = bbox[3] - bbox[1]
    text_y = y + (h - text_h) // 2 - bbox[1]

    draw.text(
        (text_x, text_y),
        current_text,
        font=font,
        fill=(30, 41, 59)
    )

    return np.array(img_pil)

# 캘리브레이션 가이드 안내 문구
def show_calibration_guide():

    start = time.time()

    while True:

        canvas = np.zeros(
            (SCREEN_H, SCREEN_W, 3),
            dtype=np.uint8
        )

        img_pil = Image.fromarray(canvas)
        draw = ImageDraw.Draw(img_pil)

        text = "점을 계속 바라보세요"

        bbox = draw.textbbox(
            (0, 0),
            text,
            font=font
        )

        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        draw.text(
            (
                (SCREEN_W - tw) // 2,
                (SCREEN_H - th) // 2
            ),
            text,
            font=font,
            fill=(255,255,255)
        )

        canvas = np.array(img_pil)

        cv2.imshow(
            "Eye Keyboard",
            canvas
        )

        if time.time() - start >= 2.0:
            break

        cv2.waitKey(1)
        
def draw_mouth_calibration_screen(
    instruction,
    mar,
    progress,
    remaining
):
    canvas = np.zeros(
        (SCREEN_H, SCREEN_W, 3),
        dtype=np.uint8
    )

    img_pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img_pil)

    title = "입벌림 캘리브레이션"

    draw.text(
        (SCREEN_W // 2, SCREEN_H // 2 - 220),
        title,
        font=font,
        fill=(255, 255, 255),
        anchor="mm"
    )

    draw.text(
        (SCREEN_W // 2, SCREEN_H // 2 - 150),
        instruction,
        font=font,
        fill=(255, 255, 255),
        anchor="mm"
    )

    draw.text(
        (SCREEN_W // 2, SCREEN_H // 2 - 60),
        f"MAR: {mar:.3f}",
        font=small_font,
        fill=(255, 180, 80),
        anchor="mm"
    )

    draw.text(
        (SCREEN_W // 2, SCREEN_H // 2 - 20),
        f"진행률: {progress:.2f}",
        font=small_font,
        fill=(200, 200, 200),
        anchor="mm"
    )

    draw.text(
        (SCREEN_W // 2, SCREEN_H // 2 + 20),
        f"남은 시간: {remaining:.1f}초",
        font=small_font,
        fill=(0, 255, 255),
        anchor="mm"
    )

    draw.text(
        (20, SCREEN_H - 40),
        "r : 다시하기   q : 종료",
        font=small_font,
        fill=(150, 150, 150)
    )

    return np.array(img_pil)
# ── 독립 타겟팅 정확도 테스트 화면 ─────────────────────────────

def _draw_centered_pil_text(
    draw,
    text,
    y,
    text_font,
    fill,
    canvas_width=SCREEN_W
):
    """PIL로 한글 문장을 화면 가로 중앙에 출력한다."""

    bbox = draw.textbbox(
        (0, 0),
        text,
        font=text_font
    )

    text_w = bbox[2] - bbox[0]

    draw.text(
        (
            (canvas_width - text_w) // 2,
            y
        ),
        text,
        font=text_font,
        fill=fill
    )


def draw_targeting_test_screen(
    targeting_runner,
    gaze_x=None,
    gaze_y=None
):
    """
    키보드와 분리된 원형 타겟팅 테스트 화면을 그린다.

    - 중간 크기의 원형 타겟
    - 진행 횟수
    - 현재 성공 횟수
    - 남은 시간
    - 드웰 진행 원
    """

    canvas = np.zeros(
        (SCREEN_H, SCREEN_W, 3),
        dtype=np.uint8
    )

    canvas[:] = (245, 246, 248)

    title_font = ImageFont.truetype(
        FONT_PATH,
        max(30, int(SCREEN_H * 0.045))
    )

    info_font = ImageFont.truetype(
        FONT_PATH,
        max(20, int(SCREEN_H * 0.027))
    )

    instruction_font = ImageFont.truetype(
        FONT_PATH,
        max(22, int(SCREEN_H * 0.030))
    )

    results = targeting_runner.get_results()

    current_trial = (
        targeting_runner.current_trial_number
    )

    success_count = results["success_count"]

    prepare_remaining = (
        targeting_runner.get_prepare_remaining_sec()
    )

    if prepare_remaining > 0.0:
        time_text = f"이동 준비 {prepare_remaining:.1f}초"
    else:
        remaining_time = (
            targeting_runner.get_remaining_time()
        )
        time_text = f"남은 시간 {remaining_time:.1f}초"

    # 상단 카드 영역
    cv2.rectangle(
        canvas,
        (0, 0),
        (SCREEN_W, 125),
        (255, 255, 255),
        -1
    )

    cv2.line(
        canvas,
        (0, 125),
        (SCREEN_W, 125),
        (215, 218, 223),
        2
    )

    img_pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img_pil)

    _draw_centered_pil_text(
        draw,
        "타겟팅 정확도 테스트",
        18,
        title_font,
        (35, 38, 45)
    )

    info_text = (
        f"진행 {current_trial} / "
        f"{targeting_runner.total_trials}"
        f"    |    성공 {success_count}"
        f"    |    {time_text}"
    )

    _draw_centered_pil_text(
        draw,
        info_text,
        75,
        info_font,
        (85, 90, 100)
    )

    canvas = np.array(img_pil)

    target = targeting_runner.current_target

    if target is not None:
        target_x, target_y = target
        radius = targeting_runner.target_radius

        # 타겟 바깥 그림자
        cv2.circle(
            canvas,
            (target_x + 5, target_y + 7),
            radius,
            (210, 213, 220),
            -1
        )

        # 중간 크기의 원형 타겟
        cv2.circle(
            canvas,
            (target_x, target_y),
            radius,
            (220, 235, 255),
            -1
        )

        # 타겟 외곽선
        cv2.circle(
            canvas,
            (target_x, target_y),
            radius,
            (70, 120, 210),
            5
        )

        # 중심 표시
        cv2.circle(
            canvas,
            (target_x, target_y),
            max(8, radius // 8),
            (70, 120, 210),
            -1
        )

        # 드웰 진행률
        dwell_ratio = (
            targeting_runner.get_dwell_ratio()
        )

        if dwell_ratio > 0:
            end_angle = int(
                -90 + 360 * dwell_ratio
            )

            cv2.ellipse(
                canvas,
                (target_x, target_y),
                (radius + 10, radius + 10),
                0,
                -90,
                end_angle,
                (40, 180, 110),
                9
            )

    # 하단 안내 카드
    cv2.rectangle(
        canvas,
        (0, SCREEN_H - 90),
        (SCREEN_W, SCREEN_H),
        (255, 255, 255),
        -1
    )

    cv2.line(
        canvas,
        (0, SCREEN_H - 90),
        (SCREEN_W, SCREEN_H - 90),
        (215, 218, 223),
        2
    )

    img_pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img_pil)

    _draw_centered_pil_text(
        draw,
        "원 안을 바라보고 일정 시간 응시하세요",
        SCREEN_H - 62,
        instruction_font,
        (55, 60, 70)
    )

    return np.array(img_pil)


def draw_targeting_result_screen(
    targeting_runner
):
    """10회 타겟팅 평가가 끝난 뒤 결과 화면을 그린다."""

    canvas = np.zeros(
        (SCREEN_H, SCREEN_W, 3),
        dtype=np.uint8
    )

    canvas[:] = (245, 246, 248)

    title_font = ImageFont.truetype(
        FONT_PATH,
        max(34, int(SCREEN_H * 0.052))
    )

    metric_font = ImageFont.truetype(
        FONT_PATH,
        max(24, int(SCREEN_H * 0.034))
    )

    guide_font = ImageFont.truetype(
        FONT_PATH,
        max(20, int(SCREEN_H * 0.026))
    )

    results = targeting_runner.get_results()

    success_count = results["success_count"]
    failure_count = results["failure_count"]
    total_trials = results["total_trials"]
    hit_rate = results["hit_rate_percent"]
    average_reaction_time = (
        results["average_reaction_time_sec"]
    )

    # 중앙 결과 카드
    card_w = int(SCREEN_W * 0.62)
    card_h = int(SCREEN_H * 0.68)

    card_x1 = (SCREEN_W - card_w) // 2
    card_y1 = (SCREEN_H - card_h) // 2
    card_x2 = card_x1 + card_w
    card_y2 = card_y1 + card_h

    cv2.rectangle(
        canvas,
        (card_x1 + 8, card_y1 + 10),
        (card_x2 + 8, card_y2 + 10),
        (215, 218, 223),
        -1
    )

    cv2.rectangle(
        canvas,
        (card_x1, card_y1),
        (card_x2, card_y2),
        (255, 255, 255),
        -1
    )

    cv2.rectangle(
        canvas,
        (card_x1, card_y1),
        (card_x2, card_y2),
        (205, 210, 218),
        2
    )

    img_pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img_pil)

    _draw_centered_pil_text(
        draw,
        "타겟팅 테스트 완료",
        card_y1 + 45,
        title_font,
        (35, 38, 45)
    )

    metrics = [
        f"성공 횟수     {success_count} / {total_trials}",
        f"실패 횟수     {failure_count} / {total_trials}",
        f"타겟 명중률   {hit_rate:.1f}%",
        (
            "평균 반응시간   "
            f"{average_reaction_time:.2f}초"
        ),
    ]

    start_y = card_y1 + 155
    line_gap = max(55, int(SCREEN_H * 0.075))

    for index, metric in enumerate(metrics):
        _draw_centered_pil_text(
            draw,
            metric,
            start_y + index * line_gap,
            metric_font,
            (55, 60, 70)
        )

    _draw_centered_pil_text(
        draw,
        "Y 키: 다시 시작    |    ESC 키: 키보드로 돌아가기",
        card_y2 - 70,
        guide_font,
        (100, 105, 115)
    )

    return np.array(img_pil)
