import cv2
import time
import numpy as np
import os

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
KEY_FONT_BASE_SIZE = max(18, int(LAYOUT["row_h"] * 0.42))

key_font = ImageFont.truetype(FONT_PATH, KEY_FONT_BASE_SIZE)

# 확대된 키는 라벨도 같이 커져야 하므로 크기별로 캐싱해 둔다.
_key_font_cache = {KEY_FONT_BASE_SIZE: key_font}


def _key_font_for(scale):
    size = max(12, int(KEY_FONT_BASE_SIZE * scale))
    cached = _key_font_cache.get(size)
    if cached is None:
        cached = ImageFont.truetype(FONT_PATH, size)
        _key_font_cache[size] = cached
    return cached


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
            CALIB_POINTS[i][0] * sw
        )

        py = int(
            CALIB_POINTS[i][1] * sh
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

    if calib.idx < len(CALIB_POINTS):

        tx = int(
            CALIB_POINTS[calib.idx][0] * sw
        )

        ty = int(
            CALIB_POINTS[calib.idx][1] * sh
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


def drawAll(
    img,
    buttonList,
    gaze_x,
    gaze_y,
    dwell_key,
    dwell_ratio,
    show_cursor,
    key_zoom=None
):

    img_pil = Image.fromarray(img)

    draw = ImageDraw.Draw(img_pil)

    # 확대된 키가 인접 키에 가려지지 않도록 마지막에 그린다.
    if key_zoom is not None:
        ordered = (
            [b for b in buttonList if not key_zoom.is_zoomed(b)]
            + [b for b in buttonList if key_zoom.is_zoomed(b)]
        )
    else:
        ordered = buttonList

    for button in ordered:

        x, y = button.pos
        w, h = button.size

        key = button.text

        # 고정 감지에 의한 확대 배율 (없으면 1.0 → 기존과 완전히 동일)
        zoom_scale = (
            key_zoom.get_scale(button)
            if key_zoom is not None
            else 1.0
        )

        on_key = (
            (
                x < gaze_x < x + w
                and
                y < gaze_y < y + h
            )
            or zoom_scale > 1.0
        )

        # k키 모드(커서 숨김)일 때의 기존 확대와 병합
        if on_key and not show_cursor:
            zoom_scale = max(zoom_scale, 1.20)

        if zoom_scale > 1.0:
            nw = int(w * zoom_scale)
            nh = int(h * zoom_scale)

            nx = x - (nw - w) // 2
            ny = y - (nh - h) // 2

        else:
            nx = x
            ny = y
            nw = w
            nh = h

        text_color = KEY_TEXT_COLOR

        if on_key and dwell_key == key and show_cursor:

            t = dwell_ratio

            bg_color = tuple(
                int(HOVER_BG[i] + (DWELL_BG_END[i] - HOVER_BG[i]) * t)
                for i in range(3)
            )

            border_color = tuple(
                int(HOVER_BORDER[i] + (DWELL_BORDER_END[i] - HOVER_BORDER[i]) * t)
                for i in range(3)
            )

            if t > 0.5:
                text_color = KEY_TEXT_COLOR_DWELL

        elif on_key and not show_cursor:

            bg_color = (255, 235, 0)
            border_color = (255, 120, 0)

        else:

            bg_color = IDLE_BG
            border_color = IDLE_BORDER

        radius = int(min(w, h) * 0.18)

        if on_key and not show_cursor:

            draw.rounded_rectangle(
                [nx-5, ny-5, nx+nw+5, ny+nh+5],
                radius=radius+5,
                outline=(255,255,0),
                width=6
            )

        draw.rounded_rectangle(
            [nx, ny, nx+nw, ny+nh],
            radius=radius,
            fill=bg_color,
            outline=border_color,
            width=2
        )

        if (
            on_key
            and
            dwell_key == key
            and
            dwell_ratio > 0
        ):

            bar_w = int(
                nw * dwell_ratio
            )

            draw.rounded_rectangle(
                [
                    nx,
                    ny+nh-6,
                    nx+bar_w,
                    ny+nh
                ],
                radius=3,
                fill=PROGRESS_BAR_COLOR
            )

        label = DISPLAY_LABELS.get(key, key)

        label_font = _key_font_for(zoom_scale)

        bbox = draw.textbbox((0, 0), label, font=label_font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        text_x = nx + (nw - text_w)//2 - bbox[0]
        text_y = ny + (nh - text_h)//2 - bbox[1]

        draw.text(
            (text_x, text_y),
            label,
            font=label_font,
            fill=text_color
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