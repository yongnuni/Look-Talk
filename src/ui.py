import cv2
import time
import numpy as np

from PIL import ImageFont, ImageDraw, Image
from src.config import SCREEN_W, SCREEN_H, FIXATION_FRAMES

from src.config import (
    FONT_PATH,
    FONT_SIZE,
    SCREEN_W,
    SCREEN_H,
    COUNTDOWN_SEC,
    CALIB_POINTS
)

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

        draw.text(
            (15, 10),
            "카메라를 정면으로 바라봐 주세요",
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

    draw.text(
        (20, sh - 35),
        "r: 재시작   q: 종료",
        font=small_font,
        fill=(120, 120, 120)
    )

    canvas[:] = np.array(img_pil)


# ── 키보드 그리기 ─────────────────────────────────────────────

def drawAll(
    img,
    buttonList,
    gaze_x,
    gaze_y,
    dwell_key,
    dwell_ratio
):

    img_pil = Image.fromarray(img)

    draw = ImageDraw.Draw(img_pil)

    for button in buttonList:

        x, y = button.pos
        w, h = button.size

        key = button.text

        on_key = (
            x < gaze_x < x+w
            and
            y < gaze_y < y+h
        )

        if on_key and dwell_key == key:

            r = int(
                255 * dwell_ratio
            )

            bg_color = (
                r,
                100,
                200
            )

        elif on_key:

            bg_color = (
                100,
                100,
                200
            )

        else:

            bg_color = (
                80,
                80,
                80
            )

        draw.rectangle(
            [x, y, x+w, y+h],
            fill=bg_color
        )

        if (
            on_key
            and
            dwell_key == key
            and
            dwell_ratio > 0
        ):

            bar_w = int(
                w * dwell_ratio
            )

            draw.rectangle(
                [
                    x,
                    y+h-6,
                    x+bar_w,
                    y+h
                ],
                fill=(0,255,180)
            )

        draw.text(
            (x+10, y+15),
            key,
            font=font,
            fill=(255,255,255)
        )

    return np.array(img_pil)

def draw_gaze_cursor(img, gaze_x, gaze_y, fixation_count):
    """시선 커서 및 십자선 렌더링."""
 
    if gaze_x < 0:
        return img
 
    cursor_color = (
        (0, 255, 120)
        if fixation_count >= FIXATION_FRAMES
        else (0, 220, 255)
    )
 
    cv2.circle(img, (gaze_x, gaze_y), 18, cursor_color, 2)
    cv2.circle(img, (gaze_x, gaze_y), 5, cursor_color, -1)
 
    cv2.putText(
        img,
        f"({gaze_x}, {gaze_y})",
        (gaze_x + 20, gaze_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1
    )
 
    cv2.line(img, (gaze_x - 26, gaze_y), (gaze_x + 26, gaze_y), cursor_color, 1)
    cv2.line(img, (gaze_x, gaze_y - 26), (gaze_x, gaze_y + 26), cursor_color, 1)
 
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
        (20, SCREEN_H - 40),
        "r: 재캘리브레이션   q: 종료",
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
    """상단 텍스트 입력 영역 렌더링."""
 
    cv2.rectangle(img, (40, 20), (SCREEN_W - 40, 100), (0, 0, 0), -1)
 
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
 
    if target_text is not None:
        draw.text(
            (55, 5),
            f"입력 문장 : {target_text}",
            font=font,
            fill=(0, 255, 0)
        )
 
    draw.text(
        (55, 25),
        current_text,
        font=font,
        fill=(255, 255, 255)
    )
 
    return np.array(img_pil)