import cv2
import time
import numpy as np

import src.hangul as hangul

from src.config import (
    SCREEN_W,
    SCREEN_H,
    SMOOTH_ALPHA,
    DWELL_SEC,
    FIXATION_RADIUS,
    FIXATION_FRAMES
)

from src.eye_tracking import (
    mp_face_mesh,
    LEFT_EYE,
    RIGHT_EYE,
    LEFT_IRIS,
    RIGHT_IRIS,
    LEFT_IRIS_RING,
    RIGHT_IRIS_RING,
    get_avg_iris,
    is_blink,
    iris_confidence,
    draw_eye_contour,
    draw_iris_ring
)

from src.calibration import Calibrator

from src.keyboard import (
    create_buttons,
    process_key,
    keys_eng_normal
)

from src.ui import (
    show_countdown,
    draw_calib_screen,
    drawAll,
    font
)

from PIL import ImageDraw, Image

def main():

    cap = cv2.VideoCapture(0)

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        640
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        480
    )

    cv2.namedWindow(
        "Eye Keyboard",
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        "Eye Keyboard",
        SCREEN_W,
        SCREEN_H
    )

    calibrator = Calibrator()

    smooth_gaze = None

    is_korean = False
    is_shift = False

    buttonList = create_buttons(
        keys_eng_normal
    )

    calib_canvas = np.zeros(
        (SCREEN_H, SCREEN_W, 3),
        dtype=np.uint8
    )

    fixation_center = None
    fixation_count = 0

    dwell_key = None
    dwell_start = None

    cooldown_end = 0

    print(
        "Eye Keyboard 시작 | r: 재캘리브레이션 | q: 종료"
    )

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as face_mesh:

        if not show_countdown(
            cap,
            face_mesh
        ):
            cap.release()
            cv2.destroyAllWindows()
            return

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.flip(
                frame,
                1
            )

            fh, fw = frame.shape[:2]

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            rgb.flags.writeable = False

            results = face_mesh.process(
                rgb
            )

            rgb.flags.writeable = True

            gaze_x = -1
            gaze_y = -1

            elapsed_ratio = 0.0

            now = time.time()

            if results.multi_face_landmarks:

                lms = results.multi_face_landmarks[0]

                draw_eye_contour(
                    frame,
                    lms,
                    LEFT_EYE,
                    fw,
                    fh
                )

                draw_eye_contour(
                    frame,
                    lms,
                    RIGHT_EYE,
                    fw,
                    fh
                )

                draw_iris_ring(
                    frame,
                    lms,
                    LEFT_IRIS,
                    LEFT_IRIS_RING,
                    fw,
                    fh,
                    (0,200,255)
                )

                draw_iris_ring(
                    frame,
                    lms,
                    RIGHT_IRIS,
                    RIGHT_IRIS_RING,
                    fw,
                    fh,
                    (0,200,255)
                )

                iris_x, iris_y = get_avg_iris(
                    lms
                )

                blink = is_blink(lms)

                conf = iris_confidence(
                    lms
                )

                if not calibrator.done:

                    if not blink:

                        elapsed_ratio = (
                            calibrator.update(
                                iris_x,
                                iris_y,
                                conf
                            )
                        )

                    draw_calib_screen(
                        calib_canvas,
                        calibrator,
                        elapsed_ratio
                    )

                    cv2.imshow(
                        "Eye Keyboard",
                        calib_canvas
                    )

                    key = (
                        cv2.waitKey(1)
                        & 0xFF
                    )

                    if key == ord('q'):
                        break

                    elif key == ord('r'):
                        calibrator.reset()

                    continue

                sx, sy = (
                    calibrator.map_to_screen(
                        iris_x,
                        iris_y
                    )
                )

                if (
                    sx is not None
                    and
                    not blink
                    and
                    conf > 0.3
                ):

                    if smooth_gaze is None:

                        smooth_gaze = [
                            float(sx),
                            float(sy)
                        ]

                    alpha = (
                        SMOOTH_ALPHA *
                        max(0.3, conf)
                    )

                    smooth_gaze[0] += (
                        alpha *
                        (
                            sx -
                            smooth_gaze[0]
                        )
                    )

                    smooth_gaze[1] += (
                        alpha *
                        (
                            sy -
                            smooth_gaze[1]
                        )
                    )

                    sx_s = smooth_gaze[0]
                    sy_s = smooth_gaze[1]

                    if fixation_center is None:
                        fixation_center = [
                            sx_s,
                            sy_s
                        ]

                        fixation_count = 1

                    else:

                        dist = np.hypot(
                            sx_s -
                            fixation_center[0],
                            sy_s -
                            fixation_center[1]
                        )

                        if dist < FIXATION_RADIUS:

                            fixation_count += 1

                            fixation_center[0] += (
                                0.05 *
                                (
                                    sx_s -
                                    fixation_center[0]
                                )
                            )

                            fixation_center[1] += (
                                0.05 *
                                (
                                    sy_s -
                                    fixation_center[1]
                                )
                            )

                        else:

                            fixation_center = [
                                sx_s,
                                sy_s
                            ]

                            fixation_count = 1

                    if fixation_count >= FIXATION_FRAMES:

                        gaze_x = int(
                            fixation_center[0]
                        )

                        gaze_y = int(
                            fixation_center[1]
                        )

                    else:

                        gaze_x = int(sx_s)
                        gaze_y = int(sy_s)

                else:

                    fixation_center = None
                    fixation_count = 0

                        # ── 드웰 클릭 로직 ────────────────────────────────

            dwell_ratio = 0.0

            hovered_key = None

            if gaze_x >= 0 and now > cooldown_end:

                for button in buttonList:

                    bx, by = button.pos
                    bw, bh = button.size

                    if (
                        bx < gaze_x < bx + bw
                        and
                        by < gaze_y < by + bh
                    ):

                        hovered_key = button.text
                        break

                if hovered_key:

                    if hovered_key != dwell_key:

                        dwell_key = hovered_key

                        dwell_start = now

                    else:

                        elapsed = now - dwell_start

                        dwell_ratio = min(
                            1.0,
                            elapsed / DWELL_SEC
                        )

                        if dwell_ratio >= 1.0:

                            (
                                is_korean,
                                is_shift,
                                buttonList
                            ) = process_key(
                                dwell_key,
                                is_korean,
                                is_shift,
                                buttonList
                            )

                            cooldown_end = now + 0.4

                            dwell_key = None
                            dwell_start = None

                else:

                    dwell_key = None
                    dwell_start = None

            # ── 렌더링 ────────────────────────────────────────

            kbd_bg = np.zeros(
                (
                    SCREEN_H,
                    SCREEN_W,
                    3
                ),
                dtype=np.uint8
            )

            kbd_bg[:] = (
                30,
                30,
                30
            )

            cv2.rectangle(
                kbd_bg,
                (40,20),
                (SCREEN_W - 40,100),
                (0,0,0),
                -1
            )

            img_pil = Image.fromarray(
                kbd_bg
            )

            draw = ImageDraw.Draw(
                img_pil
            )

            draw.text(
                (55,25),
                hangul.finalText +
                hangul.compose_jamo_buffer(),
                font=font,
                fill=(255,255,255)
            )

            kbd_bg = np.array(
                img_pil
            )

            kbd_bg = drawAll(
                kbd_bg,
                buttonList,
                gaze_x,
                gaze_y,
                dwell_key,
                dwell_ratio
            )

            # ── 시선 커서 ────────────────────────────────────

            if gaze_x >= 0:

                cursor_color = (
                    (0,255,120)
                    if fixation_count >= FIXATION_FRAMES
                    else (0,220,255)
                )

                cv2.circle(
                    kbd_bg,
                    (gaze_x, gaze_y),
                    18,
                    cursor_color,
                    2
                )

                cv2.circle(
                    kbd_bg,
                    (gaze_x, gaze_y),
                    5,
                    cursor_color,
                    -1
                )

                cv2.circle(
                    kbd_bg,
                    (gaze_x, gaze_y),
                    5,
                    cursor_color,
                    -1
                )

                cv2.putText(
                    kbd_bg,
                    f"({gaze_x}, {gaze_y})",
                    (gaze_x + 20, gaze_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1
                )

                cv2.line(
                    kbd_bg,
                    (gaze_x - 26, gaze_y),
                    (gaze_x + 26, gaze_y),
                    cursor_color,
                    1
                )

                cv2.line(
                    kbd_bg,
                    (gaze_x, gaze_y - 26),
                    (gaze_x, gaze_y + 26),
                    cursor_color,
                    1
                )

            status = (
                f"{'한글' if is_korean else 'ENG'}"
                f"  |  드웰: {DWELL_SEC}s"
                f"  |  고정: {fixation_count}f"
            )

            cv2.putText(
                kbd_bg,
                status,
                (SCREEN_W - 340, SCREEN_H - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (150,150,150),
                1
            )

            cv2.putText(
                kbd_bg,
                "r: 재캘리브레이션   q: 종료",
                (20, SCREEN_H - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (100,100,100),
                1
            )

            cv2.imshow(
                "Eye Keyboard",
                kbd_bg
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord('r'):

                calibrator.reset()

                smooth_gaze = None

                fixation_center = None

                fixation_count = 0

                if not show_countdown(
                    cap,
                    face_mesh
                ):
                    break

    cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()