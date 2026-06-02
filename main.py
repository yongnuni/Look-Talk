import cv2
import numpy as np

import src.hangul as hangul

from src.config import (
    SCREEN_W,
    SCREEN_H
)

from src.tracking.eye_tracking import (
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

from src.tracking.calibration import Calibrator
from src.tracking.gaze_pipeline import GazePipeline
from src.tracking.dwell import DwellController

from src.keyboard import (
    create_buttons,
    process_key,
    keys_kor_normal
)

from src.ui import (
    show_countdown,
    draw_calib_screen,
    drawAll,
    draw_gaze_cursor,
    draw_status_bar,
    draw_test_complete_overlay,
    draw_text_area,
    font
)

from tests.test_runner import TestRunner


def main():

    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    cv2.namedWindow(
        "Eye Keyboard",
        cv2.WINDOW_NORMAL
    )

    cv2.setWindowProperty(
        "Eye Keyboard",
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN
    )

    calibrator = Calibrator()
    gaze = GazePipeline()
    dwell = DwellController()
    tester = TestRunner()

    is_korean = True
    is_shift = False

    buttonList = create_buttons(keys_kor_normal)

    calib_canvas = np.zeros(
        (SCREEN_H, SCREEN_W, 3),
        dtype=np.uint8
    )

    print("Eye Keyboard 시작 | r: 재캘리브레이션 | q: 종료")

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as face_mesh:

        if not show_countdown(cap, face_mesh):
            cap.release()
            cv2.destroyAllWindows()
            return

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.flip(frame, 1)
            fh, fw = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = face_mesh.process(rgb)
            rgb.flags.writeable = True

            gaze_x = -1
            gaze_y = -1
            fixation_count = 0
            elapsed_ratio = 0.0

            if results.multi_face_landmarks:

                lms = results.multi_face_landmarks[0]

                draw_eye_contour(frame, lms, LEFT_EYE, fw, fh)
                draw_eye_contour(frame, lms, RIGHT_EYE, fw, fh)
                draw_iris_ring(frame, lms, LEFT_IRIS, LEFT_IRIS_RING, fw, fh, (0, 200, 255))
                draw_iris_ring(frame, lms, RIGHT_IRIS, RIGHT_IRIS_RING, fw, fh, (0, 200, 255))

                iris_x, iris_y = get_avg_iris(lms)
                blink = is_blink(lms)
                conf = iris_confidence(lms)

                # ── 캘리브레이션 ──────────────────────────────

                if not calibrator.done:

                    if not blink:
                        elapsed_ratio = calibrator.update(iris_x, iris_y, conf)

                    draw_calib_screen(calib_canvas, calibrator, elapsed_ratio)
                    cv2.imshow("Eye Keyboard", calib_canvas)

                    key = cv2.waitKey(1) & 0xFF

                    if key == ord('q'):
                        break
                    elif key == ord('r'):
                        calibrator.reset()

                    continue

                # ── 시선 파이프라인 ───────────────────────────

                sx, sy = calibrator.map_to_screen(iris_x, iris_y)

                gaze_x, gaze_y, fixation_count = gaze.update(sx, sy, conf, blink)

            # ── 드웰 클릭 ─────────────────────────────────────

            hovered_key, dwell_ratio, clicked_key = dwell.update(
                gaze_x, gaze_y, buttonList
            )

            if clicked_key:

                tester.on_key_press(clicked_key)

                (is_korean, is_shift, buttonList) = process_key(
                    clicked_key,
                    is_korean,
                    is_shift,
                    buttonList
                )

            # ── 렌더링 ────────────────────────────────────────

            kbd_bg = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
            kbd_bg[:] = (30, 30, 30)

            current_text = (
                hangul.finalText +
                hangul.compose_jamo_buffer()
            )

            target = tester.target_text if tester.active else None

            kbd_bg = draw_text_area(kbd_bg, current_text, target)

            # 테스트 완료 감지
            if tester.check_complete(current_text):
                hangul.finalText = ""
                hangul.jamo_buffer[:] = ['', '', '']

            kbd_bg = drawAll(kbd_bg, buttonList, gaze_x, gaze_y, dwell.dwell_key, dwell_ratio)

            if tester.is_showing_complete():
                kbd_bg = draw_test_complete_overlay(kbd_bg)

            kbd_bg = draw_gaze_cursor(kbd_bg, gaze_x, gaze_y, fixation_count)
            kbd_bg = draw_status_bar(kbd_bg, is_korean, fixation_count)

            cv2.imshow("Eye Keyboard", kbd_bg)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord('r'):
                calibrator.reset()
                gaze.reset()

                if not show_countdown(cap, face_mesh):
                    break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()