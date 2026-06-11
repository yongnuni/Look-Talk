import cv2
import numpy as np
import csv
import math
import time
import os
from datetime import datetime
from PIL import Image, ImageDraw
from src.calibrations.baseline_manager import save_baseline
import viz
import matplotlib.pyplot as plt

import src.hangul as hangul

from src.config import (
    SCREEN_W,
    SCREEN_H,
    PX_PER_CM
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

from src.tracking.mouth import (
    MouthClickDetector,
    draw_mouth,
    mouth_aspect_ratio
)

from src.tracking.calibration import Calibrator
from src.calibrations.mouth_calibration import MouthCalibration
from src.tracking.gaze_pipeline import GazePipeline
from src.tracking.dwell import DwellController
from src.tracking.mouth import draw_mouth
from src.tracking.head_pose import estimate_head_pose

from src.keyboard import (
    create_buttons,
    process_key,
    keys_kor_normal
)

from src.ui import (
    show_countdown,
    show_calibration_guide,
    draw_calib_screen,
    drawAll,
    draw_gaze_cursor,
    draw_status_bar,
    draw_test_complete_overlay,
    draw_text_area,
    draw_mouth_calibration_screen,
    font
)

from tests.test_runner import TestRunner
from src.metrics.collector import MetricsCollector

# 9점 테스트 (개발용)

def run_gaze_accuracy_test(
    cap,
    face_mesh,
    calibrator,
    gaze,
    collector
):

    test_points = [

        (0.1, 0.1),
        (0.5, 0.1),
        (0.9, 0.1),

        (0.1, 0.5),
        (0.5, 0.5),
        (0.9, 0.5),

        (0.1, 0.9),
        (0.5, 0.9),
        (0.9, 0.9),
    ]

    results = []

    for idx, (rx, ry) in enumerate(test_points):

        target_x = int(SCREEN_W * rx)
        target_y = int(SCREEN_H * ry)

        collector.start_target(idx, target_x, target_y)

        samples_x = []
        samples_y = []

        start_time = time.time()

        while time.time() - start_time < 3.0:

            ret, frame = cap.read()

            if not ret:
                continue

            frame = cv2.flip(frame, 1)

            # ── 프레임 단위 기본값 (STB 신호용) ──
            face_detected = False
            gaze_x = -1
            gaze_y = -1

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            result = face_mesh.process(rgb)

            canvas = np.zeros(
                (SCREEN_H, SCREEN_W, 3),
                dtype=np.uint8
            )

            cv2.circle(
                canvas,
                (target_x, target_y),
                20,
                (0,255,255),
                -1
            )

            if result.multi_face_landmarks:

                lms = result.multi_face_landmarks[0]
                face_detected = True

                iris_x, iris_y = get_avg_iris(lms)

                sx, sy = calibrator.map_to_screen(
                    iris_x,
                    iris_y
                )

                blink = is_blink(lms)
                conf = iris_confidence(lms)
                gaze_x, gaze_y, _ = gaze.update(sx, sy, conf, blink)

                elapsed = time.time() - start_time

                if elapsed >= 1.0:
                    
                    samples_x.append(gaze_x)
                    samples_y.append(gaze_y)

                    collector.add_sample(gaze_x, gaze_y, iris_x, iris_y)

            # ── STB 프레임 통계 기록 (얼굴 미검출 프레임도 포함) ──
            gaze_valid = (gaze_x >= 0 and gaze_y >= 0)
            collector.add_frame(
                face_detected=face_detected,
                gaze_valid=gaze_valid,
                timestamp=time.time()
            )

            cv2.imshow(
                "Eye Keyboard",
                canvas
            )

            cv2.waitKey(1)

        if len(samples_x) == 0:
            collector.end_target()
            continue

        pred_x = np.mean(samples_x)
        pred_y = np.mean(samples_y)

        error = math.sqrt(
            (pred_x-target_x)**2 +
            (pred_y-target_y)**2
        )

        results.append([
            idx+1,
            target_x,
            target_y,
            pred_x,
            pred_y,
            error
        ])

        collector.end_target()

    errors = [r[5] for r in results]

    avg_error = np.mean(errors)
    max_error = np.max(errors)
    min_error = np.min(errors)
    std_error = np.std(errors)

    filename = datetime.now().strftime(
        "gaze_accuracy_%Y%m%d_%H%M%S.csv"
    )

    filepath = os.path.join(
        "gaze_accuracy_results",
        filename
    )

    with open(
        filepath,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "point",
            "target_x",
            "target_y",
            "pred_x",
            "pred_y",
            "error_px"
        ])

        writer.writerows(results)

        writer.writerow([])

        writer.writerow([
            "Average Error(px)",
            avg_error
        ])

        writer.writerow([
            "Max Error(px)",
            max_error
        ])

        writer.writerow([
            "Min Error(px)",
            min_error
        ])

        writer.writerow([
            "Std Error(px)",
            std_error
        ])

    print(
        f"\nCSV 저장 완료: {filepath}"
    )

    print("\n===== GAZE TEST =====")
    print(f"Average Error : {avg_error:.2f}px")
    print(f"Max Error : {max_error:.2f}px")
    print(f"Min Error : {min_error:.2f}px")
    print(f"Std Error : {std_error:.2f}px")
    print("=====================")

    # ── collector 내보내기 ──
    out_dir = "gaze_accuracy_results"

    collector.end_session()
    collector.export_csv(
        sessions_path=os.path.join(out_dir, "sessions.csv"),
        accuracy_path=os.path.join(out_dir, "gaze_accuracy.csv")
    )
    print("[metrics] collector CSV 저장 완료:", out_dir)


# 테스트 결과 자동 시각화 (개발용)

def show_session_popup(session_id):
    try:
        viz.setup_font()
        df = viz.load_data("gaze_accuracy_results")
        s = viz.get_session(df, session_id)

        if len(s) == 0:
            print(f"[popup] 세션을 찾을 수 없음: {session_id}")
            return

        print(viz.format_summary_line(viz.summarize_session(s)))

        screen_w, screen_h = viz.infer_screen_size(df)
        viz.plot_session_overview(s, screen_w, screen_h)
        plt.show()

    except Exception as e:
        print(f"[popup] 시각화 실패: {e}")

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
    mouth_calibrator = MouthCalibration()
    gaze = GazePipeline()
    dwell = DwellController()
    mouth = MouthClickDetector()
    tester = TestRunner()

    is_korean = True
    is_shift = False
    use_pose_corrected = False
    last_session_id = None

    last_gaze_x = SCREEN_W // 2
    last_gaze_y = SCREEN_H // 2

    buttonList = create_buttons(keys_kor_normal)

    calib_canvas = np.zeros(
        (SCREEN_H, SCREEN_W, 3),
        dtype=np.uint8
    )

    print(
        "Eye Keyboard 시작 | "
        "r: 재캘리브레이션 | "
        "t: 시선정확도테스트 | "
        "q: 종료"
    )

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
        
        show_calibration_guide()

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
            mouth_click = False
            hovered_key = None
            clicked_key = None
            dwell_ratio = 0.0
            mar = 0.0

            raw_sx = None
            raw_sy = None
            corrected_sx = None
            corrected_sy = None
            sx = None
            sy = None

            corrected_iris_x = None
            corrected_iris_y = None

            iris_x = 0.0
            iris_y = 0.0
            conf = 0.0

            head_pose = {
                "valid": False,
                "yaw": 0.0,
                "pitch": 0.0,
                "roll": 0.0,
                "face_scale": 0.0,
                "tz": 0.0,
                "face_center_x": 0.5,
                "face_center_y": 0.5,
            }

            if results.multi_face_landmarks:

                lms = results.multi_face_landmarks[0]

                head_pose = estimate_head_pose(
                    lms,
                    fw,
                    fh
                )

                draw_eye_contour(frame, lms, LEFT_EYE, fw, fh)
                draw_eye_contour(frame, lms, RIGHT_EYE, fw, fh)
                draw_iris_ring(frame, lms, LEFT_IRIS, LEFT_IRIS_RING, fw, fh, (0, 200, 255))
                draw_iris_ring(frame, lms, RIGHT_IRIS, RIGHT_IRIS_RING, fw, fh, (0, 200, 255))
                draw_mouth(frame,lms,fw,fh)

                iris_x, iris_y = get_avg_iris(lms)
                blink = is_blink(lms)
                conf = iris_confidence(lms)                
                

                # ── 캘리브레이션 ──────────────────────────────

                if not calibrator.done:

                    if not blink:
                        elapsed_ratio = calibrator.update(
                            iris_x,
                            iris_y,
                            conf,
                            head_pose=head_pose
                        )

                    draw_calib_screen(calib_canvas, calibrator, elapsed_ratio)
                    cv2.imshow("Eye Keyboard", calib_canvas)

                    key = cv2.waitKey(1) & 0xFF

                    if key == ord('q'):
                        break
                    elif key == ord('r'):
                        calibrator.reset()
                   
                    continue
                # ── 입벌림 캘리브레이션 ─────────────────────────
                if not mouth_calibrator.done:
                    mar = mouth_aspect_ratio(lms)
                    mouth_progress = mouth_calibrator.update(mar)
                    if mouth_calibrator.done:
                        mouth_result = mouth_calibrator.get_result_dict()

                        print("\n===== MOUTH CALIBRATION RESULT =====")
                        print(mouth_result)
                        print("====================================\n")

                        saved_path = save_baseline(
                            mouth_result=mouth_result
                    )

                        print(f"[baseline] 저장 완료: {saved_path}")
                        mouth = MouthClickDetector()
                        dwell.reset()

                    instruction = mouth_calibrator.get_instruction()
                    remaining = mouth_calibrator.get_remaining_time()

                    mouth_canvas = draw_mouth_calibration_screen(
                        instruction,
                        mar,
                        mouth_progress,
                        remaining
                    )

                    cv2.imshow("Eye Keyboard", mouth_canvas)

                    key = cv2.waitKey(1) & 0xFF

                    if key == ord('q'):
                        break

                    elif key == ord('r'):
                        mouth_calibrator.reset()

                    continue

                # ── 시선 파이프라인 ───────────────────────────
                # 1. 기존 방식 Raw 좌표
                raw_sx, raw_sy = calibrator.map_to_screen(
                    iris_x,
                    iris_y
                )

                # 2. face center / scale 기반 iris 입력 보정
                corrected_iris_x, corrected_iris_y = calibrator.compensate_iris_by_head_pose(
                    iris_x,
                    iris_y,
                    head_pose
                )

                # 3. 보정된 iris 좌표를 다시 화면 좌표로 변환
                corrected_sx, corrected_sy = calibrator.map_to_screen(
                    corrected_iris_x,
                    corrected_iris_y
                )

                if use_pose_corrected:
                    sx, sy = corrected_sx, corrected_sy
                else:
                    sx, sy = raw_sx, raw_sy

                # 4. 최종 gaze pipeline 입력 전 좌표 유효성 검사
                # 화면 가장자리 좌표는 invalid로 보지 않음.
                # None / NaN / inf 같은 진짜 비정상값만 막음.
                screen_coord_valid = (
                    sx is not None
                    and sy is not None
                    and np.isfinite(sx)
                    and np.isfinite(sy)
                )

                tracking_valid = False

                if not screen_coord_valid:
                    # 좌표 자체가 None/NaN/inf인 경우:
                    # 커서는 마지막 정상 위치에 유지하지만, 입력은 허용하지 않음
                    gaze_x = last_gaze_x
                    gaze_y = last_gaze_y
                    fixation_count = 0
                    tracking_valid = False

                else:
                    gaze_x, gaze_y, fixation_count = gaze.update(
                        sx,
                        sy,
                        conf,
                        blink,
                        head_pose=head_pose
                    )

                    if gaze_x == -1 and gaze_y == -1:
                        # gaze.update가 추적 실패를 반환한 경우:
                        # 커서는 유지하지만, 입력은 허용하지 않음
                        gaze_x = last_gaze_x
                        gaze_y = last_gaze_y
                        fixation_count = 0
                        tracking_valid = False

                    else:
                        # 표시용으로만 화면 안쪽에 제한
                        gaze_x = int(np.clip(gaze_x, 0, SCREEN_W - 1))
                        gaze_y = int(np.clip(gaze_y, 0, SCREEN_H - 1))

                        last_gaze_x = gaze_x
                        last_gaze_y = gaze_y
                        tracking_valid = True

            # ── 드웰 클릭 ─────────────────────────────────────

                if tracking_valid:
                    hovered_key, dwell_ratio, clicked_key = dwell.update(
                        gaze_x,
                        gaze_y,
                        buttonList
                    )

                    mouth_click, mar = mouth.update(
                        lms,
                        hovered_key
                    )
                else:
                    dwell.reset()
                    hovered_key = None
                    clicked_key = None
                    dwell_ratio = 0.0
                    mouth_click = False
                    mar = 0.0

                # 기존 드웰 클릭
                if clicked_key:
                    tester.on_key_press(clicked_key)

                    (is_korean, is_shift, buttonList) = process_key(
                        clicked_key,
                        is_korean,
                        is_shift,
                        buttonList
                    )

                # 입벌림 클릭
                if mouth_click and hovered_key:

                    tester.on_key_press(hovered_key)

                    (is_korean, is_shift, buttonList) = process_key(
                        hovered_key,
                        is_korean,
                        is_shift,
                        buttonList
                    )

                    print("MOUTH INPUT:", hovered_key)
                    

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

            if gaze_x < 0 or gaze_y < 0:
                gaze_x = last_gaze_x
                gaze_y = last_gaze_y
                fixation_count = 0

            kbd_bg = drawAll(kbd_bg, buttonList, gaze_x, gaze_y, dwell.dwell_key, dwell_ratio)

            if tester.is_showing_complete():
                kbd_bg = draw_test_complete_overlay(kbd_bg)
            
            mar_text = f"MAR: {mar:.2f}"   #입벌림 지표 표시
            cv2.putText(
                kbd_bg,
                mar_text,
                (SCREEN_W // 2 - 60, SCREEN_H - 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 100, 0),
                2
            )

            pose_text = (
                f"Valid:{head_pose['valid']} "
                f"Yaw:{head_pose['yaw']:.1f} "
                f"Pitch:{head_pose['pitch']:.1f} "
                f"Roll:{head_pose['roll']:.1f} "
                f"Scale:{head_pose['face_scale']:.1f} "
                f"Center:({head_pose['face_center_x']:.2f},{head_pose['face_center_y']:.2f})"
            )
            cv2.putText(
                kbd_bg,
                pose_text,
                (30, SCREEN_H - 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255) if head_pose["valid"] else (0, 0, 255),
                2
            )

            pose_delta = calibrator.get_pose_delta(head_pose)

            if pose_delta is not None:
                delta_text = (
                    f"dCenter:({pose_delta['delta_center_x']:.4f},"
                    f"{pose_delta['delta_center_y']:.4f}) "
                    f"dScale:{pose_delta['delta_scale']:.1f}"
                )
            else:
                delta_text = "dCenter:(None,None) dScale:None"

            cv2.putText(
                kbd_bg,
                delta_text,
                (30, SCREEN_H - 210),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )

            mode_text = "Mode: PoseCorrected" if use_pose_corrected else "Mode: Raw"

            cv2.putText(
                kbd_bg,
                mode_text,
                (30, SCREEN_H - 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if use_pose_corrected else (255, 255, 255),
                2
            )

            coord_text = (
                f"Raw:({raw_sx},{raw_sy}) "
                f"PoseCorrected:({corrected_sx},{corrected_sy}) "
                f"Active:({sx},{sy}) "
                f"Gaze:({gaze_x},{gaze_y}) "
                f"Iris:({iris_x:.4f},{iris_y:.4f})"
            )

            cv2.putText(
                kbd_bg,
                coord_text,
                (30, SCREEN_H - 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )

            if corrected_iris_x is not None and corrected_iris_y is not None:
                corrected_iris_text = (
                    f"Corrected Iris:({corrected_iris_x:.4f},{corrected_iris_y:.4f})"
                )
            else:
                corrected_iris_text = "Corrected Iris:(None,None)"

            cv2.putText(
                kbd_bg,
                corrected_iris_text,
                (30, SCREEN_H - 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )

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
            elif key == ord('p'):
                use_pose_corrected = not use_pose_corrected
                gaze.reset()
                print("use_pose_corrected:", use_pose_corrected)

                show_calibration_guide()

            elif key == ord('t'):

                if calibrator.done:

                    gaze.reset()
                    collector = MetricsCollector(
                        user_id="jeesoo",
                        dev_version="v0.1-raw",
                        px_per_cm=PX_PER_CM
                    )

                    run_gaze_accuracy_test(
                        cap,
                        face_mesh,
                        calibrator,
                        gaze,
                        collector
                    )

                    last_session_id = collector.session_id

    cap.release()
    cv2.destroyAllWindows()

    # ── 종료 시 마지막 세션 결과 팝업 (측정한 적 있을 때만) ──
    if last_session_id is not None:
        show_session_popup(last_session_id)


if __name__ == "__main__":
    main()