import os
import time
import math
import csv
from datetime import datetime

import cv2
import numpy as np
import matplotlib.pyplot as plt
import src.viz.viz as viz

from src.config import SCREEN_W, SCREEN_H
from src.tracking.eye_tracking import get_avg_iris, iris_confidence
from src.tracking.head_pose import estimate_head_pose, estimate_sqpnp_headpose
from src.tracking.feature_builder import build_features
from src.metrics.collector import MetricsCollector
from src.vision.preprocessing import auto_brightness

MAX_SQPNP_DELTA_PX = 120

# 9점 테스트 (개발용)

def run_gaze_accuracy_test(
    cap,
    face_mesh,
    calibrator,
    gaze,
    collector,
    blink_detector,
    use_pose_corrected,
    use_sqpnp_corrected=False,
    use_ridge=False,
):
    os.makedirs("gaze_accuracy_results", exist_ok=True)

    if use_ridge:
        mode_name = "ridge_hybrid"
    elif use_sqpnp_corrected:
        mode_name = "sqpnp_corrected"
    elif use_pose_corrected:
        mode_name = "pose_corrected"
    else:
        mode_name = "raw"

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

            frame = auto_brightness(frame)

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

                fh, fw = frame.shape[:2]

                head_pose = estimate_head_pose(
                    lms,
                    fw,
                    fh
                )

                sqpnp_headpose = estimate_sqpnp_headpose(
                    lms,
                    fw,
                    fh
                )

                features = build_features(
                    iris_x,
                    iris_y,
                    lms,
                    head_pose,
                    frame_width=fw
                )

                # Raw 좌표
                raw_sx, raw_sy = calibrator.map_to_screen(
                    iris_x,
                    iris_y
                )

                blink_detector.update(lms)
                # 머리 자세 보정 좌표
                corrected_iris_x, corrected_iris_y = (
                    calibrator.compensate_iris_by_head_pose(
                        iris_x,
                        iris_y,
                        head_pose
                    )
                )

                corrected_sx, corrected_sy = calibrator.map_to_screen(
                    corrected_iris_x,
                    corrected_iris_y
                )

                sqpnp_corrected_iris_x, sqpnp_corrected_iris_y = (
                    calibrator.compensate_iris_by_head_pose(
                        iris_x,
                        iris_y,
                        sqpnp_headpose
                    )
                )

                sqpnp_corrected_sx, sqpnp_corrected_sy = calibrator.map_to_screen(
                    sqpnp_corrected_iris_x,
                    sqpnp_corrected_iris_y
                )

                if (
                    use_sqpnp_corrected
                    and raw_sx is not None
                    and raw_sy is not None
                    and sqpnp_corrected_sx is not None
                    and sqpnp_corrected_sy is not None
                ):
                    sqpnp_delta_x = np.clip(
                        sqpnp_corrected_sx - raw_sx,
                        -MAX_SQPNP_DELTA_PX,
                        MAX_SQPNP_DELTA_PX
                    )
                    sqpnp_delta_y = np.clip(
                        sqpnp_corrected_sy - raw_sy,
                        -MAX_SQPNP_DELTA_PX,
                        MAX_SQPNP_DELTA_PX
                    )
                    sqpnp_corrected_sx = int(raw_sx + sqpnp_delta_x)
                    sqpnp_corrected_sy = int(raw_sy + sqpnp_delta_y)

                # ── 릿지 하이브리드 좌표 ──
                ridge_sx, ridge_sy = calibrator.map_to_screen_features(
                    features
                )

                if use_ridge and ridge_sx is not None and ridge_sy is not None:
                    sx, sy = ridge_sx, ridge_sy
                elif use_ridge:
                    # 릿지 실패 프레임은 raw로 폴백 (커서 유지 원칙)
                    sx, sy = raw_sx, raw_sy
                elif use_sqpnp_corrected:
                    sx, sy = sqpnp_corrected_sx, sqpnp_corrected_sy
                elif use_pose_corrected:
                    sx, sy = corrected_sx, corrected_sy
                else:
                    sx, sy = raw_sx, raw_sy

                blink = blink_detector.is_closed
                conf = iris_confidence(lms)

                gaze_x, gaze_y, _ = gaze.update(
                    sx,
                    sy,
                    conf,
                    blink,
                    head_pose=head_pose
                )

                elapsed = time.time() - start_time

                
                if elapsed >= 1.0:

                    tracking_valid = (
                        gaze_x is not None
                        and gaze_y is not None
                        and np.isfinite(gaze_x)
                        and np.isfinite(gaze_y)
                        and not (gaze_x == -1 and gaze_y == -1)
                    )

                    if tracking_valid:
                        samples_x.append(gaze_x)
                        samples_y.append(gaze_y)

                        collector.add_sample(
                            gaze_x,
                            gaze_y,
                            iris_x * fw,
                            iris_y * fh
                        )

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
            mode_name,
            idx+1,
            target_x,
            target_y,
            pred_x,
            pred_y,
            error
        ])

        collector.end_target()

    errors = [r[6] for r in results]

    avg_error = np.mean(errors)
    max_error = np.max(errors)
    min_error = np.min(errors)
    std_error = np.std(errors)

    filename = datetime.now().strftime(
        f"gaze_accuracy_{mode_name}_%Y%m%d_%H%M%S.csv"
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
            "mode",
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
    print(f"Mode : {mode_name}")
    print(f"Average Error : {avg_error:.2f}px")
    print(f"Max Error : {max_error:.2f}px")
    print(f"Min Error : {min_error:.2f}px")
    print(f"Std Error : {std_error:.2f}px")
    print("=====================")

    # ── collector 내보내기 ──
    out_dir = "gaze_accuracy_results"

    collector.export_csv(
        sessions_path=os.path.join(
            out_dir,
            f"sessions_v{MetricsCollector.SCHEMA_VERSION}.csv"
        ),
        accuracy_path=os.path.join(
            out_dir,
            f"gaze_accuracy_v{MetricsCollector.SCHEMA_VERSION}.csv"
        ),
        export_session=False,
        export_accuracy=True
    )
    print("[metrics] collector CSV 저장 완료:", out_dir)


# 테스트 결과 자동 시각화 (개발용)

def show_session_popup(test_id):
    try:
        viz.setup_font()
        df = viz.load_data("gaze_accuracy_results")
        s = viz.get_session(df, test_id)

        if len(s) == 0:
            print(f"[popup] 세션을 찾을 수 없음: {test_id}")
            return

        print(viz.format_summary_line(viz.summarize_session(s)))

        screen_w, screen_h = viz.infer_screen_size(s)
        viz.plot_session_overview(s, screen_w, screen_h)
        plt.show()

    except Exception as e:
        print(f"[popup] 시각화 실패: {e}")
