import cv2
import numpy as np
import csv
import math
import time
import os
import uuid
import statistics
from datetime import datetime
from PIL import Image, ImageDraw
from src.calibrations.baseline_manager import save_baseline
import viz
import matplotlib.pyplot as plt

import src.hangul as hangul

from src.config import (
    SCREEN_W,
    SCREEN_H,
    PX_PER_CM,
    HEAD_TEST_PHASE_SEC,
    HEAD_TEST_PHASES,
    NINE_POINT_HEAD_TEST_TARGETS,
    HEAD_TEST_MULTI_PHASE_SEC
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
from src.tracking.head_pose import estimate_head_pose, estimate_sqpnp_headpose
from src.tracking import keyboard_session_model as ksm

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
    draw_head_movement_test_screen,
    draw_head_movement_test_multi_screen,
    draw_keyboard_session_calib_screen,
    font
)

from tests.test_runner import TestRunner
from src.metrics.csv_export import append_rows

def auto_brightness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    mean = np.mean(gray)

    target = 120

    alpha = target / max(mean, 1)

    alpha = np.clip(alpha, 0.8, 1.5)

    frame = cv2.convertScaleAbs(
        frame,
        alpha=alpha,
        beta=0
    )

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))

    return cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2BGR
    )

from src.metrics.collector import MetricsCollector

MAX_SQPNP_DELTA_PX = 120

# 9점 테스트 (개발용)

def run_gaze_accuracy_test(
    cap,
    face_mesh,
    calibrator,
    gaze,
    collector,
    use_pose_corrected,
    use_sqpnp_corrected=False
):
    os.makedirs("gaze_accuracy_results", exist_ok=True)

    if use_sqpnp_corrected:
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

                # Raw 좌표
                raw_sx, raw_sy = calibrator.map_to_screen(
                    iris_x,
                    iris_y
                )

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

                if use_sqpnp_corrected:
                    sx, sy = sqpnp_corrected_sx, sqpnp_corrected_sy
                elif use_pose_corrected:
                    sx, sy = corrected_sx, corrected_sy
                else:
                    sx, sy = raw_sx, raw_sy

                blink = is_blink(lms)
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
    print(f"Average Error : {avg_error:.2f}px")
    print(f"Max Error : {max_error:.2f}px")
    print(f"Min Error : {min_error:.2f}px")
    print(f"Std Error : {std_error:.2f}px")
    print("=====================")

    # ── collector 내보내기 ──
    out_dir = "gaze_accuracy_results"

    collector.end_session()
    collector.export_csv(
        sessions_path=os.path.join(out_dir, f"sessions_v{MetricsCollector.SCHEMA_VERSION}.csv"),
        accuracy_path=os.path.join(out_dir, f"gaze_accuracy_v{MetricsCollector.SCHEMA_VERSION}.csv")
    )
    print("[metrics] collector CSV 저장 완료:", out_dir)


# head movement 테스트 (3D/head pose 안정화 검증용, 9점 테스트와 완전 별도)
# SQPnP 관련 함수/h 키 토글/sqpnp_corrected 경로는 여기서 전혀 사용하지 않는다.

HEAD_MOVEMENT_TEST_SCHEMA_VERSION = "1.0"
HEAD_MOVEMENT_TEST_VERSION = "v1.0"


def run_head_movement_test(cap, face_mesh, calibrator):

    os.makedirs("gaze_accuracy_results", exist_ok=True)

    target_x = SCREEN_W // 2
    target_y = SCREEN_H // 2

    test_session_id = str(uuid.uuid4())

    gaze_raw_test = GazePipeline()
    gaze_pose_test = GazePipeline()

    rows = []
    aborted = False

    for phase_name, instruction in HEAD_TEST_PHASES:

        phase_start = time.time()

        while True:

            ret, frame = cap.read()

            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            frame = auto_brightness(frame)

            fh, fw = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)

            phase_elapsed = time.time() - phase_start

            face_detected = False
            iris_x = None
            iris_y = None

            raw_sx = None
            raw_sy = None
            corrected_sx = None
            corrected_sy = None

            raw_gaze_x = None
            raw_gaze_y = None
            pose_gaze_x = None
            pose_gaze_y = None

            raw_error_px = None
            pose_error_px = None

            yaw = None
            pitch = None
            roll = None
            face_scale = None
            face_center_x = None
            face_center_y = None

            delta_center_x = None
            delta_center_y = None
            scale_ratio = None
            correction_applied = False
            fallback_reason = ""

            conf = 0.0
            blink = False

            if result.multi_face_landmarks:

                lms = result.multi_face_landmarks[0]
                face_detected = True

                iris_x, iris_y = get_avg_iris(lms)
                blink = is_blink(lms)
                conf = iris_confidence(lms)

                # ITERATIVE head pose만 사용 (estimate_sqpnp_headpose 미사용)
                head_pose = estimate_head_pose(lms, fw, fh)

                yaw = head_pose["yaw"]
                pitch = head_pose["pitch"]
                roll = head_pose["roll"]
                face_scale = head_pose["face_scale"]
                face_center_x = head_pose["face_center_x"]
                face_center_y = head_pose["face_center_y"]

                raw_sx, raw_sy = calibrator.map_to_screen(iris_x, iris_y)

                corrected_iris_x, corrected_iris_y = calibrator.compensate_iris_by_head_pose(
                    iris_x,
                    iris_y,
                    head_pose
                )

                corrected_sx, corrected_sy = calibrator.map_to_screen(
                    corrected_iris_x,
                    corrected_iris_y
                )

                diag = calibrator.diagnose_pose_correction(iris_x, iris_y, head_pose)

                delta_center_x = diag["delta_center_x"]
                delta_center_y = diag["delta_center_y"]
                scale_ratio = diag["scale_ratio"]
                correction_applied = diag["would_apply"]
                fallback_reason = diag["fallback_reason"]

                if raw_sx is not None and raw_sy is not None:

                    rgx, rgy, _ = gaze_raw_test.update(
                        raw_sx,
                        raw_sy,
                        conf,
                        blink,
                        head_pose=head_pose
                    )

                    if not (rgx == -1 and rgy == -1):
                        raw_gaze_x = rgx
                        raw_gaze_y = rgy

                if corrected_sx is not None and corrected_sy is not None:

                    pgx, pgy, _ = gaze_pose_test.update(
                        corrected_sx,
                        corrected_sy,
                        conf,
                        blink,
                        head_pose=head_pose
                    )

                    if not (pgx == -1 and pgy == -1):
                        pose_gaze_x = pgx
                        pose_gaze_y = pgy

                if raw_gaze_x is not None:
                    raw_error_px = math.hypot(raw_gaze_x - target_x, raw_gaze_y - target_y)

                if pose_gaze_x is not None:
                    pose_error_px = math.hypot(pose_gaze_x - target_x, pose_gaze_y - target_y)

            else:
                fallback_reason = "face_not_detected"

            rows.append({
                "schema_version": HEAD_MOVEMENT_TEST_SCHEMA_VERSION,
                "test_version": HEAD_MOVEMENT_TEST_VERSION,
                "test_session_id": test_session_id,
                "timestamp": time.time(),
                "phase": phase_name,
                "phase_elapsed_sec": round(phase_elapsed, 3),
                "target_x_px": target_x,
                "target_y_px": target_y,
                "iris_x": iris_x,
                "iris_y": iris_y,
                "raw_sx": raw_sx,
                "raw_sy": raw_sy,
                "corrected_sx": corrected_sx,
                "corrected_sy": corrected_sy,
                "raw_gaze_x": raw_gaze_x,
                "raw_gaze_y": raw_gaze_y,
                "pose_gaze_x": pose_gaze_x,
                "pose_gaze_y": pose_gaze_y,
                "raw_error_px": round(raw_error_px, 2) if raw_error_px is not None else None,
                "pose_error_px": round(pose_error_px, 2) if pose_error_px is not None else None,
                "yaw": yaw,
                "pitch": pitch,
                "roll": roll,
                "face_scale": face_scale,
                "face_center_x": face_center_x,
                "face_center_y": face_center_y,
                "delta_center_x": delta_center_x,
                "delta_center_y": delta_center_y,
                "scale_ratio": scale_ratio,
                "correction_applied": correction_applied,
                "fallback_reason": fallback_reason,
                "conf": round(conf, 4),
                "blink": blink,
                "face_detected": face_detected,
            })

            canvas = draw_head_movement_test_screen(
                phase_name,
                instruction,
                phase_elapsed,
                HEAD_TEST_PHASE_SEC,
                target_x,
                target_y,
                raw_gaze=(raw_gaze_x, raw_gaze_y),
                pose_gaze=(pose_gaze_x, pose_gaze_y)
            )

            cv2.imshow("Eye Keyboard", canvas)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:  # 27 = ESC
                aborted = True
                break

            if phase_elapsed >= HEAD_TEST_PHASE_SEC:
                break

        if aborted:
            break

    fieldnames = [
        "schema_version", "test_version", "test_session_id",
        "timestamp", "phase", "phase_elapsed_sec",
        "target_x_px", "target_y_px",
        "iris_x", "iris_y",
        "raw_sx", "raw_sy", "corrected_sx", "corrected_sy",
        "raw_gaze_x", "raw_gaze_y", "pose_gaze_x", "pose_gaze_y",
        "raw_error_px", "pose_error_px",
        "yaw", "pitch", "roll", "face_scale", "face_center_x", "face_center_y",
        "delta_center_x", "delta_center_y", "scale_ratio",
        "correction_applied", "fallback_reason",
        "conf", "blink", "face_detected",
    ]

    csv_path = os.path.join(
        "gaze_accuracy_results",
        f"head_movement_test_v{HEAD_MOVEMENT_TEST_SCHEMA_VERSION}.csv"
    )

    append_rows(csv_path, fieldnames, rows)

    print(f"\n[head_movement_test] CSV 저장 완료: {csv_path} ({len(rows)} rows)")

    if aborted:
        print("[head_movement_test] 테스트가 도중에 중단되었습니다 (q/ESC).")

    print("\n===== HEAD MOVEMENT TEST RESULT =====")

    for phase_name, _ in HEAD_TEST_PHASES:

        phase_rows = [r for r in rows if r["phase"] == phase_name]

        if not phase_rows:
            continue

        raw_errors = [r["raw_error_px"] for r in phase_rows if r["raw_error_px"] is not None]
        pose_errors = [r["pose_error_px"] for r in phase_rows if r["pose_error_px"] is not None]

        fallback_count = sum(1 for r in phase_rows if not r["correction_applied"])
        fallback_rate = fallback_count / len(phase_rows)

        raw_mean = statistics.mean(raw_errors) if raw_errors else None
        pose_mean = statistics.mean(pose_errors) if pose_errors else None

        raw_str = f"{raw_mean:.1f}px" if raw_mean is not None else "N/A"
        pose_str = f"{pose_mean:.1f}px" if pose_mean is not None else "N/A"

        if raw_mean is not None and pose_mean is not None:
            diff_str = f"{pose_mean - raw_mean:+.1f}px"
        else:
            diff_str = "N/A"

        print(
            f"[{phase_name}] raw_error={raw_str}  pose_error={pose_str}  "
            f"diff(pose-raw)={diff_str}  fallback={fallback_rate*100:.0f}%"
        )

    print("======================================\n")


# 9점 head movement 테스트 (화면 전체 일반화 검증용, v1 center-only 테스트와 완전 별도)
# SQPnP 관련 함수/h 키 토글/sqpnp_corrected 경로는 여기서도 전혀 사용하지 않는다.

HEAD_MOVEMENT_MULTI_TEST_SCHEMA_VERSION = "1.1"
HEAD_MOVEMENT_MULTI_TEST_VERSION = "v1.1"


def run_head_movement_test_multi(cap, face_mesh, calibrator):

    os.makedirs("gaze_accuracy_results", exist_ok=True)

    all_targets_px = [
        (int(SCREEN_W * tx), int(SCREEN_H * ty))
        for tx, ty in NINE_POINT_HEAD_TEST_TARGETS
    ]

    test_session_id = str(uuid.uuid4())

    rows = []
    aborted = False
    completed_targets = 0

    for target_index, (tx_norm, ty_norm) in enumerate(NINE_POINT_HEAD_TEST_TARGETS):

        target_x, target_y = all_targets_px[target_index]

        # 타깃마다 GazePipeline을 새로 생성한다.
        # 이전 타깃을 보던 Kalman 상태를 그대로 들고 오면, 화면 위치가 바뀐 것 때문에
        # 생기는 "추격 지연"이 머리 이동과 무관한 오차로 섞여 들어간다.
        gaze_raw_test = GazePipeline()
        gaze_pose_test = GazePipeline()

        for phase_name, instruction in HEAD_TEST_PHASES:

            phase_start = time.time()

            while True:

                ret, frame = cap.read()

                if not ret:
                    continue

                frame = cv2.flip(frame, 1)
                frame = auto_brightness(frame)

                fh, fw = frame.shape[:2]

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = face_mesh.process(rgb)

                phase_elapsed = time.time() - phase_start

                face_detected = False
                iris_x = None
                iris_y = None

                raw_sx = None
                raw_sy = None
                corrected_sx = None
                corrected_sy = None

                raw_preclip_x = None
                raw_preclip_y = None
                corrected_preclip_x = None
                corrected_preclip_y = None
                raw_was_clamped = False
                corrected_was_clamped = False
                raw_mapping_valid = False
                corrected_mapping_valid = False

                raw_gaze_x = None
                raw_gaze_y = None
                pose_gaze_x = None
                pose_gaze_y = None

                raw_error_px = None
                pose_error_px = None

                yaw = None
                pitch = None
                roll = None
                face_scale = None
                face_center_x = None
                face_center_y = None

                delta_center_x = None
                delta_center_y = None
                scale_ratio = None
                correction_applied = False
                fallback_reason = ""

                conf = 0.0
                blink = False

                if result.multi_face_landmarks:

                    lms = result.multi_face_landmarks[0]
                    face_detected = True

                    iris_x, iris_y = get_avg_iris(lms)
                    blink = is_blink(lms)
                    conf = iris_confidence(lms)

                    # ITERATIVE head pose만 사용 (estimate_sqpnp_headpose 미사용)
                    head_pose = estimate_head_pose(lms, fw, fh)

                    yaw = head_pose["yaw"]
                    pitch = head_pose["pitch"]
                    roll = head_pose["roll"]
                    face_scale = head_pose["face_scale"]
                    face_center_x = head_pose["face_center_x"]
                    face_center_y = head_pose["face_center_y"]

                    raw_sx, raw_sy = calibrator.map_to_screen(iris_x, iris_y)

                    # map_to_screen()은 그대로 두고, 진단용 preclip 정보만 별도로 계산한다.
                    raw_preclip_info = calibrator.map_to_screen_with_preclip(iris_x, iris_y)
                    raw_preclip_x = raw_preclip_info["preclip_x"]
                    raw_preclip_y = raw_preclip_info["preclip_y"]
                    raw_was_clamped = raw_preclip_info["was_clamped"]
                    raw_mapping_valid = raw_preclip_info["mapping_valid"]

                    corrected_iris_x, corrected_iris_y = calibrator.compensate_iris_by_head_pose(
                        iris_x,
                        iris_y,
                        head_pose
                    )

                    corrected_sx, corrected_sy = calibrator.map_to_screen(
                        corrected_iris_x,
                        corrected_iris_y
                    )

                    corrected_preclip_info = calibrator.map_to_screen_with_preclip(
                        corrected_iris_x,
                        corrected_iris_y
                    )
                    corrected_preclip_x = corrected_preclip_info["preclip_x"]
                    corrected_preclip_y = corrected_preclip_info["preclip_y"]
                    corrected_was_clamped = corrected_preclip_info["was_clamped"]
                    corrected_mapping_valid = corrected_preclip_info["mapping_valid"]

                    diag = calibrator.diagnose_pose_correction(iris_x, iris_y, head_pose)

                    delta_center_x = diag["delta_center_x"]
                    delta_center_y = diag["delta_center_y"]
                    scale_ratio = diag["scale_ratio"]
                    correction_applied = diag["would_apply"]
                    fallback_reason = diag["fallback_reason"]

                    if raw_sx is not None and raw_sy is not None:

                        rgx, rgy, _ = gaze_raw_test.update(
                            raw_sx,
                            raw_sy,
                            conf,
                            blink,
                            head_pose=head_pose
                        )

                        if not (rgx == -1 and rgy == -1):
                            raw_gaze_x = rgx
                            raw_gaze_y = rgy

                    if corrected_sx is not None and corrected_sy is not None:

                        pgx, pgy, _ = gaze_pose_test.update(
                            corrected_sx,
                            corrected_sy,
                            conf,
                            blink,
                            head_pose=head_pose
                        )

                        if not (pgx == -1 and pgy == -1):
                            pose_gaze_x = pgx
                            pose_gaze_y = pgy

                    if raw_gaze_x is not None:
                        raw_error_px = math.hypot(raw_gaze_x - target_x, raw_gaze_y - target_y)

                    if pose_gaze_x is not None:
                        pose_error_px = math.hypot(pose_gaze_x - target_x, pose_gaze_y - target_y)

                else:
                    fallback_reason = "face_not_detected"

                rows.append({
                    "schema_version": HEAD_MOVEMENT_MULTI_TEST_SCHEMA_VERSION,
                    "test_version": HEAD_MOVEMENT_MULTI_TEST_VERSION,
                    "test_session_id": test_session_id,
                    "timestamp": time.time(),
                    "target_index": target_index,
                    "target_norm_x": tx_norm,
                    "target_norm_y": ty_norm,
                    "target_x_px": target_x,
                    "target_y_px": target_y,
                    "phase": phase_name,
                    "phase_elapsed_sec": round(phase_elapsed, 3),
                    "iris_x": iris_x,
                    "iris_y": iris_y,
                    "raw_sx": raw_sx,
                    "raw_sy": raw_sy,
                    "corrected_sx": corrected_sx,
                    "corrected_sy": corrected_sy,
                    "raw_gaze_x": raw_gaze_x,
                    "raw_gaze_y": raw_gaze_y,
                    "pose_gaze_x": pose_gaze_x,
                    "pose_gaze_y": pose_gaze_y,
                    "raw_error_px": round(raw_error_px, 2) if raw_error_px is not None else None,
                    "pose_error_px": round(pose_error_px, 2) if pose_error_px is not None else None,
                    "yaw": yaw,
                    "pitch": pitch,
                    "roll": roll,
                    "face_scale": face_scale,
                    "face_center_x": face_center_x,
                    "face_center_y": face_center_y,
                    "delta_center_x": delta_center_x,
                    "delta_center_y": delta_center_y,
                    "scale_ratio": scale_ratio,
                    "correction_applied": correction_applied,
                    "fallback_reason": fallback_reason,
                    "conf": round(conf, 4),
                    "blink": blink,
                    "face_detected": face_detected,
                    "raw_preclip_x": raw_preclip_x,
                    "raw_preclip_y": raw_preclip_y,
                    "corrected_preclip_x": corrected_preclip_x,
                    "corrected_preclip_y": corrected_preclip_y,
                    "raw_was_clamped": raw_was_clamped,
                    "corrected_was_clamped": corrected_was_clamped,
                    "raw_mapping_valid": raw_mapping_valid,
                    "corrected_mapping_valid": corrected_mapping_valid,
                })

                canvas = draw_head_movement_test_multi_screen(
                    all_targets_px,
                    target_index,
                    phase_name,
                    instruction,
                    phase_elapsed,
                    HEAD_TEST_MULTI_PHASE_SEC,
                    raw_gaze=(raw_gaze_x, raw_gaze_y),
                    pose_gaze=(pose_gaze_x, pose_gaze_y)
                )

                cv2.imshow("Eye Keyboard", canvas)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == 27:  # 27 = ESC
                    aborted = True
                    break

                if phase_elapsed >= HEAD_TEST_MULTI_PHASE_SEC:
                    break

            if aborted:
                break

        if aborted:
            break

        completed_targets += 1

    fieldnames = [
        "schema_version", "test_version", "test_session_id",
        "timestamp", "target_index", "target_norm_x", "target_norm_y",
        "target_x_px", "target_y_px",
        "phase", "phase_elapsed_sec",
        "iris_x", "iris_y",
        "raw_sx", "raw_sy", "corrected_sx", "corrected_sy",
        "raw_gaze_x", "raw_gaze_y", "pose_gaze_x", "pose_gaze_y",
        "raw_error_px", "pose_error_px",
        "yaw", "pitch", "roll", "face_scale", "face_center_x", "face_center_y",
        "delta_center_x", "delta_center_y", "scale_ratio",
        "correction_applied", "fallback_reason",
        "conf", "blink", "face_detected",
        "raw_preclip_x", "raw_preclip_y",
        "corrected_preclip_x", "corrected_preclip_y",
        "raw_was_clamped", "corrected_was_clamped",
        "raw_mapping_valid", "corrected_mapping_valid",
    ]

    csv_path = os.path.join(
        "gaze_accuracy_results",
        f"head_movement_test_multi_v{HEAD_MOVEMENT_MULTI_TEST_SCHEMA_VERSION}.csv"
    )

    append_rows(csv_path, fieldnames, rows)

    print(f"\n[head_movement_test_multi] CSV 저장 완료: {csv_path} ({len(rows)} rows)")
    print(f"[head_movement_test_multi] 완료한 타깃 수: {completed_targets}/{len(NINE_POINT_HEAD_TEST_TARGETS)}")

    if aborted:
        print("[head_movement_test_multi] 테스트가 도중에 중단되었습니다 (q/ESC).")

    print("\n===== HEAD MOVEMENT TEST (9-point) RESULT =====")

    for phase_name, _ in HEAD_TEST_PHASES:

        phase_rows = [r for r in rows if r["phase"] == phase_name]

        if not phase_rows:
            continue

        raw_errors = [r["raw_error_px"] for r in phase_rows if r["raw_error_px"] is not None]
        pose_errors = [r["pose_error_px"] for r in phase_rows if r["pose_error_px"] is not None]

        fallback_count = sum(1 for r in phase_rows if not r["correction_applied"])
        fallback_rate = fallback_count / len(phase_rows)

        raw_mean = statistics.mean(raw_errors) if raw_errors else None
        pose_mean = statistics.mean(pose_errors) if pose_errors else None

        raw_str = f"{raw_mean:.1f}px" if raw_mean is not None else "N/A"
        pose_str = f"{pose_mean:.1f}px" if pose_mean is not None else "N/A"

        if raw_mean is not None and pose_mean is not None:
            diff_str = f"{pose_mean - raw_mean:+.1f}px"
        else:
            diff_str = "N/A"

        print(
            f"[{phase_name}] raw_error={raw_str}  pose_error={pose_str}  "
            f"diff(pose-raw)={diff_str}  fallback={fallback_rate*100:.0f}%"
        )

    print("================================================\n")


# ── 세션 적응형 keyboard ROI 보정 모델 (실험 모드) ────────────────
# src/tracking/keyboard_session_model.py 참고.
# 기존 Calibrator.map_to_screen()/compensate_iris_by_head_pose()의
# 시그니처/동작은 여기서도 전혀 바꾸지 않는다(읽기 전용으로만 사용).
# estimate_sqpnp_headpose()/h 키/sqpnp_corrected 경로는 이 섹션에서
# 전혀 참조하지 않는다.

KEYBOARD_SESSION_CALIB_SCHEMA_VERSION = "1.0"
KEYBOARD_SESSION_COMPARE_SCHEMA_VERSION = "1.0"
KEYBOARD_SESSION_VALIDATION_SCHEMA_VERSION = "1.0"

KEYBOARD_SESSION_TRIAL_TIMEOUT_SEC = 4.0


def run_keyboard_session_calibration(cap, face_mesh, calibrator, button_list):
    """
    16점 캘리브레이션 완료 후, keyboard ROI 안 5개 anchor key x
    {center,left,right} 조건으로 "이동 후 유지" 방식의 짧은 추가
    캘리브레이션을 수집하고 KeyboardSessionCalibrator를 fit한다.

    Returns:
        (session_model, roi) 또는 데이터가 부족하면 (None, None)
    """

    os.makedirs("gaze_accuracy_results", exist_ok=True)

    roi, anchors = ksm.build_default(button_list)
    ksm.print_anchor_summary(anchors, roi)

    if len(anchors) < 3:
        print("[keyboard_session_calib] anchor가 부족해 실행할 수 없습니다.")
        return None, None

    calib_session_id = str(uuid.uuid4())
    session_model = ksm.KeyboardSessionCalibrator(roi)

    all_anchor_targets_px = [
        (
            int(a.pos[0] + a.size[0] / 2),
            int(a.pos[1] + a.size[1] / 2),
        )
        for a in anchors
    ]

    stage_durations = {
        "instruction": ksm.INSTRUCTION_SEC,
        "warmup": ksm.WARMUP_SEC,
        "hold": ksm.HOLD_SEC,
    }

    rows = []
    aborted = False

    for anchor_index, anchor in enumerate(anchors):

        target_x, target_y = all_anchor_targets_px[anchor_index]
        target_kx, target_ky = ksm.anchor_target_kxky(anchor, roi)

        for condition in ksm.CONDITIONS:

            stage = "instruction"
            stage_start = time.time()

            while True:

                ret, frame = cap.read()

                if not ret:
                    continue

                frame = cv2.flip(frame, 1)
                frame = auto_brightness(frame)
                fh, fw = frame.shape[:2]

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = face_mesh.process(rgb)

                now = time.time()
                stage_elapsed = now - stage_start

                if stage == "instruction" and stage_elapsed >= stage_durations["instruction"]:
                    stage = "warmup"
                    stage_start = now
                    stage_elapsed = 0.0

                elif stage == "warmup" and stage_elapsed >= stage_durations["warmup"]:
                    stage = "hold"
                    stage_start = now
                    stage_elapsed = 0.0

                elif stage == "hold" and stage_elapsed >= stage_durations["hold"]:
                    break

                raw_gaze_x = None
                raw_gaze_y = None

                if result.multi_face_landmarks:

                    lms = result.multi_face_landmarks[0]

                    iris_x, iris_y = get_avg_iris(lms)
                    blink = is_blink(lms)
                    conf = iris_confidence(lms)

                    head_pose = estimate_head_pose(lms, fw, fh)

                    raw_sx, raw_sy = calibrator.map_to_screen(iris_x, iris_y)

                    if raw_sx is not None and raw_sy is not None:
                        raw_gaze_x, raw_gaze_y = raw_sx, raw_sy

                    if stage == "hold" and raw_sx is not None and raw_sy is not None:

                        raw_kx, raw_ky = ksm.to_roi_normalized(raw_sx, raw_sy, roi)

                        diag = calibrator.diagnose_pose_correction(
                            iris_x, iris_y, head_pose
                        )

                        dyaw, dpitch, droll = ksm.compute_pose_deltas(
                            head_pose, calibrator.pose_baseline
                        )

                        # blink뿐 아니라 low confidence/invalid pose/
                        # diagnose_pose_correction의 fallback 프레임(would_apply=False)도
                        # 학습 샘플에서 제외한다. CSV1에는 sample_used로 남겨
                        # 나중에 제외 비율을 확인할 수 있게 한다.
                        sample_used = (
                            not blink
                            and conf > ksm.MIN_CONF_FOR_SAMPLE
                            and head_pose.get("valid", False)
                            and diag["would_apply"]
                        )

                        if sample_used:
                            session_model.add_frame(
                                anchor_index,
                                condition,
                                raw_kx,
                                raw_ky,
                                diag["delta_center_x"],
                                diag["delta_center_y"],
                                diag["scale_ratio"],
                                delta_yaw=dyaw,
                                delta_pitch=dpitch,
                                delta_roll=droll,
                            )

                        rows.append({
                            "schema_version": KEYBOARD_SESSION_CALIB_SCHEMA_VERSION,
                            "calib_session_id": calib_session_id,
                            "timestamp": time.time(),
                            "anchor_index": anchor_index,
                            "anchor_key_text": anchor.text,
                            "condition": condition,
                            "stage_elapsed_sec": round(stage_elapsed, 3),
                            "roi_left": roi["left"],
                            "roi_top": roi["top"],
                            "roi_width": roi["width"],
                            "roi_height": roi["height"],
                            "target_kx": target_kx,
                            "target_ky": target_ky,
                            "iris_x": iris_x,
                            "iris_y": iris_y,
                            "raw_sx": raw_sx,
                            "raw_sy": raw_sy,
                            "raw_kx": raw_kx,
                            "raw_ky": raw_ky,
                            "yaw": head_pose["yaw"],
                            "pitch": head_pose["pitch"],
                            "roll": head_pose["roll"],
                            "face_scale": head_pose["face_scale"],
                            "face_center_x": head_pose["face_center_x"],
                            "face_center_y": head_pose["face_center_y"],
                            "delta_center_x": diag["delta_center_x"],
                            "delta_center_y": diag["delta_center_y"],
                            "scale_ratio": diag["scale_ratio"],
                            "conf": round(conf, 4),
                            "blink": blink,
                            "face_detected": True,
                            "sample_used": sample_used,
                            "fallback_reason": diag["fallback_reason"],
                        })

                canvas = draw_keyboard_session_calib_screen(
                    anchor_index,
                    len(anchors),
                    condition,
                    stage,
                    stage_elapsed,
                    stage_durations[stage],
                    target_x,
                    target_y,
                    all_anchor_targets_px,
                    raw_gaze=(raw_gaze_x, raw_gaze_y),
                )

                cv2.imshow("Eye Keyboard", canvas)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == 27:  # 27 = ESC
                    aborted = True
                    break

            if aborted:
                break

            session_model.finalize_condition(
                anchor_index, condition, target_kx, target_ky
            )

        if aborted:
            break

    fieldnames = [
        "schema_version", "calib_session_id", "timestamp",
        "anchor_index", "anchor_key_text", "condition", "stage_elapsed_sec",
        "roi_left", "roi_top", "roi_width", "roi_height",
        "target_kx", "target_ky",
        "iris_x", "iris_y", "raw_sx", "raw_sy", "raw_kx", "raw_ky",
        "yaw", "pitch", "roll", "face_scale", "face_center_x", "face_center_y",
        "delta_center_x", "delta_center_y", "scale_ratio",
        "conf", "blink", "face_detected", "sample_used", "fallback_reason",
    ]

    csv_path = os.path.join(
        "gaze_accuracy_results",
        f"keyboard_session_calib_v{KEYBOARD_SESSION_CALIB_SCHEMA_VERSION}.csv"
    )

    append_rows(csv_path, fieldnames, rows)

    print(f"[keyboard_session_calib] CSV 저장 완료: {csv_path} ({len(rows)} rows)")

    if aborted:
        print("[keyboard_session_calib] 중단됨 (q/ESC). 수집된 데이터만으로 fit을 시도합니다.")

    session_model.calib_session_id = calib_session_id
    session_model.fit()

    print("[keyboard_session_calib] fit 결과 (LOO 오차는 세션 내부 참고 지표일 뿐, "
          "실사용 유효성은 'l' 키 검증 테스트로 별도 확인 필요):")
    print("  ", session_model.summary())

    return session_model, roi


def run_keyboard_session_validation_test(
    cap,
    face_mesh,
    calibrator,
    button_list,
    session_model,
    roi
):
    """
    keyboard ROI의 5개 anchor x {center,left,right} 조건마다 raw/pose/
    session_model 세 경로를 독립된 GazePipeline+DwellController로 동시에
    비교한다. 화면 평균 px 오차가 아니라 key hit rate / dwell 성공률 /
    boundary crossing / target key 유지율로 검증한다.
    """

    os.makedirs("gaze_accuracy_results", exist_ok=True)

    anchors = ksm.select_anchor_buttons(button_list, roi)

    if not anchors:
        print("[keyboard_session_validation] anchor가 없어 실행할 수 없습니다.")
        return

    validation_session_id = str(uuid.uuid4())

    rows = []
    aborted = False

    sources = ("raw", "pose", "session_model")

    for anchor_index, anchor in enumerate(anchors):

        target_x, target_y = (
            int(anchor.pos[0] + anchor.size[0] / 2),
            int(anchor.pos[1] + anchor.size[1] / 2),
        )
        target_kx, target_ky = ksm.anchor_target_kxky(anchor, roi)

        for condition in ksm.CONDITIONS:

            gaze_pipelines = {s: GazePipeline() for s in sources}
            dwell_controllers = {s: DwellController() for s in sources}

            hover_history = {s: [] for s in sources}
            completed = {s: False for s in sources}
            dwell_time = {s: None for s in sources}
            clicked_key_text = {s: None for s in sources}

            stage = "instruction"
            stage_start = time.time()

            while True:

                ret, frame = cap.read()

                if not ret:
                    continue

                frame = cv2.flip(frame, 1)
                frame = auto_brightness(frame)
                fh, fw = frame.shape[:2]

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = face_mesh.process(rgb)

                now = time.time()
                stage_elapsed = now - stage_start

                # calibration과 동일하게 instruction -> warmup -> hold로 나눈다.
                # 전환 직후(특히 warmup) 시선/필터 지연이 섞이면 hit/dwell 판정이
                # 흔들릴 수 있어 hold 전에는 dwell을 아예 갱신하지 않는다.
                if stage == "instruction" and stage_elapsed >= ksm.INSTRUCTION_SEC:
                    stage = "warmup"
                    stage_start = now
                    stage_elapsed = 0.0

                elif stage == "warmup" and stage_elapsed >= ksm.WARMUP_SEC:
                    stage = "hold"
                    stage_start = now
                    stage_elapsed = 0.0

                raw_gaze_x = None
                raw_gaze_y = None

                if result.multi_face_landmarks and stage == "hold":

                    lms = result.multi_face_landmarks[0]

                    iris_x, iris_y = get_avg_iris(lms)
                    blink = is_blink(lms)
                    conf = iris_confidence(lms)

                    head_pose = estimate_head_pose(lms, fw, fh)

                    raw_sx, raw_sy = calibrator.map_to_screen(iris_x, iris_y)

                    corrected_iris_x, corrected_iris_y = calibrator.compensate_iris_by_head_pose(
                        iris_x, iris_y, head_pose
                    )
                    pose_sx, pose_sy = calibrator.map_to_screen(
                        corrected_iris_x, corrected_iris_y
                    )

                    session_sx, session_sy = raw_sx, raw_sy

                    if raw_sx is not None and raw_sy is not None:

                        raw_gaze_x, raw_gaze_y = raw_sx, raw_sy

                        if ksm.is_inside_roi(raw_sx, raw_sy, roi):

                            raw_kx, raw_ky = ksm.to_roi_normalized(raw_sx, raw_sy, roi)

                            diag = calibrator.diagnose_pose_correction(
                                iris_x, iris_y, head_pose
                            )
                            dyaw, dpitch, droll = ksm.compute_pose_deltas(
                                head_pose, calibrator.pose_baseline
                            )
                            preclip = calibrator.map_to_screen_with_preclip(
                                iris_x, iris_y
                            )
                            sev = ksm.preclip_severity(
                                preclip["preclip_x"],
                                preclip["preclip_y"],
                                SCREEN_W,
                                SCREEN_H
                            )

                            gate = session_model.predict(
                                raw_kx, raw_ky,
                                diag["delta_center_x"],
                                diag["delta_center_y"],
                                diag["scale_ratio"],
                                delta_yaw=dyaw,
                                delta_pitch=dpitch,
                                delta_roll=droll,
                                preclip_sev=sev,
                            )

                            if gate["applied"]:
                                sess_kx = raw_kx + gate["residual_kx"]
                                sess_ky = raw_ky + gate["residual_ky"]
                                session_sx, session_sy = ksm.to_roi_pixels(
                                    sess_kx, sess_ky, roi
                                )

                    source_coords = {
                        "raw": (raw_sx, raw_sy),
                        "pose": (pose_sx, pose_sy),
                        "session_model": (session_sx, session_sy),
                    }

                    for s in sources:

                        sx_s, sy_s = source_coords[s]

                        if sx_s is None or sy_s is None:
                            continue

                        gx, gy, _ = gaze_pipelines[s].update(
                            sx_s, sy_s, conf, blink, head_pose=head_pose
                        )

                        if gx == -1 and gy == -1:
                            continue

                        hovered_key, _, clicked_key = dwell_controllers[s].update(
                            gx, gy, button_list
                        )

                        hover_history[s].append(hovered_key)

                        if clicked_key is not None and not completed[s]:
                            completed[s] = True
                            dwell_time[s] = stage_elapsed
                            clicked_key_text[s] = clicked_key

                stage_duration_for_draw = {
                    "instruction": ksm.INSTRUCTION_SEC,
                    "warmup": ksm.WARMUP_SEC,
                    "hold": KEYBOARD_SESSION_TRIAL_TIMEOUT_SEC,
                }[stage]

                canvas = draw_keyboard_session_calib_screen(
                    anchor_index,
                    len(anchors),
                    condition,
                    stage,
                    stage_elapsed,
                    stage_duration_for_draw,
                    target_x,
                    target_y,
                    [(target_x, target_y)],
                    raw_gaze=(raw_gaze_x, raw_gaze_y),
                )

                cv2.imshow("Eye Keyboard", canvas)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == 27:  # 27 = ESC
                    aborted = True
                    break

                if stage == "hold" and (
                    all(completed.values())
                    or stage_elapsed >= KEYBOARD_SESSION_TRIAL_TIMEOUT_SEC
                ):
                    break

            if aborted:
                break

            for s in sources:

                history = hover_history[s]
                n_frames = len(history)

                retention = (
                    sum(1 for h in history if h == anchor.text) / n_frames
                    if n_frames > 0
                    else 0.0
                )

                crossings = sum(
                    1
                    for prev, cur in zip(history, history[1:])
                    if cur != prev
                )

                rows.append({
                    "schema_version": KEYBOARD_SESSION_VALIDATION_SCHEMA_VERSION,
                    "validation_session_id": validation_session_id,
                    "timestamp": time.time(),
                    "anchor_index": anchor_index,
                    "target_key_text": anchor.text,
                    "target_kx": target_kx,
                    "target_ky": target_ky,
                    "source": s,
                    # hit은 마지막 hover 프레임이 아니라, dwell이 실제로 완료된
                    # 순간의 clicked_key_text 기준으로 판정한다.
                    "hit": bool(
                        completed[s]
                        and clicked_key_text[s] == anchor.text
                    ),
                    "clicked_target_key_text": clicked_key_text[s],
                    "dwell_completed": completed[s],
                    "dwell_time_sec": (
                        round(dwell_time[s], 3) if dwell_time[s] is not None else None
                    ),
                    "key_retention_ratio": round(retention, 4),
                    "boundary_crossings": crossings,
                    "head_condition": condition,
                })

        if aborted:
            break

    fieldnames = [
        "schema_version", "validation_session_id", "timestamp",
        "anchor_index", "target_key_text", "target_kx", "target_ky",
        "source", "hit", "clicked_target_key_text",
        "dwell_completed", "dwell_time_sec",
        "key_retention_ratio", "boundary_crossings", "head_condition",
    ]

    csv_path = os.path.join(
        "gaze_accuracy_results",
        f"keyboard_session_validation_v{KEYBOARD_SESSION_VALIDATION_SCHEMA_VERSION}.csv"
    )

    append_rows(csv_path, fieldnames, rows)

    print(f"[keyboard_session_validation] CSV 저장 완료: {csv_path} ({len(rows)} rows)")

    if aborted:
        print("[keyboard_session_validation] 중단됨 (q/ESC). 수집된 결과만으로 평가합니다.")

    summary = session_model.evaluate_from_validation_results(rows)

    print("\n===== KEYBOARD SESSION VALIDATION RESULT =====")

    for s in sources:

        s_summary = summary.get(s)

        if s_summary is None:
            print(f"[{s}] 데이터 없음")
            continue

        print(
            f"[{s}] hit_rate={s_summary['hit_rate']*100:.0f}%  "
            f"dwell_rate={s_summary['dwell_rate']*100:.0f}%  "
            f"mean_boundary_crossings={s_summary['mean_boundary_crossings']:.2f}  "
            f"n={s_summary['n']}"
        )

    print(f"usable_confirmed: {session_model.usable_confirmed}")
    print("===============================================\n")


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

    # 자동 노출
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)

    # 자동 화이트밸런스
    cap.set(cv2.CAP_PROP_AUTO_WB, 1)

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
    use_sqpnp_corrected = False

    # 세션 적응형 keyboard ROI 보정 모델(실험 모드). 기본은 전부 비활성.
    keyboard_session_model = None
    keyboard_roi = None
    use_keyboard_session_model = False
    keyboard_compare_rows = []

    mouth_mode = False

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
        "g: head movement 테스트(중앙 1점) | "
        "9(또는 x): head movement 테스트(9점) | "
        "m: 입벌림 입력 방식 변경 | "
        "k: keyboard session 캘리브레이션(실험) | "
        "j: keyboard session 모델 적용 on/off | "
        "l: keyboard session 검증 테스트 | "
        "q: 종료"
    )

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
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

            # 자동 밝기 + CLAHE
            frame = auto_brightness(frame)

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
            sqpnp_corrected_sx = None
            sqpnp_corrected_sy = None
            sx = None
            sy = None

            corrected_iris_x = None
            corrected_iris_y = None
            sqpnp_corrected_iris_x = None
            sqpnp_corrected_iris_y = None
            sqpnp_delta_x = None
            sqpnp_delta_y = None

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
            sqpnp_headpose = dict(head_pose)

            if results.multi_face_landmarks:

                lms = results.multi_face_landmarks[0]

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
                if mouth_mode and not mouth_calibrator.done:
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

                sqpnp_corrected_iris_x, sqpnp_corrected_iris_y = calibrator.compensate_iris_by_head_pose(
                    iris_x,
                    iris_y,
                    sqpnp_headpose
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

                if use_sqpnp_corrected:
                    sx, sy = sqpnp_corrected_sx, sqpnp_corrected_sy
                elif use_pose_corrected:
                    sx, sy = corrected_sx, corrected_sy
                else:
                    sx, sy = raw_sx, raw_sy

                # ── keyboard session model (실험 모드, ROI 안에서만) ──
                # 위의 raw/pose_corrected/sqpnp_corrected 선택 로직은 그대로
                # 끝났고, 이 블록은 그 뒤에 keyboard ROI 안에서만 독립적으로
                # 얹히는 별도 경로다. map_to_screen()/compensate_iris_by_head_pose()는
                # 다시 호출하지 않고 이미 계산된 raw_sx,raw_sy와 진단 메서드
                # (diagnose_pose_correction/map_to_screen_with_preclip)만
                # 읽기 전용으로 사용한다. use_keyboard_session_model이 꺼져
                # 있어도 CSV2 비교 로그를 위해 gate 계산 자체는 계속한다.
                ks_applied = False
                ks_gate = None
                ks_raw_kx = None
                ks_raw_ky = None
                ks_sx = None
                ks_sy = None
                ks_diag = None

                if (
                    keyboard_session_model is not None
                    and raw_sx is not None
                    and raw_sy is not None
                    and ksm.is_inside_roi(raw_sx, raw_sy, keyboard_roi)
                ):
                    ks_raw_kx, ks_raw_ky = ksm.to_roi_normalized(
                        raw_sx, raw_sy, keyboard_roi
                    )

                    ks_diag = calibrator.diagnose_pose_correction(
                        iris_x, iris_y, head_pose
                    )
                    ks_dyaw, ks_dpitch, ks_droll = ksm.compute_pose_deltas(
                        head_pose, calibrator.pose_baseline
                    )
                    ks_preclip = calibrator.map_to_screen_with_preclip(
                        iris_x, iris_y
                    )
                    ks_sev = ksm.preclip_severity(
                        ks_preclip["preclip_x"],
                        ks_preclip["preclip_y"],
                        SCREEN_W,
                        SCREEN_H
                    )

                    ks_gate = keyboard_session_model.predict(
                        ks_raw_kx,
                        ks_raw_ky,
                        ks_diag["delta_center_x"],
                        ks_diag["delta_center_y"],
                        ks_diag["scale_ratio"],
                        delta_yaw=ks_dyaw,
                        delta_pitch=ks_dpitch,
                        delta_roll=ks_droll,
                        preclip_sev=ks_sev,
                    )

                    if ks_gate["applied"]:

                        ks_kx = ks_raw_kx + ks_gate["residual_kx"]
                        ks_ky = ks_raw_ky + ks_gate["residual_ky"]
                        ks_sx, ks_sy = ksm.to_roi_pixels(ks_kx, ks_ky, keyboard_roi)

                        # SQPnP 경로와는 절대 동시에 적용하지 않는다. use_sqpnp_corrected가
                        # 켜져 있으면 use_keyboard_session_model 값과 무관하게 sx,sy는
                        # 건드리지 않는다(위의 h/j 상호 배제 가드와 별개의 방어선).
                        if use_keyboard_session_model and not use_sqpnp_corrected:
                            sx, sy = int(ks_sx), int(ks_sy)
                            ks_applied = True

                    if use_sqpnp_corrected:
                        ks_active_source = "sqpnp"
                    elif use_pose_corrected:
                        ks_active_source = "pose"
                    else:
                        ks_active_source = "raw"

                    if ks_applied:
                        ks_active_source = "session_model"

                    keyboard_compare_rows.append({
                        "schema_version": KEYBOARD_SESSION_COMPARE_SCHEMA_VERSION,
                        "calib_session_id": getattr(
                            keyboard_session_model, "calib_session_id", ""
                        ),
                        "timestamp": time.time(),
                        "raw_sx": raw_sx,
                        "raw_sy": raw_sy,
                        "raw_kx": ks_raw_kx,
                        "raw_ky": ks_raw_ky,
                        "corrected_sx": corrected_sx,
                        "corrected_sy": corrected_sy,
                        "session_model_sx": ks_sx,
                        "session_model_sy": ks_sy,
                        "residual_kx": ks_gate["residual_kx"],
                        "residual_ky": ks_gate["residual_ky"],
                        "applied": ks_gate["applied"],
                        "variant_used": ks_gate["variant_used"],
                        "preclip_severity": ks_gate["preclip_severity"],
                        "gate_model_usable": ks_gate["gate_model_usable"],
                        "gate_preclip": ks_gate["gate_preclip"],
                        "gate_feature_range": ks_gate["gate_feature_range"],
                        "gate_residual_size": ks_gate["gate_residual_size"],
                        "fallback_reason": ks_gate["fallback_reason"],
                        "delta_center_x": ks_diag["delta_center_x"],
                        "delta_center_y": ks_diag["delta_center_y"],
                        "scale_ratio": ks_diag["scale_ratio"],
                        "active_source": ks_active_source,
                    })

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

            if use_sqpnp_corrected:
                mode_text = "Mode: SQPnP"
            else:
                mode_text = "Mode: PoseCorrected" if use_pose_corrected else "Mode: Raw"

            cv2.putText(
                kbd_bg,
                mode_text,
                (30, SCREEN_H - 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if (use_pose_corrected or use_sqpnp_corrected) else (255, 255, 255),
                2
            )

            if sqpnp_delta_x is not None and sqpnp_delta_y is not None:
                sqpnp_delta_text = f"d:({int(sqpnp_delta_x)},{int(sqpnp_delta_y)})"
            else:
                sqpnp_delta_text = "d:(None,None)"

            sqpnp_mode_text = (
                f"SQPnP: ON {sqpnp_delta_text}"
                if use_sqpnp_corrected
                else "SQPnP: OFF"
            )

            cv2.putText(
                kbd_bg,
                sqpnp_mode_text,
                (30, SCREEN_H - 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0) if use_sqpnp_corrected else (255, 255, 255),
                2
            )

            coord_text = (
                f"Raw:({raw_sx},{raw_sy}) "
                f"PoseCorrected:({corrected_sx},{corrected_sy}) "
                f"SQPnP:({sqpnp_corrected_sx},{sqpnp_corrected_sy}) "
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

            elif key == ord('h'):
                use_sqpnp_corrected = not use_sqpnp_corrected
                gaze.reset()
                print("use_sqpnp_corrected:", use_sqpnp_corrected)

            elif key == ord('m'):
                mouth_mode = True
                mouth_calibrator.reset()

                print("입벌림 캘리브레이션 시작")

            elif key == ord('t'):

                 if calibrator.done:

                    gaze.reset()

                    if use_sqpnp_corrected:
                        version_name = "v0.1-sqpnp-corrected"
                    elif use_pose_corrected:
                        version_name = "v0.1-pose-corrected"
                    else:
                        version_name = "v0.1-raw"

                    collector = MetricsCollector(
                        user_id="yejin",
                        dev_version=version_name,
                        px_per_cm=PX_PER_CM,
                        calib_id=calibrator.calib_id,
                        calib_reproj_rmse_px=calibrator.calib_reproj_rmse_px,
                    )

                    run_gaze_accuracy_test(
                        cap,
                        face_mesh,
                        calibrator,
                        gaze,
                        collector,
                        use_pose_corrected,
                        use_sqpnp_corrected
                    )

                    last_session_id = collector.session_id

            elif key == ord('g'):

                if calibrator.done and calibrator.pose_baseline is not None:

                    run_head_movement_test(cap, face_mesh, calibrator)

                else:
                    print(
                        "[head_movement_test] 캘리브레이션 또는 pose_baseline이 없어 "
                        "실행할 수 없습니다. 먼저 'r'로 캘리브레이션을 완료하세요."
                    )

            elif key == ord('9') or key == ord('x'):
                # '9' 키가 환경/키보드 배열에 따라 잘 안 잡히는 경우를 대비해
                # 'x'로도 동일하게 실행되도록 해둔다.

                if calibrator.done and calibrator.pose_baseline is not None:

                    run_head_movement_test_multi(cap, face_mesh, calibrator)

                else:
                    print(
                        "[head_movement_test_multi] 캘리브레이션 또는 pose_baseline이 없어 "
                        "실행할 수 없습니다. 먼저 'r'로 캘리브레이션을 완료하세요."
                    )

            elif key == ord('k'):
                # 세션 적응형 keyboard ROI 보정 모델: mini calibration 수집(실험 모드)

                if calibrator.done and calibrator.pose_baseline is not None:

                    fitted_model, fitted_roi = run_keyboard_session_calibration(
                        cap, face_mesh, calibrator, buttonList
                    )

                    if fitted_model is not None:
                        keyboard_session_model = fitted_model
                        keyboard_roi = fitted_roi
                        use_keyboard_session_model = False

                else:
                    print(
                        "[keyboard_session_calib] 캘리브레이션 또는 pose_baseline이 없어 "
                        "실행할 수 없습니다. 먼저 'r'로 캘리브레이션을 완료하세요."
                    )

            elif key == ord('j'):
                # 세션 적응형 keyboard ROI 보정 모델 적용 on/off (실험 모드)
                # h(SQPnP)와는 동시에 켜지지 않는다. h 키 자체는 다른 팀원
                # 작업이라 건드리지 않고, 여기서 켜는 시점만 막는다
                # (SQPnP가 이미 켜져 있는 동안 켜졌더라도 실시간 루프의
                # gate가 use_sqpnp_corrected일 때 session_model 적용을
                # 항상 스킵하므로 실제로 동시 적용되는 경우는 없다).

                if use_sqpnp_corrected:
                    print(
                        "[keyboard_session_model] SQPnP 모드(h)가 켜져 있어 "
                        "session_model을 켤 수 없습니다. session_model과 SQPnP 경로는 "
                        "동시에 쓸 수 없습니다. 먼저 'h'로 SQPnP를 끄세요."
                    )

                elif (
                    keyboard_session_model is not None
                    and keyboard_session_model.usable_provisional
                ):
                    use_keyboard_session_model = not use_keyboard_session_model

                    if use_keyboard_session_model:

                        if keyboard_session_model.usable_confirmed is True:
                            print("use_keyboard_session_model: True (검증 통과)")
                        else:
                            print(
                                "[keyboard_session_model] 경고: 검증 전 실험 적용입니다 "
                                f"(usable_confirmed={keyboard_session_model.usable_confirmed}). "
                                "'l' 키로 raw/pose/session_model 검증 테스트를 먼저 "
                                "실행하는 것을 권장합니다."
                            )

                    else:
                        print("use_keyboard_session_model: False")

                else:
                    print(
                        "[keyboard_session_model] 아직 fit되지 않았거나 "
                        "usable_provisional=False입니다. 먼저 'k'로 세션 캘리브레이션을 "
                        "완료하세요."
                    )

            elif key == ord('l'):
                # raw/pose/session_model 병렬 검증 테스트(실험 모드)

                if keyboard_session_model is not None:
                    run_keyboard_session_validation_test(
                        cap,
                        face_mesh,
                        calibrator,
                        buttonList,
                        keyboard_session_model,
                        keyboard_roi
                    )

                else:
                    print(
                        "[keyboard_session_validation] keyboard_session_model이 없어 "
                        "실행할 수 없습니다. 먼저 'k'로 세션 캘리브레이션을 완료하세요."
                    )

    cap.release()
    cv2.destroyAllWindows()

    if keyboard_compare_rows:

        ks_compare_fieldnames = [
            "schema_version", "calib_session_id", "timestamp",
            "raw_sx", "raw_sy", "raw_kx", "raw_ky",
            "corrected_sx", "corrected_sy",
            "session_model_sx", "session_model_sy",
            "residual_kx", "residual_ky",
            "applied", "variant_used", "preclip_severity",
            "gate_model_usable", "gate_preclip",
            "gate_feature_range", "gate_residual_size",
            "fallback_reason",
            "delta_center_x", "delta_center_y", "scale_ratio",
            "active_source",
        ]

        ks_compare_csv_path = os.path.join(
            "gaze_accuracy_results",
            f"keyboard_session_realtime_compare_v{KEYBOARD_SESSION_COMPARE_SCHEMA_VERSION}.csv"
        )

        append_rows(ks_compare_csv_path, ks_compare_fieldnames, keyboard_compare_rows)

        print(
            f"[keyboard_session_compare] CSV 저장 완료: {ks_compare_csv_path} "
            f"({len(keyboard_compare_rows)} rows)"
        )

    # ── 종료 시 마지막 세션 결과 팝업 (측정한 적 있을 때만) ──
    if last_session_id is not None:
        show_session_popup(last_session_id)


if __name__ == "__main__":
    main()
