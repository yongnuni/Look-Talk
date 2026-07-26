import cv2
import numpy as np
import csv
import math
import time
import os
from datetime import datetime
from PIL import Image, ImageDraw
from src.calibrations.baseline_manager import save_baseline
from src.tracking.blink import BlinkDetector, BlinkKind
import src.viz.viz as viz
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
    iris_confidence,
    draw_eye_contour,
    draw_iris_ring
)

from src.tracking.mouth import (
    MouthClickDetector,
    draw_mouth,
    mouth_aspect_ratio
)

from src.calibrations.mouth_calibration import MouthCalibration
from src.tracking.gaze_pipeline import GazePipeline
from src.tracking.dwell import DwellController
from src.tracking.head_pose import estimate_head_pose, estimate_sqpnp_headpose

from src.tracking.mappers.factory import (
    create_mapper,
    MODE_CALIBRATED,
    MODE_NO_CALIBRATION,
    AVAILABLE_MODES,
    DEFAULT_NO_CALIBRATION_STRATEGY,
)
from src.tracking.mappers.strategies import available_strategies
from src.metrics.session_logger import SessionLogger

# ── 하이브리드(백본+릿지) 모듈 ──
from src.tracking.gaze_backbone import GazeBackbone
from src.tracking.feature_builder import build_features

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

# 백본 추론 주기 (프레임). CPU 부담을 줄이기 위해 N프레임마다 1회 추론하고
# 그 사이에는 마지막 시선 벡터를 재사용합니다. 1이면 매 프레임 추론.
BACKBONE_INFER_EVERY = 2

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
    backbone=None
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

    backbone_frame_count = 0
    last_gaze_vec = None

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

                # ── 하이브리드: 시선 벡터 + 특징 벡터 ──
                if backbone is not None and backbone.available:
                    if backbone_frame_count % BACKBONE_INFER_EVERY == 0:
                        last_gaze_vec = backbone.predict(frame, lms)
                    backbone_frame_count += 1

                features = build_features(
                    iris_x,
                    iris_y,
                    head_pose,
                    gaze_vec=last_gaze_vec,
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

    collector.end_session()
    collector.export_csv(
        sessions_path=os.path.join(out_dir, f"sessions_v{MetricsCollector.SCHEMA_VERSION}.csv"),
        accuracy_path=os.path.join(out_dir, f"gaze_accuracy_v{MetricsCollector.SCHEMA_VERSION}.csv")
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

        screen_w, screen_h = viz.infer_screen_size(s)
        viz.plot_session_overview(s, screen_w, screen_h)
        plt.show()

    except Exception as e:
        print(f"[popup] 시각화 실패: {e}")

def main(mode=MODE_CALIBRATED, strategy_name=None):

    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

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

    mapper = create_mapper(mode, strategy_name)
    mouth_calibrator = MouthCalibration()
    gaze = GazePipeline()
    dwell = DwellController()
    mouth = MouthClickDetector()
    tester = TestRunner()
    blink_detector = BlinkDetector(
        detect_natural=True,
        detect_intentional=True
    )

    # ── 하이브리드 백본 초기화 ──
    # 모델 파일이 없거나 onnxruntime 미설치면 available=False로만 남고
    # 기존 파이프라인은 그대로 동작합니다 (릿지는 기하 특징만으로 학습됨).
    backbone = GazeBackbone()

    is_korean = True
    is_shift = False
    show_all_markers = False   # b 키로 토글: 후보 좌표를 동시에 마커로 표시

    mouth_mode = False
    show_debug_overlay = False   # d 키로 토글: head pose/gaze 진단 텍스트 표시 여부

    session_logger = SessionLogger(
        mode=mode,
        mapper_metadata=mapper.get_metadata(),
        screen_w=SCREEN_W,
        screen_h=SCREEN_H,
    )

    last_session_id = None

    last_gaze_x = SCREEN_W // 2
    last_gaze_y = SCREEN_H // 2

    # 백본 추론 스로틀링 상태
    backbone_frame_count = 0
    last_gaze_vec = None

    buttonList = create_buttons(keys_kor_normal)

    calib_canvas = np.zeros(
        (SCREEN_H, SCREEN_W, 3),
        dtype=np.uint8
    )

    print(
        f"Eye Keyboard 시작 [mode={mode}] | "
        "r: 재캘리브레이션/리셋 | "
        "t: 시선정확도테스트(calibrated 전용) | "
        "m: 입벌림 입력 방식 변경 | "
        "p/h/g/o: 매핑 방식 전환(calibrated 전용) | "
        "b: 후보 마커 토글 | "
        "d: 디버그 오버레이 토글 | "
        "q: 종료"
    )

    if mode == MODE_NO_CALIBRATION:
        print(f"[mode] no_calibration strategy: {mapper.active_method}")
        print(f"[mode] strategy params: {mapper.get_metadata()['strategy_params']}")

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

        if mode == MODE_CALIBRATED:
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
            blink_event = None

            mapping_result = None
            sx = None
            sy = None

            features = None

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

            # 얼굴 미검출 프레임에도 캘리브레이션 화면 유지 (깜빡임 방지)
            # no_calibration 모드는 mapper.ready가 항상 True라 이 블록에
            # 들어오지 않는다.
            if not mapper.ready and not results.multi_face_landmarks:

                draw_calib_screen(calib_canvas, mapper.calibrator, elapsed_ratio)
                cv2.imshow("Eye Keyboard", calib_canvas)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord('r'):
                    mapper.reset()

                continue

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
                blink_event = blink_detector.update(lms)
                blink = blink_detector.is_closed
                conf = iris_confidence(lms)                
                
                # ── 하이브리드: 시선 벡터 추론 (스로틀링) + 특징 벡터 ──
                # 캘리브레이션 단계에서도 릿지 학습용 특징을 쌓아야 하므로
                # calibrator.done 여부와 무관하게 여기서 계산합니다.
                if backbone.available:
                    if backbone_frame_count % BACKBONE_INFER_EVERY == 0:
                        last_gaze_vec = backbone.predict(frame, lms)
                    backbone_frame_count += 1

                features = build_features(
                    iris_x,
                    iris_y,
                    head_pose,
                    gaze_vec=last_gaze_vec,
                    frame_width=fw
                )

                # ── 캘리브레이션 (calibrated 모드에서만 진입) ──────
                # no_calibration 모드는 mapper.ready가 항상 True라
                # 이 블록에 들어오지 않는다.

                if not mapper.ready:

                    if not blink:
                        elapsed_ratio = mapper.update_initialization(
                            iris_x,
                            iris_y,
                            conf,
                            head_pose=head_pose,
                            features=features
                        )

                    draw_calib_screen(calib_canvas, mapper.calibrator, elapsed_ratio)
                    cv2.imshow("Eye Keyboard", calib_canvas)

                    key = cv2.waitKey(1) & 0xFF

                    if key == ord('q'):
                        break
                    elif key == ord('r'):
                        mapper.reset()

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

                # ── 시선 매핑 (calibrated/no_calibration 공통 경계) ──
                # 최종 좌표 선택 알고리즘은 여기서 알지 못한다 — mapper가 다
                # 계산해서 MappingResult로 돌려준다.
                mapping_result = mapper.map(
                    iris_x,
                    iris_y,
                    head_pose=head_pose,
                    features=features,
                    sqpnp_head_pose=sqpnp_headpose,
                )

                sx, sy = mapping_result.x, mapping_result.y

                # 4. 최종 gaze pipeline 입력 전 좌표 유효성 검사
                # 화면 가장자리 좌표는 invalid로 보지 않음.
                # None / NaN / inf 같은 진짜 비정상값만 막음.
                screen_coord_valid = mapping_result.valid

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
            kbd_bg[:] = (245, 246, 248)

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

            # ── 공통 세션 로깅 (calibrated/no_calibration 공통, mapper와 무관) ──
            session_logger.log_frame(
                active_method=(
                    mapping_result.active_method if mapping_result is not None else None
                ),
                raw_iris_x=iris_x,
                raw_iris_y=iris_y,
                mapped_sx=sx,
                mapped_sy=sy,
                mapping_valid=(
                    mapping_result.valid if mapping_result is not None else False
                ),
                gaze_x=gaze_x,
                gaze_y=gaze_y,
                hovered_key=hovered_key,
                dwell_ratio=dwell_ratio,
                clicked_key=clicked_key,
                input_text_len=len(current_text),
            )

            # ── 디버그 오버레이 (head pose / gaze 좌표 진단) ──────────
            # d 키로 표시 여부만 토글. 계산 로직 자체는 항상 실행됨.
            # 내용은 모드별로 다르다 — main.py는 mapping_result만 읽고,
            # raw/pose/sqpnp/ridge 후보 계산 자체는 하지 않는다.
            if show_debug_overlay:

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

                if mode == MODE_CALIBRATED:

                    pose_delta = mapper.calibrator.get_pose_delta(head_pose)

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

                    active_method = mapper.active_method
                    mode_text = f"Mode: {active_method}"

                    cv2.putText(
                        kbd_bg,
                        mode_text,
                        (30, SCREEN_H - 180),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0) if active_method != "raw" else (255, 255, 255),
                        2
                    )

                    meta = mapping_result.metadata if mapping_result is not None else {}
                    sqpnp_delta_x = meta.get("sqpnp_delta_x")
                    sqpnp_delta_y = meta.get("sqpnp_delta_y")

                    if sqpnp_delta_x is not None and sqpnp_delta_y is not None:
                        sqpnp_delta_text = f"d:({int(sqpnp_delta_x)},{int(sqpnp_delta_y)})"
                    else:
                        sqpnp_delta_text = "d:(None,None)"

                    sqpnp_mode_text = (
                        f"SQPnP: ON {sqpnp_delta_text}"
                        if active_method == "sqpnp_corrected"
                        else "SQPnP: OFF"
                    )

                    cv2.putText(
                        kbd_bg,
                        sqpnp_mode_text,
                        (30, SCREEN_H - 240),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0) if active_method == "sqpnp_corrected" else (255, 255, 255),
                        2
                    )

                    # ── 릿지/백본 상태 표시 ──
                    if last_gaze_vec is not None:
                        gaze_vec_text = f"GazeVec:({last_gaze_vec[0]:.1f},{last_gaze_vec[1]:.1f})"
                    else:
                        gaze_vec_text = "GazeVec:(None)"

                    ridge_sx, ridge_sy = (
                        (mapping_result.candidates.get("ridge_hybrid") or (None, None))
                        if mapping_result is not None else (None, None)
                    )

                    ridge_status_text = (
                        f"Ridge: {'FIT' if mapper.calibrator.ridge.fitted else 'NOT_FIT'} "
                        f"Backbone: {'ON' if backbone.available else 'OFF'} "
                        f"{gaze_vec_text} "
                        f"RidgeXY:({ridge_sx},{ridge_sy})"
                    )

                    cv2.putText(
                        kbd_bg,
                        ridge_status_text,
                        (30, SCREEN_H - 270),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0) if active_method == "ridge_hybrid" else (255, 255, 255),
                        2
                    )

                    candidates = mapping_result.candidates if mapping_result is not None else {}
                    raw_xy = candidates.get("raw") or (None, None)
                    pose_xy = candidates.get("pose_corrected") or (None, None)
                    sqpnp_xy = candidates.get("sqpnp_corrected") or (None, None)

                    coord_text = (
                        f"Raw:{raw_xy} "
                        f"PoseCorrected:{pose_xy} "
                        f"SQPnP:{sqpnp_xy} "
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

                    corrected_iris_x = meta.get("corrected_iris_x")
                    corrected_iris_y = meta.get("corrected_iris_y")

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

                else:
                    # ── no_calibration 모드용 간단 디버그 텍스트 ──
                    meta = mapping_result.metadata if mapping_result is not None else {}
                    method = mapping_result.active_method if mapping_result is not None else "N/A"

                    nc_text = (
                        f"Mode: no_calibration ({method}) "
                        f"Valid:{screen_coord_valid} "
                        f"Mapped:({sx},{sy}) "
                        f"Gaze:({gaze_x},{gaze_y}) "
                        f"Iris:({iris_x:.4f},{iris_y:.4f})"
                    )

                    cv2.putText(
                        kbd_bg,
                        nc_text,
                        (30, SCREEN_H - 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        2
                    )

                    params_text = f"Params: {meta.get('strategy_params')}"

                    cv2.putText(
                        kbd_bg,
                        params_text,
                        (30, SCREEN_H - 150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (200, 200, 200),
                        2
                    )

            # ── 후보 좌표 마커 동시 표시 (비교용, 클릭에는 영향 없음) ──
            if show_all_markers and mapping_result is not None:
                # 알려진 calibrated 후보 이름은 기존 색/라벨을 그대로 쓰고,
                # 새 strategy 등 미지의 후보 이름은 기본 색으로 대체한다.
                marker_style = {
                    "raw": ((255, 255, 255), "R"),
                    "pose_corrected": ((0, 255, 0), "P"),
                    "sqpnp_corrected": ((0, 165, 255), "S"),
                    "ridge_hybrid": ((255, 0, 255), "G"),
                }

                for name, xy in mapping_result.candidates.items():
                    if xy is None:
                        continue

                    mx, my = xy
                    if mx is None or my is None:
                        continue

                    color, label = marker_style.get(
                        name, ((0, 200, 200), name[:1].upper() if name else "?")
                    )

                    cv2.circle(kbd_bg, (int(mx), int(my)), 12, color, 2)
                    cv2.putText(
                        kbd_bg, label, (int(mx) + 14, int(my) - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                    )

            kbd_bg = draw_gaze_cursor(kbd_bg, gaze_x, gaze_y, fixation_count)
            kbd_bg = draw_status_bar(kbd_bg, is_korean, fixation_count)

            cv2.imshow("Eye Keyboard", kbd_bg)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord('r'):
                mapper.reset()
                gaze.reset()

                if mode == MODE_CALIBRATED:
                    if not show_countdown(cap, face_mesh):
                        break
                else:
                    print("[mode] no_calibration strategy/gaze 상태를 초기화했습니다.")

            elif key == ord('p'):
                if mode == MODE_CALIBRATED:
                    new_method = (
                        "raw" if mapper.active_method == "pose_corrected" else "pose_corrected"
                    )
                    mapper.set_active_method(new_method)
                    gaze.reset()
                    print("active_method:", mapper.active_method)
                    show_calibration_guide()
                else:
                    print("[mode] p 키는 calibrated 모드 전용입니다.")

            elif key == ord('h'):
                if mode == MODE_CALIBRATED:
                    new_method = (
                        "raw" if mapper.active_method == "sqpnp_corrected" else "sqpnp_corrected"
                    )
                    mapper.set_active_method(new_method)
                    gaze.reset()
                    print("active_method:", mapper.active_method)
                else:
                    print("[mode] h 키는 calibrated 모드 전용입니다.")

            elif key == ord('g'):
                if mode == MODE_CALIBRATED:
                    # 릿지 하이브리드 모드 토글 (fitted 여부 경고는 set_active_method 내부에서 처리)
                    new_method = (
                        "raw" if mapper.active_method == "ridge_hybrid" else "ridge_hybrid"
                    )
                    mapper.set_active_method(new_method)
                    gaze.reset()
                    print("active_method:", mapper.active_method)
                else:
                    print("[mode] g 키는 calibrated 모드 전용입니다.")

            elif key == ord('o'):
                if mode == MODE_CALIBRATED:
                    # raw 전용 키: 모든 보정 모드를 꺼서 순수 raw로 전환
                    mapper.set_active_method("raw")
                    gaze.reset()
                    print("[mode] Raw로 전환")
                else:
                    print("[mode] o 키는 calibrated 모드 전용입니다.")

            elif key == ord('b'):
                show_all_markers = not show_all_markers
                print("show_all_markers:", show_all_markers)

            elif key == ord('d'):
                show_debug_overlay = not show_debug_overlay
                print("show_debug_overlay:", show_debug_overlay)

            elif key == ord('m'):
                mouth_mode = True
                mouth_calibrator.reset()

                print("입벌림 캘리브레이션 시작")

            elif key == ord('t'):

                 if mode != MODE_CALIBRATED:
                    print("[test] 정확도 테스트는 현재 calibrated 모드에서만 지원됩니다.")

                 elif mapper.ready:

                    gaze.reset()

                    use_pose_corrected = mapper.active_method == "pose_corrected"
                    use_sqpnp_corrected = mapper.active_method == "sqpnp_corrected"
                    use_ridge = mapper.active_method == "ridge_hybrid"

                    if use_ridge:
                        version_name = "v0.2-ridge-hybrid"
                    elif use_sqpnp_corrected:
                        version_name = "v0.1-sqpnp-corrected"
                    elif use_pose_corrected:
                        version_name = "v0.1-pose-corrected"
                    else:
                        version_name = "v0.1-raw"

                    collector = MetricsCollector(
                        user_id="yejin",
                        dev_version=version_name,
                        px_per_cm=PX_PER_CM,
                        calib_id=mapper.calibrator.calib_id,
                        calib_reproj_rmse_px=mapper.calibrator.calib_reproj_rmse_px,
                        use_pose_corrected=use_pose_corrected,
                        use_sqpnp_corrected=use_sqpnp_corrected,
                    )

                    run_gaze_accuracy_test(
                        cap,
                        face_mesh,
                        mapper.calibrator,
                        gaze,
                        collector,
                        blink_detector,
                        use_pose_corrected,
                        use_sqpnp_corrected,
                        use_ridge=use_ridge,
                        backbone=backbone
                    )

                    last_session_id = collector.session_id

    session_logger.close()

    cap.release()
    cv2.destroyAllWindows()

    # ── 종료 시 마지막 세션 결과 팝업 (측정한 적 있을 때만) ──
    if last_session_id is not None:
        show_session_popup(last_session_id)


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Look-Talk Eye Keyboard")

    parser.add_argument(
        "--gaze-mode",
        dest="gaze_mode",
        choices=list(AVAILABLE_MODES),
        default=MODE_CALIBRATED,
        help=(
            "시선 매핑 모드. 'calibrated'(기본값)는 기존 16점 캘리브레이션을 "
            "사용하고, 'no_calibration'은 캘리브레이션 화면 없이 바로 "
            "키보드로 진입한다."
        ),
    )

    parser.add_argument(
        "--strategy",
        dest="strategy",
        default=None,
        help=(
            "no_calibration 모드에서 사용할 strategy 이름. 생략하면 "
            f"기본값('{DEFAULT_NO_CALIBRATION_STRATEGY}')을 사용한다. "
            f"사용 가능한 strategy: {', '.join(available_strategies()) or '(없음)'}"
        ),
    )

    args = parser.parse_args()

    if args.gaze_mode == MODE_NO_CALIBRATION:
        name = args.strategy or DEFAULT_NO_CALIBRATION_STRATEGY
        if name not in available_strategies():
            available = ", ".join(available_strategies()) or "(없음)"
            parser.error(
                f"'{name}'은(는) 등록된 no_calibration strategy가 아닙니다. "
                f"사용 가능한 strategy: {available}"
            )

    return args.gaze_mode, args.strategy


if __name__ == "__main__":
    _mode, _strategy_name = _parse_args()
    main(mode=_mode, strategy_name=_strategy_name)