import numpy as np

"""
릿지 회귀 매핑용 특징 벡터 조립기 (하이브리드 방식의 접착제).

특징 순서는 절대 바꾸면 안 됩니다.
캘리브레이션 때 학습한 순서와 런타임 순서가 다르면
회귀 모델이 엉뚱한 좌표를 출력합니다.

FEATURE_NAMES가 단일 진실 공급원(single source of truth)입니다.
"""

FEATURE_NAMES = [
    "iris_x",          # 0: 기존 파이프라인의 홍채 x (0~1)
    "iris_y",          # 1: 기존 파이프라인의 홍채 y (0~1)
    "head_yaw",        # 2: solvePnP head yaw (deg)
    "head_pitch",      # 3
    "head_roll",       # 4
    "face_center_x",   # 5: 얼굴 중심 (0~1) → 화면 내 평행이동 보상용
    "face_center_y",   # 6
    "face_scale_norm", # 7: 눈 사이 거리 / 프레임 폭 → 거리(스케일) 보상용
    "left_eye_x",
    "left_eye_y",
    "right_eye_x",
    "right_eye_y",

    "left_rel_x",
    "left_rel_y",
    "right_rel_x",
    "right_rel_y"
]

FEATURE_DIM = len(FEATURE_NAMES)


def build_features(
    iris_x,
    iris_y,
    landmarks,
    head_pose,
    frame_width=640
):
    """
    Args:
        iris_x, iris_y: get_avg_iris() 결과 (0~1 정규화 좌표)
        head_pose: estimate_head_pose() 반환 dict
        frame_width: face_scale 정규화용 프레임 폭(px)

    Returns:
        np.ndarray shape (FEATURE_DIM,) 또는 None (필수값 결손 시)
    """

    if iris_x is None or iris_y is None:
        return None

    if head_pose is None or not head_pose.get("valid", False):
        # 머리 자세가 invalid면 특징 벡터를 만들지 않습니다.
        # (invalid 프레임을 학습/추론에 섞으면 회귀가 오염됩니다)
        return None

    face_scale = head_pose.get("face_scale", 0.0)
    face_scale_norm = face_scale / max(frame_width, 1)

    lm = landmarks.landmark

    left_eye_x = (lm[33].x + lm[133].x)/2
    left_eye_y = (lm[159].y + lm[145].y)/2

    right_eye_x = (lm[362].x + lm[263].x)/2
    right_eye_y = (lm[386].y + lm[374].y)/2

    # -------------------------
    # Iris center
    # -------------------------

    left_iris_x = np.mean([lm[i].x for i in [468,469,470,471,472]])
    left_iris_y = np.mean([lm[i].y for i in [468,469,470,471,472]])

    right_iris_x = np.mean([lm[i].x for i in [473,474,475,476,477]])
    right_iris_y = np.mean([lm[i].y for i in [473,474,475,476,477]])

    # -------------------------
    # Eye size
    # -------------------------

    left_width = max(abs(lm[133].x - lm[33].x), 1e-6)
    right_width = max(abs(lm[263].x - lm[362].x), 1e-6)

    left_height = max(
        lm[145].y - lm[159].y,
        1e-6
    )

    right_height = max(
        lm[374].y - lm[386].y,
        1e-6
    )

    # -------------------------
    # Relative iris
    # -------------------------

    left_rel_x = (left_iris_x - left_eye_x) / (left_width * 0.5)
    left_rel_y = (left_iris_y - left_eye_y) / (left_height * 0.5)

    right_rel_x = (right_iris_x - right_eye_x) / (right_width * 0.5)
    right_rel_y = (right_iris_y - right_eye_y) / (right_height * 0.5)

    left_rel_x = np.clip(left_rel_x, -1.2, 1.2)
    left_rel_y = np.clip(left_rel_y, -1.2, 1.2)

    right_rel_x = np.clip(right_rel_x, -1.2, 1.2)
    right_rel_y = np.clip(right_rel_y, -1.2, 1.2)

    feat = np.array(
        [
            iris_x,
            iris_y,

            head_pose.get("yaw", 0.0),
            head_pose.get("pitch", 0.0),
            head_pose.get("roll", 0.0),

            head_pose.get("face_center_x", 0.5),
            head_pose.get("face_center_y", 0.5),

            face_scale_norm,

            left_eye_x,
            left_eye_y,
            right_eye_x,
            right_eye_y,

            left_rel_x,
            left_rel_y,
            right_rel_x,
            right_rel_y
        ],
        dtype=np.float64
    )

    if not np.all(np.isfinite(feat)):
        return None

    return feat