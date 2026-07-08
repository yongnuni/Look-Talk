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
    "gaze_yaw",        # 2: 백본 시선 yaw (deg). 백본 없으면 0.0
    "gaze_pitch",      # 3: 백본 시선 pitch (deg). 백본 없으면 0.0
    "head_yaw",        # 4: solvePnP head yaw (deg)
    "head_pitch",      # 5
    "head_roll",       # 6
    "face_center_x",   # 7: 얼굴 중심 (0~1) → 화면 내 평행이동 보상용
    "face_center_y",   # 8
    "face_scale_norm", # 9: 눈 사이 거리 / 프레임 폭 → 거리(스케일) 보상용
]

FEATURE_DIM = len(FEATURE_NAMES)


def build_features(
    iris_x,
    iris_y,
    head_pose,
    gaze_vec=None,
    frame_width=640
):
    """
    Args:
        iris_x, iris_y: get_avg_iris() 결과 (0~1 정규화 좌표)
        head_pose: estimate_head_pose() 반환 dict
        gaze_vec: GazeBackbone.predict() 반환 (yaw, pitch) 또는 None
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

    if gaze_vec is not None:
        gaze_yaw, gaze_pitch = gaze_vec
    else:
        gaze_yaw, gaze_pitch = 0.0, 0.0

    face_scale = head_pose.get("face_scale", 0.0)
    face_scale_norm = face_scale / max(frame_width, 1)

    feat = np.array(
        [
            iris_x,
            iris_y,
            gaze_yaw,
            gaze_pitch,
            head_pose.get("yaw", 0.0),
            head_pose.get("pitch", 0.0),
            head_pose.get("roll", 0.0),
            head_pose.get("face_center_x", 0.5),
            head_pose.get("face_center_y", 0.5),
            face_scale_norm,
        ],
        dtype=np.float64
    )

    if not np.all(np.isfinite(feat)):
        return None

    return feat