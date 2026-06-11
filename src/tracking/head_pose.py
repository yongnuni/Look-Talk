import cv2
import numpy as np


# MediaPipe FaceMesh 기준 주요 얼굴 landmark
FACE_2D_INDICES = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "left_mouth": 61,
    "right_mouth": 291,
}


# 대략적인 3D 얼굴 모델 좌표
# 실제 cm 단위라기보다 상대적인 얼굴 구조를 표현하는 기준 좌표입니다.
FACE_3D_MODEL = np.array(
    [
        [0.0, 0.0, 0.0],          # nose tip
        [0.0, -63.6, -12.5],      # chin
        [-43.3, 32.7, -26.0],     # left eye outer
        [43.3, 32.7, -26.0],      # right eye outer
        [-28.9, -28.9, -24.1],    # left mouth
        [28.9, -28.9, -24.1],     # right mouth
    ],
    dtype=np.float64
)


def _invalid_result():
    return {
        "valid": False,
        "yaw": 0.0,
        "pitch": 0.0,
        "roll": 0.0,
        "face_scale": 0.0,
        "tz": 0.0,
        "face_center_x": 0.5,
        "face_center_y": 0.5,
    }


def rotation_matrix_to_euler_angles(rotation_matrix):
    """
    회전 행렬을 yaw, pitch, roll 각도로 변환합니다.
    반환값 단위는 degree입니다.
    """

    sy = np.sqrt(
        rotation_matrix[0, 0] ** 2 +
        rotation_matrix[1, 0] ** 2
    )

    singular = sy < 1e-6

    if not singular:
        pitch = np.arctan2(
            rotation_matrix[2, 1],
            rotation_matrix[2, 2]
        )

        yaw = np.arctan2(
            -rotation_matrix[2, 0],
            sy
        )

        roll = np.arctan2(
            rotation_matrix[1, 0],
            rotation_matrix[0, 0]
        )

    else:
        pitch = np.arctan2(
            -rotation_matrix[1, 2],
            rotation_matrix[1, 1]
        )

        yaw = np.arctan2(
            -rotation_matrix[2, 0],
            sy
        )

        roll = 0.0

    return (
        float(np.degrees(yaw)),
        float(np.degrees(pitch)),
        float(np.degrees(roll))
    )


def estimate_head_pose(landmarks, frame_width, frame_height):
    """
    MediaPipe FaceMesh landmarks를 받아 head pose와 얼굴 중심 정보를 계산합니다.

    Returns:
        {
            "valid": bool,
            "yaw": float,
            "pitch": float,
            "roll": float,
            "face_scale": float,
            "tz": float,
            "face_center_x": float,
            "face_center_y": float
        }
    """

    if landmarks is None:
        return _invalid_result()

    if frame_width <= 0 or frame_height <= 0:
        return _invalid_result()

    try:
        lm = landmarks.landmark

        image_points = np.array(
            [
                [
                    lm[FACE_2D_INDICES["nose_tip"]].x * frame_width,
                    lm[FACE_2D_INDICES["nose_tip"]].y * frame_height
                ],
                [
                    lm[FACE_2D_INDICES["chin"]].x * frame_width,
                    lm[FACE_2D_INDICES["chin"]].y * frame_height
                ],
                [
                    lm[FACE_2D_INDICES["left_eye_outer"]].x * frame_width,
                    lm[FACE_2D_INDICES["left_eye_outer"]].y * frame_height
                ],
                [
                    lm[FACE_2D_INDICES["right_eye_outer"]].x * frame_width,
                    lm[FACE_2D_INDICES["right_eye_outer"]].y * frame_height
                ],
                [
                    lm[FACE_2D_INDICES["left_mouth"]].x * frame_width,
                    lm[FACE_2D_INDICES["left_mouth"]].y * frame_height
                ],
                [
                    lm[FACE_2D_INDICES["right_mouth"]].x * frame_width,
                    lm[FACE_2D_INDICES["right_mouth"]].y * frame_height
                ],
            ],
            dtype=np.float64
        )

        focal_length = frame_width

        camera_matrix = np.array(
            [
                [focal_length, 0, frame_width / 2],
                [0, focal_length, frame_height / 2],
                [0, 0, 1],
            ],
            dtype=np.float64
        )

        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rotation_vector, translation_vector = cv2.solvePnP(
            FACE_3D_MODEL,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return _invalid_result()

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        yaw, pitch, roll = rotation_matrix_to_euler_angles(rotation_matrix)

        # pitch 값은 환경에 따라 160~180도처럼 나올 수 있으므로
        # 절대값 기준 invalid 필터는 사용하지 않습니다.
        # 필요한 경우 baseline과의 delta로 판단해야 합니다.

        left_eye = image_points[2]
        right_eye = image_points[3]

        face_scale = float(
            np.linalg.norm(
                right_eye - left_eye
            )
        )

        tz = float(
            translation_vector[2][0]
        )

        # 얼굴 중심 좌표.
        # 0~1 범위의 MediaPipe 정규화 좌표를 그대로 사용합니다.
        # 카메라 화면 안에서 얼굴 전체가 이동한 정도를 파악하는 데 사용합니다.
        face_center_x = float(
            (
                lm[FACE_2D_INDICES["nose_tip"]].x +
                lm[FACE_2D_INDICES["left_eye_outer"]].x +
                lm[FACE_2D_INDICES["right_eye_outer"]].x +
                lm[FACE_2D_INDICES["left_mouth"]].x +
                lm[FACE_2D_INDICES["right_mouth"]].x
            ) / 5
        )

        face_center_y = float(
            (
                lm[FACE_2D_INDICES["nose_tip"]].y +
                lm[FACE_2D_INDICES["left_eye_outer"]].y +
                lm[FACE_2D_INDICES["right_eye_outer"]].y +
                lm[FACE_2D_INDICES["left_mouth"]].y +
                lm[FACE_2D_INDICES["right_mouth"]].y
            ) / 5
        )

        return {
            "valid": True,
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
            "face_scale": face_scale,
            "tz": tz,
            "face_center_x": face_center_x,
            "face_center_y": face_center_y,
        }

    except Exception as e:
        print("head pose error:", e)
        return _invalid_result()