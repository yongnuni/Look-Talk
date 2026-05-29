import cv2
import mediapipe as mp
import numpy as np

# ── MediaPipe 초기화 ──────────────────────────────────────────

mp_face_mesh = mp.solutions.face_mesh

LEFT_IRIS = 468
RIGHT_IRIS = 473

LEFT_EYE = [
    33, 133,
    159, 145,
    160, 161,
    246
]

RIGHT_EYE = [
    362, 263,
    386, 374,
    385, 384,
    466
]

LEFT_IRIS_RING = [
    469,
    470,
    471,
    472
]

RIGHT_IRIS_RING = [
    474,
    475,
    476,
    477
]


# ── 홍채 중심 계산 ───────────────────────────────────────────

def get_avg_iris(landmarks):

    lx = landmarks.landmark[LEFT_IRIS].x
    ly = landmarks.landmark[LEFT_IRIS].y

    rx = landmarks.landmark[RIGHT_IRIS].x
    ry = landmarks.landmark[RIGHT_IRIS].y

    return (
        (lx + rx) / 2,
        (ly + ry) / 2
    )


# ── EAR 계산 ────────────────────────────────────────────────

def eye_aspect_ratio(
    landmarks,
    outer_idx,
    inner_idx,
    top_idx,
    bottom_idx
):

    lm = landmarks.landmark

    eye_width = abs(
        lm[outer_idx].x -
        lm[inner_idx].x
    )

    eye_height = abs(
        lm[top_idx].y -
        lm[bottom_idx].y
    )

    if eye_width <= 0.001:
        return 0.0

    return eye_height / eye_width


# ── 눈 깜빡임 검출 ───────────────────────────────────────────

def is_blink(landmarks):

    left_ear = eye_aspect_ratio(
        landmarks,
        33,
        133,
        159,
        145
    )

    right_ear = eye_aspect_ratio(
        landmarks,
        263,
        362,
        386,
        374
    )

    avg_ear = (
        left_ear +
        right_ear
    ) / 2

    return avg_ear < 0.18


# ── 홍채 추적 신뢰도 계산 ────────────────────────────────────

def iris_confidence(landmarks):

    left_ear = eye_aspect_ratio(
        landmarks,
        33,
        133,
        159,
        145
    )

    right_ear = eye_aspect_ratio(
        landmarks,
        263,
        362,
        386,
        374
    )

    avg_ear = (
        left_ear +
        right_ear
    ) / 2

    if avg_ear < 0.18:
        return 0.0

    lm = landmarks.landmark

    left_iris_x = lm[LEFT_IRIS].x
    right_iris_x = lm[RIGHT_IRIS].x

    def center_score(
        iris_x,
        outer_idx,
        inner_idx,
        top_idx,
        bottom_idx
    ):

        center_x = (
            lm[outer_idx].x +
            lm[inner_idx].x
        ) / 2

        eye_width = abs(
            lm[outer_idx].x -
            lm[inner_idx].x
        )

        return max(
            0.0,
            1.0 -
            abs(
                iris_x - center_x
            ) /
            (
                eye_width / 2 + 1e-6
            )
        )

    left_score = center_score(
        left_iris_x,
        33,
        133,
        159,
        145
    )

    right_score = center_score(
        right_iris_x,
        263,
        362,
        386,
        374
    )

    score = (
        avg_ear * 3 *
        (
            left_score +
            right_score
        ) / 2
    )

    return min(
        1.0,
        score
    )


# ── 눈 윤곽선 그리기 ─────────────────────────────────────────

def draw_eye_contour(
    frame,
    landmarks,
    indices,
    width,
    height
):

    points = np.array(
        [
            (
                int(
                    landmarks.landmark[i].x
                    * width
                ),
                int(
                    landmarks.landmark[i].y
                    * height
                )
            )
            for i in indices
        ],
        dtype=np.int32
    )

    cv2.polylines(
        frame,
        [points],
        True,
        (147,112,219),
        1
    )


# ── 홍채 링 그리기 ───────────────────────────────────────────

def draw_iris_ring(
    frame,
    landmarks,
    center_idx,
    ring_indices,
    width,
    height,
    color
):

    center_x = int(
        landmarks.landmark[center_idx].x
        * width
    )

    center_y = int(
        landmarks.landmark[center_idx].y
        * height
    )

    if ring_indices:

        points = [

            (
                int(
                    landmarks.landmark[i].x
                    * width
                ),
                int(
                    landmarks.landmark[i].y
                    * height
                )
            )

            for i in ring_indices
        ]

        radii = [

            int(
                np.hypot(
                    p[0] - center_x,
                    p[1] - center_y
                )
            )

            for p in points
        ]

        radius = max(
            3,
            int(
                np.mean(radii)
            )
        )

    else:

        radius = 5

    cv2.circle(
        frame,
        (center_x, center_y),
        radius,
        color,
        1
    )

    cv2.circle(
        frame,
        (center_x, center_y),
        2,
        color,
        -1
    )