import cv2
import numpy as np
import time

from src.config import (
    CALIB_POINTS,
    SCREEN_W,
    SCREEN_H,
    CALIB_STABILIZE_SEC,
    CALIB_COLLECT_SEC,
    CALIB_STD_X,
    CALIB_STD_Y
)


class Calibrator:

    def __init__(self):
        self.reset()

    def reset(self):

        self.idx = 0

        self.samples = []
        self.pose_samples = []

        self.iris_pts = []
        self.screen_pts = []
        self.pose_pts = []

        self.pose_baseline = None

        self.H = None

        self.done = False

        self.hold_start = None

        self.warning = ""
        self.warning_start = None

    def update(
        self,
        iris_x,
        iris_y,
        conf,
        head_pose=None
    ):

        now = time.time()

        if self.hold_start is None:
            self.hold_start = now

        elapsed = now - self.hold_start

        # 안정화 시간이 지난 뒤부터 샘플 수집
        if (
            elapsed >= CALIB_STABILIZE_SEC
            and conf > 0.4
        ):

            self.samples.append(
                (
                    iris_x,
                    iris_y
                )
            )

            if head_pose is not None and head_pose.get("valid", False):
                self.pose_samples.append(
                    (
                        head_pose.get("yaw", 0.0),
                        head_pose.get("pitch", 0.0),
                        head_pose.get("roll", 0.0),
                        head_pose.get("face_scale", 0.0),
                        head_pose.get("tz", 0.0),
                        head_pose.get("face_center_x", 0.5),
                        head_pose.get("face_center_y", 0.5),
                    )
                )

        if elapsed >= (
            CALIB_STABILIZE_SEC +
            CALIB_COLLECT_SEC
        ):

            if len(self.samples) > 5:

                xs = sorted(
                    s[0]
                    for s in self.samples
                )

                ys = sorted(
                    s[1]
                    for s in self.samples
                )

                n = len(xs)

                lo = int(n * 0.2)
                hi = int(n * 0.8)

                avg_x = np.mean(
                    xs[lo:hi]
                )

                avg_y = np.mean(
                    ys[lo:hi]
                )

            else:

                avg_x = np.mean(
                    [
                        s[0]
                        for s in self.samples
                    ]
                ) if self.samples else 0.5

                avg_y = np.mean(
                    [
                        s[1]
                        for s in self.samples
                    ]
                ) if self.samples else 0.5

            # 시선 흔들림 검사
            xs_raw = [
                s[0]
                for s in self.samples
            ]

            ys_raw = [
                s[1]
                for s in self.samples
            ]

            std_x = np.std(xs_raw) if xs_raw else 0.0
            std_y = np.std(ys_raw) if ys_raw else 0.0

            if (
                std_x > CALIB_STD_X
                or std_y > CALIB_STD_Y
            ):

                self.warning = "시선이 불안정합니다"
                self.warning_start = time.time()

                self.samples = []
                self.pose_samples = []
                self.hold_start = None

                return elapsed / (
                    CALIB_STABILIZE_SEC +
                    CALIB_COLLECT_SEC
                )

            self.warning = ""

            self.iris_pts.append(
                [
                    avg_x,
                    avg_y
                ]
            )

            # 해당 캘리브레이션 점의 head pose 평균 저장
            if self.pose_samples:
                pose_avg = np.mean(
                    np.array(self.pose_samples, dtype=np.float32),
                    axis=0
                ).tolist()

                self.pose_pts.append(pose_avg)

            sx, sy = CALIB_POINTS[self.idx]

            self.screen_pts.append(
                [
                    sx * SCREEN_W,
                    sy * SCREEN_H
                ]
            )

            self.idx += 1

            self.samples = []
            self.pose_samples = []
            self.hold_start = None

            if self.idx >= len(CALIB_POINTS):

                src = np.array(
                    self.iris_pts,
                    dtype=np.float32
                )

                dst = np.array(
                    self.screen_pts,
                    dtype=np.float32
                )

                self.H, _ = cv2.findHomography(
                    src,
                    dst
                )

                # 캘리브레이션 당시 평균 head pose baseline 저장
                valid_pose_pts = [
                    p for p in self.pose_pts
                    if p is not None and len(p) >= 7 and p[3] > 0
                ]

                if valid_pose_pts:
                    self.pose_baseline = np.mean(
                        np.array(valid_pose_pts, dtype=np.float32),
                        axis=0
                    ).tolist()
                else:
                    self.pose_baseline = None

                print("[calibration] pose_baseline:", self.pose_baseline)

                self.done = True

        return elapsed / (
            CALIB_STABILIZE_SEC +
            CALIB_COLLECT_SEC
        )

    def get_pose_delta(self, head_pose):
        """
        캘리브레이션 당시 head pose와 현재 head pose의 차이를 반환합니다.

        pose_baseline 구조:
            [0] yaw
            [1] pitch
            [2] roll
            [3] face_scale
            [4] tz
            [5] face_center_x
            [6] face_center_y
        """

        if self.pose_baseline is None:
            return None

        if head_pose is None or not head_pose.get("valid", False):
            return None

        if len(self.pose_baseline) < 7:
            return None

        base_yaw = self.pose_baseline[0]
        base_pitch = self.pose_baseline[1]
        base_roll = self.pose_baseline[2]
        base_scale = self.pose_baseline[3]
        base_center_x = self.pose_baseline[5]
        base_center_y = self.pose_baseline[6]

        current_yaw = head_pose.get("yaw", base_yaw)
        current_pitch = head_pose.get("pitch", base_pitch)
        current_roll = head_pose.get("roll", base_roll)
        current_scale = head_pose.get("face_scale", base_scale)
        current_center_x = head_pose.get("face_center_x", base_center_x)
        current_center_y = head_pose.get("face_center_y", base_center_y)

        return {
            "delta_yaw": current_yaw - base_yaw,
            "delta_pitch": current_pitch - base_pitch,
            "delta_roll": current_roll - base_roll,
            "delta_scale": current_scale - base_scale,
            "delta_center_x": current_center_x - base_center_x,
            "delta_center_y": current_center_y - base_center_y,
        }

    def compensate_iris_by_head_pose(
        self,
        iris_x,
        iris_y,
        head_pose
    ):
        """
        회귀 모델 전 단계의 규칙 기반 보정입니다.

        screen 좌표를 직접 움직이지 않고,
        map_to_screen()에 들어가기 전의 iris 입력값을 약하게 보정합니다.

        목적:
            얼굴 전체가 카메라 화면 안에서 이동한 양을 iris 좌표에서 일부 제거합니다.
            이렇게 하면 얼굴 이동이 커서 이동으로 섞이는 현상을 줄일 수 있습니다.

        주의:
            이 함수는 완전한 3D gaze 보정이 아닙니다.
            face_center와 face_scale을 이용한 약한 보정입니다.
        """

        if iris_x is None or iris_y is None:
            return iris_x, iris_y

        if self.pose_baseline is None:
            return iris_x, iris_y

        if head_pose is None or not head_pose.get("valid", False):
            return iris_x, iris_y

        if len(self.pose_baseline) < 7:
            return iris_x, iris_y

        base_scale = self.pose_baseline[3]
        base_center_x = self.pose_baseline[5]
        base_center_y = self.pose_baseline[6]

        current_scale = head_pose.get("face_scale", base_scale)
        current_center_x = head_pose.get("face_center_x", base_center_x)
        current_center_y = head_pose.get("face_center_y", base_center_y)

        if base_scale <= 0 or current_scale <= 0:
            return iris_x, iris_y

        # 얼굴 중심 이동량
        delta_center_x = current_center_x - base_center_x
        delta_center_y = current_center_y - base_center_y

        # 작은 얼굴 중심 흔들림은 보정하지 않음
        # MediaPipe landmark 노이즈가 커서에 섞이는 것을 방지합니다.
        center_deadband_x = 0.005
        center_deadband_y = 0.005

        if abs(delta_center_x) < center_deadband_x:
            delta_center_x = 0.0

        if abs(delta_center_y) < center_deadband_y:
            delta_center_y = 0.0

        # 얼굴 크기 변화 비율
        scale_ratio = current_scale / base_scale

        # 너무 큰 자세/거리 변화는 보정하지 않음
        # 이 경우는 보정보다 재캘리브레이션 대상에 가깝습니다.
        if abs(delta_center_x) > 0.15:
            return iris_x, iris_y

        if abs(delta_center_y) > 0.15:
            return iris_x, iris_y

        if scale_ratio < 0.7 or scale_ratio > 1.4:
            return iris_x, iris_y

        # 보정 강도
        # 처음에는 약하게 시작해야 합니다.
        center_gain_x = 0.08
        center_gain_y = 0.08

        corrected_iris_x = iris_x - delta_center_x * center_gain_x
        corrected_iris_y = iris_y - delta_center_y * center_gain_y

        # 얼굴 거리 변화 보정
        # 얼굴이 가까워져 face_scale이 커지면 iris 변화가 과장될 수 있으므로
        # 0.5 중심 기준으로 아주 약하게 정규화합니다.
        scale_gain = 0.0

        corrected_iris_x = 0.5 + (
            corrected_iris_x - 0.5
        ) / (
            1.0 + (scale_ratio - 1.0) * scale_gain
        )

        corrected_iris_y = 0.5 + (
            corrected_iris_y - 0.5
        ) / (
            1.0 + (scale_ratio - 1.0) * scale_gain
        )

        corrected_iris_x = float(
            np.clip(
                corrected_iris_x,
                0.0,
                1.0
            )
        )

        corrected_iris_y = float(
            np.clip(
                corrected_iris_y,
                0.0,
                1.0
            )
        )

        return corrected_iris_x, corrected_iris_y

    def apply_head_pose_correction(
        self,
        screen_x,
        screen_y,
        head_pose
    ):
        """
        screen 좌표에 head pose를 직접 더하는 방식은 사용하지 않습니다.

        이전 방식:
            corrected_x = screen_x + delta_yaw * alpha

        문제:
            커서가 시선이 아니라 고개 움직임을 따라갔습니다.

        현재 방식:
            screen 좌표 보정은 하지 않습니다.
            대신 compensate_iris_by_head_pose()에서
            map_to_screen() 이전 iris 입력값을 약하게 보정합니다.
        """

        return screen_x, screen_y

    def map_to_screen(
        self,
        iris_x,
        iris_y
    ):

        if self.H is None:
            return None, None

        pt = np.array(
            [[[iris_x, iris_y]]],
            dtype=np.float32
        )

        result = cv2.perspectiveTransform(
            pt,
            self.H
        )

        screen_x = int(
            np.clip(
                result[0][0][0],
                0,
                SCREEN_W - 1
            )
        )

        screen_y = int(
            np.clip(
                result[0][0][1],
                0,
                SCREEN_H - 1
            )
        )

        return (
            screen_x,
            screen_y
        )