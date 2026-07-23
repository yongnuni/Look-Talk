import cv2
import numpy as np
from collections import deque


from src.config import (
    GAZE_AVG_WINDOW,
    SMOOTH_ALPHA,
    FIXATION_RADIUS,
    FIXATION_FRAMES
)



class GazePipeline:

    def __init__(self):
        self.fixation_center = None
        self.fixation_count = 0
        self.last_output = None

        # 최근 화면 좌표를 저장하는 이동평균 버퍼
        self.gaze_buffer = deque(
            maxlen=GAZE_AVG_WINDOW
        )

        self.kalman = cv2.KalmanFilter(4, 2)

        self.kalman.measurementMatrix = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0]
            ],
            np.float32
        )

        self.kalman.transitionMatrix = np.array(
            [
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ],
            np.float32
        )

        self.kalman.processNoiseCov = (
            np.eye(4, dtype=np.float32) * 0.003
        )

        self.kalman.measurementNoiseCov = (
            np.eye(2, dtype=np.float32) * 0.3
        )

        self.initialized = False

    def reset(self):
        self.fixation_center = None
        self.fixation_count = 0
        self.last_output = None
        self.gaze_buffer.clear()

        self.kalman.statePost = np.zeros(
            (4, 1),
            np.float32
        )

        self.initialized = False

    def _reset_tracking_state(self):
        """
        현재 gaze가 유효하지 않을 때 fixation과 smoothing 상태를 초기화합니다.
        """
        self.fixation_center = None
        self.fixation_count = 0
        self.last_output = None
        self.gaze_buffer.clear()
        self.initialized = False

    def _is_head_pose_valid(self, head_pose):
        """
        head_pose 기반으로 현재 얼굴 자세가 시선 입력에 적절한지 판단합니다.

        아직은 좌표 보정이 아니라,
        너무 비정면이면 gaze 입력을 무효 처리하는 용도입니다.
        """

        if head_pose is None:
            return True

        if not head_pose.get("valid", False):
            return False

        yaw = abs(head_pose.get("yaw", 0.0))
        pitch = abs(head_pose.get("pitch", 0.0))
        roll = abs(head_pose.get("roll", 0.0))

        # 임시 기준값입니다.
        # 실제 테스트하면서 20~35도 사이로 조정하면 됩니다.
        if yaw > 25:
            return False

        if pitch > 25:
            return False

        if roll > 25:
            return False

        return True

    def update(self, sx, sy, conf, blink, head_pose=None):
        """
        캘리브레이션된 화면 좌표(sx, sy)를 받아
        head pose 유효성 검사, Kalman smoothing, fixation 감지 후
        최종 시선 좌표를 반환합니다.

        Args:
            sx: 캘리브레이션된 화면 x 좌표
            sy: 캘리브레이션된 화면 y 좌표
            conf: 홍채 추적 신뢰도
            blink: 눈 깜빡임 여부
            head_pose: estimate_head_pose()의 반환값

        Returns:
            (gaze_x, gaze_y, fixation_count)
            유효하지 않은 경우 gaze_x, gaze_y = -1
        """

        if sx is None or sy is None or blink or conf <= 0.3:
            self._reset_tracking_state()
            return -1, -1, 0

        #if not self._is_head_pose_valid(head_pose):
        #    self._reset_tracking_state()
        #    return -1, -1, 0
        # 최근 화면 좌표를 버퍼에 저장
        self.gaze_buffer.append(
            (
                float(sx),
                float(sy)
            )
        )

        # 현재까지 저장된 좌표들의 평균을 Kalman 입력으로 사용
        buffer_array = np.asarray(
            self.gaze_buffer,
            dtype=np.float32
        )

        sx = float(
            np.mean(buffer_array[:, 0])
        )

        sy = float(
            np.mean(buffer_array[:, 1])
        )
        # Kalman Filter
        if not self.initialized:

            self.kalman.statePost = np.array(
                [
                    [np.float32(sx)],
                    [np.float32(sy)],
                    [0],
                    [0]
                ],
                dtype=np.float32
            )

            self.initialized = True

        self.kalman.predict()

        measurement = np.array(
            [
                [np.float32(sx)],
                [np.float32(sy)]
            ],
            dtype=np.float32
        )

        estimated = self.kalman.correct(measurement)

        sx_s = float(estimated[0][0])
        sy_s = float(estimated[1][0])

        alpha = 0.35

        if self.last_output is None:
        # 첫 유효 좌표는 비교할 이전 좌표가 없으므로 그대로 사용
            sx_s = float(sx_s)
            sy_s = float(sy_s)
            self.last_output = [sx_s, sy_s]

        else:
        # Dead Zone 계산 전에 이전 최종 출력 좌표 보존
            prev_x, prev_y = self.last_output

        # Kalman 결과에 EMA 적용
            ema_x = (alpha * sx_s + (1 - alpha) * prev_x)
            ema_y = (alpha * sy_s + (1 - alpha) * prev_y)

        # 이전 최종 좌표와 새로운 EMA 좌표 사이의 실제 거리
            movement = np.hypot(ema_x - prev_x,ema_y - prev_y)

            # 이동량에 따라 Dead Zone 크기 결정
            if movement < 10:
                dead_zone = 5
            elif movement < 40:
                dead_zone = 10
            else:
                dead_zone = 20

            if movement < dead_zone:
            # 작은 흔들림은 무시
                sx_s = prev_x
                sy_s = prev_y
            else:
            # 의미 있는 이동만 반영
                sx_s = ema_x
                sy_s = ema_y
                self.last_output = [sx_s, sy_s]

        # Fixation 감지
        if self.fixation_center is None:
            self.fixation_center = [sx_s, sy_s]
            self.fixation_count = 1

        else:
            dist = np.hypot(
                sx_s - self.fixation_center[0],
                sy_s - self.fixation_center[1]
            )

            if dist < FIXATION_RADIUS:
                self.fixation_count += 1
                self.fixation_center[0] += 0.05 * (
                    sx_s - self.fixation_center[0]
                )
                self.fixation_center[1] += 0.05 * (
                    sy_s - self.fixation_center[1]
                )

            else:
                self.fixation_center = [sx_s, sy_s]
                self.fixation_count = 1

        if self.fixation_count >= FIXATION_FRAMES:
            gaze_x = int(self.fixation_center[0])
            gaze_y = int(self.fixation_center[1])
        else:
            gaze_x = int(sx_s)
            gaze_y = int(sy_s)

        return gaze_x, gaze_y, self.fixation_count