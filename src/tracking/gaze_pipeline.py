import cv2
import numpy as np

from src.config import (
    SMOOTH_ALPHA,
    FIXATION_RADIUS,
    FIXATION_FRAMES
)


class GazePipeline:

    def __init__(self):
        self.fixation_center = None
        self.fixation_count = 0
        self.last_output = None

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
            np.eye(4, dtype=np.float32) * 0.03
        )

        self.kalman.measurementNoiseCov = (
            np.eye(2, dtype=np.float32) * 0.5
        )

        self.initialized = False
        

    def reset(self):
        self.fixation_center = None
        self.fixation_count = 0
        self.last_output = None

        self.kalman.statePost = np.zeros(
        (4, 1),
        np.float32
        )

        self.initialized = False
        

    def update(self, sx, sy, conf, blink):
        """
        캘리브레이션된 화면 좌표(sx, sy)를 받아
        스무딩 및 Fixation 감지 후 최종 시선 좌표 반환.

        Returns:
            (gaze_x, gaze_y, fixation_count)
            유효하지 않은 경우 gaze_x, gaze_y = -1
        """

        if sx is None or blink or conf <= 0.3:
            self.fixation_center = None
            self.fixation_count = 0
            return -1, -1, 0


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
        if self.last_output is None:
            self.last_output = [sx_s, sy_s]

        dead_zone = 15

        dist = np.hypot(
            sx_s - self.last_output[0],
            sy_s - self.last_output[1]
        )

        if dist < dead_zone:
            sx_s = self.last_output[0]
            sy_s = self.last_output[1]
        else:
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
                self.fixation_center[0] += 0.05 * (sx_s - self.fixation_center[0])
                self.fixation_center[1] += 0.05 * (sy_s - self.fixation_center[1])

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