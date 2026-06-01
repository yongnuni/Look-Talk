import numpy as np

from src.config import (
    SMOOTH_ALPHA,
    FIXATION_RADIUS,
    FIXATION_FRAMES
)


class GazePipeline:

    def __init__(self):
        self.smooth_gaze = None
        self.fixation_center = None
        self.fixation_count = 0

    def reset(self):
        self.smooth_gaze = None
        self.fixation_center = None
        self.fixation_count = 0

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

        # EMA 스무딩
        if self.smooth_gaze is None:
            self.smooth_gaze = [float(sx), float(sy)]

        alpha = SMOOTH_ALPHA * max(0.3, conf)

        self.smooth_gaze[0] += alpha * (sx - self.smooth_gaze[0])
        self.smooth_gaze[1] += alpha * (sy - self.smooth_gaze[1])

        sx_s = self.smooth_gaze[0]
        sy_s = self.smooth_gaze[1]

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