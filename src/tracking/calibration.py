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

        self.iris_pts = []
        self.screen_pts = []

        self.H = None

        self.done = False

        self.hold_start = None

        self.warning = ""
        self.warning_start = None

    def update(
        self,
        iris_x,
        iris_y,
        conf
    ):

        now = time.time()

        if self.hold_start is None:
            self.hold_start = now

        elapsed = now - self.hold_start

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

            self.iris_pts.append(
                [
                    avg_x,
                    avg_y
                ]
            )

            sx, sy = CALIB_POINTS[self.idx]

            self.screen_pts.append(
                [
                    sx * SCREEN_W,
                    sy * SCREEN_H
                ]
            )

            xs_raw = [s[0] for s in self.samples]
            ys_raw = [s[1] for s in self.samples]

            std_x = np.std(xs_raw)
            std_y = np.std(ys_raw)

            if (
                std_x > CALIB_STD_X
                or std_y > CALIB_STD_Y
            ):

                self.warning = "시선이 불안정합니다"
                self.warning_start = time.time()

                self.samples = []
                self.hold_start = None

                return elapsed / (
                    CALIB_STABILIZE_SEC +
                    CALIB_COLLECT_SEC
                )

            self.warning = ""

            self.idx += 1

            self.samples = []

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

                self.done = True

        return elapsed / (
            CALIB_STABILIZE_SEC +
            CALIB_COLLECT_SEC
        )

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