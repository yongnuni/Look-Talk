import cv2
import numpy as np
import time

from src.config import (
    CALIB_POINTS,
    CALIB_HOLD_SEC,
    SCREEN_W,
    SCREEN_H
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

        if conf > 0.4:

            self.samples.append(
                (
                    iris_x,
                    iris_y
                )
            )

        if elapsed >= CALIB_HOLD_SEC:

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

        return elapsed / CALIB_HOLD_SEC

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