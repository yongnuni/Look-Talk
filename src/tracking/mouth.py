import math
import time
import cv2
UPPER_LIP = 13
LOWER_LIP = 14

LEFT_MOUTH = 78
RIGHT_MOUTH = 308
MOUTH_OUTLINE = [
    61,146,91,181,84,17,314,405,
    321,375,291,308,324,318,402,
    317,14,87,178,88,95
]



def distance(p1, p2):

    return math.hypot(
        p1.x - p2.x,
        p1.y - p2.y
    )


def mouth_aspect_ratio(landmarks):

    lm = landmarks.landmark

    mouth_height = distance(
        lm[UPPER_LIP],
        lm[LOWER_LIP]
    )

    mouth_width = distance(
        lm[LEFT_MOUTH],
        lm[RIGHT_MOUTH]
    )

    if mouth_width < 1e-6:
        return 0.0

    return mouth_height / mouth_width


def is_mouth_open(
    landmarks,
    threshold=0.30
):

    mar = mouth_aspect_ratio(
        landmarks
    )

    return (
        mar > threshold,
        mar
    )


class MouthClickDetector:

    def __init__(
        self,
        threshold=0.30,
        hold_time=0.3 
    ):

        self.threshold = threshold
        self.hold_time = hold_time

        self.open_start = None
        self.clicked = False
        self.start_key = None

    def update(
        self,
        landmarks,
        hovered_key=None
    ):

        mouth_open, mar = is_mouth_open(
            landmarks,
            self.threshold
        )

        now = time.time()

        click = False

        if mouth_open:

            if self.open_start is None:
                self.open_start = now
                self.start_key = hovered_key

            elif (
                now - self.open_start
                >= self.hold_time
                and not self.clicked
            ):

                if hovered_key == self.start_key:

                    click = True

                self.clicked = True

        else:

            self.open_start = None
            self.clicked = False
            self.start_key = None

        return click, mar
    

def draw_mouth(frame, landmarks, fw, fh):

    ids = [13, 14, 78, 308]

    for idx in ids:

        lm = landmarks.landmark[idx]

        x = int(lm.x * fw)
        y = int(lm.y * fh)

        cv2.circle(
            frame,
            (x, y),
            6,
            (0,255,0),
            -1
        )

    upper = landmarks.landmark[13]
    lower = landmarks.landmark[14]

    cv2.line(
        frame,
        (int(upper.x*fw), int(upper.y*fh)),
        (int(lower.x*fw), int(lower.y*fh)),
        (0,255,255),
        2
    )