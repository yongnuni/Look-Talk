import cv2
import numpy as np


def auto_brightness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    mean = np.mean(gray)

    target = 120

    alpha = target / max(mean, 1)

    alpha = np.clip(alpha, 0.8, 1.5)

    frame = cv2.convertScaleAbs(
        frame,
        alpha=alpha,
        beta=0
    )

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))

    return cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2BGR
    )
