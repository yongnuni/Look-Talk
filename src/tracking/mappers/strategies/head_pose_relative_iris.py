"""
HeadPoseRelativeIrisStrategy — 첫 번째 baseline NoCalibration strategy.

배경:
    get_avg_iris()가 반환하는 iris_x/iris_y는 카메라 프레임 전체 기준
    정규화 좌표다(src/tracking/eye_tracking.py). 안구 회전보다 얼굴(머리) 위치
    변화가 훨씬 크게 반영되므로, 이 값을 그대로 화면에 고정 배율로 매핑하면
    사실상 "머리 위치 추적"에 가까워진다.

    이 strategy는 그 문제를 완전히 해결하지는 못하지만, 다음 두 가지로
    최소한의 개인차 흡수를 시도한다.

    1. iris 좌표에서 head_pose['face_center_x/y']를 빼서 "얼굴 중심 대비
       상대 iris 위치"로 바꾼다(머리 이동 성분을 상당 부분 제거).
    2. head_pose['face_scale']로 그 상대값을 정규화해 카메라와의 거리 변화
       (앞뒤로 기대는 동작)에 대한 민감도를 낮춘다.

    16점 화면 응시를 요구하지 않기 위해, "정지 상태에서의 상대 iris 위치"를
    시작 직후 몇 프레임(anchor_warmup_frames) 동안 자동으로 평균 내
    중립점(anchor)으로 삼는다. 이건 화면의 특정 지점을 보라고 안내하는
    캘리브레이션 화면이 아니라, 완전히 자동/무음으로 이루어지는 초기화다.

주의:
    gain_x_ratio/gain_y_ratio는 실측 데이터 없이 정한 실험용 baseline
    추정치다(주석 참고). 실제 카메라로 테스트해 조정이 필요하다.
"""

from dataclasses import dataclass, asdict

import numpy as np

from src.config import SCREEN_W, SCREEN_H
from src.tracking.mappers.strategies import register_strategy
from src.tracking.mappers.strategies.base import NoCalibrationStrategy


@dataclass
class HeadPoseRelativeIrisParams:
    # ── 실험용 baseline 추정치, 검증되지 않음 ──────────────────
    # 안구가 눈구멍 안에서 좌우로 움직일 수 있는 범위는(머리 고정 상태) 대략
    # 프레임 정규화 좌표로 ±0.01~0.03 수준일 것으로 추정했다(직접 측정한
    # 값이 아니라, MediaPipe 눈 랜드마크 크기가 전형적인 웹캠 거리에서
    # 프레임 폭의 수 %인 점에서 역산한 대략적인 가정치). 이 좁은 범위를
    # 화면 폭의 상당 부분(초기값: 좌우 각각 약 40%)까지 늘리기 위해
    # gain을 크게 잡았다. 실제 카메라로 좌우 끝까지 응시해보고
    # 화면 도달 범위를 관찰한 뒤 재조정이 필요하다.
    gain_x_ratio: float = 18.0
    gain_y_ratio: float = 22.0

    invert_x: bool = False
    invert_y: bool = False

    center_x_ratio: float = 0.5
    center_y_ratio: float = 0.5

    # 시작/reset 직후 몇 프레임을 "중립 위치"로 자동 평균 낼지.
    # main.py 프레임레이트 기준 약 0.5~1초 정도를 목표로 잡은 값.
    anchor_warmup_frames: int = 15

    # face_scale 비율 정규화 시 발산 방지용 clamp
    min_scale_ratio: float = 0.3
    max_scale_ratio: float = 3.0


@register_strategy("head_pose_relative_iris")
class HeadPoseRelativeIrisStrategy(NoCalibrationStrategy):

    def __init__(self, params: HeadPoseRelativeIrisParams | None = None):
        self.params = params or HeadPoseRelativeIrisParams()
        self._anchor_samples_x = []
        self._anchor_samples_y = []
        self._anchor_samples_scale = []
        self._anchor_rel_x = None
        self._anchor_rel_y = None
        self._anchor_scale = None
        self._anchored = False

    def reset(self):
        self._anchor_samples_x = []
        self._anchor_samples_y = []
        self._anchor_samples_scale = []
        self._anchor_rel_x = None
        self._anchor_rel_y = None
        self._anchor_scale = None
        self._anchored = False

    def map(self, iris_x, iris_y, head_pose=None, features=None):

        if head_pose is None or not head_pose.get("valid", False):
            return None, None

        if iris_x is None or iris_y is None:
            return None, None

        if not (np.isfinite(iris_x) and np.isfinite(iris_y)):
            return None, None

        face_scale = head_pose.get("face_scale", 0.0)
        face_center_x = head_pose.get("face_center_x")
        face_center_y = head_pose.get("face_center_y")

        if face_scale is None or face_scale <= 0:
            return None, None

        if face_center_x is None or face_center_y is None:
            return None, None

        rel_x = iris_x - face_center_x
        rel_y = iris_y - face_center_y

        if not self._anchored:

            self._anchor_samples_x.append(rel_x)
            self._anchor_samples_y.append(rel_y)
            self._anchor_samples_scale.append(face_scale)

            if len(self._anchor_samples_x) >= self.params.anchor_warmup_frames:
                self._anchor_rel_x = float(np.mean(self._anchor_samples_x))
                self._anchor_rel_y = float(np.mean(self._anchor_samples_y))
                self._anchor_scale = float(np.mean(self._anchor_samples_scale))
                self._anchored = True

            return None, None

        scale_ratio = (
            face_scale / self._anchor_scale
            if self._anchor_scale > 0
            else 1.0
        )
        scale_ratio = float(
            np.clip(scale_ratio, self.params.min_scale_ratio, self.params.max_scale_ratio)
        )

        dx = (rel_x - self._anchor_rel_x) / scale_ratio
        dy = (rel_y - self._anchor_rel_y) / scale_ratio

        if self.params.invert_x:
            dx = -dx

        if self.params.invert_y:
            dy = -dy

        screen_x = (
            SCREEN_W * self.params.center_x_ratio
            + dx * self.params.gain_x_ratio * SCREEN_W
        )
        screen_y = (
            SCREEN_H * self.params.center_y_ratio
            + dy * self.params.gain_y_ratio * SCREEN_H
        )

        screen_x = int(np.clip(screen_x, 0, SCREEN_W - 1))
        screen_y = int(np.clip(screen_y, 0, SCREEN_H - 1))

        return screen_x, screen_y

    def get_params(self):
        d = asdict(self.params)
        d["anchored"] = self._anchored
        if self._anchored:
            d["anchor_rel_x"] = round(self._anchor_rel_x, 5)
            d["anchor_rel_y"] = round(self._anchor_rel_y, 5)
            d["anchor_scale"] = round(self._anchor_scale, 3)
        return d
