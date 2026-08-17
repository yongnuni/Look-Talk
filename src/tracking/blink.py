"""
blink.py — 눈 깜빡임 검출 전용 모듈

자연 깜빡임(NATURAL)과 의도 깜빡임(INTENTIONAL)을
'감은 상태의 지속 시간'으로 구분하는 상태 머신 기반 검출기.
"""

import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


# 랜드마크 인덱스 (이 모듈이 자체 보유 — 기존 파일에 의존하지 않음)

LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133
LEFT_LID_UPPER = 159
LEFT_LID_LOWER = 145

RIGHT_EYE_OUTER = 263
RIGHT_EYE_INNER = 362
RIGHT_LID_UPPER = 386
RIGHT_LID_LOWER = 374


# EAR 계산 (기존 파일에서 이관)

def eye_aspect_ratio(landmarks, outer_idx, inner_idx, top_idx, bottom_idx):
    lm = landmarks.landmark
    eye_width = abs(lm[outer_idx].x - lm[inner_idx].x)
    eye_height = abs(lm[top_idx].y - lm[bottom_idx].y)
    if eye_width <= 1e-3:
        return 0.0
    return eye_height / eye_width


def average_ear(landmarks):
    """양안 평균 EAR. iris_confidence 등 외부에서도 재사용."""
    left = eye_aspect_ratio(
        landmarks,
        LEFT_EYE_OUTER, LEFT_EYE_INNER,
        LEFT_LID_UPPER, LEFT_LID_LOWER
    )
    right = eye_aspect_ratio(
        landmarks,
        RIGHT_EYE_OUTER, RIGHT_EYE_INNER,
        RIGHT_LID_UPPER, RIGHT_LID_LOWER
    )
    return (left + right) / 2.0


# 이벤트 정의

class BlinkKind(Enum):
    NATURAL = auto()        # 자연 깜빡임 (짧음)
    INTENTIONAL = auto()    # 의도 깜빡임 (길게 감았다 뜸)
    LONG_CLOSURE = auto()   # 너무 오래 감음 (졸음/일시정지 등)


@dataclass
class BlinkEvent:
    kind: BlinkKind
    duration: float     # 감고 있던 시간(초)
    timestamp: float    # 이벤트 발생 시각 (time.monotonic 기준)


class _State(Enum):
    OPEN = auto()
    CLOSED = auto()


class BlinkDetector:
    """
    매 프레임 update(landmarks)를 호출하면,
    깜빡임이 '완결'된 프레임에서만 BlinkEvent를 반환하고
    그 외에는 None을 반환한다.

    detect_natural / detect_intentional 플래그로
    어떤 종류를 이벤트로 받을지 선택한다.
    (예: 시선 추적 게이팅만 필요하면 둘 다 False로 두고
     is_closed 프로퍼티만 사용해도 됨)
    """

    def __init__(
        self,
        close_threshold: float = 0.18,   # 이 미만이면 '감음'
        open_threshold: float = 0.22,    # 이 초과면 '뜸' (히스테리시스)
        natural_max_sec: float = 0.25,   # 이 이하 지속 → 자연
        intent_min_sec: float = 0.30,    # 이 이상 지속 → 의도
        intent_max_sec: float = 0.80,    # 이 초과 → LONG_CLOSURE
        refractory_sec: float = 0.10,    # 이벤트 직후 재검출 금지 시간
        detect_natural: bool = True,
        detect_intentional: bool = True,
    ):
        self.close_threshold = close_threshold
        self.open_threshold = open_threshold
        self.natural_max_sec = natural_max_sec
        self.intent_min_sec = intent_min_sec
        self.intent_max_sec = intent_max_sec
        self.refractory_sec = refractory_sec
        self.detect_natural = detect_natural
        self.detect_intentional = detect_intentional

        self._state = _State.OPEN
        self._closed_at = 0.0
        self._last_event_time = -1e9
        self._long_fired = False

    # 외부에서 "지금 감겨 있나"를 물을 때 (시선 업데이트 게이팅용)
    @property
    def is_closed(self) -> bool:
        return self._state == _State.CLOSED

    def update(self, landmarks, now: float = None) -> Optional[BlinkEvent]:
        if now is None:
            now = time.monotonic()

        ear = average_ear(landmarks)

        if self._state == _State.OPEN:
            # 불응기 안에서는 새 닫힘을 시작하지 않음 (떨림 방지)
            if (ear < self.close_threshold
                    and now - self._last_event_time >= self.refractory_sec):
                self._state = _State.CLOSED
                self._closed_at = now
                self._long_fired = False
            return None

        # CLOSED 상태
        duration = now - self._closed_at

        if ear > self.open_threshold:
            # 눈을 떴다 → 깜빡임 완결, 분류 시도
            self._state = _State.OPEN
            event = self._classify(duration, now)
            if event is not None:
                self._last_event_time = now
            return event

        # 아직 감은 채 — 한계 초과 시 LONG_CLOSURE를 1회만 발화
        if duration > self.intent_max_sec and not self._long_fired:
            self._long_fired = True
            self._last_event_time = now
            return BlinkEvent(BlinkKind.LONG_CLOSURE, duration, now)

        return None

    def _classify(self, duration: float, now: float) -> Optional[BlinkEvent]:
        # 이미 LONG_CLOSURE로 발화했으면, 뜨는 순간 중복 이벤트 금지
        if self._long_fired:
            return None

        if duration <= self.natural_max_sec:
            if self.detect_natural:
                return BlinkEvent(BlinkKind.NATURAL, duration, now)
            return None

        if self.intent_min_sec <= duration <= self.intent_max_sec:
            if self.detect_intentional:
                return BlinkEvent(BlinkKind.INTENTIONAL, duration, now)
            return None

        # natural_max ~ intent_min 사이 데드존 → 의도 모호, 버림
        return None