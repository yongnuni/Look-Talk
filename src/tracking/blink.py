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


# 눈을 감기 전에 한 타깃을 얼마나 안정적으로 바라봐야 선택 후보로 잠글지.
# MouthClickDetector.lock_time(0.25초)과 같은 값 — 두 제스처 모드의 잠금
# 타이밍을 일치시킨다.
BLINK_LOCK_TIME_SEC = 0.25


class BlinkSelectionController:
    """기존 ``BlinkDetector`` 이벤트를 현재 키보드 타깃 선택에 연결한다.

    looktalk-frontend의 ``src/features/multimodalInput/BlinkController.ts``와
    같은 잠금 모델을 쓴다. EAR 히스테리시스와 깜빡임 분류는 ``BlinkDetector``가,
    타깃 hit-test는 fixation 파이프라인이 담당하고 이 클래스는 둘을 결합한다.

    동작 순서
    1. 눈을 뜬 상태에서 같은 타깃을 ``lock_time`` 이상 응시하면 ``locked_target``
       으로 잠근다. 잠깐 시선이 흔들려 hover가 끊겨도 이미 성립한 잠금은 풀지
       않고 후보 누적만 초기화한다.
    2. 눈을 감는 순간 그 잠금을 ``start_target``으로 확정한다.
    3. 의도 깜빡임으로 눈을 뜨면 ``start_target``을 그대로 선택한다 —
       **눈을 뜬 프레임의 hover로 재검증하지 않는다.**

    (3)이 중요하다. 눈을 다시 뜬 첫 프레임은 GazePipeline이 눈 감김 동안 버퍼를
    비웠다가 막 재개한 좌표이고 홍채 신뢰도도 아직 낮아 ``hovered_target``이
    None이 되기 쉽다. 그 프레임의 hover로 선택을 재검증하면 사용자가 의도적으로
    깜빡였는데도 입력이 조용히 사라진다. MouthClickDetector가 ``start_key``를
    제스처 시작 시점에 확정하고 끝에서 재검증하지 않는 것과 같은 이유다.
    """

    def __init__(self, lock_time=BLINK_LOCK_TIME_SEC):
        # 눈을 감기 전 잠금에 필요한 최소 응시 시간
        self.lock_time = lock_time

        self.reset()

    @property
    def locked_target(self):
        return self._locked_target

    def reset(self):
        """진행 중인 깜빡임 선택 상태를 초기화한다.

        모드 전환이나 추적 실패 뒤에는 눈이 열린 프레임을 먼저 확인해야
        (``_armed``) 다시 입력을 받는다. 이미 감겨 있던 눈을 새 깜빡임의
        시작으로 오인하지 않기 위한 안전장치다.
        """

        self._candidate_target = None
        self._candidate_started_at = None
        self._locked_target = None

        # 눈을 감는 순간 확정되는 제스처 대상.
        # 눈을 뜰 때 이 값을 그대로 선택한다.
        self._start_target = None

        self._was_closed = False
        self._armed = False

    def _update_gaze_lock(self, hovered_target, now):
        """같은 타깃을 ``lock_time`` 이상 응시하면 잠근다.

        MouthClickDetector._update_gaze_lock()과 동일한 규칙이다.
        """

        if hovered_target is None:
            # 후보만 무효화하고 이미 성립한 잠금은 유지한다
            # (시선이 잠깐 키 사이를 지나가는 경우).
            self._candidate_target = None
            self._candidate_started_at = None
            return

        # 이미 잠근 타깃을 계속 보고 있으면 후보를 새로 누적할 필요가 없다.
        if hovered_target == self._locked_target:
            self._candidate_target = hovered_target
            self._candidate_started_at = now
            return

        # 새로운 타깃을 보기 시작한 경우
        if hovered_target != self._candidate_target:
            self._candidate_target = hovered_target
            self._candidate_started_at = now
            return

        if (
            self._candidate_started_at is not None
            and now - self._candidate_started_at >= self.lock_time
        ):
            self._locked_target = hovered_target

    def _clear_lock(self):
        self._candidate_target = None
        self._candidate_started_at = None
        self._locked_target = None
        self._start_target = None

    def update(self, hovered_target, blink_event, is_closed, now=None):
        """한 프레임의 hover/깜빡임 상태를 받아 선택된 타깃을 돌려준다.

        시선 좌표가 무효한 프레임(눈 감김 등)에서도 ``hovered_target=None``으로
        매 프레임 호출해야 한다. 좌표가 무효하다는 이유로 ``reset()``하면
        잠금과 ``_armed``가 통째로 날아가 모든 깜빡임이 무효가 된다.
        """

        if now is None:
            now = time.monotonic()

        if is_closed:
            if not self._was_closed:
                # 여기서 대상을 확정한다 —
                # 눈을 뜬 뒤에는 다시 검증하지 않는다.
                # 아직 눈 뜬 프레임을 본 적이 없으면(_armed=False) 이미 감겨
                # 있던 눈이므로 이번 감김은 제스처로 세지 않는다.
                self._start_target = (
                    self._locked_target if self._armed else None
                )

            self._was_closed = True

            if (
                blink_event is not None
                and blink_event.kind == BlinkKind.LONG_CLOSURE
            ):
                # 너무 오래 감은 것은 입력 의도가 아니므로 제스처를 폐기한다.
                self._clear_lock()

            return None

        was_closed = self._was_closed

        self._was_closed = False
        self._armed = True

        if not was_closed:
            # 계속 눈을 뜨고 있는 프레임 — 잠금만 갱신한다.
            self._update_gaze_lock(hovered_target, now)
            return None

        # 눈을 막 뜬 프레임. 잠근 대상을 hover로 재검증하지 않는다.
        selected_target = None

        if (
            blink_event is not None
            and blink_event.kind == BlinkKind.INTENTIONAL
            and self._start_target is not None
        ):
            selected_target = self._start_target

        # 입벌림 모드가 입을 닫는 순간 잠금을 전부 초기화하는 것과 대칭이다.
        # 다음 입력에는 다시 lock_time만큼 응시해야 하며, 시선을 옮긴 뒤의
        # 자연 깜빡임이 남아 있던 잠금을 눌러버리는 사고를 구조적으로 막는다.
        self._clear_lock()

        return selected_target
