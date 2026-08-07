"""
src/tracking/fixation.py

고정(fixation) 감지 전용 모듈 — I-VT(속도 기반) + I-DT(분산 기반) 하이브리드.

설계 원칙
---------
1. 이 모듈은 GazePipeline / calibrator / mapper / DwellController 중
   어떤 것도 수정하거나 참조하지 않는다.
   입력은 "이미 확정이 끝난 최종 화면 좌표 (gaze_x, gaze_y)" 하나뿐이다.

2. 출력도 좌표를 바꾸지 않는다. "지금 고정 중인가"라는 상태 신호만 낸다.
   따라서 이 모듈을 통째로 삭제해도 아이트래킹 성능과 기존 입력 로직은
   1도 변하지 않는다.

3. 판정은 독자적으로 한다.
   GazePipeline에도 fixation_count가 있지만 그건 스무딩/커서 표시용
   프레임 카운터라 시간 단위 판정에 쓸 수 없고, 손대면 좌표 자체가
   변하므로 참조하지 않는다.

판정 로직
---------
IDLE ──(속도 < v_enter)──> CANDIDATE ──(분산 유지 min_fixation_sec)──> FIXATED
FIXATED ──(속도 > v_exit  또는  중심에서 exit_radius 이탈)──> IDLE

- I-VT : 프레임 간 이동 속도(px/s). 도약안구운동(saccade) 구간을 걸러낸다.
- I-DT : 최근 좌표들이 dispersion_radius 안에 모여 있는지. 실제 고정 판정.
- 진입/이탈 임계값을 다르게 둔 히스테리시스 구조라 경계에서 깜빡이지 않는다.
"""

import math
import time
from collections import deque

from src.config import PX_PER_CM


# ── 시야각(deg) → 픽셀 환산 ──────────────────────────────────
# 가이드라인의 "반경 1°", "5~50°/s" 기준을 그대로 쓰기 위한 환산.
# PX_PER_CM은 config.MONITOR_DIAGONAL_INCH에서 계산된 값이다.

VIEWING_DISTANCE_CM = 60.0   # 사용자-모니터 거리 추정치


def px_per_deg(distance_cm=VIEWING_DISTANCE_CM):
    return PX_PER_CM * distance_cm * math.tan(math.radians(1.0))


# ── 튜닝 파라미터 (조정 지점은 문서 하단 참고) ────────────────

VELOCITY_ENTER_DEG_S = 12.0    # 이 속도 아래면 "멈추는 중" (후보 시작)
VELOCITY_EXIT_DEG_S = 20.0     # 이 속도 위면 saccade로 보고 고정 해제
DISPERSION_RADIUS_DEG = 1.2    # I-DT 반경. 이 안에 모여 있어야 고정
EXIT_RADIUS_DEG = 2.2          # 고정 후 중심에서 이만큼 벗어나면 해제
MIN_FIXATION_SEC = 0.15        # 이 시간만큼 분산 조건을 유지해야 고정 확정
LOSS_GRACE_SEC = 0.20          # 추적 실패(깜빡임 등) 허용 시간
MAX_FRAME_GAP_SEC = 0.30       # 프레임 간격이 이보다 크면 불연속으로 보고 재시작
VELOCITY_MEDIAN_N = 3          # 속도 중앙값 필터 창
CENTER_ADAPT = 0.05            # 고정 중 중심 미세 추종 계수
WINDOW_MAX_SEC = 1.0           # I-DT 창에 보관할 최대 시간

IDLE = "idle"
CANDIDATE = "candidate"
FIXATED = "fixated"


class FixationState:
    """한 프레임의 고정 판정 결과 (읽기 전용 스냅샷)."""

    __slots__ = (
        "fixated",
        "phase",
        "center",
        "duration",
        "velocity",
        "dispersion",
    )

    def __init__(
        self,
        fixated=False,
        phase=IDLE,
        center=None,
        duration=0.0,
        velocity=0.0,
        dispersion=0.0,
    ):
        self.fixated = fixated
        self.phase = phase
        self.center = center
        self.duration = duration
        self.velocity = velocity
        self.dispersion = dispersion

    def __repr__(self):
        cx = f"({self.center[0]:.0f},{self.center[1]:.0f})" if self.center else "None"
        return (
            f"<Fixation {self.phase} center={cx} "
            f"dur={self.duration:.2f}s v={self.velocity:.0f}px/s "
            f"disp={self.dispersion:.0f}px>"
        )


class FixationDetector:

    def __init__(
        self,
        velocity_enter_deg_s=VELOCITY_ENTER_DEG_S,
        velocity_exit_deg_s=VELOCITY_EXIT_DEG_S,
        dispersion_radius_deg=DISPERSION_RADIUS_DEG,
        exit_radius_deg=EXIT_RADIUS_DEG,
        min_fixation_sec=MIN_FIXATION_SEC,
        loss_grace_sec=LOSS_GRACE_SEC,
        viewing_distance_cm=VIEWING_DISTANCE_CM,
    ):
        ppd = px_per_deg(viewing_distance_cm)

        self.px_per_deg = ppd

        self.v_enter = velocity_enter_deg_s * ppd
        self.v_exit = velocity_exit_deg_s * ppd
        self.dispersion_radius = dispersion_radius_deg * ppd
        self.exit_radius = exit_radius_deg * ppd
        self.min_fixation_sec = min_fixation_sec
        self.loss_grace_sec = loss_grace_sec

        self.reset()

    # ── 상태 관리 ────────────────────────────────────────────

    def reset(self):
        """전체 초기화. 재캘리브레이션 등에서 호출."""
        self._soft_reset()
        self._clear_motion()
        self._last_valid_t = 0.0

    def _soft_reset(self):
        """고정 판정만 초기화 (속도 연속성은 유지)."""
        self.phase = IDLE
        self.center = None
        self.window = []          # [(t, x, y)]
        self.candidate_start = None
        self.fixation_start = None
        self._dispersion = 0.0

    def _clear_motion(self):
        """속도 추정 상태까지 초기화 (프레임 불연속 시)."""
        self._last_pt = None      # (t, x, y)
        self._vel_buf = deque(maxlen=VELOCITY_MEDIAN_N)
        self._velocity = 0.0

    # ── 메인 갱신 ────────────────────────────────────────────

    def update(self, gaze_x, gaze_y, now=None):
        """
        Args:
            gaze_x, gaze_y: 최종 화면 좌표. 추적 실패면 -1 또는 None.
            now: 테스트용 시각 주입. 생략하면 time.time().

        Returns:
            FixationState
        """

        if now is None:
            now = time.time()

        # ── 추적 실패 프레임 ──
        if (
            gaze_x is None
            or gaze_y is None
            or gaze_x < 0
            or gaze_y < 0
        ):
            if now - self._last_valid_t > self.loss_grace_sec:
                self._soft_reset()
                self._clear_motion()

            return self._state(now)

        # ── 프레임 불연속(캘리브레이션 화면 복귀, 정확도 테스트 종료 등) ──
        if (
            self._last_pt is not None
            and now - self._last_pt[0] > MAX_FRAME_GAP_SEC
        ):
            self._soft_reset()
            self._clear_motion()

        self._last_valid_t = now

        # ── I-VT: 순간 속도 (중앙값 필터) ──
        if self._last_pt is None:
            velocity = 0.0

        else:
            pt, px, py = self._last_pt
            dt = now - pt

            if dt <= 1e-4:
                velocity = self._velocity
            else:
                velocity = math.hypot(gaze_x - px, gaze_y - py) / dt

        self._vel_buf.append(velocity)

        sorted_v = sorted(self._vel_buf)
        self._velocity = sorted_v[len(sorted_v) // 2]

        self._last_pt = (now, float(gaze_x), float(gaze_y))

        # ── 이미 고정 상태: 이탈 조건만 검사 ──
        if self.phase == FIXATED:

            dist = math.hypot(
                gaze_x - self.center[0],
                gaze_y - self.center[1]
            )

            if self._velocity > self.v_exit or dist > self.exit_radius:
                self._soft_reset()
                return self._state(now)

            # 고정 중 미세 드리프트 추종
            self.center[0] += CENTER_ADAPT * (gaze_x - self.center[0])
            self.center[1] += CENTER_ADAPT * (gaze_y - self.center[1])

            return self._state(now)

        # ── IDLE / CANDIDATE ──
        if self._velocity > self.v_enter:
            # 아직 이동 중(saccade). 후보 창을 버린다.
            self.phase = IDLE
            self.window = []
            self.candidate_start = None
            self._dispersion = 0.0
            return self._state(now)

        # 느린 구간 → I-DT 창에 축적
        self.window.append((now, float(gaze_x), float(gaze_y)))

        # 오래된 점 제거
        while (
            len(self.window) > 1
            and now - self.window[0][0] > WINDOW_MAX_SEC
        ):
            self.window.pop(0)

        # 분산이 반경을 넘으면 넘지 않을 때까지 앞쪽을 잘라낸다
        while (
            len(self.window) > 1
            and self._dispersion_radius() > self.dispersion_radius
        ):
            self.window.pop(0)

        self._dispersion = self._dispersion_radius()
        self.candidate_start = self.window[0][0]
        self.phase = CANDIDATE

        # ── 고정 확정 ──
        if now - self.candidate_start >= self.min_fixation_sec:

            cx, cy = self._centroid()

            self.phase = FIXATED
            self.fixation_start = self.candidate_start
            self.center = [cx, cy]

        return self._state(now)

    # ── 내부 계산 ────────────────────────────────────────────

    def _centroid(self):
        n = len(self.window)
        sx = sum(p[1] for p in self.window)
        sy = sum(p[2] for p in self.window)
        return sx / n, sy / n

    def _dispersion_radius(self):
        if not self.window:
            return 0.0

        cx, cy = self._centroid()

        return max(
            math.hypot(p[1] - cx, p[2] - cy)
            for p in self.window
        )

    def _state(self, now):

        fixated = self.phase == FIXATED

        if fixated and self.fixation_start is not None:
            duration = now - self.fixation_start
        else:
            duration = 0.0

        return FixationState(
            fixated=fixated,
            phase=self.phase,
            center=(self.center[0], self.center[1]) if self.center else None,
            duration=duration,
            velocity=self._velocity,
            dispersion=self._dispersion,
        )