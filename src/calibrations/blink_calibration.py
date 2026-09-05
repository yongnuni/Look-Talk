"""Personal EAR calibration restored from the project's standalone blink test.

The state machine intentionally keeps the policy used by
``blink_calibration_test.py`` in the repository history: five rounds of open-eye
collection followed by one blink, median-based thresholds, and the detector's
existing defaults when the measured EAR span is too small.
"""

from dataclasses import asdict, dataclass
import statistics
import time


@dataclass(frozen=True)
class BlinkCalibrationResult:
    close_threshold: float
    open_threshold: float
    open_ear_median: float
    closed_ear_median: float
    calibration_failed_fallback: bool
    total_trials: int
    closed_sample_count: int

    def to_dict(self):
        return asdict(self)


class BlinkCalibration:
    """Collect open/closed EAR samples without owning a camera or window."""

    def __init__(
        self,
        total_trials=5,
        open_collect_sec=1.2,
        open_min_samples=15,
        blink_window_sec=2.0,
        rest_sec=0.6,
        max_attempts=3,
        default_close_threshold=0.18,
        default_open_threshold=0.22,
    ):
        self.total_trials = total_trials
        self.open_collect_sec = open_collect_sec
        self.open_min_samples = open_min_samples
        self.blink_window_sec = blink_window_sec
        self.rest_sec = rest_sec
        self.max_attempts = max_attempts
        self.default_close_threshold = default_close_threshold
        self.default_open_threshold = default_open_threshold
        self.reset()

    def reset(self, now=None):
        if now is None:
            now = time.monotonic()

        self.state = "open"
        self.current_trial_index = 0
        self.state_started_at = now
        self.open_samples = []
        self.round_open_samples = []
        self.closed_minima = []
        self.current_minimum = None
        self.failed_attempts = 0
        self.failed = False
        self.done = False
        self.result = None

        # 얼굴이 보이지 않는 구간을 단계 경과 시간에서 빼기 위한 상태.
        # 첫 프레임은 직전 간격이 없으므로 얼굴이 있었다고 보고 시작한다.
        self._last_now = None
        self._had_face = True

    def _advance_clock(self, now, has_face):
        """얼굴이 없던 구간만큼 단계 시작 시각을 뒤로 민다.

        측정 도중 얼굴이 잠깐 사라졌다고 해서 깜빡임 대기 창(2초)이 흘러가
        버리면 사용자 잘못이 아닌 이유로 시행이 실패한다. 얼굴이 없는 동안
        경과 시간을 멈춰 두면 ``state_started_at``의 의미(벽시계 기준 단계
        시작 시각)는 그대로 유지된다.
        """

        if (
            self._last_now is not None
            and now > self._last_now
            and not (has_face and self._had_face)
        ):
            self.state_started_at += now - self._last_now

        self._last_now = now
        self._had_face = has_face

    def update(self, ear, now=None):
        if now is None:
            now = time.monotonic()

        self._advance_clock(now, ear is not None)

        if self.done or self.failed:
            return self.get_progress(now)

        elapsed = max(0.0, now - self.state_started_at)

        if self.state == "open":
            if ear is not None:
                self.round_open_samples.append(float(ear))

            if (
                elapsed >= self.open_collect_sec
                and len(self.round_open_samples) >= self.open_min_samples
            ):
                self.open_samples.extend(self.round_open_samples)
                self.current_minimum = None
                self._start_state("blink", now)

        elif self.state == "blink":
            if ear is not None:
                value = float(ear)
                self.current_minimum = (
                    value
                    if self.current_minimum is None
                    else min(self.current_minimum, value)
                )

            open_estimate = self._median(self.open_samples)
            blinked = (
                self.current_minimum is not None
                and self.current_minimum < open_estimate * 0.6
                and ear is not None
                and ear > open_estimate * 0.8
            )

            if blinked:
                # 실제로 감았다 뜬 것이 확인된 시행만 통계에 넣는다. 대기 창이
                # 그냥 만료된 회차의 최소값은 눈을 뜬 채로 관측한 값이라, 그걸
                # closed 표본에 섞으면 closed_median이 위로 끌려 올라가
                # threshold가 통째로 어긋난다.
                self.closed_minima.append(self.current_minimum)
                self.current_trial_index += 1
                self.failed_attempts = 0

                if self.current_trial_index >= self.total_trials:
                    self._finish()
                else:
                    self._start_state("rest", now)

            elif elapsed >= self.blink_window_sec:
                self._fail_attempt(now)

        elif self.state == "rest" and elapsed >= self.rest_sec:
            self._begin_open_collection(now)

        return self.get_progress(now)

    def _fail_attempt(self, now):
        """깜빡임을 확인하지 못한 회차를 재시도한다.

        ``max_attempts``번 연속으로 놓치면 조용히 잘못된 threshold를 만들지
        않고 실패 상태로 멈춘다. 사용자는 다시 측정하거나 기본값으로 계속할 수
        있다(``continue_with_defaults``).
        """

        self.failed_attempts += 1

        if self.failed_attempts >= self.max_attempts:
            self.state = "failed"
            self.failed = True
            return

        self._begin_open_collection(now)

    def _begin_open_collection(self, now):
        self.round_open_samples = []
        self.current_minimum = None
        self._start_state("open", now)

    def continue_with_defaults(self):
        """측정에 실패해도 기존 detector 기본값으로 진행한다."""

        self.result = BlinkCalibrationResult(
            close_threshold=self.default_close_threshold,
            open_threshold=self.default_open_threshold,
            open_ear_median=round(self._median(self.open_samples), 4),
            closed_ear_median=round(self._median(self.closed_minima), 4),
            calibration_failed_fallback=True,
            total_trials=self.total_trials,
            closed_sample_count=len(self.closed_minima),
        )
        self.state = "done"
        self.failed = False
        self.done = True
        return self.result

    def _start_state(self, state, now):
        self.state = state
        self.state_started_at = now

    def _finish(self):
        open_median = self._median(self.open_samples)
        closed_median = self._median(self.closed_minima)
        span = open_median - closed_median
        fallback = not self.closed_minima or span < 0.05

        if fallback:
            close_threshold = self.default_close_threshold
            open_threshold = self.default_open_threshold
        else:
            close_threshold = round(closed_median + span * 0.35, 4)
            open_threshold = round(closed_median + span * 0.55, 4)

        self.result = BlinkCalibrationResult(
            close_threshold=close_threshold,
            open_threshold=open_threshold,
            open_ear_median=round(open_median, 4),
            closed_ear_median=round(closed_median, 4),
            calibration_failed_fallback=fallback,
            total_trials=self.total_trials,
            closed_sample_count=len(self.closed_minima),
        )
        self.state = "done"
        self.done = True

    @staticmethod
    def _median(values):
        return float(statistics.median(values)) if values else 0.0

    def get_progress(self, now=None):
        if self.done:
            return 1.0
        if self.failed:
            return 0.0
        if now is None:
            now = time.monotonic()

        elapsed = max(0.0, now - self.state_started_at)
        if self.state == "open":
            return min(
                1.0,
                elapsed / self.open_collect_sec,
                len(self.round_open_samples) / self.open_min_samples,
            )
        if self.state == "blink":
            return min(1.0, elapsed / self.blink_window_sec)
        if self.state == "rest":
            return min(1.0, elapsed / self.rest_sec)
        return 0.0

    def get_remaining_time(self, now=None):
        if self.done or self.failed:
            return 0.0
        if now is None:
            now = time.monotonic()

        durations = {
            "open": self.open_collect_sec,
            "blink": self.blink_window_sec,
            "rest": self.rest_sec,
        }
        duration = durations.get(self.state, 0.0)
        return max(0.0, duration - (now - self.state_started_at))

    def get_instruction(self):
        if self.state == "open":
            return "눈을 편하게 뜨고 정면을 바라봐 주세요."
        if self.state == "blink":
            return "지금 한 번 자연스럽게 눈을 깜빡여 주세요."
        if self.state == "rest":
            return "좋습니다. 눈을 뜨고 잠시 기다려 주세요."
        if self.state == "failed":
            return (
                "눈 깜빡임을 확인하지 못했습니다. "
                "r: 다시 측정 / c: 기본값으로 계속"
            )
        if self.state == "done":
            return "눈 깜빡임 캘리브레이션 완료"
        return ""

    def get_result_dict(self):
        return self.result.to_dict() if self.result is not None else None
