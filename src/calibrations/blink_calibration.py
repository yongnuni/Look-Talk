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
        default_close_threshold=0.18,
        default_open_threshold=0.22,
    ):
        self.total_trials = total_trials
        self.open_collect_sec = open_collect_sec
        self.open_min_samples = open_min_samples
        self.blink_window_sec = blink_window_sec
        self.rest_sec = rest_sec
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
        self.done = False
        self.result = None

    def update(self, ear, now=None):
        if now is None:
            now = time.monotonic()

        if self.done:
            return 1.0

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

            if blinked or elapsed >= self.blink_window_sec:
                if self.current_minimum is not None:
                    self.closed_minima.append(self.current_minimum)
                self._start_state("rest", now)

        elif self.state == "rest" and elapsed >= self.rest_sec:
            self.current_trial_index += 1
            self.round_open_samples = []

            if self.current_trial_index >= self.total_trials:
                self._finish()
            else:
                self._start_state("open", now)

        return self.get_progress(now)

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
        if self.done:
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
        if self.state == "done":
            return "눈 깜빡임 캘리브레이션 완료"
        return ""

    def get_result_dict(self):
        return self.result.to_dict() if self.result is not None else None
