from dataclasses import dataclass, asdict
import time
import statistics


@dataclass
class MouthTrialResult:
    """
    입벌림 캘리브레이션 1회 시행 결과
    """

    success: bool
    reaction_time: float = 0.0
    activation_duration: float = 0.0
    peak_mar: float = 0.0
    activation_amplitude: float = 0.0


@dataclass
class MouthCalibrationResult:
    """
    입벌림 캘리브레이션 최종 결과
    """

    mar_baseline: float
    activation_threshold: float

    mouth_success_rate: float
    activation_amplitude_mean: float
    activation_duration_mean: float
    open_close_speed_mean: float

    mouth_consistency: float
    mouth_contrast_ratio: float
    mouth_false_trigger_rate: float
    amplitude_decay: float

    mouth_min_hold_duration: float
    mouth_init_score: float

    total_trials: int
    success_count: int
    false_trigger_count: int

    def to_dict(self):
        return asdict(self)


class MouthCalibration:
    """
    입벌림 캘리브레이션 로직 담당 클래스

    진행 순서:
    1. rest_collect: 입을 닫은 기본 상태 MAR baseline 수집
    2. trial_ready: 다음 시행 준비
    3. trial_wait: 입 벌림 입력 대기
    4. trial_active: 입 벌림 유지 중
    5. done: 캘리브레이션 완료
    """

    def __init__(
        self,
        total_trials=5,
        rest_collect_sec=4.0,
        ready_sec=2.0,
        response_timeout_sec=5.0,
        min_open_hold_sec=1.2,
        close_confirm_sec=1.0,
    ):
        self.min_open_hold_sec = min_open_hold_sec
        self.total_trials = total_trials
        self.rest_collect_sec = rest_collect_sec
        self.ready_sec = ready_sec
        self.response_timeout_sec = response_timeout_sec
        self.close_confirm_sec = close_confirm_sec

        self.reset()

    def reset(self):
        self.state = "rest_collect"

        self.started_at = time.time()
        self.state_started_at = time.time()

        self.rest_mar_samples = []

        self.mar_baseline = None
        self.activation_threshold = None

        self.current_trial_index = 0
        self.trials = []

        self.false_trigger_count = 0
        self._false_trigger_active = False

        self.trial_started_at = None
        self.activation_started_at = None
        self.activation_peak_mar = 0.0
        self.activation_last_open_time = None

        self.done = False
        self.result = None

    def update(self, mar):
        """
        매 프레임마다 현재 MAR 값을 넣어 호출한다.

        Parameters
        ----------
        mar : float
            현재 프레임의 Mouth Aspect Ratio

        Returns
        -------
        float
            현재 단계 진행률 0.0 ~ 1.0
        """

        now = time.time()

        if mar is None:
            return 0.0

        if self.state == "rest_collect":
            return self._update_rest_collect(mar, now)

        if self.state == "trial_ready":
            return self._update_trial_ready(mar, now)

        if self.state == "trial_wait":
            return self._update_trial_wait(mar, now)

        if self.state == "trial_active":
            return self._update_trial_active(mar, now)

        if self.state == "done":
            return 1.0

        return 0.0

    def _update_rest_collect(self, mar, now):
        self.rest_mar_samples.append(mar)

        elapsed = now - self.state_started_at
        progress = min(elapsed / self.rest_collect_sec, 1.0)

        if elapsed >= self.rest_collect_sec:
            self._finish_rest_collect()
            self._start_trial_ready()

        return progress

    def _finish_rest_collect(self):
        if not self.rest_mar_samples:
            self.mar_baseline = 0.0
        else:
            self.mar_baseline = statistics.mean(self.rest_mar_samples)

        # threshold는 파일럿 후 조정 가능
        # baseline보다 충분히 커야 입 벌림으로 인정
        self.activation_threshold = max(
            self.mar_baseline * 1.6,
            self.mar_baseline + 0.08
        )

    def _start_trial_ready(self):
        self.state = "trial_ready"
        self.state_started_at = time.time()
        self._false_trigger_active = False

    def _update_trial_ready(self, mar, now):
        """
        사용자가 다음 시행 전에 준비하는 구간.
        이때 입벌림이 감지되면 false trigger로 카운트.
        """

        self._check_false_trigger(mar)

        elapsed = now - self.state_started_at
        progress = min(elapsed / self.ready_sec, 1.0)

        if elapsed >= self.ready_sec:
            self._start_trial_wait()

        return progress

    def _start_trial_wait(self):
        self.state = "trial_wait"
        self.state_started_at = time.time()
        self.trial_started_at = time.time()

        self.activation_started_at = None
        self.activation_peak_mar = 0.0
        self.activation_last_open_time = None

    def _update_trial_wait(self, mar, now):
        """
        '입을 벌려주세요' 안내 후 실제 입벌림을 기다리는 단계.
        """

        elapsed = now - self.trial_started_at
        progress = min(elapsed / self.response_timeout_sec, 1.0)

        if self._is_mouth_open(mar):
            self.state = "trial_active"
            self.state_started_at = now

            self.activation_started_at = now
            self.activation_peak_mar = mar
            self.activation_last_open_time = now

            return progress

        if elapsed >= self.response_timeout_sec:
            self._add_failed_trial()
            self._move_next_or_finish()

        return progress

    def _update_trial_active(self, mar, now):
        """
        입벌림이 감지된 뒤 최소 유지 시간 동안 입을 벌린 상태를 유지하게 한다.
        최소 유지 시간이 지난 뒤에만 닫힘 확인을 시작한다.
        """

        if mar > self.activation_peak_mar:
            self.activation_peak_mar = mar

        open_elapsed = now - self.activation_started_at

        # 아직 최소 입벌림 유지 시간이 지나지 않았으면 계속 유지
        if open_elapsed < self.min_open_hold_sec:
            if self._is_mouth_open(mar):
                self.activation_last_open_time = now

            return min(open_elapsed / self.min_open_hold_sec, 1.0)

        # 최소 유지 시간이 지난 뒤부터 닫힘 확인
        if self._is_mouth_open(mar):
            self.activation_last_open_time = now
        else:
            if (
                self.activation_last_open_time is not None
                and now - self.activation_last_open_time >= self.close_confirm_sec
            ):
                self._add_success_trial(now)
                self._move_next_or_finish()

        return 1.0

    def _is_mouth_open(self, mar):
        if self.activation_threshold is None:
            return False

        return mar >= self.activation_threshold

    def _check_false_trigger(self, mar):
        """
        지시 전 준비 구간에서 발생한 입벌림을 false trigger로 계산.
        한 번 열린 상태가 계속 유지되어도 1회만 카운트한다.
        """

        if self.activation_threshold is None:
            return

        is_open = self._is_mouth_open(mar)

        if is_open and not self._false_trigger_active:
            self.false_trigger_count += 1
            self._false_trigger_active = True

        if not is_open:
            self._false_trigger_active = False

    def _add_failed_trial(self):
        self.trials.append(
            MouthTrialResult(
                success=False
            )
        )

    def _add_success_trial(self, now):
        reaction_time = self.activation_started_at - self.trial_started_at
        activation_duration = now - self.activation_started_at
        activation_amplitude = self.activation_peak_mar - self.mar_baseline

        self.trials.append(
            MouthTrialResult(
                success=True,
                reaction_time=reaction_time,
                activation_duration=activation_duration,
                peak_mar=self.activation_peak_mar,
                activation_amplitude=activation_amplitude,
            )
        )

    def _move_next_or_finish(self):
        self.current_trial_index += 1

        if self.current_trial_index >= self.total_trials:
            self._finish_calibration()
        else:
            self._start_trial_ready()

    def _finish_calibration(self):
        self.result = self._calculate_result()
        self.done = True
        self.state = "done"

    def _calculate_result(self):
        success_trials = [
            trial
            for trial in self.trials
            if trial.success
        ]

        success_count = len(success_trials)
        mouth_success_rate = success_count / self.total_trials

        amplitudes = [
            trial.activation_amplitude
            for trial in success_trials
        ]

        durations = [
            trial.activation_duration
            for trial in success_trials
        ]

        if amplitudes:
            activation_amplitude_mean = statistics.mean(amplitudes)
        else:
            activation_amplitude_mean = 0.0

        if durations:
            activation_duration_mean = statistics.mean(durations)
        else:
            activation_duration_mean = 0.0

        if activation_duration_mean > 0:
            open_close_speed_mean = (
                activation_amplitude_mean / activation_duration_mean
            )
        else:
            open_close_speed_mean = 0.0

        if len(amplitudes) >= 2:
            amplitude_mean = statistics.mean(amplitudes)
            amplitude_std = statistics.stdev(amplitudes)

            if amplitude_mean > 0:
                mouth_consistency = 1.0 - min(
                    amplitude_std / amplitude_mean,
                    1.0
                )
            else:
                mouth_consistency = 0.0
        else:
            mouth_consistency = 0.0

        if self.mar_baseline and self.mar_baseline > 0:
            mouth_contrast_ratio = (
                activation_amplitude_mean / self.mar_baseline
            )
        else:
            mouth_contrast_ratio = 0.0

        mouth_false_trigger_rate = (
            self.false_trigger_count / self.total_trials
        )

        amplitude_decay = self._calculate_amplitude_decay(amplitudes)

        mouth_min_hold_duration = max(
            0.15,
            activation_duration_mean * 0.5
        )

        contrast_normalized = min(
            mouth_contrast_ratio / 1.5,
            1.0
        )

        mouth_init_score = (
            mouth_success_rate * 0.6
            + contrast_normalized * 0.4
        )

        return MouthCalibrationResult(
            mar_baseline=self.mar_baseline,
            activation_threshold=self.activation_threshold,

            mouth_success_rate=mouth_success_rate,
            activation_amplitude_mean=activation_amplitude_mean,
            activation_duration_mean=activation_duration_mean,
            open_close_speed_mean=open_close_speed_mean,

            mouth_consistency=mouth_consistency,
            mouth_contrast_ratio=mouth_contrast_ratio,
            mouth_false_trigger_rate=mouth_false_trigger_rate,
            amplitude_decay=amplitude_decay,

            mouth_min_hold_duration=mouth_min_hold_duration,
            mouth_init_score=mouth_init_score,

            total_trials=self.total_trials,
            success_count=success_count,
            false_trigger_count=self.false_trigger_count,
        )

    def _calculate_amplitude_decay(self, amplitudes):
        """
        반복 후 입 벌림 크기 감소율.
        값이 클수록 후반부 amplitude가 줄었다는 의미.
        """

        if len(amplitudes) < 2:
            return 0.0

        first = amplitudes[0]
        last = amplitudes[-1]

        if first <= 0:
            return 0.0

        decay = (first - last) / first

        return max(decay, 0.0)

    def get_instruction(self):
        if self.state == "rest_collect":
            return "입을 편하게 닫고 정면을 바라봐 주세요."

        if self.state == "trial_ready":
            return (
                f"입벌림 캘리브레이션 "
                f"{self.current_trial_index + 1}/{self.total_trials}회차 준비"
            )

        if self.state == "trial_wait":
            return "입을 벌려 주세요."

        if self.state == "trial_active":
            return "입을 벌린 상태를 유지한 뒤 천천히 닫아 주세요."

        if self.state == "done":
            return "입벌림 캘리브레이션 완료"

        return ""

    def get_result_dict(self):
        if self.result is None:
            return None

        return self.result.to_dict()

    def get_remaining_time(self):

        now = time.time()

        if self.state == "rest_collect":
            return max(
                0,
                self.rest_collect_sec -
                (now - self.state_started_at)
            )   

        if self.state == "trial_ready":
            return max(
                0,
                self.ready_sec -
                (now - self.state_started_at)
            )

        if self.state == "trial_wait":
            return max(
                0,
                self.response_timeout_sec -
                (now - self.trial_started_at)
            )

        return 0