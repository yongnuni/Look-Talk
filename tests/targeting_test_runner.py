import math
import random
import time
from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class TargetAttempt:
    """타겟 한 회차의 평가 결과."""

    trial: int
    target_x: int
    target_y: int
    selected_x: Optional[int]
    selected_y: Optional[int]
    success: bool
    reaction_time_sec: float
    reason: str


class TargetingTestRunner:
    """
    키보드와 분리된 독립 원형 타겟팅 테스트.

    - 화면 전체에 중간 크기의 원형 타겟을 표시
    - 총 10회 평가
    - 드웰: 타겟 내부를 일정 시간 응시하면 성공
    - 입벌림: 선택 순간 시선 좌표가 타겟 내부면 성공
    - 제한시간이 지나면 실패
    """

    def __init__(
        self,
        screen_w: int,
        screen_h: int,
        total_trials: int = 10,
        target_radius: Optional[int] = None,
        dwell_sec: float = 1.0,
        timeout_sec: float = 5.0,
        randomize: bool = True,
        prepare_sec: float = 2.0
    ):
        if not 1 <= total_trials <= 10:
            raise ValueError(
                "total_trials는 1 이상 10 이하여야 합니다."
            )

        self.screen_w = screen_w
        self.screen_h = screen_h
        self.total_trials = total_trials
        self.dwell_sec = dwell_sec
        self.timeout_sec = timeout_sec
        self.randomize = randomize
        self.prepare_sec = prepare_sec

        # 1536×864 화면 기준 약 69px,
        # 1280×720 화면 기준 약 60px 반지름
        self.target_radius = (
            target_radius
            if target_radius is not None
            else max(
                60,
                min(
                    80,
                    int(min(screen_w, screen_h) * 0.08)
                )
            )
        )

        self.base_targets = self._create_target_positions()
        self.targets = []

        self.active = False
        self.completed = False

        self.current_index = 0
        self.trial_started_at: Optional[float] = None
        self.prepare_started_at: Optional[float] = None
        self.dwell_started_at: Optional[float] = None
        self.complete_time: Optional[float] = None

        self.attempts: list[TargetAttempt] = []

    def _create_target_positions(
        self
    ) -> list[tuple[int, int]]:
        """
        화면 상단 안내문 영역을 피하면서
        화면 전체에 분산된 10개의 타겟 위치를 만든다.
        """

        normalized_positions = [
            (0.15, 0.20),
            (0.50, 0.20),
            (0.85, 0.20),

            (0.15, 0.50),
            (0.50, 0.50),
            (0.85, 0.50),

            (0.15, 0.80),
            (0.50, 0.80),
            (0.85, 0.80),

            (0.70, 0.35),
        ]

        margin_x = self.target_radius + 30
        margin_top = self.target_radius + 80
        margin_bottom = self.target_radius + 30

        positions = []

        for normalized_x, normalized_y in normalized_positions:
            x = int(self.screen_w * normalized_x)
            y = int(self.screen_h * normalized_y)

            x = max(
                margin_x,
                min(
                    self.screen_w - margin_x,
                    x
                )
            )

            y = max(
                margin_top,
                min(
                    self.screen_h - margin_bottom,
                    y
                )
            )

            positions.append((x, y))

        return positions

    @property
    def current_target(
        self
    ) -> Optional[tuple[int, int]]:
        if (
            not self.active
            or self.current_index >= len(self.targets)
        ):
            return None

        return self.targets[self.current_index]

    @property
    def current_trial_number(self) -> int:
        if self.completed:
            return self.total_trials

        return min(
            self.current_index + 1,
            self.total_trials
        )

    def start(
        self,
        now: Optional[float] = None
    ) -> None:
        """타겟팅 테스트를 처음부터 시작한다."""

        if now is None:
            now = time.monotonic()

        self.targets = list(
            self.base_targets[:self.total_trials]
        )

        if self.randomize:
            random.shuffle(self.targets)

        self.active = True
        self.completed = False

        self.current_index = 0
        self.attempts = []

        self.trial_started_at = None
        self.prepare_started_at = now
        self.dwell_started_at = None
        self.complete_time = None

    def reset(self) -> None:
        """테스트 진행 상태를 모두 초기화한다."""

        self.targets = []
        self.active = False
        self.completed = False

        self.current_index = 0
        self.attempts = []

        self.trial_started_at = None
        self.prepare_started_at = None
        self.dwell_started_at = None
        self.complete_time = None

    @property
    def is_preparing(self) -> bool:
        return self.get_prepare_remaining_sec() > 0.0

    def is_inside_target(
        self,
        gaze_x: Optional[int],
        gaze_y: Optional[int]
    ) -> bool:
        """시선 좌표가 현재 원형 타겟 내부인지 확인한다."""

        target = self.current_target

        if (
            target is None
            or gaze_x is None
            or gaze_y is None
            or gaze_x < 0
            or gaze_y < 0
        ):
            return False

        target_x, target_y = target

        distance = math.hypot(
            gaze_x - target_x,
            gaze_y - target_y
        )

        return distance <= self.target_radius

    def update_dwell(
        self,
        gaze_x: Optional[int],
        gaze_y: Optional[int],
        tracking_valid: bool,
        now: Optional[float] = None
    ) -> Optional[TargetAttempt]:
        """
        드웰 방식의 타겟팅 상태를 갱신한다.

        타겟 내부를 dwell_sec 동안 연속 응시하면 성공한다.
        timeout_sec가 지나면 실패한다.
        """

        if not self.active:
            return None

        if now is None:
            now = time.monotonic()

        if not self._start_trial_if_ready(now):
            self.dwell_started_at = None
            return None

        if self._is_timed_out(now):
            return self._finish_current_trial(
                success=False,
                selected_x=gaze_x,
                selected_y=gaze_y,
                reason="timeout",
                now=now
            )

        if (
            not tracking_valid
            or not self.is_inside_target(gaze_x, gaze_y)
        ):
            self.dwell_started_at = None
            return None

        if self.dwell_started_at is None:
            self.dwell_started_at = now
            return None

        dwell_duration = now - self.dwell_started_at

        if dwell_duration >= self.dwell_sec:
            return self._finish_current_trial(
                success=True,
                selected_x=gaze_x,
                selected_y=gaze_y,
                reason="dwell",
                now=now
            )

        return None

    def register_selection(
        self,
        gaze_x: Optional[int],
        gaze_y: Optional[int],
        now: Optional[float] = None
    ) -> Optional[TargetAttempt]:
        """
        입벌림 등의 선택 이벤트를 처리한다.

        선택 순간 시선이 타겟 내부면 성공,
        타겟 외부면 실패로 기록한다.
        """

        if not self.active:
            return None

        if now is None:
            now = time.monotonic()

        if not self._start_trial_if_ready(now):
            return None

        success = self.is_inside_target(
            gaze_x,
            gaze_y
        )

        return self._finish_current_trial(
            success=success,
            selected_x=gaze_x,
            selected_y=gaze_y,
            reason=(
                "selection_hit"
                if success
                else "selection_miss"
            ),
            now=now
        )

    def get_dwell_ratio(
        self,
        now: Optional[float] = None
    ) -> float:
        """현재 타겟 내부 드웰 진행률을 0~1로 반환한다."""

        if not self.active:
            return 0.0

        if now is None:
            now = time.monotonic()

        if (
            not self._start_trial_if_ready(now)
            or self.dwell_started_at is None
        ):
            return 0.0

        elapsed = now - self.dwell_started_at

        return max(
            0.0,
            min(1.0, elapsed / self.dwell_sec)
        )

    def get_remaining_time(
        self,
        now: Optional[float] = None
    ) -> float:
        """현재 회차의 남은 제한시간을 반환한다."""

        if not self.active:
            return 0.0

        if now is None:
            now = time.monotonic()

        if not self._start_trial_if_ready(now):
            return 0.0

        elapsed = now - self.trial_started_at

        return max(
            0.0,
            self.timeout_sec - elapsed
        )

    def get_results(self) -> dict:
        """현재까지의 타겟팅 평가 결과를 반환한다."""

        success_count = sum(
            attempt.success
            for attempt in self.attempts
        )

        failure_count = (
            len(self.attempts) - success_count
        )

        if self.attempts:
            average_reaction_time_sec = sum(
                attempt.reaction_time_sec
                for attempt in self.attempts
            ) / len(self.attempts)
        else:
            average_reaction_time_sec = 0.0

        hit_rate_percent = (
            success_count / self.total_trials * 100.0
            if self.total_trials > 0
            else 0.0
        )

        return {
            "total_trials": self.total_trials,
            "completed_trials": len(self.attempts),
            "success_count": success_count,
            "failure_count": failure_count,
            "hit_rate_percent": hit_rate_percent,
            "average_reaction_time_sec": (
                average_reaction_time_sec
            ),
            "attempts": [
                asdict(attempt)
                for attempt in self.attempts
            ],
        }

    def get_prepare_remaining_sec(
        self,
        now: Optional[float] = None
    ) -> float:
        if (
            not self.active
            or self.prepare_started_at is None
            or self.trial_started_at is not None
        ):
            return 0.0

        if now is None:
            now = time.monotonic()

        return max(
            0.0,
            self.prepare_sec - (now - self.prepare_started_at)
        )

    def _start_trial_if_ready(self, now: float) -> bool:
        if not self.active:
            return False

        if self.trial_started_at is not None:
            return True

        if self.prepare_started_at is None:
            self.trial_started_at = now
            return True

        if now - self.prepare_started_at < self.prepare_sec:
            return False

        self.trial_started_at = (
            self.prepare_started_at + self.prepare_sec
        )
        return True

    def _is_timed_out(self, now: float) -> bool:
        if self.trial_started_at is None:
            return False

        return (
            now - self.trial_started_at
            >= self.timeout_sec
        )

    def _finish_current_trial(
        self,
        success: bool,
        selected_x: Optional[int],
        selected_y: Optional[int],
        reason: str,
        now: float
    ) -> TargetAttempt:
        target = self.current_target

        if target is None:
            raise RuntimeError(
                "진행 중인 타겟이 없습니다."
            )

        target_x, target_y = target

        reaction_time_sec = 0.0

        if self.trial_started_at is not None:
            reaction_time_sec = max(
                0.0,
                now - self.trial_started_at
            )

        attempt = TargetAttempt(
            trial=self.current_index + 1,
            target_x=target_x,
            target_y=target_y,
            selected_x=selected_x,
            selected_y=selected_y,
            success=success,
            reaction_time_sec=reaction_time_sec,
            reason=reason
        )

        self.attempts.append(attempt)
        self.current_index += 1
        self.dwell_started_at = None

        if self.current_index >= self.total_trials:
            self.active = False
            self.completed = True
            self.complete_time = now

        else:
            self.trial_started_at = None
            self.prepare_started_at = now

        return attempt
