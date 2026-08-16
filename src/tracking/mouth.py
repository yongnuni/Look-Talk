import math
import time

import cv2


# =========================================================
# MediaPipe Mouth Landmark
# =========================================================

UPPER_LIP = 13
LOWER_LIP = 14

LEFT_MOUTH = 78
RIGHT_MOUTH = 308

MOUTH_OUTLINE = [
    61, 146, 91, 181, 84, 17, 314, 405,
    321, 375, 291, 308, 324, 318, 402,
    317, 14, 87, 178, 88, 95
]


# =========================================================
# MAR 계산
# =========================================================

def distance(p1, p2):
    """
    두 MediaPipe landmark 사이의 유클리드 거리 계산
    """

    return math.hypot(
        p1.x - p2.x,
        p1.y - p2.y
    )


def mouth_aspect_ratio(landmarks):
    """
    Mouth Aspect Ratio(MAR)를 계산한다.

    MAR =
        입의 세로 길이
        ----------------
        입의 가로 길이
    """

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


# =========================================================
# 사용자별 MAR 캘리브레이션
# =========================================================

class MouthCalibration:
    """
    사용자별 MAR 캘리브레이션

    진행 순서

    1. 입을 편하게 닫은 상태 측정
    2. 입을 충분히 벌린 상태 측정
    3. Closed MAR / Open MAR 계산
    4. 사용자별 Open / Close Threshold 계산

    Open Threshold
        = Closed MAR
        + (Open MAR - Closed MAR) * open_ratio

    Close Threshold
        = Closed MAR
        + (Open MAR - Closed MAR) * close_ratio
    """

    def __init__(
        self,
        sample_duration=2.0,
        open_ratio=0.55,
        close_ratio=0.30,
        min_gap=0.03
    ):

        self.sample_duration = sample_duration

        self.open_ratio = open_ratio
        self.close_ratio = close_ratio

        # 입 닫힘/벌림 차이가 이 값보다 작으면
        # 잘못된 캘리브레이션으로 판단
        self.min_gap = min_gap

        # 기본 Threshold
        self.default_open_threshold = 0.30
        self.default_close_threshold = 0.23

        self.reset()

    # -----------------------------------------------------
    # main.py 호환용 속성
    # -----------------------------------------------------

    @property
    def done(self):
        """
        기존 main.py의

        mouth_calibrator.done

        코드와 호환하기 위한 속성
        """

        return self.finished

    # -----------------------------------------------------
    # 초기화
    # -----------------------------------------------------

    def reset(self):
        """
        캘리브레이션 상태 초기화

        reset 후 바로 입 닫힘 측정을 시작하도록 구성한다.
        """

        self.state = "closed"

        self.start_time = time.time()

        self.closed_samples = []
        self.open_samples = []

        self.closed_mar = None
        self.open_mar = None

        self.open_threshold = self.default_open_threshold
        self.close_threshold = self.default_close_threshold

        self.finished = False

        self.calibration_valid = False

    # -----------------------------------------------------
    # 입 닫힘 캘리브레이션 시작
    # -----------------------------------------------------

    def start_closed_calibration(self):

        self.state = "closed"
        self.start_time = time.time()

        self.closed_samples.clear()
        self.open_samples.clear()

        self.closed_mar = None
        self.open_mar = None

        self.finished = False
        self.calibration_valid = False

    # -----------------------------------------------------
    # 입 벌림 캘리브레이션 시작
    # -----------------------------------------------------

    def start_open_calibration(self):

        self.state = "open"

        self.start_time = time.time()

        self.open_samples.clear()

    # -----------------------------------------------------
    # 평균
    # -----------------------------------------------------

    @staticmethod
    def _mean(values):

        if not values:
            return 0.0

        return sum(values) / len(values)

    # -----------------------------------------------------
    # Threshold 계산
    # -----------------------------------------------------

    def _calculate_thresholds(self):

        if (
            self.closed_mar is None
            or self.open_mar is None
        ):
            return

        gap = (
            self.open_mar
            - self.closed_mar
        )

        # -----------------------------------------
        # 잘못된 캘리브레이션
        # -----------------------------------------

        if gap <= self.min_gap:

            print(
                "[mouth calibration] "
                "입 닫힘/벌림 MAR 차이가 너무 작습니다."
            )

            print(
                "[mouth calibration] "
                f"Closed={self.closed_mar:.3f}, "
                f"Open={self.open_mar:.3f}, "
                f"Gap={gap:.3f}"
            )

            self.open_threshold = (
                self.default_open_threshold
            )

            self.close_threshold = (
                self.default_close_threshold
            )

            self.calibration_valid = False

            # 캘리브레이션 과정 자체는 종료
            self.finished = True
            self.state = "finished"

            return

        # -----------------------------------------
        # 개인별 Threshold 계산
        # -----------------------------------------

        self.open_threshold = (
            self.closed_mar
            + gap * self.open_ratio
        )

        self.close_threshold = (
            self.closed_mar
            + gap * self.close_ratio
        )

        self.calibration_valid = True

        self.finished = True
        self.state = "finished"

    # -----------------------------------------------------
    # 매 프레임 업데이트
    # -----------------------------------------------------

    def update(self, mar):
        """
        main.py에서 매 프레임 호출

        현재 main.py 구조:

            mar = mouth_aspect_ratio(lms)
            mouth_progress = mouth_calibrator.update(mar)

        따라서 landmarks가 아닌
        계산된 MAR 값을 입력받는다.

        return:
            progress: 0.0 ~ 1.0
        """

        if self.finished:
            return 1.0

        now = time.time()

        if self.start_time is None:
            self.start_time = now

        elapsed = (
            now
            - self.start_time
        )

        progress = min(
            elapsed / self.sample_duration,
            1.0
        )

        # =================================================
        # 1단계: 입 닫힘 측정
        # =================================================

        if self.state == "closed":

            self.closed_samples.append(
                mar
            )

            if elapsed >= self.sample_duration:

                self.closed_mar = self._mean(
                    self.closed_samples
                )

                print(
                    "[mouth calibration] "
                    f"Closed MAR = {self.closed_mar:.3f}"
                )

                # 바로 다음 단계 시작
                self.start_open_calibration()

                # 새 단계 시작이므로 진행률 초기화
                return 0.0

        # =================================================
        # 2단계: 입 벌림 측정
        # =================================================

        elif self.state == "open":

            self.open_samples.append(
                mar
            )

            if elapsed >= self.sample_duration:

                self.open_mar = self._mean(
                    self.open_samples
                )

                print(
                    "[mouth calibration] "
                    f"Open MAR = {self.open_mar:.3f}"
                )

                self._calculate_thresholds()

                return 1.0

        return progress

    # -----------------------------------------------------
    # UI 안내 문구
    # -----------------------------------------------------

    def get_instruction(self):
        """
        기존 main.py의
        draw_mouth_calibration_screen()에 전달할 안내 문구
        """

        if self.state == "closed":

            return (
                "입을 편하게 닫고 "
                "그 상태를 유지해주세요."
            )

        if self.state == "open":

            return (
                "입을 충분히 벌리고 "
                "그 상태를 유지해주세요."
            )

        if self.state == "finished":

            if self.calibration_valid:
                return "입벌림 캘리브레이션이 완료되었습니다."

            return (
                "캘리브레이션 값 차이가 작아 "
                "기본값을 적용합니다."
            )

        return "입벌림 캘리브레이션 준비 중입니다."

    # -----------------------------------------------------
    # 남은 시간
    # -----------------------------------------------------

    def get_remaining_time(self):

        if self.finished:
            return 0.0

        if self.start_time is None:
            return self.sample_duration

        elapsed = (
            time.time()
            - self.start_time
        )

        return max(
            0.0,
            self.sample_duration - elapsed
        )

    # -----------------------------------------------------
    # 결과 저장용 Dictionary
    # -----------------------------------------------------

    def get_result_dict(self):
        """
        baseline.json 및 CSV 저장용 결과값

        main.py에서:

            mouth_result =
                mouth_calibrator.get_result_dict()

        형태로 사용
        """

        gap = None

        if (
            self.closed_mar is not None
            and self.open_mar is not None
        ):

            gap = (
                self.open_mar
                - self.closed_mar
            )

        return {
            "closed_mar": self.closed_mar,
            "open_mar": self.open_mar,
            "mar_gap": gap,

            "open_threshold": (
                self.open_threshold
            ),

            "close_threshold": (
                self.close_threshold
            ),

            "open_ratio": (
                self.open_ratio
            ),

            "close_ratio": (
                self.close_ratio
            ),

            "sample_duration": (
                self.sample_duration
            ),

            "calibration_valid": (
                self.calibration_valid
            ),
        }


# =========================================================
# 입벌림 클릭 감지
# =========================================================

class MouthClickDetector:

    def __init__(
        self,
        open_threshold=0.30,
        close_threshold=0.23,
        hold_time=0.30,
        cooldown=0.50,
        lock_time=0.25
    ):

        # 개인별 Threshold
        self.open_threshold = (
            open_threshold
        )

        self.close_threshold = (
            close_threshold
        )

        # 입을 얼마나 유지해야 선택할지
        self.hold_time = hold_time

        # 클릭 이후 추가 입력 제한 시간
        self.cooldown = cooldown

        # 입을 벌리기 전에 한 키를 얼마나 안정적으로
        # 바라봐야 선택 후보로 잠글지
        self.lock_time = lock_time

        # 현재 입 상태
        self.is_open = False

        # 입 벌림 시작 시간
        self.open_start = None

        # 입을 벌리기 전에 안정적으로 바라보고 있는
        # 후보 키와 후보 시작 시간
        self.candidate_key = None
        self.candidate_start = None

        # 입을 벌리기 전에 최종적으로 잠근 키
        self.locked_key = None

        # 이번 입벌림 동작에서 실제 선택된 키
        # main.py에서 mouth.selected_key로 사용
        self.selected_key = None

        # 입 벌림 시작 시점에 사용할 잠긴 키
        self.start_key = None

        # 현재 입벌림 동작에서
        # 이미 클릭했는지 여부
        self.clicked = False

        # 마지막 클릭 시간
        self.last_click_time = 0.0

    # -----------------------------------------------------
    # 개인 Threshold 적용
    # -----------------------------------------------------

    def set_thresholds(
        self,
        open_threshold,
        close_threshold
    ):
        """
        MouthCalibration에서 계산한
        사용자별 Threshold를 적용한다.
        """

        self.open_threshold = float(
            open_threshold
        )

        self.close_threshold = float(
            close_threshold
        )

        self.reset()

        print(
            "[mouth] Threshold 적용 | "
            f"open={self.open_threshold:.3f}, "
            f"close={self.close_threshold:.3f}"
        )

    # -----------------------------------------------------
    # 상태 초기화
    # -----------------------------------------------------

    def reset(self):
        """
        진행 중인 입벌림 선택 상태 초기화
        """

        self.is_open = False

        self.open_start = None
        self.start_key = None

        self.candidate_key = None
        self.candidate_start = None
        self.locked_key = None
        self.selected_key = None

        self.clicked = False

    # -----------------------------------------------------
    # 입벌림 전 시선 키 잠금
    # -----------------------------------------------------

    def _update_gaze_lock(
        self,
        hovered_key,
        now
    ):
        """
        입이 닫혀 있는 동안 시선이 일정 시간 같은 키에
        머물면 그 키를 locked_key로 저장한다.

        이미 잠긴 키가 있는 상태에서 시선이 잠깐 흔들려도
        기존 locked_key는 바로 해제하지 않는다.
        다른 키를 lock_time 이상 안정적으로 바라본 경우에만
        새로운 키로 잠금을 갱신한다.
        """

        if hovered_key is None:
            self.candidate_key = None
            self.candidate_start = None
            return

        # 이미 잠긴 키를 계속 보고 있으면
        # 별도의 후보 누적이 필요하지 않음
        if hovered_key == self.locked_key:
            self.candidate_key = hovered_key
            self.candidate_start = now
            return

        # 새로운 키를 보기 시작한 경우
        if hovered_key != self.candidate_key:
            self.candidate_key = hovered_key
            self.candidate_start = now
            return

        # 같은 후보 키를 lock_time 이상 바라보면 잠금
        if (
            self.candidate_start is not None
            and now - self.candidate_start >= self.lock_time
        ):
            self.locked_key = hovered_key

    # -----------------------------------------------------
    # 클릭 판정
    # -----------------------------------------------------

    def update(
        self,
        landmarks,
        hovered_key=None
    ):
        """
        입벌림 선택 판정

        동작 순서
        1. 입을 닫은 상태에서 같은 키를 lock_time 동안 응시
        2. 해당 키를 locked_key로 잠금
        3. 입벌림이 시작되면 현재 시선이 아닌 locked_key 사용
        4. hold_time 이상 입을 유지하면 locked_key를 선택

        return:
            click
            mar

        실제 선택된 키는 self.selected_key에 저장된다.
        """

        mar = mouth_aspect_ratio(
            landmarks
        )

        now = time.time()

        click = False

        # 매 프레임 선택 결과는 초기화
        # 클릭이 발생한 프레임에서만 실제 키가 들어간다.
        self.selected_key = None

        was_open = self.is_open

        # =================================================
        # 입이 닫혀 있는 동안 시선 키 잠금
        # =================================================

        if not self.is_open:
            self._update_gaze_lock(
                hovered_key,
                now
            )

        # =================================================
        # 히스테리시스 기반 입 상태 판정
        # =================================================

        if not self.is_open:

            # 현재 닫힌 상태
            # Open Threshold 이상이 되어야 열림 판정
            if mar >= self.open_threshold:

                self.is_open = True

        else:

            # 현재 열린 상태
            # Close Threshold 이하가 되어야 닫힘 판정
            if mar <= self.close_threshold:

                self.is_open = False

        # =================================================
        # 입을 막 닫은 순간
        # =================================================

        if was_open and not self.is_open:

            self.open_start = None
            self.start_key = None

            self.candidate_key = None
            self.candidate_start = None
            self.locked_key = None

            self.clicked = False

            return click, mar

        # =================================================
        # 입 벌림 상태
        # =================================================

        if self.is_open:

            # ---------------------------------------------
            # 처음 입을 벌린 순간
            # ---------------------------------------------

            if self.open_start is None:

                self.open_start = now

                # 현재 hovered_key를 사용하지 않고
                # 입을 벌리기 전에 잠가 둔 키를 사용
                self.start_key = self.locked_key

            # ---------------------------------------------
            # Hold Time 이상 유지
            # ---------------------------------------------

            elif (
                now - self.open_start
                >= self.hold_time
                and not self.clicked
            ):

                # -----------------------------------------
                # Cooldown 검사
                # -----------------------------------------

                cooldown_ready = (
                    now - self.last_click_time
                    >= self.cooldown
                )

                # 입을 벌리는 동안 현재 시선 위치는
                # 비교하지 않는다.
                # 입벌림 전에 잠가 둔 키가 있을 때만 선택한다.
                if (
                    self.start_key is not None
                    and cooldown_ready
                ):

                    click = True

                    self.selected_key = self.start_key

                    self.last_click_time = now

                # 같은 입벌림 동작에서는
                # 추가 클릭을 막는다.
                self.clicked = True

        # =================================================
        # 계속 입이 닫혀 있는 상태
        # =================================================

        else:

            self.open_start = None
            self.start_key = None
            self.clicked = False

        return click, mar


# =========================================================
# 입 랜드마크 시각화
# =========================================================

def draw_mouth(
    frame,
    landmarks,
    fw,
    fh
):

    ids = [
        UPPER_LIP,
        LOWER_LIP,
        LEFT_MOUTH,
        RIGHT_MOUTH
    ]

    for idx in ids:

        lm = landmarks.landmark[idx]

        x = int(
            lm.x * fw
        )

        y = int(
            lm.y * fh
        )

        cv2.circle(
            frame,
            (x, y),
            6,
            (0, 255, 0),
            -1
        )

    upper = landmarks.landmark[
        UPPER_LIP
    ]

    lower = landmarks.landmark[
        LOWER_LIP
    ]

    cv2.line(
        frame,
        (
            int(upper.x * fw),
            int(upper.y * fh)
        ),
        (
            int(lower.x * fw),
            int(lower.y * fh)
        ),
        (0, 255, 255),
        2
    )


# =========================================================
# 캘리브레이션 디버그 정보
# =========================================================

def draw_mouth_calibration_info(
    frame,
    calibration,
    mar
):
    """
    캘리브레이션 진행 상태를
    카메라 프레임에 표시한다.
    """

    y = 40

    # -----------------------------------------------------
    # 현재 MAR
    # -----------------------------------------------------

    cv2.putText(
        frame,
        f"MAR: {mar:.3f}",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    y += 40

    # -----------------------------------------------------
    # 현재 상태
    # -----------------------------------------------------

    cv2.putText(
        frame,
        f"State: {calibration.state}",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # -----------------------------------------------------
    # Closed MAR
    # -----------------------------------------------------

    if calibration.closed_mar is not None:

        y += 40

        cv2.putText(
            frame,
            (
                "Closed MAR: "
                f"{calibration.closed_mar:.3f}"
            ),
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    # -----------------------------------------------------
    # Open MAR
    # -----------------------------------------------------

    if calibration.open_mar is not None:

        y += 40

        cv2.putText(
            frame,
            (
                "Open MAR: "
                f"{calibration.open_mar:.3f}"
            ),
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    # -----------------------------------------------------
    # Threshold
    # -----------------------------------------------------

    if calibration.finished:

        y += 40

        cv2.putText(
            frame,
            (
                "Open Threshold: "
                f"{calibration.open_threshold:.3f}"
            ),
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        y += 40

        cv2.putText(
            frame,
            (
                "Close Threshold: "
                f"{calibration.close_threshold:.3f}"
            ),
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        y += 40

        valid_text = (
            "Calibration: VALID"
            if calibration.calibration_valid
            else "Calibration: DEFAULT"
        )

        cv2.putText(
            frame,
            valid_text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (
                (0, 255, 0)
                if calibration.calibration_valid
                else (0, 0, 255)
            ),
            2
        )