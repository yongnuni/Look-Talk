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

        # 추천 목록처럼 선택 대상의 의미가 바뀌는 외부 문맥
        self.target_context = None

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

    def reset_target_lock(self):
        """입 상태는 유지하면서 현재 시선 대상 잠금만 안전하게 취소합니다.

        입을 벌린 도중 추천 목록이 바뀌면 그 제스처로 새 후보가 선택되지
        않도록, 입을 다시 닫을 때까지 현재 제스처를 소비된 상태로 둡니다.
        """

        self.candidate_key = None
        self.candidate_start = None
        self.locked_key = None
        self.selected_key = None
        self.start_key = None

        if self.is_open:
            self.clicked = True

    def set_target_context(self, context):
        """선택 대상 목록 변경 시 기존 hover/lock을 무효화합니다."""

        if context == self.target_context:
            return False

        self.target_context = context
        self.reset_target_lock()
        return True

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
