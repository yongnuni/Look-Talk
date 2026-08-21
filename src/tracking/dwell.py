import time

from src.config import DWELL_SEC
from src.tracking.fixation import IDLE_STATE, FixationHitbox


class DwellController:

    def __init__(self, fixation_hitbox=None):
        self.dwell_key = None
        self.dwell_start = None
        self.cooldown_end = 0

        # 세 트리거(시선 지속 시간 / 깜빡임 / 입 개폐)가 모두 이 컨트롤러에서
        # hover 키를 받아 가므로, 고정 감지형 히트박스 확장도 여기 한 곳에만
        # 물린다. 판정 로직 자체는 src/tracking/fixation.py가 독자적으로
        # 수행하고, dwell은 결과 키만 넘겨받는다.
        self.fixation_hitbox = (
            fixation_hitbox
            if fixation_hitbox is not None
            else FixationHitbox()
        )

        self.fixation_state = IDLE_STATE

    def reset(self):
        """
        현재 dwell 상태를 초기화합니다.
        gaze가 유효하지 않거나 hover 상태를 초기화해야 할 때 호출합니다.

        고정 상태는 여기서 건드리지 않습니다 — 입벌림 모드는 매 프레임
        reset()을 호출하므로, 함께 지우면 고정이 절대 성립하지 못합니다.
        고정 상태를 통째로 비우려면 fixation_hitbox.reset()을 쓰세요.
        """
        self.dwell_key = None
        self.dwell_start = None

    def update(self, gaze_x, gaze_y, buttonList, now=None):
        """
        현재 시선 좌표와 버튼 리스트를 받아 드웰 상태를 갱신합니다.

        now를 넘기면 그 시각을 기준으로 판정합니다(테스트용). 실사용에서는
        생략해 time.time()을 쓰며, dwell과 고정 판정이 같은 시계를 씁니다.

        Returns:
            (hovered_key, dwell_ratio, clicked_key)

            hovered_key:
                현재 hover 중인 키

            dwell_ratio:
                dwell 진행률, 0.0 ~ 1.0

            clicked_key:
                dwell 완료 시 입력할 키
        """

        if now is None:
            now = time.time()

        dwell_ratio = 0.0
        hovered_key = None
        clicked_key = None

        gaze_valid = not (gaze_x < 0 or gaze_y < 0)

        # 고정 판정은 dwell 상태·cooldown과 무관한 독립 레이어다.
        # cooldown 중에도 시선은 계속 흐르므로 매 프레임 먼저 갱신한다.
        self.fixation_state = self.fixation_hitbox.update(
            gaze_x,
            gaze_y,
            buttonList,
            valid=gaze_valid,
            now=now,
        )

        # gaze가 유효하지 않거나 cooldown 중이면 dwell 상태 초기화
        if not gaze_valid or now <= self.cooldown_end:
            self.reset()
            return hovered_key, dwell_ratio, clicked_key

        # 렌더링과 같은 Button.rect를 기준으로 판정한다. 고정이 성립한
        # 키만 판정 영역이 넓어지고(겹치는 구간에서는 그 키가 이긴다),
        # 확장이 없는 곳에서는 종전처럼 가장 가까운 키로 보정하지 않고
        # 어떤 키도 선택하지 않는다.
        hovered_button = self.fixation_hitbox.hit_test(
            buttonList,
            gaze_x,
            gaze_y,
        )
        if hovered_button is not None:
            hovered_key = hovered_button.text

        # dwell 진행
        if hovered_key:

            if hovered_key != self.dwell_key:
                self.dwell_key = hovered_key
                self.dwell_start = now

            else:
                elapsed = now - self.dwell_start
                dwell_ratio = min(
                    1.0,
                    elapsed / DWELL_SEC
                )

                if dwell_ratio >= 1.0:
                    clicked_key = self.dwell_key
                    self.cooldown_end = now + 0.4
                    self.reset()

        else:
            self.reset()

        return hovered_key, dwell_ratio, clicked_key
