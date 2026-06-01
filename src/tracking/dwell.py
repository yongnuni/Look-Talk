import time

from src.config import DWELL_SEC


class DwellController:

    def __init__(self):
        self.dwell_key = None
        self.dwell_start = None
        self.cooldown_end = 0

    def update(self, gaze_x, gaze_y, buttonList):
        """
        현재 시선 좌표와 버튼 리스트를 받아 드웰 상태 갱신.

        Returns:
            (hovered_key, dwell_ratio, clicked_key)
            clicked_key는 드웰 완료 시에만 값, 나머지는 None
        """

        now = time.time()

        dwell_ratio = 0.0
        hovered_key = None
        clicked_key = None

        if gaze_x < 0 or now <= self.cooldown_end:
            self.dwell_key = None
            self.dwell_start = None
            return hovered_key, dwell_ratio, clicked_key

        for button in buttonList:
            bx, by = button.pos
            bw, bh = button.size

            if (
                bx < gaze_x < bx + bw
                and
                by < gaze_y < by + bh
            ):
                hovered_key = button.text
                break

        if hovered_key:

            if hovered_key != self.dwell_key:
                self.dwell_key = hovered_key
                self.dwell_start = now

            else:
                elapsed = now - self.dwell_start
                dwell_ratio = min(1.0, elapsed / DWELL_SEC)

                if dwell_ratio >= 1.0:
                    clicked_key = self.dwell_key
                    self.cooldown_end = now + 0.4
                    self.dwell_key = None
                    self.dwell_start = None

        else:
            self.dwell_key = None
            self.dwell_start = None

        return hovered_key, dwell_ratio, clicked_key