"""
src/key_zoom.py

고정 감지 결과를 받아 "어느 키를, 지금 몇 배로 그릴지"만 결정하는 모듈.

역할 경계
---------
- fixation.py : 시선이 고정됐는지만 판단 (좌표계 지식 없음)
- key_zoom.py : 고정된 좌표에 어떤 키가 있는지 독자적으로 찾고 배율 계산
- ui.py       : key_zoom이 준 배율대로 그리기만 함
- dwell.py    : key_zoom이 준 배율만큼 히트박스 반경만 넓힘

키 선택은 dwell의 hover_lock / assist_radius를 전혀 참조하지 않고
독자적으로 판단한다. 즉 이 모듈을 제거해도 dwell 판정 결과는
확대 이전 상태로 그대로 돌아간다.
"""

import math
import time


# ── 튜닝 파라미터 ────────────────────────────────────────────

ZOOM_SCALE = 1.3          # 최대 확대 배율 (요구사항: 1.3배)
ZOOM_IN_SEC = 0.12        # 확대 애니메이션 길이 (80~150ms 권장 구간)
ZOOM_OUT_SEC = 0.08       # 축소 애니메이션 길이
SNAP_MARGIN_PX = 30       # 키 사각형에서 이 거리 안이면 그 키로 인정
ZOOM_AFFECTS_HITBOX = True  # False로 두면 시각 확대만, 판정은 완전 불변


def _ease_out_cubic(t):
    t = max(0.0, min(1.0, t))
    return 1.0 - (1.0 - t) ** 3


class KeyZoomController:

    def __init__(
        self,
        scale=ZOOM_SCALE,
        zoom_in_sec=ZOOM_IN_SEC,
        zoom_out_sec=ZOOM_OUT_SEC,
        snap_margin_px=SNAP_MARGIN_PX,
        affects_hitbox=ZOOM_AFFECTS_HITBOX,
    ):
        self.scale = scale
        self.zoom_in_sec = zoom_in_sec
        self.zoom_out_sec = zoom_out_sec
        self.snap_margin_px = snap_margin_px
        self.affects_hitbox = affects_hitbox

        self.reset()

    def reset(self):
        # 버튼 객체는 process_key에서 매번 재생성되므로 참조 대신
        # 좌상단 좌표로 식별한다 (레이아웃은 고정, 라벨만 바뀜).
        self.target_pos = None
        self.target_text = None
        self._progress = 0.0
        self._last_now = None

    # ── 매 프레임 갱신 ───────────────────────────────────────

    def update(self, fixation_state, buttonList, now=None):
        """
        Args:
            fixation_state: FixationDetector.update()의 반환값 (None 허용)
            buttonList: 현재 화면의 버튼 리스트
        """

        if now is None:
            now = time.time()

        if self._last_now is None:
            dt = 0.0
        else:
            dt = max(0.0, now - self._last_now)

        self._last_now = now

        target = None

        if (
            fixation_state is not None
            and fixation_state.fixated
            and fixation_state.center is not None
        ):
            target = self._pick_button(
                fixation_state.center,
                buttonList
            )

        if target is not None:

            pos = (int(target.pos[0]), int(target.pos[1]))

            if pos != self.target_pos:
                # 다른 키로 고정이 옮겨가면 애니메이션을 새로 1회 재생
                self.target_pos = pos
                self._progress = 0.0

            self.target_text = target.text

            if self.zoom_in_sec <= 0:
                self._progress = 1.0
            else:
                self._progress = min(
                    1.0,
                    self._progress + dt / self.zoom_in_sec
                )

        else:

            if self.zoom_out_sec <= 0:
                self._progress = 0.0
            else:
                self._progress = max(
                    0.0,
                    self._progress - dt / self.zoom_out_sec
                )

            if self._progress <= 0.0:
                self.target_pos = None
                self.target_text = None

    # ── 조회 API ─────────────────────────────────────────────

    def get_scale(self, button):
        """이 버튼의 현재 렌더링 배율 (1.0 ~ scale)."""

        if self.target_pos is None or self._progress <= 0.0:
            return 1.0

        if (int(button.pos[0]), int(button.pos[1])) != self.target_pos:
            return 1.0

        return 1.0 + (self.scale - 1.0) * _ease_out_cubic(self._progress)

    def get_hit_scale(self, button):
        """이 버튼의 판정(히트박스) 배율. affects_hitbox=False면 항상 1.0."""

        if not self.affects_hitbox:
            return 1.0

        return self.get_scale(button)

    def is_zoomed(self, button):
        return self.get_scale(button) > 1.001

    def get_rect(self, button):
        """확대 반영된 (x, y, w, h). 중심을 유지한 채 확장."""

        x, y = button.pos
        w, h = button.size

        s = self.get_scale(button)

        if s <= 1.0:
            return int(x), int(y), int(w), int(h)

        nw = int(w * s)
        nh = int(h * s)

        nx = int(x - (nw - w) // 2)
        ny = int(y - (nh - h) // 2)

        return nx, ny, nw, nh

    # ── 내부: 고정 좌표에 해당하는 키 찾기 ───────────────────

    def _pick_button(self, center, buttonList):

        px, py = center

        best = None
        best_dist = float("inf")

        for button in buttonList:

            bx, by = button.pos
            bw, bh = button.size

            # 사각형까지의 거리 (내부면 0)
            dx = max(bx - px, 0.0, px - (bx + bw))
            dy = max(by - py, 0.0, py - (by + bh))

            dist = math.hypot(dx, dy)

            if dist < best_dist:
                best_dist = dist
                best = button

        if best is not None and best_dist <= self.snap_margin_px:
            return best

        return None