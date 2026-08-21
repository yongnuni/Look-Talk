"""고정(fixation) 감지 기반 히트박스 확장 — I-VT(속도) + I-DT(분산) 하이브리드.

이 모듈은 시선 추적 파이프라인 **바깥**에 있다. GazePipeline이 확정한 최종
좌표를 읽기만 하고, 좌표를 고치거나 되돌려 주지 않는다. GazePipeline 내부의
fixation_center/fixation_count는 커서 표시를 안정화하기 위한 값이라 여기서
쓰지 않고, 판정 전용 상태를 독자적으로 유지한다. 따라서 이 모듈을 꺼도
(FIXATION_HITBOX_ENABLED=False) 시선 좌표와 기존 판정은 한 픽셀도 달라지지
않는다.

하는 일은 하나다: "사용자가 목표를 이미 찾아서 응시하기 시작했는가"를 스스로
판정하고, 그 순간의 키 하나만 히트박스를 넓혀 준다.

- 탐색(saccade) 중에는 어떤 확장도 걸지 않는다. 탐색 단계에서 판정 영역이
  흔들리면 전체 배치 파악을 방해한다는 선행 연구 결과를 그대로 따른 것이다.
- 화면에 그려지는 키 크기·위치는 바꾸지 않는다. 확장은 판정 영역에만
  적용되고 키캡 레이아웃은 정적으로 고정된다(모션에 의한 주의 포착·공간기억
  붕괴 리스크 회피). 확장 때문에 주변 키가 밀리거나 줄어드는 일은 없다.
- 확장 영역은 인접 키를 최대 1/3까지 덮으며, 겹치는 구간에서는 확장된 쪽이
  이긴다. 인접 키의 나머지 2/3는 그대로 자기 영역이므로 통째로 가려지는
  키는 생기지 않는다.

세 트리거(시선 지속 시간 / 깜빡임 / 입 개폐)는 모두 DwellController.update()
에서 hover 키를 받아 가므로, 이 레이어를 그 한 곳에 물려 두면 세 트리거 모두에
동시에 적용된다.

키보드 쪽 자료구조는 읽기만 한다 — Button.rect(KeyRect)의 좌표를 참조할 뿐,
src/keyboard.py의 어떤 함수·클래스도 수정하지 않는다.
"""

import math
import time
from dataclasses import dataclass
from typing import Optional

from src.config import (
    FIXATION_DISPERSION_DEG,
    FIXATION_HITBOX_ENABLED,
    FIXATION_HITBOX_EXPAND_RATIO,
    FIXATION_HITBOX_MAX_OVERLAP_RATIO,
    FIXATION_MAX_GAP_SEC,
    FIXATION_MIN_DURATION_SEC,
    FIXATION_RELEASE_DEG,
    FIXATION_VELOCITY_DEG_PER_SEC,
    FIXATION_VISUAL_ANIM_SEC,
    FIXATION_VISUAL_EXPAND_ENABLED,
    FIXATION_VISUAL_EXPAND_RATIO,
    PX_PER_DEG,
)


# =========================================================
# 고정 상태
# =========================================================

@dataclass(frozen=True)
class FixationState:
    """한 프레임의 고정 판정 결과. 표시·로깅용으로도 그대로 쓴다.

    fixation_id는 고정이 새로 시작될 때마다 증가한다. "같은 고정이 이어지는
    중인지, 도약 뒤에 새로 시작된 고정인지"를 호출부가 구분할 때 쓴다.
    """

    active: bool
    center_x: Optional[float]
    center_y: Optional[float]
    duration_sec: float
    velocity_deg_per_sec: Optional[float]
    fixation_id: int = 0


def is_valid_gaze(x, y, valid=True):
    """이번 프레임의 좌표를 고정 판정에 쓸 수 있는지.

    main.py의 tracking_valid와 동일한 의미의 음수 좌표 규칙을 함께 본다.
    """
    return (
        valid
        and x is not None
        and y is not None
        and x >= 0
        and y >= 0
    )


IDLE_STATE = FixationState(
    active=False,
    center_x=None,
    center_y=None,
    duration_sec=0.0,
    velocity_deg_per_sec=None,
    fixation_id=0,
)


# =========================================================
# I-VT + I-DT 고정 검출기
# =========================================================

class FixationDetector:
    """속도(I-VT)와 분산(I-DT)을 함께 보는 고정 검출기.

    두 기준을 모두 쓰는 이유:

    - I-VT 단독: 웹캠 노이즈로 속도가 순간적으로 튀면 고정이 끊긴다.
    - I-DT 단독: 아주 느린 표류(drift)를 고정으로 오인한다.

    그래서 "속도가 임계값 이하이고(도약이 아니고), 고정 중심에서 일정 반경
    안에 최소 지속 시간 이상 머무를 것"을 함께 요구한다.

    성립 이후에는 해제 반경(release_deg)을 조금 더 크게 잡아, 미세한 흔들림
    때문에 확장이 켜졌다 꺼졌다 하지 않게 한다.
    """

    def __init__(
        self,
        px_per_deg=PX_PER_DEG,
        velocity_deg_per_sec=FIXATION_VELOCITY_DEG_PER_SEC,
        dispersion_deg=FIXATION_DISPERSION_DEG,
        release_deg=FIXATION_RELEASE_DEG,
        min_duration_sec=FIXATION_MIN_DURATION_SEC,
        max_gap_sec=FIXATION_MAX_GAP_SEC,
    ):
        self.px_per_deg = float(px_per_deg)
        self.velocity_deg_per_sec = float(velocity_deg_per_sec)
        self.dispersion_deg = float(dispersion_deg)
        self.release_deg = float(release_deg)
        self.min_duration_sec = float(min_duration_sec)
        self.max_gap_sec = float(max_gap_sec)

        # 고정이 새로 시작될 때마다 증가. reset()으로 되돌리지 않는다 —
        # 되돌리면 "새 고정"과 "직전 고정"을 구분할 수 없게 된다.
        self.fixation_id = 0

        self.reset()

    # -----------------------------------------------------
    # 임계값의 px 환산
    # -----------------------------------------------------

    @property
    def dispersion_px(self):
        return self.dispersion_deg * self.px_per_deg

    @property
    def release_px(self):
        return self.release_deg * self.px_per_deg

    # -----------------------------------------------------
    # 상태 초기화
    # -----------------------------------------------------

    def reset(self):
        self._active = False
        self._start_ts = None
        self._last_ts = None
        self._last_point = None
        self._sum_x = 0.0
        self._sum_y = 0.0
        self._count = 0
        self._velocity = None
        self.state = IDLE_STATE

    def _begin(self, x, y, now):
        """현재 좌표를 시작점으로 새 고정 후보를 연다."""
        self._active = False
        self._start_ts = now
        self._sum_x = float(x)
        self._sum_y = float(y)
        self._count = 1
        self.fixation_id += 1

    @property
    def center(self):
        if self._count == 0:
            return None
        return (
            self._sum_x / self._count,
            self._sum_y / self._count,
        )

    # -----------------------------------------------------
    # 프레임 갱신
    # -----------------------------------------------------

    def update(self, x, y, valid=True, now=None):
        """시선 좌표 한 점을 받아 고정 상태를 갱신한다.

        Args:
            x, y: GazePipeline이 확정한 화면 좌표
            valid: 이번 프레임의 추적이 유효한지(main.py의 tracking_valid)
            now: 시각(초). 호출부와 시계를 맞추기 위해 명시적으로 받는다.

        Returns:
            FixationState
        """

        if now is None:
            now = time.monotonic()

        if not is_valid_gaze(x, y, valid):
            self.reset()
            return self.state

        gap = (
            None
            if self._last_ts is None
            else now - self._last_ts
        )

        # 루프가 끊긴 구간(모드 전환·캘리브레이션 등)은 이어 붙이지 않는다.
        # 큰 dt로 속도를 계산하면 실제로는 튄 시선이 "느린 이동"으로 보인다.
        if gap is not None and (gap < 0 or gap > self.max_gap_sec):
            self.reset()
            gap = None

        velocity = self._velocity

        if (
            gap is not None
            and gap > 0
            and self._last_point is not None
        ):
            distance = math.hypot(
                x - self._last_point[0],
                y - self._last_point[1],
            )
            velocity = distance / gap / self.px_per_deg

        saccade = (
            velocity is not None
            and velocity > self.velocity_deg_per_sec
        )

        if self._count == 0:
            self._begin(x, y, now)

        else:
            center_x, center_y = self.center

            limit = (
                self.release_px
                if self._active
                else self.dispersion_px
            )

            drifted = (
                math.hypot(x - center_x, y - center_y) > limit
            )

            if saccade or drifted:
                # 도약이 시작됐거나 응시점이 벗어났다 → 고정 종료 후 재시작
                self._begin(x, y, now)

            else:
                self._sum_x += float(x)
                self._sum_y += float(y)
                self._count += 1

        duration = now - self._start_ts

        if (
            not self._active
            and not saccade
            and duration >= self.min_duration_sec
        ):
            self._active = True

        self._last_ts = now
        self._last_point = (float(x), float(y))
        self._velocity = velocity

        center = self.center

        self.state = FixationState(
            active=self._active,
            center_x=center[0],
            center_y=center[1],
            duration_sec=duration,
            velocity_deg_per_sec=velocity,
            fixation_id=self.fixation_id,
        )

        return self.state


# =========================================================
# 확장 판정 지오메트리
# =========================================================

def expanded_bounds(
    target,
    targets,
    expand_ratio,
    radius_px=None,
    max_overlap_ratio=FIXATION_HITBOX_MAX_OVERLAP_RATIO,
):
    """확장된 판정 사각형을 (left, top, right, bottom)으로 계산한다.

    target/targets는 rect(KeyRect)를 가진 선택 대상이면 무엇이든 된다 —
    키캡 버튼과 추천(자동완성) 슬롯이 같은 규칙으로 확장된다.

    확장량은 대상 한 변 x expand_ratio다 — 0.5면 좌우로 각각 폭의 절반이 붙어
    판정 폭이 "폭 + 폭"이 된다. 고정 px가 아니라 대상 크기에서 뽑기 때문에
    해상도나 레이아웃(쿼티/천지인)이 달라져도 UI와 같은 비율을 유지하고,
    키보다 훨씬 넓은 추천 슬롯에도 그대로 적용된다. radius_px(개인별 시선
    오차 반경 등)가 주어지면 그보다 작아지지 않게 올린다.

    확장은 이웃을 덮어도 되지만, 어떤 대상도 그 대상의 max_overlap_ratio
    (기본 1/3)보다 깊게 침범하지 않는다. 한도는 변마다 따로 계산한다 —
    오른쪽에는 좁은 기능키가, 왼쪽에는 스페이스바가, 위에는 추천 슬롯이 올 수
    있어서 하나로 묶으면 좁은 쪽 기준으로 네 변이 전부 깎이기 때문이다.

    rect는 읽기만 한다. 화면에 그려지는 위치와 크기는 이 계산에 전혀
    영향받지 않는다 — 레이아웃은 정적으로 고정되고, 넓어지는 것은 판정
    영역뿐이다.
    """

    rect = target.rect

    desired_x = rect.width * expand_ratio
    desired_y = rect.height * expand_ratio

    if radius_px is not None:
        desired_x = max(desired_x, radius_px)
        desired_y = max(desired_y, radius_px)

    left = right = desired_x
    top = bottom = desired_y

    for other in targets:
        if other is target:
            continue

        other_rect = other.rect

        # 가로 방향: 세로 구간이 겹치는 대상만 좌우 이웃으로 본다.
        if (
            other_rect.y < rect.bottom
            and other_rect.bottom > rect.y
        ):
            if other_rect.x >= rect.right:
                right = min(
                    right,
                    (other_rect.x - rect.right)
                    + other_rect.width * max_overlap_ratio,
                )

            elif other_rect.right <= rect.x:
                left = min(
                    left,
                    (rect.x - other_rect.right)
                    + other_rect.width * max_overlap_ratio,
                )

        # 세로 방향: 가로 구간이 겹치는 대상만 위아래 이웃으로 본다.
        if (
            other_rect.x < rect.right
            and other_rect.right > rect.x
        ):
            if other_rect.y >= rect.bottom:
                bottom = min(
                    bottom,
                    (other_rect.y - rect.bottom)
                    + other_rect.height * max_overlap_ratio,
                )

            elif other_rect.bottom <= rect.y:
                top = min(
                    top,
                    (rect.y - other_rect.bottom)
                    + other_rect.height * max_overlap_ratio,
                )

    return (
        rect.x - max(0.0, left),
        rect.y - max(0.0, top),
        rect.right + max(0.0, right),
        rect.bottom + max(0.0, bottom),
    )


def lerp_bounds(start, end, t):
    """두 사각형 사이를 t(0~1)로 보간한다. 확대 애니메이션용."""

    return tuple(
        start[i] + (end[i] - start[i]) * t
        for i in range(4)
    )


def ease_out(t):
    """짧은 1회성 확대에 쓰는 ease-out 곡선(cubic)."""

    t = min(1.0, max(0.0, t))
    return 1.0 - (1.0 - t) ** 3


def bounds_contain(bounds, point_x, point_y):
    """확장 사각형 안에 있는지 검사한다.

    경계 규칙은 KeyRect.contains와 같은 반열린 구간을 따른다.
    """

    left, top, right, bottom = bounds

    return (
        left <= point_x < right
        and top <= point_y < bottom
    )


# =========================================================
# 고정 → 확장 대상 키 결정
# =========================================================

class FixationHitbox:
    """고정이 시작된 키 하나의 히트박스만 넓혀 주는 판정 보조 레이어.

    동작 순서
    1. 매 프레임 FixationDetector로 고정 여부를 독자 판정한다.
    2. 고정이 성립하면 그 시점의 고정 중심에 해당하는 키를 확장 대상(앵커)
       으로 잡는다. 탐색(도약) 중에는 어떤 확장도 걸지 않는다.
    3. 확장 영역은 인접 키를 최대 1/3까지 덮으며, 겹치는 구간에서는 확장된
       쪽이 이긴다. 확장은 "이미 목표를 찾아 응시 중인 키"에만 걸리므로 그
       구간에 떨어진 좌표는 그 키를 노린 것으로 본다. 인접 키의 나머지 2/3는
       그대로 자기 영역이라 완전히 가려지는 키는 없다.
    4. 걸린 확장은 시선이 확장 영역 안에 머무는 동안 유지된다. 겹침 구간에
       잠깐 들어가는 것만으로는 확장이 넘어가지 않고, 다른 키에 새 고정이
       성립해야 그쪽으로 옮겨 간다.

    화면 렌더링은 이 레이어를 참조하지 않는다 — 키캡의 위치와 크기는 정적으로
    고정되고, 확장은 판정 영역에만 존재한다.

    앵커를 객체가 아니라 **좌표**로 기억하는 이유: process_key()가
    Shift/한영 전환 때 buttonList를, 추천 갱신 때 슬롯 목록을 통째로 새로
    만들기 때문에, 객체를 들고 있으면 화면에 없는 대상을 가리키게 된다.
    좌표로 들고 있다가 매 프레임 현재 목록에서 다시 찾으면 그 문제가 없다.
    """

    def __init__(
        self,
        enabled=FIXATION_HITBOX_ENABLED,
        detector=None,
        expand_ratio=FIXATION_HITBOX_EXPAND_RATIO,
        personal_radius_px=None,
        max_overlap_ratio=FIXATION_HITBOX_MAX_OVERLAP_RATIO,
        visual_enabled=FIXATION_VISUAL_EXPAND_ENABLED,
        visual_expand_ratio=FIXATION_VISUAL_EXPAND_RATIO,
        visual_anim_sec=FIXATION_VISUAL_ANIM_SEC,
    ):
        self.enabled = bool(enabled)
        self.detector = detector or FixationDetector()
        self.expand_ratio = float(expand_ratio)
        self.max_overlap_ratio = float(max_overlap_ratio)

        # 시각 확대 — 판정과 분리된 표시 전용 값이다. 꺼도 히트박스 확장은
        # 그대로 동작한다.
        self.visual_enabled = bool(visual_enabled)
        self.visual_expand_ratio = float(visual_expand_ratio)
        self.visual_anim_sec = float(visual_anim_sec)

        # 개인별 시선 오차 반경(px). 초기 캘리브레이션에서 산출한 95퍼센타일
        # 같은 값을 넣으면 그 반경 이상으로 확장이 보장된다. None이면
        # 키 크기 비율만 사용한다.
        self.personal_radius_px = personal_radius_px

        self._anchor_center = None
        self._anchor_bounds = None
        self._anchor_visual_bounds = None
        self._anchor_started_at = None
        self._anchored_fixation_id = None
        self.anchor_target = None
        self.state = IDLE_STATE

    def reset(self):
        self.detector.reset()
        self._clear_anchor()
        self.state = IDLE_STATE

    def _clear_anchor(self):
        self._anchor_center = None
        self._anchor_bounds = None
        self._anchor_visual_bounds = None
        self._anchor_started_at = None
        self.anchor_target = None

    @property
    def active(self):
        """지금 확장이 실제로 걸려 있는지."""
        return self.anchor_target is not None

    @property
    def anchor_key(self):
        return (
            self.anchor_target.text
            if self.anchor_target is not None
            else None
        )

    def bounds_of(self, target, targets):
        """이 레이어의 설정으로 계산한 대상의 확장 사각형."""

        return expanded_bounds(
            target,
            targets,
            self.expand_ratio,
            self.personal_radius_px,
            self.max_overlap_ratio,
        )

    # -----------------------------------------------------
    # 프레임 갱신
    # -----------------------------------------------------

    def update(self, gaze_x, gaze_y, targets, valid=True, now=None):
        """고정 여부를 갱신하고 확장 대상 키를 정한다."""

        if not self.enabled:
            self._clear_anchor()
            self._anchored_fixation_id = None
            self.state = IDLE_STATE
            return self.state

        self.state = self.detector.update(
            gaze_x,
            gaze_y,
            valid=valid,
            now=now,
        )

        if not is_valid_gaze(gaze_x, gaze_y, valid):
            self._clear_anchor()
            self._anchored_fixation_id = None
            return self.state

        anchor_target = None

        # ── 새 고정이 성립하면 확장 대상을 그쪽으로 옮긴다 ──
        # 도약이 끝난 시점 = 사용자가 목표를 이미 찾은 시점.
        if (
            self.state.active
            and self._anchored_fixation_id != self.state.fixation_id
        ):
            anchor_target = self._find_anchor_target(
                targets,
                self.state.center_x,
                self.state.center_y,
            )

            if anchor_target is not None:
                self._anchor_center = (
                    self.state.center_x,
                    self.state.center_y,
                )
                self._anchored_fixation_id = self.state.fixation_id

        # ── 새 고정이 없으면 걸려 있던 확장을 이어 간다 ──
        # 시선이 확장 영역을 벗어날 때 내린다. 각도 기준 고정 반경만으로
        # 판단하면(1.5°가 키 반폭보다 작을 수 있다) 시선이 키 가장자리로
        # 흐르는 순간 — 확장이 가장 필요한 순간 — 풀려 버린다.
        if anchor_target is None and self._anchor_center is not None:
            held_target = self._find_anchor_target(
                targets,
                *self._anchor_center
            )

            if held_target is not None and bounds_contain(
                self.bounds_of(held_target, targets),
                gaze_x,
                gaze_y,
            ):
                anchor_target = held_target

            else:
                self._anchor_center = None

        previous_target = self.anchor_target
        self.anchor_target = anchor_target

        self._anchor_bounds = (
            self.bounds_of(anchor_target, targets)
            if anchor_target is not None
            else None
        )

        self._update_visual_bounds(
            previous_target,
            anchor_target,
            targets,
            now,
        )

        return self.state

    def _update_visual_bounds(
        self,
        previous_target,
        anchor_target,
        targets,
        now,
    ):
        """확대 애니메이션을 한 프레임 진행시킨다.

        판정과 분리된 표시 전용 계산이다. 확대는 키가 바뀔 때마다 한 번만
        재생되고 되돌아가지 않는다(반복 pulsing은 시각 유발성 멀미 리스크).
        """

        if anchor_target is None or not self.visual_enabled:
            self._anchor_visual_bounds = None
            self._anchor_started_at = None
            return

        if now is None:
            now = time.monotonic()

        anchor_changed = (
            previous_target is None
            or previous_target is not anchor_target
        )

        if anchor_changed or self._anchor_started_at is None:
            self._anchor_started_at = now

        rect = anchor_target.rect

        target = expanded_bounds(
            anchor_target,
            targets,
            self.visual_expand_ratio,
            self.personal_radius_px,
            self.max_overlap_ratio,
        )

        if self.visual_anim_sec <= 0:
            progress = 1.0
        else:
            progress = ease_out(
                (now - self._anchor_started_at) / self.visual_anim_sec
            )

        self._anchor_visual_bounds = lerp_bounds(
            (rect.x, rect.y, rect.right, rect.bottom),
            target,
            progress,
        )

    def anchor_visual_rect(self):
        """확대해서 그릴 키캡 사각형 (x, y, right, bottom). 없으면 None.

        렌더링에서만 쓴다 — 판정은 언제나 anchor_debug_rect()/hit_test()가
        쓰는 히트박스 쪽이며, 그쪽이 이 사각형보다 넓다(암묵 확장).
        """

        if self._anchor_visual_bounds is None:
            return None

        return tuple(
            int(round(value))
            for value in self._anchor_visual_bounds
        )

    def _find_anchor_target(self, targets, center_x, center_y):
        """고정 중심에 해당하는 대상을 현재 목록에서 찾는다.

        중심이 대상 사이 여백에 떨어진 경우가 이 기능이 실제로 필요한
        상황이다. 그때는 확장 영역이 중심을 포함하는 대상 중 중심에서 가장
        가까운 것을 고른다. 확장 영역 밖이면 아무것도 고르지 않는다
        (무리한 보정 금지).
        """

        for target in targets:
            if target.rect.contains(center_x, center_y):
                return target

        nearest_target = None
        nearest_distance = None

        for target in targets:
            if not bounds_contain(
                self.bounds_of(target, targets),
                center_x,
                center_y,
            ):
                continue

            target_center_x, target_center_y = target.rect.center

            distance = math.hypot(
                center_x - target_center_x,
                center_y - target_center_y,
            )

            if (
                nearest_distance is None
                or distance < nearest_distance
            ):
                nearest_target = target
                nearest_distance = distance

        return nearest_target

    # -----------------------------------------------------
    # 확장을 반영한 hit-test
    # -----------------------------------------------------

    def hit_test(self, targets, point_x, point_y):
        """확장 영역이 최우선, 그 밖에서는 기존 판정 그대로.

        겹치는 구간에서는 확장된 대상이 이긴다. 확장 중이 아니면 목록 순서대로
        실제 사각형을 검사하는 기존 판정과 결과가 완전히 같다(그래서 호출부는
        우선순위가 높은 대상을 목록 앞에 둔다 — 예: 추천 슬롯 먼저, 키 나중).
        """

        if (
            self.anchor_target is not None
            and self._anchor_bounds is not None
            and bounds_contain(self._anchor_bounds, point_x, point_y)
        ):
            return self.anchor_target

        for target in targets:
            if target.rect.contains(point_x, point_y):
                return target

        return None

    def anchor_debug_rect(self):
        """디버그 오버레이용 확장 사각형 (x, y, right, bottom). 없으면 None."""

        if self._anchor_bounds is None:
            return None

        return tuple(int(value) for value in self._anchor_bounds)
