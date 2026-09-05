"""BlinkSelectionController 동작 검증.

looktalk-frontend의 src/features/multimodalInput/BlinkController.test.ts와 같은
시나리오를 유지한다. 프론트엔드는 EAR을 직접 넣지만 여기서는 BlinkDetector가
이미 분류한 (blink_event, is_closed)를 받으므로, 그 두 값을 프레임 단위로
흘려보내며 같은 선택 결과가 나오는지 확인한다.
"""

from src.tracking.blink import (
    BlinkEvent,
    BlinkKind,
    BlinkSelectionController,
)


BASE = 100.0
LOCK_SEC = 0.25


def _event(kind, duration=0.4):
    return BlinkEvent(kind=kind, duration=duration, timestamp=1.0)


def _lock_on(controller, target, start=BASE):
    """LOCK_SEC 이상 같은 타깃을 응시해 잠금을 성립시킨다."""

    controller.update(target, None, is_closed=False, now=start)
    controller.update(target, None, is_closed=False, now=start + LOCK_SEC)


def test_intentional_blink_selects_target_locked_before_eye_closure():
    controller = BlinkSelectionController()
    target = object()

    _lock_on(controller, target)
    assert controller.locked_target is target

    assert (
        controller.update(None, None, is_closed=True, now=BASE + 0.26) is None
    )
    assert controller.locked_target is target

    assert (
        controller.update(
            target,
            _event(BlinkKind.INTENTIONAL),
            is_closed=False,
            now=BASE + 0.57,
        )
        is target
    )


def test_blink_selects_locked_target_even_if_gaze_moved_on_reopen():
    """눈을 뜬 프레임의 hover로 재검증하지 않는다.

    눈을 다시 뜬 첫 프레임은 GazePipeline이 막 재개한 좌표라 hover가 옆 키로
    튀거나 아예 None이 되기 쉽다. 그 프레임으로 재검증하면 사용자가 의도적으로
    깜빡였는데도 입력이 조용히 사라진다.
    """

    first = object()
    second = object()

    drifted = BlinkSelectionController()
    _lock_on(drifted, first)
    drifted.update(None, None, is_closed=True, now=BASE + 0.26)

    assert (
        drifted.update(
            second,
            _event(BlinkKind.INTENTIONAL),
            is_closed=False,
            now=BASE + 0.57,
        )
        is first
    )

    invalid = BlinkSelectionController()
    _lock_on(invalid, first)
    invalid.update(None, None, is_closed=True, now=BASE + 0.26)

    assert (
        invalid.update(
            None,
            _event(BlinkKind.INTENTIONAL),
            is_closed=False,
            now=BASE + 0.57,
        )
        is first
    )


def test_natural_or_long_closure_never_selects_a_target():
    controller = BlinkSelectionController()
    target = object()

    _lock_on(controller, target)
    controller.update(None, None, is_closed=True, now=BASE + 0.26)
    assert (
        controller.update(
            target, _event(BlinkKind.NATURAL), False, now=BASE + 0.46
        )
        is None
    )

    _lock_on(controller, target, start=BASE + 0.50)
    controller.update(None, None, is_closed=True, now=BASE + 0.80)
    assert (
        controller.update(
            None, _event(BlinkKind.LONG_CLOSURE, 1.3), True, now=BASE + 2.11
        )
        is None
    )
    assert controller.locked_target is None
    assert controller.update(target, None, False, now=BASE + 2.20) is None


def test_short_gaze_does_not_lock_a_target():
    """LOCK_SEC 미만만 응시했으면 의도 깜빡임에도 선택하지 않는다."""

    controller = BlinkSelectionController()
    target = object()

    controller.update(target, None, is_closed=False, now=BASE)
    controller.update(target, None, is_closed=False, now=BASE + 0.10)
    assert controller.locked_target is None

    controller.update(None, None, is_closed=True, now=BASE + 0.11)
    assert (
        controller.update(
            target,
            _event(BlinkKind.INTENTIONAL),
            is_closed=False,
            now=BASE + 0.42,
        )
        is None
    )


def test_lock_moves_to_another_target_after_lock_time():
    controller = BlinkSelectionController()
    first = object()
    second = object()

    _lock_on(controller, first)

    controller.update(second, None, is_closed=False, now=BASE + 0.30)
    controller.update(second, None, is_closed=False, now=BASE + 0.55)
    assert controller.locked_target is second

    controller.update(None, None, is_closed=True, now=BASE + 0.56)
    assert (
        controller.update(
            second,
            _event(BlinkKind.INTENTIONAL),
            is_closed=False,
            now=BASE + 0.87,
        )
        is second
    )


def test_brief_hover_gap_keeps_an_established_lock():
    """시선이 잠깐 키 사이를 지나가도 이미 성립한 잠금은 유지한다."""

    controller = BlinkSelectionController()
    target = object()

    _lock_on(controller, target)
    controller.update(None, None, is_closed=False, now=BASE + 0.30)

    assert controller.locked_target is target


def test_lock_is_released_after_a_selection():
    controller = BlinkSelectionController()
    target = object()

    _lock_on(controller, target)
    controller.update(None, None, is_closed=True, now=BASE + 0.26)
    assert (
        controller.update(
            target,
            _event(BlinkKind.INTENTIONAL),
            is_closed=False,
            now=BASE + 0.57,
        )
        is target
    )
    assert controller.locked_target is None

    # 곧바로 다시 감으면 잠금이 아직 재성립하지 않아 선택되지 않는다.
    controller.update(None, None, is_closed=True, now=BASE + 0.70)
    assert (
        controller.update(
            target,
            _event(BlinkKind.INTENTIONAL),
            is_closed=False,
            now=BASE + 1.01,
        )
        is None
    )


def test_reset_requires_an_open_frame_before_a_new_blink():
    """reset 뒤 이미 감겨 있던 눈은 먼저 다시 뜨기 전까지 제스처로 세지 않는다."""

    controller = BlinkSelectionController()
    target = object()

    _lock_on(controller, target)
    controller.update(None, None, is_closed=True, now=BASE + 0.26)
    controller.reset()

    controller.update(None, None, is_closed=True, now=BASE + 0.35)
    assert (
        controller.update(
            target,
            _event(BlinkKind.INTENTIONAL),
            is_closed=False,
            now=BASE + 0.75,
        )
        is None
    )

    # 다시 뜬 뒤 LOCK_SEC을 응시해 잠근 다음에야 새 깜빡임이 성립한다.
    _lock_on(controller, target, start=BASE + 0.80)
    controller.update(None, None, is_closed=True, now=BASE + 1.06)
    assert (
        controller.update(
            target,
            _event(BlinkKind.INTENTIONAL),
            is_closed=False,
            now=BASE + 1.37,
        )
        is target
    )
