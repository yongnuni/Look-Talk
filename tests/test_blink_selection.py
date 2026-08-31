from src.tracking.blink import (
    BlinkEvent,
    BlinkKind,
    BlinkSelectionController,
)


def _event(kind, duration=0.4):
    return BlinkEvent(kind=kind, duration=duration, timestamp=1.0)


def test_intentional_blink_selects_target_locked_before_eye_closure():
    controller = BlinkSelectionController()
    target = object()

    assert controller.update(target, None, is_closed=False) is None
    assert controller.update(None, None, is_closed=True) is None
    assert controller.locked_target is target
    assert (
        controller.update(target, _event(BlinkKind.INTENTIONAL), is_closed=False)
        is target
    )


def test_blink_does_not_select_when_reopened_on_a_different_target():
    controller = BlinkSelectionController()
    first = object()
    second = object()

    controller.update(first, None, is_closed=False)
    controller.update(None, None, is_closed=True)

    assert (
        controller.update(second, _event(BlinkKind.INTENTIONAL), is_closed=False)
        is None
    )


def test_natural_or_long_closure_never_selects_a_target():
    controller = BlinkSelectionController()
    target = object()

    controller.update(target, None, is_closed=False)
    controller.update(None, None, is_closed=True)
    assert controller.update(target, _event(BlinkKind.NATURAL), False) is None

    controller.update(target, None, is_closed=False)
    controller.update(None, None, is_closed=True)
    assert controller.update(None, _event(BlinkKind.LONG_CLOSURE, 0.9), True) is None
    assert controller.update(target, None, False) is None
