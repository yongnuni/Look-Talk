from src.tracking.mappers.factory import (
    MODE_CALIBRATED,
    MODE_NO_CALIBRATION,
    AVAILABLE_MODES,
    DEFAULT_NO_CALIBRATION_STRATEGY,
)
from src.tracking.mappers.strategies import available_strategies
from src.keyboard import KEYBOARD_LAYOUT_QWERTY, KEYBOARD_LAYOUT_CHEONJIIN


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Look-Talk Eye Keyboard")

    parser.add_argument(
        "--gaze-mode",
        dest="gaze_mode",
        choices=list(AVAILABLE_MODES),
        default=MODE_CALIBRATED,
        help=(
            "시선 매핑 모드. 'calibrated'(기본값)는 기존 16점 캘리브레이션을 "
            "사용하고, 'no_calibration'은 캘리브레이션 화면 없이 바로 "
            "키보드로 진입한다."
        ),
    )

    parser.add_argument(
        "--strategy",
        dest="strategy",
        default=None,
        help=(
            "no_calibration 모드에서 사용할 strategy 이름. 생략하면 "
            f"기본값('{DEFAULT_NO_CALIBRATION_STRATEGY}')을 사용한다. "
            f"사용 가능한 strategy: {', '.join(available_strategies()) or '(없음)'}"
        ),
    )

    parser.add_argument(
        "--keyboard-layout",
        dest="keyboard_layout",
        choices=[
            KEYBOARD_LAYOUT_QWERTY,
            KEYBOARD_LAYOUT_CHEONJIIN,
        ],
        default=KEYBOARD_LAYOUT_QWERTY,
        help=(
            "키보드 배열. 'qwerty'는 기존 쿼티 배열이며, "
            "'cheonjiin'은 천지인 3×4 배열을 사용한다."
        ),
    )

    parser.add_argument(
        "--user-id",
        dest="user_id",
        default="yejin",
        help=(
            "sessions.csv 등에 기록될 참가자 식별자. 팀원별로 실행 결과를 "
            "구분하려면 실행 시 지정한다(기본값은 기존 동작과 동일하게 "
            "'yejin' 고정값을 유지)."
        ),
    )

    parser.add_argument(
        "--condition-label",
        dest="condition_label",
        default="",
        help=(
            "sessions.csv에 기록될 실험 조건 라벨(자유 문자열, 기본값 빈 "
            "문자열). config_hash는 파라미터만 포착하므로, 코드를 바꾸고 "
            "파라미터가 그대로면 두 조건이 같은 해시로 묶인다 — 'baseline' / "
            "'new-smoothing'처럼 실험자가 직접 구분하려면 지정한다."
        ),
    )

    args = parser.parse_args()

    if args.gaze_mode == MODE_NO_CALIBRATION:
        name = args.strategy or DEFAULT_NO_CALIBRATION_STRATEGY
        if name not in available_strategies():
            available = ", ".join(available_strategies()) or "(없음)"
            parser.error(
                f"'{name}'은(는) 등록된 no_calibration strategy가 아닙니다. "
                f"사용 가능한 strategy: {available}"
            )

    return (
        args.gaze_mode,
        args.strategy,
        args.keyboard_layout,
        args.user_id,
        args.condition_label
    )
