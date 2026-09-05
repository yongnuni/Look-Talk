from src.tracking.mappers.factory import (
    MODE_CALIBRATED,
    MODE_NO_CALIBRATION,
    AVAILABLE_MODES,
    DEFAULT_NO_CALIBRATION_STRATEGY,
)
from src.tracking.mappers.strategies import available_strategies
from src.keyboard import KEYBOARD_LAYOUT_QWERTY, KEYBOARD_LAYOUT_CHEONJIIN
from tests.test_sentences import TEST_SENTENCE_GROUPS


def _resolve_test_sentence(raw_value, parser):
    """'<인덱스>' 또는 '<그룹>:<인덱스>' 형식을 실제 문장 문자열로 바꾼다.

    그룹을 생략하면 'legacy'로 취급한다(TEST_SENTENCE_GROUPS['legacy']==
    TEST_SENTENCES). 잘못된 형식/그룹/인덱스는 parser.error로 즉시 종료한다.
    """

    if raw_value is None:
        return None

    if ":" in raw_value:
        group_name, _, index_text = raw_value.partition(":")
    else:
        group_name, index_text = "legacy", raw_value

    if group_name not in TEST_SENTENCE_GROUPS:
        available = ", ".join(sorted(TEST_SENTENCE_GROUPS))
        parser.error(
            f"--test-sentence의 그룹 '{group_name}'이(가) 존재하지 않습니다. "
            f"사용 가능한 그룹: {available}"
        )

    try:
        index = int(index_text)
    except ValueError:
        parser.error(
            f"--test-sentence의 인덱스 '{index_text}'는 정수가 아닙니다."
        )

    sentences = TEST_SENTENCE_GROUPS[group_name]
    if index < 0 or index >= len(sentences):
        parser.error(
            f"--test-sentence의 인덱스 {index}가 그룹 '{group_name}' 범위"
            f"(0~{len(sentences) - 1})를 벗어났습니다."
        )

    return sentences[index]


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

    parser.add_argument(
        "--calib-points",
        dest="calib_points",
        type=int,
        choices=[9, 16],
        default=16,
        help=(
            "calibrated 모드에서 사용할 캘리브레이션 점 개수. "
            "9 또는 16을 선택할 수 있으며 기본값은 16이다."
        ),
    )

    parser.add_argument(
        "--performance-flow",
        dest="performance_flow",
        action="store_true",
        help=(
            "16점 시선 캘리브레이션부터 gaze/blink/mouth 실입력 테스트와 "
            "결과 대시보드까지 통합 성능 플로우로 실행한다."
        ),
    )

    parser.add_argument(
        "--autocomplete",
        dest="autocomplete",
        choices=["on", "off"],
        default="on",
        help=(
            "초성 자동완성 사용 여부. 기본값 'on'. 'off'로 끄면 이 플래그가 "
            "config_snapshot에 등재되지 않아 config_hash로 조건이 구분되지 "
            "않으므로 --condition-label을 함께 지정해야 한다."
        ),
    )

    parser.add_argument(
        "--test-sentence",
        dest="test_sentence",
        default=None,
        help=(
            "문장 테스트 목표 문장을 '<인덱스>' 또는 '<그룹>:<인덱스>' 형식으로 "
            "결정적으로 지정한다(그룹: legacy/hit/miss, tests/test_sentences.py "
            "TEST_SENTENCE_GROUPS 참고). 생략하면 기존과 동일하게 "
            "TEST_SENTENCES 중에서 무작위로 고른다."
        ),
    )

    args = parser.parse_args()

    if args.performance_flow and (
        args.gaze_mode != MODE_CALIBRATED or args.calib_points != 16
    ):
        parser.error(
            "--performance-flow는 --gaze-mode calibrated 및 "
            "--calib-points 16에서만 사용할 수 있습니다."
        )

    if args.gaze_mode == MODE_NO_CALIBRATION:
        name = args.strategy or DEFAULT_NO_CALIBRATION_STRATEGY
        if name not in available_strategies():
            available = ", ".join(available_strategies()) or "(없음)"
            parser.error(
                f"'{name}'은(는) 등록된 no_calibration strategy가 아닙니다. "
                f"사용 가능한 strategy: {available}"
            )

    if args.autocomplete == "off" and not args.condition_label:
        parser.error("--autocomplete off 실행은 --condition-label 필수")

    resolved_test_sentence = _resolve_test_sentence(args.test_sentence, parser)

    return (
        args.gaze_mode,
        args.strategy,
        args.keyboard_layout,
        args.user_id,
        args.condition_label,
        args.calib_points,
        args.performance_flow,
        args.autocomplete,
        resolved_test_sentence,
    )
