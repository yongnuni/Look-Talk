"""
식별자 발급 모듈.

세 함수로 나눈 이유: 호출부에서 어떤 종류의 id를 발급받는지 함수 이름만으로
읽히게 하기 위함이다. 각 id의 수명은 다음과 같다.

    new_run_id()    앱 실행 1회. main() 진입 시 1회 발급되어 이후 모든 수집기가
                     공유하게 될 값이다(1단계에서 배선). 앱을 껐다 켜면 새 값이 된다.
    new_calib_id()  캘리브레이션 1회. 'r' 키로 재캘리브레이션할 때마다 새로 발급된다
                     (기존 Calibrator.calib_id의 후신).
    new_test_id()   9점 테스트 1회. 't' 키를 누를 때마다 새로 발급된다
                     (기존 MetricsCollector.session_id의 후신).

전부 str(uuid.uuid4())이며 접두어를 붙이지 않는다 — 기존 CSV에 이미 접두어 없는
uuid 문자열이 쌓여 있어, 형식을 그대로 통일하기 위함이다.
"""

import uuid


def new_run_id():
    """앱 실행 1회를 식별하는 uuid4 문자열을 발급한다."""
    return str(uuid.uuid4())


def new_calib_id():
    """캘리브레이션 1회를 식별하는 uuid4 문자열을 발급한다."""
    return str(uuid.uuid4())


def new_test_id():
    """9점 테스트 1회를 식별하는 uuid4 문자열을 발급한다."""
    return str(uuid.uuid4())
