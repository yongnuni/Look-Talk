"""
프로세스 전역 단조 시계(monotonic clock) 어댑터.

배경: 기존 코드베이스는 time.time() / time.monotonic() / datetime.now(timezone.utc) /
datetime.now()(naive) 4종의 시계를 혼용해 로그 간 시간 정렬이 불가능하다
(docs/current_state_report.md 2-3절 참고). 이 모듈이 앞으로 모든 로그 타임스탬프의
단일 출처가 된다.

설계:
    - init() 시점에 t0_monotonic(time.monotonic() 기준점)과 t0_utc(그 순간의
      tz-aware UTC 시각)를 쌍으로 고정한다.
    - now_ms()는 t0_monotonic 대비 경과 시간(ms)만 반환한다 — monotonic 기반이므로
      시스템 시각이 바뀌어도(NTP 보정, 서머타임 등) 영향받지 않는다.
    - monotonic 값 자체는 절대시각이 아니고 프로세스 재시작 시 리셋된다. 특정
      ts_ms를 실제 벽시계 시각으로 복원하려면 t0_utc_iso() 시각에 ts_ms(밀리초)를
      더해야 한다 — t0 쌍을 함께 저장해 두는 이유가 이것이다. 세션(sessions) 행에는
      t0_utc_iso()를 세션당 1회만 기록하고, 프레임 단위 로그는 now_ms()만 쓰면 된다.
"""

import time
from datetime import datetime, timezone

_t0_monotonic = None
_t0_utc = None


def init():
    """프로세스 시작 시 1회 호출한다. t0_monotonic과 t0_utc를 쌍으로 고정한다.

    중복 호출은 에러 대신 경고 후 무시한다 — 재초기화로 t0가 바뀌면 이미 기록된
    ts_ms가 전부 무효가 되기 때문이다(같은 t0 기준이어야 서로 비교 가능하다).
    """
    global _t0_monotonic, _t0_utc

    if _t0_monotonic is not None:
        print("[clock] init()이 이미 호출되었습니다 — 재초기화를 무시합니다.")
        return

    _t0_monotonic = time.monotonic()
    _t0_utc = datetime.now(timezone.utc)


def now_ms():
    """t0 대비 경과 시간(ms)을 float으로 반환한다.

    init() 이전 호출은 조용히 0을 반환하지 않고 명시적으로 에러를 던진다.
    """
    if _t0_monotonic is None:
        raise RuntimeError(
            "clock.init()을 먼저 호출해야 합니다 (now_ms 호출 전)."
        )

    return (time.monotonic() - _t0_monotonic) * 1000.0


def t0_utc_iso():
    """t0 시점의 tz-aware UTC ISO 문자열을 반환한다. sessions 행에 세션당 1회 기록용.

    init() 이전 호출은 명시적으로 에러를 던진다.
    """
    if _t0_utc is None:
        raise RuntimeError(
            "clock.init()을 먼저 호출해야 합니다 (t0_utc_iso 호출 전)."
        )

    return _t0_utc.isoformat()


def _reset():
    """테스트 전용 내부 리셋. 프로덕션 코드에서는 호출하지 말 것."""
    global _t0_monotonic, _t0_utc
    _t0_monotonic = None
    _t0_utc = None
