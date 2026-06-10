import cv2
import tkinter as tk

# ── 폰트 설정 ─────────────────────────────────────────────────
FONT_PATH = "malgun.ttf"
FONT_SIZE = 40

# ── 화면 해상도 자동 감지 ─────────────────────────────────────

root = tk.Tk()
root.withdraw()

SCREEN_W = root.winfo_screenwidth()
SCREEN_H = root.winfo_screenheight()

root.destroy()

# ── 화면 물리 정보 (ACC-06 cm 환산용) ──────────────────
# 풀스크린이고 SCREEN_W/H가 실제 모니터 px 해상도이므로,
# 9점 테스트 오차 px와 같은 좌표계 → 별도 보정 불필요.
# 수동 입력 필요한 값은 대각 인치 하나뿐.
# 정식 웹캠 모니터 도착하면 그 값으로 교체. 분산 테스트 중엔 각자 자기 기기 값으로.

MONITOR_DIAGONAL_INCH = 16.0   # ← 측정자가 자기 모니터 대각 크기(인치)로 수정

# px_per_cm = 대각선 px / 대각선 cm
_diag_px = (SCREEN_W ** 2 + SCREEN_H ** 2) ** 0.5
_diag_cm = MONITOR_DIAGONAL_INCH * 2.54
PX_PER_CM = _diag_px / _diag_cm

# ── 캘리브레이션 설정 ─────────────────────────────────────────

MARGIN = 0.08
_M = MARGIN
_T = 1 - MARGIN

CALIB_POINTS = [
    (_M,               _M),
    (_M + (_T-_M)/3,   _M),
    (_M + (_T-_M)*2/3, _M),
    (_T,               _M),

    (_M,               _M + (_T-_M)/3),
    (_M + (_T-_M)/3,   _M + (_T-_M)/3),
    (_M + (_T-_M)*2/3, _M + (_T-_M)/3),
    (_T,               _M + (_T-_M)/3),

    (_M,               _M + (_T-_M)*2/3),
    (_M + (_T-_M)/3,   _M + (_T-_M)*2/3),
    (_M + (_T-_M)*2/3, _M + (_T-_M)*2/3),
    (_T,               _M + (_T-_M)*2/3),

    (_M,               _T),
    (_M + (_T-_M)/3,   _T),
    (_M + (_T-_M)*2/3, _T),
    (_T,               _T),
]

CALIB_HOLD_SEC = 2.0
SMOOTH_ALPHA = 0.20
COUNTDOWN_SEC = 3
DWELL_SEC = 1.2

# 캘리브레이션 안정화
CALIB_STABILIZE_SEC = 1.0   # 점 응시 안정화
CALIB_COLLECT_SEC = 2.0     # 실제 데이터 수집

# 시선 편차 허용 범위
CALIB_STD_X = 0.008
CALIB_STD_Y = 0.008

# ── 시선 안정화 설정 ──────────────────────────────────────────

FIXATION_RADIUS = 40
FIXATION_FRAMES = 6
