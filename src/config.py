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
