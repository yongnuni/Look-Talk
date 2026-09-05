import math

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


# ── 16점 캘리브레이션 (4 × 4) ────────────────────────────────
CALIB_POINTS_16 = [
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


# ── 9점 캘리브레이션 (3 × 3) ─────────────────────────────────
# 화면 전체를 좌/중앙/우 × 상/중앙/하 형태로 사용
CALIB_POINTS_9 = [
    # 1행
    (_M,   _M),
    (0.50, _M),
    (_T,   _M),

    # 2행
    (_M,   0.50),
    (0.50, 0.50),
    (_T,   0.50),

    # 3행
    (_M,   _T),
    (0.50, _T),
    (_T,   _T),
]


# 기존 코드 호환용 기본 캘리브레이션
# 별도 설정이 없으면 기존과 동일하게 16점 캘리브레이션 사용
CALIB_POINTS = CALIB_POINTS_16


SMOOTH_ALPHA = 0.35
GAZE_AVG_WINDOW = 3
COUNTDOWN_SEC = 3
DWELL_SEC = 0.8

# 캘리브레이션 안정화
CALIB_STABILIZE_SEC = 1.0   # 점 응시 안정화
CALIB_COLLECT_SEC = 2.0     # 실제 데이터 수집

# 시선 편차 허용 범위
CALIB_STD_X = 0.008
CALIB_STD_Y = 0.008

# ── 시선 안정화 설정 ──────────────────────────────────────────

FIXATION_RADIUS = 40
FIXATION_FRAMES = 6

# ── 고정(fixation) 감지 기반 히트박스 확장 ────────────────────
# 위의 FIXATION_RADIUS/FRAMES는 GazePipeline이 커서 좌표를 안정화할 때 쓰는
# 값이고, 아래 값들은 그와 완전히 분리된 판정 전용 레이어의 설정이다.
# 시선 좌표 자체는 전혀 바꾸지 않고, 고정이 시작된 키 하나의 판정 영역만
# 넓힌다(화면에 그려지는 키 크기는 그대로).

VIEWING_DISTANCE_CM = 60.0   # ← 측정자가 눈~화면 실제 거리(cm)로 수정

# 시야각 1°가 화면에서 차지하는 px. 속도(°/s)·분산(°) 임계값의 px 환산에 쓴다.
PX_PER_DEG = (
    2.0 * VIEWING_DISTANCE_CM * math.tan(math.radians(0.5)) * PX_PER_CM
)

FIXATION_HITBOX_ENABLED = True

# I-VT: 점간 이동 속도가 이 값 이하로 떨어지면 고정 후보.
# 연구에서 쓰이는 5~50°/s 범위의 중간값.
FIXATION_VELOCITY_DEG_PER_SEC = 30.0

# I-DT: 고정 중심에서 이 반경 안에 머물러야 고정으로 본다(체스 연구 기준 약 1°).
FIXATION_DISPERSION_DEG = 1.0

# 고정 성립에 필요한 최소 지속 시간(100~200ms 범위).
# dwell 판정(DWELL_SEC)보다 훨씬 짧아야 dwell이 차기 전에 확장이 걸린다.
FIXATION_MIN_DURATION_SEC = 0.15

# 고정 해제 반경. 성립 이후에는 조금 관대하게 두어 미세한 흔들림 때문에
# 확장이 켜졌다 꺼졌다 하지 않게 한다(히스테리시스).
FIXATION_RELEASE_DEG = 1.5

# 프레임 간격이 이보다 크면(모드 전환·캘리브레이션 등으로 루프가 끊긴 경우)
# 연속된 시선으로 이어 붙이지 않고 고정 상태를 새로 시작한다.
FIXATION_MAX_GAP_SEC = 0.3

# 확장량 = 키 한 변 × 이 비율(각 변 바깥으로).
# 0.5면 판정 폭이 "키 폭 + 키 폭"(= 2배)이 된다.
FIXATION_HITBOX_EXPAND_RATIO = 0.5

# 확장이 인접 키를 덮을 수 있는 최대 깊이(그 키 한 변 대비 비율).
# 겹치는 구간에서는 확장된 쪽이 이기므로, 인접 키가 통째로 먹히지 않도록
# 침범 깊이를 제한한다. 1/3이면 인접 키의 나머지 2/3는 그대로 남는다.
FIXATION_HITBOX_MAX_OVERLAP_RATIO = 1.0 / 3.0

# 고정된 키캡을 화면에서도 실제로 키울지. 끄면 판정 영역만 넓어진다.
FIXATION_VISUAL_EXPAND_ENABLED = True

# 시각 확대 비율. 히트박스 비율보다 작게 두면 "보이는 것보다 판정이 조금 더
# 너그러운" 암묵 확장이 된다. 침범 한도(1/3)는 시각 확대에도 똑같이 걸리므로
# 인접 키가 통째로 가려지지 않는다.
FIXATION_VISUAL_EXPAND_RATIO = 0.35

# 확대 애니메이션 길이(초). 80~150ms의 짧은 1회성 ease-out —
# 없으면 반응 체감이 약하고, 길거나 반복되면 시각 유발성 멀미 리스크가 커진다.
FIXATION_VISUAL_ANIM_SEC = 0.12

# ── 릿지 회귀 설정 ─────────────────────────────────────────────

RIDGE_ALPHA = 1.0   # 릿지 회귀 정규화 계수
RIDGE_DEGREE = 2    # 릿지 회귀 다항 특징 차수