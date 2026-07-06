"""
blink_calibration_test.py — 깜빡임 캘리브레이션 + 판정 테스트 (독립 실행)

blink.py 와 같은 폴더에 두고 실행:
    python blink_calibration_test.py

흐름:
  1) 캘리브레이션 5라운드
     각 라운드 = [눈 뜨고 정면 응시 1.2초] -> [깜빡임 1회]
     뜬 눈 EAR 중앙값 / 감은 눈 EAR 중앙값으로 임계값 개인화
  2) 테스트 화면
     - 최근 깜빡임 판정(NATURAL / INTENTIONAL / LONG CLOSURE) 크게 표시
     - 최근 5개 깜빡임 이력 (지속시간 + 판정)
     - 실시간 EAR 게이지 바 + 임계값 마커
     - r: 재캘리브레이션 / q: 종료
  3) 캘리브레이션 결과는 blink_calib.json 으로 저장
     (메인 프로그램에서 읽어 BlinkDetector에 적용 가능)
"""

import json
import time
from collections import deque

import cv2
import numpy as np
import mediapipe as mp

from src.tracking.blink import BlinkDetector, BlinkKind, average_ear


# ── 설정 ─────────────────────────────────────────────

ROUNDS = 5                 # 캘리브레이션 라운드 수
OPEN_COLLECT_SEC = 1.2     # 라운드당 '뜬 눈' 수집 시간
OPEN_MIN_SAMPLES = 15      # 뜬 눈 최소 샘플 수 (얼굴 미검출 시 자동 연장)
BLINK_WINDOW_SEC = 2.0     # 라운드당 깜빡임 대기 시간
REST_SEC = 0.6             # 라운드 사이 휴식
HISTORY_SIZE = 5           # 이력 표시 개수
BANNER_HOLD_SEC = 2.0      # 판정 배너 강조 유지 시간
CALIB_FILE = "blink_calib.json"

KIND_STYLE = {
    BlinkKind.NATURAL:      ("NATURAL",      (190, 190, 190)),
    BlinkKind.INTENTIONAL:  ("INTENTIONAL",  (80, 220, 255)),
    BlinkKind.LONG_CLOSURE: ("LONG CLOSURE", (60, 60, 255)),
}


# ── 그리기 헬퍼 ───────────────────────────────────────

def put(frame, text, org, scale=0.7, color=(255, 255, 255), thick=2):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thick, cv2.LINE_AA)


def draw_ear_bar(frame, ear, close_t, open_t, x=20, y=None, w=260, h=18):
    """실시간 EAR 게이지. 임계값 위치를 마커로 표시."""
    if y is None:
        y = frame.shape[0] - 50

    ear_max = max(0.45, open_t * 1.6)  # 게이지 스케일 상한

    cv2.rectangle(frame, (x, y), (x + w, y + h), (70, 70, 70), -1)

    fill = int(np.clip(ear / ear_max, 0, 1) * w)
    bar_color = (90, 200, 90) if ear > open_t else \
                (60, 60, 230) if ear < close_t else (80, 200, 230)
    cv2.rectangle(frame, (x, y), (x + fill, y + h), bar_color, -1)

    for t, col in ((close_t, (60, 60, 255)), (open_t, (90, 220, 90))):
        tx = x + int(np.clip(t / ear_max, 0, 1) * w)
        cv2.line(frame, (tx, y - 4), (tx, y + h + 4), col, 2)

    put(frame, f"EAR {ear:.3f}", (x + w + 12, y + h - 2), 0.55)


def draw_history(frame, history):
    """최근 깜빡임 이력 (우측 상단, 최신이 위)."""
    x = frame.shape[1] - 295
    y = 30
    put(frame, "Recent blinks", (x, y), 0.6, (200, 200, 200))
    for i, ev in enumerate(reversed(history)):
        label, color = KIND_STYLE[ev.kind]
        put(frame, f"{ev.duration:.2f}s  {label}",
            (x, y + 28 + i * 26), 0.55, color, 1)


def median(values):
    return float(np.median(values)) if values else 0.0


# ── 캘리브레이션 상태 ─────────────────────────────────

class Calibration:
    def __init__(self):
        self.reset()

    def reset(self):
        self.phase = "open"          # open -> blink -> rest -> (다음 라운드 / done)
        self.round_idx = 0
        self.phase_start = time.monotonic()
        self.open_samples = []       # 전 라운드 누적 (뜬 눈 EAR)
        self.round_open = []         # 현재 라운드 뜬 눈 EAR
        self.round_minima = []       # 라운드별 최저 EAR (감은 눈)
        self.cur_min = None
        self.result = None           # (close_t, open_t, open_med, closed_med)
        self.failed = False

    def _next_phase(self, phase):
        self.phase = phase
        self.phase_start = time.monotonic()

    def update(self, ear):
        """매 프레임 호출. ear=None 이면 얼굴 미검출."""
        now = time.monotonic()
        elapsed = now - self.phase_start

        if self.phase == "open":
            if ear is not None:
                self.round_open.append(ear)
            if elapsed >= OPEN_COLLECT_SEC and len(self.round_open) >= OPEN_MIN_SAMPLES:
                self.open_samples.extend(self.round_open)
                self.cur_min = None
                self._next_phase("blink")

        elif self.phase == "blink":
            if ear is not None:
                self.cur_min = ear if self.cur_min is None else min(self.cur_min, ear)

            open_est = median(self.open_samples)
            blinked = (self.cur_min is not None
                       and self.cur_min < open_est * 0.6
                       and ear is not None
                       and ear > open_est * 0.8)

            if blinked or elapsed >= BLINK_WINDOW_SEC:
                if self.cur_min is not None:
                    self.round_minima.append(self.cur_min)
                self._next_phase("rest")

        elif self.phase == "rest":
            if elapsed >= REST_SEC:
                self.round_idx += 1
                self.round_open = []
                if self.round_idx >= ROUNDS:
                    self._finish()
                else:
                    self._next_phase("open")

    def _finish(self):
        open_med = median(self.open_samples)
        closed_med = median(self.round_minima)
        span = open_med - closed_med

        if span < 0.05 or not self.round_minima:
            # 뜬 눈/감은 눈 차이가 너무 작음 -> 검출 불안정, 기본값 사용
            self.failed = True
            self.result = (0.18, 0.22, open_med, closed_med)
        else:
            close_t = closed_med + span * 0.35
            open_t = closed_med + span * 0.55
            self.result = (round(close_t, 4), round(open_t, 4),
                           round(open_med, 4), round(closed_med, 4))

        self.phase = "done"

    def draw(self, frame, ear):
        h, w = frame.shape[:2]
        put(frame, f"CALIBRATION  round {self.round_idx + 1}/{ROUNDS}",
            (20, 40), 0.8, (80, 220, 255))

        if self.phase == "open":
            put(frame, "Keep eyes OPEN, look straight", (20, 80), 0.7)
            done = min(len(self.round_open) / OPEN_MIN_SAMPLES,
                       (time.monotonic() - self.phase_start) / OPEN_COLLECT_SEC)
            cv2.rectangle(frame, (20, 95), (20 + int(220 * min(done, 1.0)), 107),
                          (90, 200, 90), -1)
        elif self.phase == "blink":
            put(frame, "Blink ONCE now", (20, 80), 0.9, (80, 220, 255))
        elif self.phase == "rest":
            put(frame, "OK", (20, 80), 0.9, (90, 220, 90))

        if ear is not None:
            draw_ear_bar(frame, ear, 0.18, 0.22)
        else:
            put(frame, "NO FACE", (20, h - 40), 0.8, (60, 60, 255))


# ── 메인 ─────────────────────────────────────────────

def main():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    calib = Calibration()
    detector = None
    history = deque(maxlen=HISTORY_SIZE)
    last_event = None
    last_event_time = 0.0
    closed_since = None
    prev_frame_time = time.monotonic()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # 거울 모드
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        landmarks = None
        ear = None
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            ear = average_ear(landmarks)

        now = time.monotonic()
        fps = 1.0 / max(now - prev_frame_time, 1e-6)
        prev_frame_time = now

        # ── 캘리브레이션 단계 ──
        if calib.phase != "done":
            calib.update(ear)
            calib.draw(frame, ear)

            if calib.phase == "done":
                close_t, open_t, open_med, closed_med = calib.result
                detector = BlinkDetector(
                    close_threshold=close_t,
                    open_threshold=open_t,
                    detect_natural=True,
                    detect_intentional=True,
                )
                history.clear()
                last_event = None
                closed_since = None

                payload = {
                    "close_threshold": close_t,
                    "open_threshold": open_t,
                    "open_ear_median": open_med,
                    "closed_ear_median": closed_med,
                    "calibration_failed_fallback": calib.failed,
                    "timestamp": time.time(),
                }
                with open(CALIB_FILE, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
                print(f"[calib] saved -> {CALIB_FILE}: {payload}")
                if calib.failed:
                    print("[calib] 경고: 뜬눈/감은눈 EAR 차이가 작아 기본 임계값 사용")

        # ── 테스트 단계 ──
        else:
            if landmarks is not None:
                event = detector.update(landmarks)

                if detector.is_closed and closed_since is None:
                    closed_since = now
                elif not detector.is_closed:
                    closed_since = None

                if event is not None:
                    history.append(event)
                    last_event = event
                    last_event_time = now
            else:
                closed_since = None

            put(frame, "TEST  (r: recalibrate / q: quit)", (20, 35),
                0.7, (200, 200, 200))
            put(frame,
                f"thr close={detector.close_threshold:.3f}  "
                f"open={detector.open_threshold:.3f}",
                (20, 65), 0.55, (160, 160, 160), 1)
            if calib.failed:
                put(frame, "calib failed -> default thresholds",
                    (20, 90), 0.55, (60, 60, 255), 1)

            # 현재 상태
            if landmarks is None:
                put(frame, "NO FACE", (20, 130), 0.8, (60, 60, 255))
            elif detector.is_closed:
                dur = now - closed_since if closed_since else 0.0
                put(frame, f"CLOSED  {dur:.2f}s", (20, 130), 0.8, (80, 200, 230))
            else:
                put(frame, "OPEN", (20, 130), 0.8, (90, 220, 90))

            # 최근 판정 배너 (이벤트 직후 강조, 이후 흐리게 유지)
            if last_event is not None:
                label, color = KIND_STYLE[last_event.kind]
                fresh = (now - last_event_time) < BANNER_HOLD_SEC
                scale = 1.4 if fresh else 1.0
                col = color if fresh else tuple(int(c * 0.55) for c in color)
                put(frame, f"{label}  ({last_event.duration:.2f}s)",
                    (20, 200), scale, col, 3 if fresh else 2)

            draw_history(frame, history)

            if ear is not None:
                draw_ear_bar(frame, ear,
                             detector.close_threshold,
                             detector.open_threshold)

        put(frame, f"{fps:4.1f} fps", (frame.shape[1] - 110,
                                       frame.shape[0] - 20), 0.5,
            (140, 140, 140), 1)

        cv2.imshow("Blink Calibration & Test", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            calib.reset()
            detector = None

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()