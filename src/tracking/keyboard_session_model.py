"""
Session-adaptive keyboard ROI pose-aware correction model.

이 모듈은 3D/head-pose gaze 안정화 담당 파트의 실험 모드로 추가되는 독립 코드다.

- Calibrator.map_to_screen() / compensate_iris_by_head_pose()의 시그니처와 동작은
  이 모듈이 추가되어도 전혀 바뀌지 않는다. 이 모듈은 그 두 메서드와
  diagnose_pose_correction() / map_to_screen_with_preclip()의 반환값을
  읽기 전용으로 소비하기만 한다.
- estimate_sqpnp_headpose(), h 키, sqpnp_corrected 경로는 이 모듈에서
  import하지도, 참조하지도 않는다.
- 실시간 경로이므로 sklearn을 쓰지 않고 Ridge를 numpy로 직접 구현한다.
- 모든 좌표는 keyboard ROI 기준 정규화 좌표(kx, ky ∈ [0,1])로만 다룬다.
  화면 픽셀 좌표는 ROI 계산과 최종 실시간 적용 시 변환에만 쓰인다.
"""

import numpy as np


# ── 수집 phase 타이밍 (이동 후 유지 방식) ───────────────────────
# center/left/right 조건 공통. "계속 움직이는" 방식이 아니라
# 안내대로 이동한 뒤 정지 상태를 유지시켜 워밍업 이후 구간만 수집한다.
INSTRUCTION_SEC = 0.7   # 안내 문구 + 고개 이동 구간 (수집 안 함)
WARMUP_SEC = 0.5        # 유지 시작 직후 워밍업 구간 (수집 안 함)
HOLD_SEC = 1.0          # 실제 데이터 수집 구간
CONDITION_TOTAL_SEC = INSTRUCTION_SEC + WARMUP_SEC + HOLD_SEC

CONDITIONS = ("center", "left", "right")

# ── preclip severity 임계값 (px, 화면 경계 기준 clamp 이전 좌표) ──
# was_clamped=True를 무조건 폴백시키지 않고 심각도로 나눈다.
# ok: 클램프 없음. mild_clamp: 경계 살짝 밖(모델 적용 가능).
# severe_divergence: 큰 폭 외삽(과거 분석에서 수천 px까지 확인됨, 폴백 대상).
PRECLIP_MILD_MARGIN_PX = 150.0
PRECLIP_SEVERE_MARGIN_PX = 400.0

# ── 실시간 적용 gate 임계값 ─────────────────────────────────────
FEATURE_RANGE_MARGIN_RATIO = 0.15   # 수집 시 관측 min/max에 15% 여유
RESIDUAL_MAX_RATIO = 0.20           # ROI 크기 대비 residual 상한(비율)

# ── 학습 샘플 채택 기준 (mini calibration 수집 시) ───────────────
# blink뿐 아니라 low confidence/invalid pose/diagnose_pose_correction의
# would_apply=False(=fallback) 프레임도 학습 샘플에서 제외한다.
# calibrator.update()의 기존 conf 게이트(conf > 0.4)와 동일한 값을 쓴다.
MIN_CONF_FOR_SAMPLE = 0.4

RIDGE_ALPHA_GRID = (0.1, 0.3, 1.0, 3.0, 10.0)

MIN_ANCHORS_FOR_FIT = 3
MIN_SAMPLES_FOR_3D_VARIANT = 3   # delta_yaw/pitch/roll이 유효한 샘플 최소 개수

# 3d variant는 basic보다 LOO 오차가 이 비율 미만일 때만(=충분히 나을 때만) 선택한다.
# 작은 샘플 수에서 yaw/pitch/roll 추가로 인한 근소한 LOO 개선은 과적합일 가능성이
# 높으므로, 사소한 차이로는 3d를 고르지 않고 basic을 기본값으로 유지한다.
MIN_3D_IMPROVEMENT_RATIO = 0.97

# 앵커 후보에서 제외할 키(중심 좌표가 넓은 폭 때문에 왜곡됨)
_ANCHOR_EXCLUDE_TEXTS = {"Enter", "Shift", "한/영", "Del", " "}

BASIC_FEATURES = (
    "raw_kx", "raw_ky",
    "delta_center_x", "delta_center_y",
    "scale_ratio_minus_1",
)

FEATURES_3D = BASIC_FEATURES + (
    "delta_yaw", "delta_pitch", "delta_roll",
)


# ── keyboard ROI / anchor 선택 ──────────────────────────────────

def compute_keyboard_roi(button_list):
    """
    실제 온스크린 키보드(src/keyboard.py의 buttonList)의 bounding box를
    keyboard ROI로 계산한다. placeholder를 쓰지 않는다.
    """

    lefts = [b.pos[0] for b in button_list]
    tops = [b.pos[1] for b in button_list]
    rights = [b.pos[0] + b.size[0] for b in button_list]
    bottoms = [b.pos[1] + b.size[1] for b in button_list]

    left = min(lefts)
    top = min(tops)
    right = max(rights)
    bottom = max(bottoms)

    return {
        "left": float(left),
        "top": float(top),
        "right": float(right),
        "bottom": float(bottom),
        "width": float(right - left),
        "height": float(bottom - top),
    }


def to_roi_normalized(px_x, px_y, roi):
    """px 좌표 -> keyboard ROI 기준 정규화 좌표(kx, ky)."""

    kx = (px_x - roi["left"]) / roi["width"]
    ky = (px_y - roi["top"]) / roi["height"]
    return kx, ky


def to_roi_pixels(kx, ky, roi):
    """keyboard ROI 정규화 좌표 -> px 좌표."""

    px_x = roi["left"] + kx * roi["width"]
    px_y = roi["top"] + ky * roi["height"]
    return px_x, px_y


def is_inside_roi(px_x, px_y, roi, margin_px=0.0):
    return (
        roi["left"] - margin_px <= px_x <= roi["right"] + margin_px
        and roi["top"] - margin_px <= px_y <= roi["bottom"] + margin_px
    )


def select_anchor_buttons(button_list, roi):
    """
    ROI 네 모서리 + 중심에 가장 가까운 실제 키 5개를 anchor로 선택한다.

    특정 레이아웃(QWERTY 등)을 가정하지 않고 "ROI 정규화 좌표 상 목표점에
    가장 가까운 버튼" 규칙만 쓰므로 키보드 레이아웃이 바뀌어도 동일 절차로
    재사용 가능하다. 폭이 넓은 특수키(Enter/Shift/한영/Del/Space)는
    중심 좌표가 왜곡되므로 후보에서 제외한다.
    """

    candidates = [
        b for b in button_list
        if b.text not in _ANCHOR_EXCLUDE_TEXTS
    ]

    if not candidates:
        candidates = list(button_list)

    candidate_norm = []

    for b in candidates:
        center_x = b.pos[0] + b.size[0] / 2.0
        center_y = b.pos[1] + b.size[1] / 2.0
        nx, ny = to_roi_normalized(center_x, center_y, roi)
        candidate_norm.append((b, nx, ny))

    target_points = [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (1.0, 1.0),
        (0.5, 0.5),
    ]

    chosen = []
    used_ids = set()

    for tx, ty in target_points:

        best_button = None
        best_dist = None

        for b, nx, ny in candidate_norm:

            if id(b) in used_ids:
                continue

            dist = (nx - tx) ** 2 + (ny - ty) ** 2

            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_button = b

        if best_button is not None:
            chosen.append(best_button)
            used_ids.add(id(best_button))

    return chosen


def describe_anchors(anchors, roi):
    """
    선택된 anchor들의 key text/center px/정규화 kx,ky를 확인용으로 반환한다.
    ROI 전역을 잘 커버하는지(코너 4개 + 중심 1개가 실제로 넓게 퍼져 있는지)
    눈으로 확인할 수 있게 하기 위함.
    """

    described = []

    for anchor in anchors:
        center_x = anchor.pos[0] + anchor.size[0] / 2.0
        center_y = anchor.pos[1] + anchor.size[1] / 2.0
        kx, ky = to_roi_normalized(center_x, center_y, roi)

        described.append({
            "text": anchor.text,
            "center_x": center_x,
            "center_y": center_y,
            "kx": kx,
            "ky": ky,
        })

    return described


def print_anchor_summary(anchors, roi):
    """select_anchor_buttons() 결과를 콘솔에 확인용으로 출력한다."""

    print("[keyboard_session_model] anchor 5개 선택 결과:")

    for info in describe_anchors(anchors, roi):
        print(
            f"  key='{info['text']}' "
            f"center_px=({info['center_x']:.1f},{info['center_y']:.1f}) "
            f"norm=({info['kx']:.3f},{info['ky']:.3f})"
        )


def build_default(button_list):
    """main.py에서 한 번에 ROI+anchor를 얻기 위한 편의 함수."""

    roi = compute_keyboard_roi(button_list)
    anchors = select_anchor_buttons(button_list, roi)
    return roi, anchors


def anchor_target_kxky(anchor_button, roi):
    center_x = anchor_button.pos[0] + anchor_button.size[0] / 2.0
    center_y = anchor_button.pos[1] + anchor_button.size[1] / 2.0
    return to_roi_normalized(center_x, center_y, roi)


# ── head pose delta (yaw/pitch/roll) ────────────────────────────

def compute_pose_deltas(head_pose, pose_baseline):
    """
    calibrator.pose_baseline 구조([0]yaw [1]pitch [2]roll ...)를 그대로 읽어
    delta_yaw/pitch/roll을 계산한다. head_pose/pose_baseline이 유효하지 않으면
    (None, None, None)을 반환한다 (model_3d feature 결측 처리용).
    """

    if head_pose is None or not head_pose.get("valid", False):
        return None, None, None

    if pose_baseline is None or len(pose_baseline) < 3:
        return None, None, None

    delta_yaw = head_pose.get("yaw", 0.0) - pose_baseline[0]
    delta_pitch = head_pose.get("pitch", 0.0) - pose_baseline[1]
    delta_roll = head_pose.get("roll", 0.0) - pose_baseline[2]

    return delta_yaw, delta_pitch, delta_roll


# ── preclip severity ─────────────────────────────────────────────

def preclip_severity(preclip_x, preclip_y, screen_w, screen_h):
    """
    map_to_screen_with_preclip()의 preclip 좌표가 화면 경계를 얼마나
    벗어났는지로 심각도를 3단계로 나눈다.

    Returns: "ok" | "mild_clamp" | "severe_divergence"
    """

    if preclip_x is None or preclip_y is None:
        return "severe_divergence"

    over_x = max(0.0, -preclip_x, preclip_x - (screen_w - 1))
    over_y = max(0.0, -preclip_y, preclip_y - (screen_h - 1))
    over = max(over_x, over_y)

    if over <= 0.0:
        return "ok"

    if over <= PRECLIP_MILD_MARGIN_PX:
        return "mild_clamp"

    if over <= PRECLIP_SEVERE_MARGIN_PX:
        return "mild_clamp"

    return "severe_divergence"


# ── Ridge regression (수동 구현, feature standardization 포함) ──

class RidgeModel:
    """
    표준화된 feature에 대해 closed-form ridge를 직접 계산한다.
    fit 시 계산한 mean/std를 저장해두고 predict에서도 동일하게 사용한다
    (세션마다 다시 fit되므로 mean/std도 세션 로컬 값).
    """

    def __init__(self, alpha):
        self.alpha = alpha
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.coef_ = None

    def fit(self, X, Y):

        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)

        self.x_mean = X.mean(axis=0)

        std = X.std(axis=0)
        std[std < 1e-8] = 1.0
        self.x_std = std

        Xs = (X - self.x_mean) / self.x_std

        self.y_mean = Y.mean(axis=0)
        Ys = Y - self.y_mean

        n_features = Xs.shape[1]

        A = Xs.T @ Xs + self.alpha * np.eye(n_features)
        b = Xs.T @ Ys

        self.coef_ = np.linalg.solve(A, b)

        return self

    def predict(self, x):

        x = np.asarray(x, dtype=np.float64)
        xs = (x - self.x_mean) / self.x_std

        return xs @ self.coef_ + self.y_mean


def _fit_ridge_with_loo_alpha_search(X, Y, groups, alpha_grid):
    """
    leave-one-anchor-out(groups 기준)으로 alpha_grid 중 최적값을 고르고,
    그 alpha로 전체 데이터에 다시 fit한 최종 모델과 LOO 오차(평균 유클리드
    거리, kx/ky 정규화 단위)를 함께 반환한다.
    """

    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    groups = np.asarray(groups)

    unique_groups = np.unique(groups)

    if len(unique_groups) < 2 or len(X) < 3:
        return None, None

    best_alpha = None
    best_loo_error = None

    for alpha in alpha_grid:

        errors = []

        for held_out in unique_groups:

            train_mask = groups != held_out
            test_mask = groups == held_out

            if train_mask.sum() < 2 or test_mask.sum() == 0:
                continue

            model = RidgeModel(alpha).fit(X[train_mask], Y[train_mask])
            pred = model.predict(X[test_mask])

            residual = pred - Y[test_mask]
            errors.extend(np.hypot(residual[:, 0], residual[:, 1]).tolist())

        if not errors:
            continue

        mean_error = float(np.mean(errors))

        if best_loo_error is None or mean_error < best_loo_error:
            best_loo_error = mean_error
            best_alpha = alpha

    if best_alpha is None:
        return None, None

    final_model = RidgeModel(best_alpha).fit(X, Y)

    return final_model, best_loo_error


# ── 세션 적응형 keyboard ROI 보정 모델 ───────────────────────────

class KeyboardSessionCalibrator:
    """
    16점 캘리브레이션이 끝난 뒤, keyboard ROI 안에서만 동작하는
    session-adaptive residual correction model.

    사용 흐름:
        1. compute_keyboard_roi/select_anchor_buttons로 roi/anchor 준비
        2. KeyboardSessionCalibrator(roi) 생성
        3. anchor x condition 반복하며 add_frame()으로 hold 구간 프레임 누적,
           조건 하나가 끝날 때마다 finalize_condition() 호출
        4. 전부 끝나면 fit() 호출 -> model_basic/model_3d 비교 후 active 선택
        5. 실시간 루프에서 predict()로 gate 통과한 경우에만 residual 적용
        6. 검증 테스트가 끝나면 evaluate_from_validation_results()로
           usable_confirmed 갱신
    """

    def __init__(self, roi):

        self.roi = roi

        self._frame_buffer = {}   # (anchor_index, condition) -> list[dict]
        self.samples = []         # 조건별 평균이 끝난 학습 샘플들

        self.models = {}          # variant_name -> RidgeModel
        self.loo_error = {}       # variant_name -> float (정규화 kx/ky 단위)
        self.feature_ranges = {}  # variant_name -> {feature: (lo, hi)}

        self.raw_baseline_error = None

        self.active_variant = None
        self.usable_provisional = False
        self.usable_confirmed = None  # None=미평가, True/False=검증 후 결정

    # ── 데이터 수집 ──────────────────────────────────────────

    def add_frame(
        self,
        anchor_index,
        condition,
        raw_kx,
        raw_ky,
        delta_center_x,
        delta_center_y,
        scale_ratio,
        delta_yaw=None,
        delta_pitch=None,
        delta_roll=None,
    ):
        """hold 구간(워밍업 이후) 동안 매 프레임 호출."""

        key = (anchor_index, condition)

        self._frame_buffer.setdefault(key, []).append({
            "raw_kx": raw_kx,
            "raw_ky": raw_ky,
            "delta_center_x": delta_center_x,
            "delta_center_y": delta_center_y,
            "scale_ratio": scale_ratio,
            "delta_yaw": delta_yaw,
            "delta_pitch": delta_pitch,
            "delta_roll": delta_roll,
        })

    def finalize_condition(self, anchor_index, condition, target_kx, target_ky):
        """
        조건 하나(예: anchor 2의 left)의 hold 구간 수집을 마무리하고
        프레임 평균을 학습 샘플 하나로 확정한다.
        """

        key = (anchor_index, condition)
        frames = self._frame_buffer.pop(key, [])

        if not frames:
            return None

        def _mean(field):
            values = [f[field] for f in frames if f[field] is not None]
            return float(np.mean(values)) if values else None

        sample = {
            "anchor_index": anchor_index,
            "condition": condition,
            "target_kx": target_kx,
            "target_ky": target_ky,
            "raw_kx": _mean("raw_kx"),
            "raw_ky": _mean("raw_ky"),
            "delta_center_x": _mean("delta_center_x"),
            "delta_center_y": _mean("delta_center_y"),
            "scale_ratio_minus_1": (
                _mean("scale_ratio") - 1.0
                if _mean("scale_ratio") is not None
                else None
            ),
            "delta_yaw": _mean("delta_yaw"),
            "delta_pitch": _mean("delta_pitch"),
            "delta_roll": _mean("delta_roll"),
        }

        if sample["raw_kx"] is None or sample["raw_ky"] is None:
            return None

        self.samples.append(sample)
        return sample

    # ── 모델 fit ─────────────────────────────────────────────

    def _build_arrays(self, feature_names):

        X = []
        Y = []
        groups = []

        for s in self.samples:

            values = [s.get(f) for f in feature_names]

            if any(v is None for v in values):
                continue

            X.append(values)
            Y.append([
                s["target_kx"] - s["raw_kx"],
                s["target_ky"] - s["raw_ky"],
            ])
            groups.append(s["anchor_index"])

        return np.array(X), np.array(Y), np.array(groups)

    def _compute_feature_ranges(self, X, feature_names):

        ranges = {}

        for i, name in enumerate(feature_names):

            col = X[:, i]
            lo = float(col.min())
            hi = float(col.max())

            if hi - lo < 1e-6:
                pad = max(abs(hi) * 0.5, 0.01)
                lo -= pad
                hi += pad

            margin = (hi - lo) * FEATURE_RANGE_MARGIN_RATIO
            ranges[name] = (lo - margin, hi + margin)

        return ranges

    def fit(self):
        """
        model_basic / model_3d를 모두 fit한다.

        LOO(leave-one-anchor-out) 오차는 세션 내부 데이터 5~15개 샘플로
        계산한 참고 지표일 뿐, 실사용 성능(hit rate 등)의 근거가 아니다.
        어느 variant가 최종적으로 유효한지는 evaluate_from_validation_results()의
        결과로만 판단해야 한다.

        active_variant 선택은 보수적으로 한다:
          - basic이 raw_baseline_error보다 못하면 아예 usable_provisional=False.
          - basic이 raw보다 나으면 기본값은 항상 basic.
          - 3d는 3d_error < basic_error * MIN_3D_IMPROVEMENT_RATIO 일 때만,
            즉 basic보다 최소 마진 이상 나을 때만 선택한다.
            근소한 차이로 3d를 고르면 샘플이 적은 상태에서 yaw/pitch/roll에
            과적합했을 가능성이 커서 기본값을 basic으로 둔다.
        """

        n_anchors = len(set(s["anchor_index"] for s in self.samples))

        if n_anchors < MIN_ANCHORS_FOR_FIT:
            self.usable_provisional = False
            self.active_variant = None
            return self

        X_basic, Y_basic, groups_basic = self._build_arrays(BASIC_FEATURES)

        if len(X_basic) < 3:
            self.usable_provisional = False
            self.active_variant = None
            return self

        raw_errors = np.hypot(Y_basic[:, 0], Y_basic[:, 1])
        self.raw_baseline_error = float(np.mean(raw_errors))

        model_basic, loo_basic = _fit_ridge_with_loo_alpha_search(
            X_basic, Y_basic, groups_basic, RIDGE_ALPHA_GRID
        )

        if model_basic is not None:
            self.models["basic"] = model_basic
            self.loo_error["basic"] = loo_basic
            self.feature_ranges["basic"] = self._compute_feature_ranges(
                X_basic, BASIC_FEATURES
            )

        X_3d, Y_3d, groups_3d = self._build_arrays(FEATURES_3D)

        if len(X_3d) >= MIN_SAMPLES_FOR_3D_VARIANT:

            model_3d, loo_3d = _fit_ridge_with_loo_alpha_search(
                X_3d, Y_3d, groups_3d, RIDGE_ALPHA_GRID
            )

            if model_3d is not None:
                self.models["3d"] = model_3d
                self.loo_error["3d"] = loo_3d
                self.feature_ranges["3d"] = self._compute_feature_ranges(
                    X_3d, FEATURES_3D
                )

        basic_error = self.loo_error.get("basic")
        threed_error = self.loo_error.get("3d")

        best_variant = None

        if basic_error is not None and basic_error < self.raw_baseline_error:
            best_variant = "basic"

            if (
                threed_error is not None
                and threed_error < basic_error * MIN_3D_IMPROVEMENT_RATIO
            ):
                best_variant = "3d"

        self.active_variant = best_variant
        self.usable_provisional = best_variant is not None

        return self

    # ── 실시간 적용 ──────────────────────────────────────────

    def _feature_vector(self, variant, features):

        names = BASIC_FEATURES if variant == "basic" else FEATURES_3D
        values = [features.get(name) for name in names]

        if any(v is None for v in values):
            return None

        return names, np.array(values, dtype=np.float64)

    def _in_feature_range(self, variant, names, values):

        ranges = self.feature_ranges.get(variant)

        if ranges is None:
            return False

        for name, value in zip(names, values):
            lo, hi = ranges[name]
            if not (lo <= value <= hi):
                return False

        return True

    def predict(
        self,
        raw_kx,
        raw_ky,
        delta_center_x,
        delta_center_y,
        scale_ratio,
        delta_yaw=None,
        delta_pitch=None,
        delta_roll=None,
        preclip_sev="ok",
    ):
        """
        gate 4종(모델 usable / preclip severity / feature range / residual
        size)을 개별적으로 평가해 dict로 반환한다. CSV2(실시간 병렬 비교
        로그)에 그대로 컬럼으로 남겨 나중에 "왜 이번 프레임엔 모델이
        적용되지 않았는지"를 재구성할 수 있게 하기 위함이다.

        gate는 앞 단계가 실패하면 뒤 단계는 평가하지 않고 None으로 남는다
        (예: model_not_usable이면 preclip/feature_range/residual_size는
        애초에 검사할 필요가 없으므로 None).

        Returns:
            {
                "applied": bool,
                "residual_kx": float,
                "residual_ky": float,
                "variant_used": "basic" | "3d" | None,
                "preclip_severity": str,
                "gate_model_usable": bool,
                "gate_preclip": bool or None,
                "gate_feature_range": bool or None,
                "gate_residual_size": bool or None,
                "fallback_reason": str,  # applied=True면 ""
            }
        """

        result = {
            "applied": False,
            "residual_kx": 0.0,
            "residual_ky": 0.0,
            "variant_used": None,
            "preclip_severity": preclip_sev,
            "gate_model_usable": self.usable_provisional and self.active_variant is not None,
            "gate_preclip": None,
            "gate_feature_range": None,
            "gate_residual_size": None,
            "fallback_reason": "",
        }

        if not result["gate_model_usable"]:
            result["fallback_reason"] = "model_not_usable"
            return result

        result["gate_preclip"] = preclip_sev != "severe_divergence"

        if not result["gate_preclip"]:
            result["fallback_reason"] = "preclip_severe"
            return result

        features = {
            "raw_kx": raw_kx,
            "raw_ky": raw_ky,
            "delta_center_x": delta_center_x,
            "delta_center_y": delta_center_y,
            "scale_ratio_minus_1": (
                scale_ratio - 1.0 if scale_ratio is not None else None
            ),
            "delta_yaw": delta_yaw,
            "delta_pitch": delta_pitch,
            "delta_roll": delta_roll,
        }

        variant = self.active_variant
        feature_vec = self._feature_vector(variant, features)

        if feature_vec is None and variant == "3d" and "basic" in self.models:
            # 이번 프레임에 pose delta가 없으면(예: head_pose invalid) basic으로 강등
            variant = "basic"
            feature_vec = self._feature_vector(variant, features)

        if feature_vec is None:
            result["fallback_reason"] = "feature_missing"
            return result

        names, x = feature_vec

        result["gate_feature_range"] = self._in_feature_range(variant, names, x)

        if not result["gate_feature_range"]:
            result["fallback_reason"] = "feature_out_of_range"
            return result

        residual = self.models[variant].predict(x)
        residual_kx, residual_ky = float(residual[0]), float(residual[1])

        result["gate_residual_size"] = (
            abs(residual_kx) <= RESIDUAL_MAX_RATIO
            and abs(residual_ky) <= RESIDUAL_MAX_RATIO
        )

        if not result["gate_residual_size"]:
            result["fallback_reason"] = "residual_too_large"
            return result

        result["applied"] = True
        result["variant_used"] = variant
        result["residual_kx"] = residual_kx
        result["residual_ky"] = residual_ky

        return result

    # ── 검증 결과 반영 ───────────────────────────────────────

    def evaluate_from_validation_results(self, trial_rows):
        """
        run_keyboard_session_validation_test()가 만든 시도 단위 결과
        (dict list, 각 dict에 최소 source/hit/dwell_completed/
        boundary_crossings 키 필요)를 받아 raw/pose/session_model을
        key hit rate, dwell 성공률, boundary crossing으로 비교하고
        usable_confirmed를 확정한다.

        화면 평균 px 오차가 아니라 이 세 지표로만 판단한다.
        """

        summary = {}

        for source in ("raw", "pose", "session_model"):

            rows = [r for r in trial_rows if r.get("source") == source]

            if not rows:
                summary[source] = None
                continue

            hit_rate = sum(1 for r in rows if r.get("hit")) / len(rows)
            dwell_rate = (
                sum(1 for r in rows if r.get("dwell_completed")) / len(rows)
            )
            mean_crossings = float(
                np.mean([r.get("boundary_crossings", 0) for r in rows])
            )

            summary[source] = {
                "hit_rate": hit_rate,
                "dwell_rate": dwell_rate,
                "mean_boundary_crossings": mean_crossings,
                "n": len(rows),
            }

        session = summary.get("session_model")
        raw = summary.get("raw")
        pose = summary.get("pose")

        if session is None or raw is None:
            self.usable_confirmed = False
            return summary

        beats_raw = (
            session["hit_rate"] >= raw["hit_rate"]
            and session["dwell_rate"] >= raw["dwell_rate"]
            and session["mean_boundary_crossings"] <= raw["mean_boundary_crossings"]
        )

        beats_pose = True

        if pose is not None:
            beats_pose = (
                session["hit_rate"] >= pose["hit_rate"]
                and session["dwell_rate"] >= pose["dwell_rate"]
            )

        self.usable_confirmed = bool(beats_raw and beats_pose)

        return summary

    # ── 디버그/로그용 요약 ───────────────────────────────────

    def summary(self):

        return {
            "n_samples": len(self.samples),
            "raw_baseline_error": self.raw_baseline_error,
            "loo_error": dict(self.loo_error),
            "active_variant": self.active_variant,
            "usable_provisional": self.usable_provisional,
            "usable_confirmed": self.usable_confirmed,
        }
