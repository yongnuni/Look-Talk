import cv2
import numpy as np
import os
import time
import uuid

from src.metrics.csv_export import append_rows
from src.config import (
    CALIB_POINTS,
    SCREEN_W,
    SCREEN_H,
    CALIB_STABILIZE_SEC,
    CALIB_COLLECT_SEC,
    CALIB_STD_X,
    CALIB_STD_Y
)

from src.tracking.feature_builder import FEATURE_DIM

# ── 릿지 회귀 설정 ────────────────────────────────────────────
# config.py에 없어도 동작하도록 기본값을 둡니다.
# RIDGE_ALPHA: L2 정규화 강도. 캘리브레이션 점이 적을수록 키우세요 (0.5~10 권장 범위).
# RIDGE_DEGREE: 다항 차수. 2 고정 권장. 3 이상은 16점 캘리브레이션에서 가장자리 발산 위험.
try:
    from src.config import RIDGE_ALPHA
except ImportError:
    RIDGE_ALPHA = 1.0

try:
    from src.config import RIDGE_DEGREE
except ImportError:
    RIDGE_DEGREE = 2


class PolyRidgeMapper:
    """
    다항 특징 확장(2차) + 릿지 회귀로
    특징 벡터 → 화면 좌표(x, y)를 매핑합니다.

    sklearn이 있으면 sklearn을 사용하고,
    없으면 동일한 수식의 numpy 폐형해(closed-form)로 대체합니다.
    두 경로의 수학은 동일합니다: (X'X + αI)^-1 X'y

    홈그래피와 달리:
        - 입력이 2차원(홍채)이 아니라 FEATURE_DIM차원(홍채+시선벡터+머리자세)
        - 머리 자세 변화가 특징에 포함되어 있어 매핑 자체가 자세를 흡수
    """

    def __init__(self, degree=RIDGE_DEGREE, alpha=RIDGE_ALPHA):
        self.degree = degree
        self.alpha = alpha
        self.fitted = False

        self._impl = None           # "sklearn" 또는 "numpy"
        self._model = None          # sklearn pipeline
        self._W = None              # numpy 경로 가중치
        self._mean = None           # numpy 경로 표준화 파라미터
        self._std = None

    # ── 다항 확장 (numpy 경로) ──────────────────────────────

    @staticmethod
    def _poly_expand(X, degree):
        """
        1차 + 2차(제곱 및 상호작용) 항 생성.
        sklearn PolynomialFeatures(include_bias=True)와 동일한 구성입니다.
        """
        n, d = X.shape
        cols = [np.ones((n, 1)), X]

        if degree >= 2:
            for i in range(d):
                for j in range(i, d):
                    cols.append((X[:, i] * X[:, j]).reshape(-1, 1))

        return np.hstack(cols)

    # ── 학습 ────────────────────────────────────────────────

    def fit(self, X, Y):
        """
        Args:
            X: (N, FEATURE_DIM) 특징 행렬
            Y: (N, 2) 화면 좌표 타깃
        """

        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)

        if X.ndim != 2 or X.shape[0] < 8:
            print("[ridge] 학습 샘플 부족:", X.shape)
            return False

        try:
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler, PolynomialFeatures
            from sklearn.linear_model import Ridge

            self._model = make_pipeline(
                StandardScaler(),
                PolynomialFeatures(
                    degree=self.degree,
                    include_bias=False
                ),
                Ridge(alpha=self.alpha)
            )

            self._model.fit(X, Y)
            self._impl = "sklearn"
            self.fitted = True
            return True

        except ImportError:
            pass

        # ── numpy 폐형해 경로 ──
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0)
        self._std[self._std < 1e-8] = 1.0

        Xs = (X - self._mean) / self._std
        Phi = self._poly_expand(Xs, self.degree)

        n_feat = Phi.shape[1]

        A = Phi.T @ Phi + self.alpha * np.eye(n_feat)
        # bias 항은 정규화하지 않음 (관례)
        A[0, 0] -= self.alpha

        try:
            self._W = np.linalg.solve(A, Phi.T @ Y)
        except np.linalg.LinAlgError:
            print("[ridge] 선형계 해 실패")
            return False

        self._impl = "numpy"
        self.fitted = True
        return True

    # ── 예측 ────────────────────────────────────────────────

    def predict(self, feat):
        """
        Args:
            feat: (FEATURE_DIM,) 단일 특징 벡터
        Returns:
            (screen_x, screen_y) float, 또는 None
        """

        if not self.fitted or feat is None:
            return None

        feat = np.asarray(feat, dtype=np.float64).reshape(1, -1)

        if not np.all(np.isfinite(feat)):
            return None

        if self._impl == "sklearn":
            out = self._model.predict(feat)[0]
        else:
            Xs = (feat - self._mean) / self._std
            Phi = self._poly_expand(Xs, self.degree)
            out = (Phi @ self._W)[0]

        return float(out[0]), float(out[1])


class Calibrator:

    CALIB_SCHEMA_VERSION = "1.3"

    def __init__(self):
        # 마지막으로 품질 기준을 통과한 Homography 보관
        # reset()을 실행해도 유지되어야 함
        self.last_good_H = None
        self.last_good_calib_rmse_px = None
        self.last_good_calib_id = None
        self.reset()

    def reset(self):

        self.idx = 0

        self.samples = []
        self.pose_samples = []
        self.sample_confs = []

        # ── 릿지용 원시 샘플 (점당 평균이 아니라 프레임 단위로 쌓음) ──
        # 홈그래피는 점당 대표값 16개로 충분하지만,
        # 릿지는 샘플 수가 많을수록 정규화가 안정되므로
        # 수집 구간의 모든 유효 프레임 특징을 그대로 사용합니다.
        self.feature_samples = []      # 현재 점에서 수집 중인 특징들
        self.ridge_X = []              # 전체 학습 특징
        self.ridge_Y = []              # 전체 학습 타깃 (화면 px)

        self.iris_pts = []
        self.screen_pts = []
        self.pose_pts = []

        self.point_records = []
        self.point_retry_counts = {}
        self.calib_id = str(uuid.uuid4())
        self.calib_reproj_rmse_px = None
        self.ridge_reproj_rmse_px = None
        self.calibration_fallback_used = False
        self.rejected_calib_rmse_px = None
        self.edge_mean_reproj_error_px = None
        self.center_mean_reproj_error_px = None
        self.applied_calib_rmse_px = None

        self.pose_baseline = None

        self.H = None
        self.ridge = PolyRidgeMapper()

        self.done = False

        self.hold_start = None

        self.warning = ""
        self.warning_start = None

        # EMA용 이전 보정값
        self.prev_corrected_iris_x = None
        self.prev_corrected_iris_y = None

    def update(
        self,
        iris_x,
        iris_y,
        conf,
        head_pose=None,
        features=None
    ):

        now = time.time()

        if self.hold_start is None:
            self.hold_start = now

        elapsed = now - self.hold_start

        # 안정화 시간이 지난 뒤부터 샘플 수집
        if (
            elapsed >= CALIB_STABILIZE_SEC
            and conf > 0.4
        ):

            self.samples.append(
                (
                    iris_x,
                    iris_y
                )
            )
            self.sample_confs.append(conf)

            # ── 릿지용 특징 샘플 수집 (기존 로직에 추가만 됨) ──
            if features is not None:
                self.feature_samples.append(np.asarray(features, dtype=np.float64))

            if head_pose is not None and head_pose.get("valid", False):
                self.pose_samples.append(
                    (
                        head_pose.get("yaw", 0.0),
                        head_pose.get("pitch", 0.0),
                        head_pose.get("roll", 0.0),
                        head_pose.get("face_scale", 0.0),
                        head_pose.get("tz", 0.0),
                        head_pose.get("face_center_x", 0.5),
                        head_pose.get("face_center_y", 0.5),
                    )
                )

        if elapsed >= (
            CALIB_STABILIZE_SEC +
            CALIB_COLLECT_SEC
        ):
            if len(self.samples) > 5:

                xs = sorted(
                    s[0]
                    for s in self.samples
                )

                ys = sorted(
                    s[1]
                    for s in self.samples
                )

                n = len(xs)

                lo = int(n * 0.2)
                hi = int(n * 0.8)

                avg_x = np.median(
                    xs[lo:hi]
                )

                avg_y = np.median(
                    ys[lo:hi]
                )

            else:

                avg_x = np.median(
                    [
                        s[0]
                        for s in self.samples
                    ]
                ) if self.samples else 0.5

                avg_y = np.median(
                    [
                        s[1]
                        for s in self.samples
                    ]
                ) if self.samples else 0.5

            # 시선 흔들림 검사
            # 1. 수집된 원본 좌표 분리
            xs_raw = [
                s[0]
                for s in self.samples
            ]

            ys_raw = [
                s[1]
                for s in self.samples
            ]
            # 2. 표준편차 계산
            std_x = np.std(xs_raw) if xs_raw else 0.0
            std_y = np.std(ys_raw) if ys_raw else 0.0

            sample_count = len(self.samples)

            if sample_count > 5:
                trimmed_count = max(0, hi - lo)
            else:
                trimmed_count = sample_count

            removed_count = sample_count - trimmed_count

            print(
                f"[calibration point {self.idx}] "
                f"samples={sample_count}, "
                f"used={trimmed_count}, "
                f"removed={removed_count}, "
                f"std_x={std_x:.6f}, "
                f"std_y={std_y:.6f}, "
                f"median=({avg_x:.6f}, {avg_y:.6f})"
            )
            # 3. 불안정 여부 검사
            is_unstable = (
                std_x > CALIB_STD_X
                or std_y > CALIB_STD_Y
            )

            retry_count = self.point_retry_counts.get(
                self.idx,
                0
            )

            if is_unstable and retry_count < 1:

                self.point_retry_counts[self.idx] = (
                    retry_count + 1
                )

                self.warning = (
                    "시선이 불안정합니다. "
                    "한 번 더 측정합니다."
                )
                self.warning_start = time.time()

                print(
                    f"[calibration point {self.idx}] "
                    f"unstable: "
                    f"std_x={std_x:.6f}, "
                    f"std_y={std_y:.6f} "
                    f"-> retry 1"
                )

                self.samples = []
                self.pose_samples = []
                self.sample_confs = []
                self.feature_samples = []
                self.hold_start = None

                return elapsed / (
                    CALIB_STABILIZE_SEC +
                    CALIB_COLLECT_SEC
                )

            if is_unstable:
                print(
                    f"[calibration point {self.idx}] "
                    f"still unstable after retry; "
                    f"accepted with warning"
                )

                self.warning = (
                    "시선이 다소 불안정하지만 "
                    "현재 측정값을 사용합니다."
                )
                self.warning_start = time.time()

            else:
                self.warning = ""

            self.iris_pts.append(
                [
                    avg_x,
                    avg_y
                ]
            )

            # 해당 캘리브레이션 점의 head pose 평균 저장
            if self.pose_samples:
                pose_avg = np.mean(
                    np.array(self.pose_samples, dtype=np.float32),
                    axis=0
                ).tolist()

                self.pose_pts.append(pose_avg)

            sx, sy = CALIB_POINTS[self.idx]

            self.screen_pts.append(
                [
                    sx * SCREEN_W,
                    sy * SCREEN_H
                ]
            )

            # ── 릿지 학습 데이터 확정 (이 점은 안정 판정을 통과했음) ──
            target_px = [sx * SCREEN_W, sy * SCREEN_H]

            for f in self.feature_samples:
                if f is not None and len(f) == FEATURE_DIM:
                    self.ridge_X.append(f)
                    self.ridge_Y.append(target_px)

            

            mean_confidence = (
                float(np.mean(self.sample_confs))
                if self.sample_confs
                else None
            )
            is_edge_point = (
                sx <= 0.2
                or sx >= 0.8
                or sy <= 0.2
                or sy >= 0.8
            )

            self.point_records.append(
                {
                    "calib_point_index": self.idx,
                    "point_x_norm": sx,
                    "point_y_norm": sy,
                    "region": (
                        "edge"
                        if is_edge_point
                        else "center"
                    ),
                    "reproj_error_px": None,
                    "ridge_reproj_error_px": None,
                    "iris_std_x_px": float(std_x),
                    "iris_std_y_px": float(std_y),
                    "sample_count": len(self.samples),
                    "used_sample_count": trimmed_count,
                    "removed_sample_count": removed_count,
                    "calibration_retry_count": (
                        self.point_retry_counts.get(
                            self.idx,
                            0
                        )
                    ),
                    "feature_sample_count": len(self.feature_samples),
                    "mean_confidence": mean_confidence, 
                }
            )

            self.idx += 1

            self.samples = []
            self.pose_samples = []
            self.sample_confs = []
            self.feature_samples = []
            self.hold_start = None

            if self.idx >= len(CALIB_POINTS):

                src = np.array(
                    self.iris_pts,
                    dtype=np.float32
                )

                dst = np.array(
                    self.screen_pts,
                    dtype=np.float32
                )

                self.H, _ = cv2.findHomography(
                    src,
                    dst
                )

                # STB-11 재투영 오차: 학습에 쓴 홍채 좌표를 H로 다시 화면 좌표로
                # 투영해 실제 타깃 좌표와의 거리를 잰다. map_to_screen()의 clip은
                # 쓰지 않는다 (경계 밖 오차가 잘려 과소평가되는 것을 막기 위함).
                # findHomography가 퇴화 대응점 등으로 None을 반환하면 계산을
                # 건너뛰고 reproj_error_px는 None으로 남긴다 (STB-12/13은 유지).
                if self.H is not None:

                    reproj = cv2.perspectiveTransform(
                        src.reshape(-1, 1, 2),
                        self.H
                    ).reshape(-1, 2)

                    for i, rec in enumerate(self.point_records):
                        pred_x, pred_y = reproj[i]
                        target_x, target_y = self.screen_pts[i]

                        rec["reproj_error_px"] = float(
                            np.hypot(
                                pred_x - target_x,
                                pred_y - target_y
                            )
                        )

                        point_x_norm = rec["point_x_norm"]
                        point_y_norm = rec["point_y_norm"]

                        is_edge = (
                            point_x_norm <= 0.2
                            or point_x_norm >= 0.8
                            or point_y_norm <= 0.2
                            or point_y_norm >= 0.8
                        )

                        region = "edge" if is_edge else "center"

                        print(
                            f"[calibration reproj {i}] "
                            f"region={region}, "
                            f"target=({target_x:.1f}, {target_y:.1f}), "
                            f"mapped=({pred_x:.1f}, {pred_y:.1f}), "
                            f"error={rec['reproj_error_px']:.1f}px"
                        )

                    # STB-11 세션 요약: 16점 재투영 오차의 RMSE
                    self.calib_reproj_rmse_px = float(
                        np.sqrt(
                            np.mean(
                                [rec["reproj_error_px"] ** 2 for rec in self.point_records]
                            )
                        )
                    )
                    edge_errors = []
                    center_errors = []

                    for rec in self.point_records:
                        error = rec["reproj_error_px"]

                        if error is None:
                            continue

                        is_edge = (
                            rec["point_x_norm"] <= 0.2
                            or rec["point_x_norm"] >= 0.8
                            or rec["point_y_norm"] <= 0.2
                            or rec["point_y_norm"] >= 0.8
                        )

                        if is_edge:
                            edge_errors.append(error)
                        else:
                            center_errors.append(error)

                    edge_mean_error = (
                        float(np.mean(edge_errors))
                        if edge_errors
                        else None
                    )

                    center_mean_error = (
                        float(np.mean(center_errors))
                        if center_errors
                        else None
                    )
                    self.edge_mean_reproj_error_px = edge_mean_error
                    self.center_mean_reproj_error_px = center_mean_error

                    edge_text = (
                        f"{edge_mean_error:.1f}px"
                        if edge_mean_error is not None
                        else "N/A"
                    )

                    center_text = (
                        f"{center_mean_error:.1f}px"
                        if center_mean_error is not None
                        else "N/A"
                    )

                print(
                    "[calibration region error] "
                    f"edge_mean={edge_text}, "
                    f"center_mean={center_text}"
                )
               # 캘리브레이션 품질 검사
               # 캘리브레이션 품질 기준
                max_calib_rmse_px = 150.0

                calibration_valid = (
                    self.H is not None
                    and self.calib_reproj_rmse_px is not None
                    and self.calib_reproj_rmse_px <= max_calib_rmse_px
                )

                if calibration_valid:

                    self.calibration_fallback_used = False
                    self.rejected_calib_rmse_px = None
                    self.applied_calib_rmse_px = self.calib_reproj_rmse_px

                    # 새 캘리브레이션 결과가 정상이면 백업 갱신
                    self.last_good_H = self.H.copy()
                    self.last_good_calib_rmse_px = (
                        self.calib_reproj_rmse_px
                    )
                    self.last_good_calib_id = self.calib_id

                    print(
                        "[calibration] 새 캘리브레이션 적용: "
                        f"homography RMSE "
                        f"{self.calib_reproj_rmse_px:.1f}px"
                    )

                else:

                    rmse_text = (
                        f"{self.calib_reproj_rmse_px:.1f}px"
                        if self.calib_reproj_rmse_px is not None
                        else "계산 불가"
                    )

                    # 이전 정상 결과가 있으면 새 불량 결과 대신 복구
                    if self.last_good_H is not None:

                        rejected_calib_id = self.calib_id
                        rejected_rmse = self.calib_reproj_rmse_px

                        self.calibration_fallback_used = True
                        self.rejected_calib_rmse_px = rejected_rmse
                        self.applied_calib_rmse_px = (
                            self.last_good_calib_rmse_px
                        )

                        self.H = self.last_good_H.copy()
                        self.calib_reproj_rmse_px = (
                            self.last_good_calib_rmse_px
                        )

                        print(
                            "[calibration warning] "
                            f"새 캘리브레이션 품질 불량: "
                            f"RMSE {rmse_text}"
                        )
                        print(
                            "[calibration fallback] "
                            "마지막 정상 Homography를 계속 사용합니다."
                        )
                        print(
                            "[calibration fallback] "
                            f"rejected_calib_id={rejected_calib_id}, "
                            f"restored_calib_id="
                            f"{self.last_good_calib_id}"
                        )

                    # 이전 정상 결과가 없는 첫 실행
                    elif self.H is not None:
                        self.calibration_fallback_used = False
                        self.rejected_calib_rmse_px = None
                        self.applied_calib_rmse_px = (
                            self.calib_reproj_rmse_px
                        )

                        print(
                            "[calibration warning] "
                            f"첫 캘리브레이션 RMSE가 높습니다: "
                            f"{rmse_text}"
                        )
                        print(
                            "[calibration warning] "
                            "이전 정상 결과가 없어 현재 결과를 "
                            "임시로 사용합니다."
                        )
                        print(
                            "[calibration warning] "
                            "커서 상태가 좋지 않으면 r 키로 "
                            "다시 캘리브레이션하세요."
                        )

                    # Homography 계산 자체가 실패했고 백업도 없는 경우
                    else:

                        print(
                            "[calibration] "
                            "Homography 계산에 실패했고 "
                            "이전 정상 결과도 없습니다."
                        )
                        print(
                            "[calibration] "
                            "캘리브레이션을 다시 시작합니다."
                        )

                        self.reset()
                        return 0.0

               

                # ── 릿지 회귀 학습 (홈그래피와 병렬, 서로 독립) ──
                self._fit_ridge()

                # 캘리브레이션 당시 평균 head pose baseline 저장
                valid_pose_pts = [
                    p for p in self.pose_pts
                    if p is not None and len(p) >= 7 and p[3] > 0
                ]

                if valid_pose_pts:
                    self.pose_baseline = np.mean(
                        np.array(valid_pose_pts, dtype=np.float32),
                        axis=0
                    ).tolist()
                else:
                    self.pose_baseline = None

                print("[calibration] pose_baseline:", self.pose_baseline)

                self.export_calibration_quality_csv()

                self.done = True

        return elapsed / (
            CALIB_STABILIZE_SEC +
            CALIB_COLLECT_SEC
        )

    # ── 릿지 학습 및 재투영 오차 ─────────────────────────────

    def _fit_ridge(self):
        """
        캘리브레이션 완료 시점에 홈그래피와 병렬로 릿지 매퍼를 학습합니다.
        특징 샘플이 부족하면(백본/머리자세 invalid로 features가 안 쌓인 경우)
        조용히 건너뛰고, 릿지 모드에서 홈그래피 좌표로 폴백됩니다.
        """

        if len(self.ridge_X) < 8:
            print(f"[ridge] 특징 샘플 부족({len(self.ridge_X)}) → 릿지 미학습 (raw/pose 모드는 정상)")
            return

        X = np.array(self.ridge_X, dtype=np.float64)
        Y = np.array(self.ridge_Y, dtype=np.float64)

        ok = self.ridge.fit(X, Y)

        if not ok:
            print("[ridge] 학습 실패")
            return

        # 릿지 재투영 오차: 점별 평균 특징으로 예측한 좌표와 타깃 거리
        # (홈그래피 STB-11과 같은 정의를 유지하되, 릿지는 프레임 단위로 학습했으므로
        #  점별로 프레임 예측 오차를 평균하여 기록)
        errors_by_point = {}

        preds = []
        for f in X:
            p = self.ridge.predict(f)
            preds.append(p if p is not None else (np.nan, np.nan))

        preds = np.array(preds, dtype=np.float64)
        errs = np.hypot(preds[:, 0] - Y[:, 0], preds[:, 1] - Y[:, 1])

        # 타깃 좌표를 키로 점 단위 평균 오차 계산
        for e, y in zip(errs, Y):
            key = (round(y[0], 1), round(y[1], 1))
            errors_by_point.setdefault(key, []).append(e)

        for rec in self.point_records:
            key = (
                round(rec["point_x_norm"] * SCREEN_W, 1),
                round(rec["point_y_norm"] * SCREEN_H, 1),
            )
            if key in errors_by_point:
                rec["ridge_reproj_error_px"] = float(
                    np.mean(errors_by_point[key])
                )

        valid_errs = errs[np.isfinite(errs)]

        if len(valid_errs) > 0:
            self.ridge_reproj_rmse_px = float(
                np.sqrt(np.mean(valid_errs ** 2))
            )

        print(
            f"[ridge] 학습 완료: 샘플 {len(X)}개, "
            f"reproj RMSE {self.ridge_reproj_rmse_px:.1f}px "
            f"(homography RMSE: {self.calib_reproj_rmse_px})"
        )

    def get_pose_delta(self, head_pose):
        """
        캘리브레이션 당시 head pose와 현재 head pose의 차이를 반환합니다.

        pose_baseline 구조:
            [0] yaw
            [1] pitch
            [2] roll
            [3] face_scale
            [4] tz
            [5] face_center_x
            [6] face_center_y
        """

        if self.pose_baseline is None:
            return None

        if head_pose is None or not head_pose.get("valid", False):
            return None

        if len(self.pose_baseline) < 7:
            return None

        base_yaw = self.pose_baseline[0]
        base_pitch = self.pose_baseline[1]
        base_roll = self.pose_baseline[2]
        base_scale = self.pose_baseline[3]
        base_center_x = self.pose_baseline[5]
        base_center_y = self.pose_baseline[6]

        current_yaw = head_pose.get("yaw", base_yaw)
        current_pitch = head_pose.get("pitch", base_pitch)
        current_roll = head_pose.get("roll", base_roll)
        current_scale = head_pose.get("face_scale", base_scale)
        current_center_x = head_pose.get("face_center_x", base_center_x)
        current_center_y = head_pose.get("face_center_y", base_center_y)

        return {
            "delta_yaw": current_yaw - base_yaw,
            "delta_pitch": current_pitch - base_pitch,
            "delta_roll": current_roll - base_roll,
            "delta_scale": current_scale - base_scale,
            "delta_center_x": current_center_x - base_center_x,
            "delta_center_y": current_center_y - base_center_y,
        }

    def compensate_iris_by_head_pose(
        self,
        iris_x,
        iris_y,
        head_pose
    ):
        """
        회귀 모델 전 단계의 규칙 기반 보정입니다.

        screen 좌표를 직접 움직이지 않고,
        map_to_screen()에 들어가기 전의 iris 입력값을 약하게 보정합니다.

        목적:
            얼굴 전체가 카메라 화면 안에서 이동한 양을 iris 좌표에서 일부 제거합니다.
            이렇게 하면 얼굴 이동이 커서 이동으로 섞이는 현상을 줄일 수 있습니다.

        주의:
            이 함수는 완전한 3D gaze 보정이 아닙니다.
            face_center와 face_scale을 이용한 약한 보정입니다.
        """

        if iris_x is None or iris_y is None:
            return iris_x, iris_y

        if self.pose_baseline is None:
            return iris_x, iris_y

        if head_pose is None or not head_pose.get("valid", False):
            return iris_x, iris_y

        if len(self.pose_baseline) < 7:
            return iris_x, iris_y

        base_yaw = self.pose_baseline[0]
        base_pitch = self.pose_baseline[1]

        base_scale = self.pose_baseline[3]
        base_center_x = self.pose_baseline[5]
        base_center_y = self.pose_baseline[6]

        current_yaw = head_pose.get("yaw", base_yaw)
        current_pitch = head_pose.get("pitch", base_pitch)

        current_scale = head_pose.get("face_scale", base_scale)
        current_center_x = head_pose.get("face_center_x", base_center_x)
        current_center_y = head_pose.get("face_center_y", base_center_y)

        if base_scale <= 0 or current_scale <= 0:
            return iris_x, iris_y
        
        delta_yaw = current_yaw - base_yaw
        delta_pitch = current_pitch - base_pitch

        # 얼굴 중심 이동량
        delta_center_x = current_center_x - base_center_x
        delta_center_y = current_center_y - base_center_y

        # 작은 얼굴 중심 흔들림은 보정하지 않음
        # MediaPipe landmark 노이즈가 커서에 섞이는 것을 방지합니다.
        center_deadband_x = 0.015
        center_deadband_y = 0.015

        if abs(delta_center_x) < center_deadband_x:
            delta_center_x = 0.0

        if abs(delta_center_y) < center_deadband_y:
            delta_center_y = 0.0

        # 얼굴 크기 변화 비율
        scale_ratio = current_scale / base_scale

        # 너무 큰 자세/거리 변화는 보정하지 않음
        # 이 경우는 보정보다 재캘리브레이션 대상에 가깝습니다.
        if abs(delta_center_x) > 0.25:
            return iris_x, iris_y

        if abs(delta_center_y) > 0.25:
            return iris_x, iris_y

        if scale_ratio < 0.7 or scale_ratio > 1.4:
            return iris_x, iris_y

        # 보정 강도
        # 처음에는 약하게 시작해야 합니다.
        center_gain_x = 0.30
        center_gain_y = 0.30

        yaw_gain = 0.0025
        pitch_gain = 0.0020

        corrected_iris_x = (
            iris_x
            - delta_center_x * center_gain_x
            - delta_yaw * yaw_gain
            - (scale_ratio - 1.0) * 0.04
        )

        corrected_iris_y = (
            iris_y
            - delta_center_y * center_gain_y
            - delta_pitch * pitch_gain
            - (scale_ratio - 1.0) * 0.02
        )

        # 얼굴 거리 변화 보정
        # 얼굴이 가까워져 face_scale이 커지면 iris 변화가 과장될 수 있으므로
        # 0.5 중심 기준으로 아주 약하게 정규화합니다.
        scale_gain = 0.35

        corrected_iris_x = 0.5 + (
            corrected_iris_x - 0.5
        ) / (
            1.0 + (scale_ratio - 1.0) * scale_gain
        )

        corrected_iris_y = 0.5 + (
            corrected_iris_y - 0.5
        ) / (
            1.0 + (scale_ratio - 1.0) * scale_gain
        )

        corrected_iris_x = float(
            np.clip(corrected_iris_x,0.0,1.0)
        )

        corrected_iris_y = float(
            np.clip(corrected_iris_y,0.0,1.0)
        )

        # -----------------------------
        # EMA
        # -----------------------------
        alpha = 0.35

        if self.prev_corrected_iris_x is None:

            self.prev_corrected_iris_x = corrected_iris_x
            self.prev_corrected_iris_y = corrected_iris_y

        else:

            corrected_iris_x = (
                alpha * corrected_iris_x +
                (1-alpha) * self.prev_corrected_iris_x
            )

            corrected_iris_y = (
                alpha * corrected_iris_y +
                (1-alpha) * self.prev_corrected_iris_y
            )

            self.prev_corrected_iris_x = corrected_iris_x
            self.prev_corrected_iris_y = corrected_iris_y

        return corrected_iris_x, corrected_iris_y

    def apply_head_pose_correction(
        self,
        screen_x,
        screen_y,
        head_pose
    ):
        """
        screen 좌표에 head pose를 직접 더하는 방식은 사용하지 않습니다.

        이전 방식:
            corrected_x = screen_x + delta_yaw * alpha

        문제:
            커서가 시선이 아니라 고개 움직임을 따라갔습니다.

        현재 방식:
            screen 좌표 보정은 하지 않습니다.
            대신 compensate_iris_by_head_pose()에서
            map_to_screen() 이전 iris 입력값을 약하게 보정합니다.
        """

        return screen_x, screen_y

    def map_to_screen(
        self,
        iris_x,
        iris_y
    ):

        if self.H is None:
            return None, None

        pt = np.array(
            [[[iris_x, iris_y]]],
            dtype=np.float32
        )

        result = cv2.perspectiveTransform(
            pt,
            self.H
        )

        screen_x = int(
            np.clip(
                result[0][0][0],
                0,
                SCREEN_W - 1
            )
        )

        screen_y = int(
            np.clip(
                result[0][0][1],
                0,
                SCREEN_H - 1
            )
        )

        return (
            screen_x,
            screen_y
        )

    def map_to_screen_features(self, features):
        """
        하이브리드(릿지) 매핑: 특징 벡터 → 화면 좌표.

        map_to_screen()과 동일한 계약(반환/클립)을 지킵니다:
            - 미학습/입력 결손 시 (None, None)
            - 정상 시 화면 범위로 clip된 int 좌표

        main.py에서 릿지 모드일 때 이 함수 결과가 None이면
        기존 map_to_screen() 결과로 폴백하도록 되어 있으므로,
        릿지가 실패해도 커서가 죽지 않습니다.
        """

        if not self.ridge.fitted or features is None:
            return None, None

        pred = self.ridge.predict(features)

        if pred is None:
            return None, None

        px, py = pred

        if not (np.isfinite(px) and np.isfinite(py)):
            return None, None

        screen_x = int(np.clip(px, 0, SCREEN_W - 1))
        screen_y = int(np.clip(py, 0, SCREEN_H - 1))

        return screen_x, screen_y

    def export_calibration_quality_csv(
        self,
        path=None
    ):
        """
        STB-11(재투영 오차)/STB-12(입력 신호 안정성)/STB-13(수집 품질)을
        캘리브레이션 포인트당 1행으로 저장합니다. findHomography 직후 호출됩니다.

        v1.1: ridge_reproj_error_px, feature_sample_count 컬럼이 추가되었습니다.
        """

        if path is None:
            path = (
                f"gaze_accuracy_results/calibration_quality_v"
                f"{self.CALIB_SCHEMA_VERSION}.csv"
            )

        os.makedirs(
            os.path.dirname(path),
            exist_ok=True
        )

        fieldnames = [
            "calib_id",
            "calib_point_index",
            "point_x_norm",
            "point_y_norm",
            "region",
            "reproj_error_px",
            "ridge_reproj_error_px",
            "iris_std_x_norm",
            "iris_std_y_norm",
            "sample_count",
            "used_sample_count",
            "removed_sample_count",
            "calibration_retry_count",
            "feature_sample_count",
            "mean_confidence",
            "schema_version",
        ]

        rows = [
            {
                "calib_id": self.calib_id,
                "calib_point_index": rec["calib_point_index"],
                "point_x_norm": rec["point_x_norm"],
                "point_y_norm": rec["point_y_norm"],
                "region": rec.get("region"),
                "reproj_error_px": (
                    round(rec["reproj_error_px"], 2)
                    if rec["reproj_error_px"] is not None
                    else None
                ),
                "ridge_reproj_error_px": (
                    round(rec["ridge_reproj_error_px"], 2)
                    if rec.get("ridge_reproj_error_px") is not None
                    else None
                ),
                "iris_std_x_norm": round(rec["iris_std_x_px"], 5),
                "iris_std_y_norm": round(rec["iris_std_y_px"], 5),
                "sample_count": rec["sample_count"],
                "used_sample_count": rec.get(
                    "used_sample_count",
                    rec["sample_count"]
                ),
                "removed_sample_count": rec.get(
                    "removed_sample_count",
                    0
                ),
                "calibration_retry_count": rec.get(
                    "calibration_retry_count",
                    0
                ),
                "feature_sample_count": rec.get("feature_sample_count", 0),
                "mean_confidence": (
                    round(rec["mean_confidence"], 4)
                    if rec["mean_confidence"] is not None
                    else None
                ),
                "schema_version": self.CALIB_SCHEMA_VERSION,
            }
            for rec in self.point_records
        ]

        append_rows(path, fieldnames, rows)