"""head pose 집계 순수 함수 모듈.

현재 상태: 이 모듈을 import하는 코드는 아직 없음.
지표 체계 개편 4~5단계에서 frames의 head_yaw/pitch/roll 및 calibration_quality의 head_*_mean 컬럼과 연결할 예정.

주의: 아래 설명에 등장하는 스키마 버전은 작성 시점 기준이며 현행과 다르다.
배선 시점에 calibration_quality / gaze_accuracy의 실제 버전으로 재확인할 것.

ddof=1을 기본값으로 쓰는 이유: STB-14와 STB-15는 서로 다른 파일(캘리브레이션 시점 vs
9점 테스트 시점)에 기록되지만 직접 비교 대상이므로 표본표준편차로 통일한다. 기존
STB-12(calibration.py의 iris std_x/y)는 np.std() 기본값인 ddof=0을 쓰는데, 이 모듈은
새 지표라 하위 호환을 고려할 필요가 없어 그것과 의도적으로 다르게 간다.

n 미달 시(mean은 n>=1, std는 n>=2 필요) 0.0이 아니라 float('nan')을 반환한다. 측정이
불가능한 상태를 정상값(0.0)으로 위장하지 않기 위함이며, 과거 iris_std가 샘플 부족 시
0.0을 반환해 "흔들림 없음"으로 잘못 읽힌 사례를 반복하지 않기 위해서다.

flip 필터: PnP 기반 pose 추정은 정면 부근에서 종종 각도를 반대쪽으로 잘못 잡는
("flip") 프레임을 섞어 넣는다. 이를 걸러내지 않으면 median이 아니라 mean이 크게
왜곡된다. 판정은 프레임 단위로 하며(세 축 중 하나라도 median 대비 90도를 넘게
벗어나면 그 프레임 전체를 세 축 집계 모두에서 제외), 세 축이 항상 같은 프레임
집합을 서술하도록 보장한다. 다만 이 판정 자체가 median에 의존하므로, flip된
프레임이 과반을 넘으면 median이 flip 클러스터 쪽에 앉아 정상 프레임이 오히려
제외될 수 있다 — pose_flip_n이 큰 행은 분석 단계에서 신뢰할 수 없는 것으로
취급해야 한다.

face_scale은 눈 사이 랜드마크 거리(px) 기반이라 카메라 해상도에 절대값이 종속된다.
해상도가 바뀌면(예: 1280x720 → 다른 값) face_scale 절대값을 세션 간 직접 비교할 수 없다.

face_center_x/y는 MediaPipe 정규화 좌표(0~1) 평균이다. 부호 방향: 사용자가 자신의
왼쪽으로 이동하면 x는 감소한다. yaw/pitch의 양의 방향이 실제로 어느 쪽을 의미하는지는
아직 실측으로 확정되지 않았다.
"""

from typing import Sequence

import numpy as np

NAN = float("nan")


def _normalize_angle(v: float) -> float:
    """각도를 (-180, 180] 범위로 되돌린다."""
    r = ((v + 180.0) % 360.0) - 180.0
    if r == -180.0:
        r = 180.0
    return r


def _circular_median(values: Sequence[float]) -> float:
    """표본 제한 원형 median: 후보를 표본값으로 한정해
    Σ|circ_delta(x, cand)|가 최소인 표본을 반환. 동률 시 첫 번째.
    산술 median은 짝수 n에서 ±180 경계 걸침 시 중점이 클러스터
    밖(예: 179와 -179의 중점 0)에 앉아 전 프레임을 flip으로 오판한다."""
    best, best_cost = values[0], float("inf")
    for cand in values:
        cost = sum(abs(((v - cand + 180.0) % 360.0) - 180.0) for v in values)
        if cost < best_cost:
            best, best_cost = cand, cost
    return float(best)


def aggregate_angles(frames: Sequence[tuple], ddof: int = 1) -> dict:
    """frames: (yaw, pitch, roll) 시퀀스 (단위: 도).

    내부 헬퍼 — 외부 소비자(calibration.py, collector.py)는 aggregate_pose_frames를
    사용할 것. 이 함수의 반환 키는 공개 계약이 아님.

    반환 키: yaw_mean, yaw_std, pitch_mean, pitch_std, roll_mean, roll_std, flip_n.
    """
    if len(frames) == 0:
        return {
            "yaw_mean": NAN, "yaw_std": NAN,
            "pitch_mean": NAN, "pitch_std": NAN,
            "roll_mean": NAN, "roll_std": NAN,
            "flip_n": 0,
        }

    n = len(frames)
    yaws = [f[0] for f in frames]
    pitches = [f[1] for f in frames]
    rolls = [f[2] for f in frames]

    med_yaw = _circular_median(yaws)
    med_pitch = _circular_median(pitches)
    med_roll = _circular_median(rolls)

    delta_yaw = [((yaws[i] - med_yaw + 180.0) % 360.0) - 180.0 for i in range(n)]
    delta_pitch = [((pitches[i] - med_pitch + 180.0) % 360.0) - 180.0 for i in range(n)]
    delta_roll = [((rolls[i] - med_roll + 180.0) % 360.0) - 180.0 for i in range(n)]

    # 프레임 단위 flip 판정 — 세 축 중 하나라도 median 대비 90도를 넘으면
    # 그 프레임을 세 축 통계 모두에서 함께 제외한다 (축별 개별 제외 금지).
    keep = [
        i for i in range(n)
        if abs(delta_yaw[i]) <= 90.0
        and abs(delta_pitch[i]) <= 90.0
        and abs(delta_roll[i]) <= 90.0
    ]
    flip_n = n - len(keep)

    def _mean_std(med, deltas):
        kept = [deltas[i] for i in keep]
        n_kept = len(kept)
        mean = _normalize_angle(med + float(np.mean(kept))) if n_kept >= 1 else NAN
        std = float(np.std(kept, ddof=ddof)) if n_kept >= 2 else NAN
        return mean, std

    yaw_mean, yaw_std = _mean_std(med_yaw, delta_yaw)
    pitch_mean, pitch_std = _mean_std(med_pitch, delta_pitch)
    roll_mean, roll_std = _mean_std(med_roll, delta_roll)

    return {
        "yaw_mean": yaw_mean, "yaw_std": yaw_std,
        "pitch_mean": pitch_mean, "pitch_std": pitch_std,
        "roll_mean": roll_mean, "roll_std": roll_std,
        "flip_n": flip_n,
    }


def aggregate_scalars(frames: Sequence[tuple], ddof: int = 1) -> dict:
    """frames: (face_scale, face_center_x, face_center_y) 시퀀스.

    flip 필터는 적용하지 않는다 — face_scale/face_center는 landmark 직접 산출이라
    PnP flip에 오염되지 않기 때문이다.

    내부 헬퍼 — 외부 소비자(calibration.py, collector.py)는 aggregate_pose_frames를
    사용할 것. 이 함수의 반환 키는 공개 계약이 아님.

    반환 키: scale_mean, scale_std, center_x_mean, center_x_std, center_y_mean, center_y_std.
    """
    n = len(frames)
    if n == 0:
        return {
            "scale_mean": NAN, "scale_std": NAN,
            "center_x_mean": NAN, "center_x_std": NAN,
            "center_y_mean": NAN, "center_y_std": NAN,
        }

    def _mean_std(values):
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=ddof)) if n >= 2 else NAN
        return mean, std

    scale_mean, scale_std = _mean_std([f[0] for f in frames])
    center_x_mean, center_x_std = _mean_std([f[1] for f in frames])
    center_y_mean, center_y_std = _mean_std([f[2] for f in frames])

    return {
        "scale_mean": scale_mean, "scale_std": scale_std,
        "center_x_mean": center_x_mean, "center_x_std": center_x_std,
        "center_y_mean": center_y_mean, "center_y_std": center_y_std,
    }


def aggregate_pose_frames(frames: Sequence[tuple], ddof: int = 1) -> dict:
    """공용 진입점.

    frames: (yaw, pitch, roll, face_scale, tz, face_center_x, face_center_y) 7-tuple
    시퀀스 (calibration.py의 self.pose_samples 원소와 동일 형식). tz는 이 집계에서
    의도적으로 제외한다.
    """
    angles = aggregate_angles([(f[0], f[1], f[2]) for f in frames], ddof=ddof)
    scalars = aggregate_scalars([(f[3], f[5], f[6]) for f in frames], ddof=ddof)

    return {
        "pose_yaw_mean_deg": angles["yaw_mean"],
        "pose_yaw_std_deg": angles["yaw_std"],
        "pose_pitch_mean_deg": angles["pitch_mean"],
        "pose_pitch_std_deg": angles["pitch_std"],
        "pose_roll_mean_deg": angles["roll_mean"],
        "pose_roll_std_deg": angles["roll_std"],
        "face_scale_mean": scalars["scale_mean"],
        "face_scale_std": scalars["scale_std"],
        "face_center_x_mean": scalars["center_x_mean"],
        "face_center_x_std": scalars["center_x_std"],
        "face_center_y_mean": scalars["center_y_mean"],
        "face_center_y_std": scalars["center_y_std"],
        "pose_valid_n": len(frames),
        "pose_flip_n": angles["flip_n"],
    }
