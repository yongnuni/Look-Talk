"""pose_agg.py 검증 스크립트.

pytest가 설치되어 있지 않아(2026-07-08 확인, test_notion_export.py 참고) 표준 라이브러리
assert로 작성.
실행: python -m tests.test_pose_agg
"""
import math

from src.metrics.pose_agg import aggregate_angles, aggregate_pose_frames, aggregate_scalars

EXPECTED_KEYS = {
    "pose_yaw_mean_deg", "pose_yaw_std_deg",
    "pose_pitch_mean_deg", "pose_pitch_std_deg",
    "pose_roll_mean_deg", "pose_roll_std_deg",
    "face_scale_mean", "face_scale_std",
    "face_center_x_mean", "face_center_x_std",
    "face_center_y_mean", "face_center_y_std",
    "pose_valid_n", "pose_flip_n",
}


def test_circular_boundary_no_flip():
    """179/-179/178은 실제로는 179도대 인접값 — 경계 넘김을 flip으로 오판하면 안 된다."""
    frames = [(179, 0, 0), (-179, 0, 0), (178, 0, 0)]
    r = aggregate_angles(frames)
    assert r["flip_n"] == 0, r
    assert abs(r["yaw_mean"] - 179.3333333) < 0.01, r
    # 원형 median(179) 기준 kept delta=[0,2,-1] → stdev(ddof=1) 독립 계산값과 대조
    assert abs(r["yaw_std"] - 1.5275) < 0.01, r


def test_even_n_boundary_pair():
    """짝수 n에서 산술 median(0.0)을 쓰면 179/-179 둘 다 flip으로 오판되는 버그의 회귀 테스트."""
    frames = [(179, 0, 0), (-179, 0, 0)]
    r = aggregate_angles(frames)
    assert r["flip_n"] == 0, r
    assert abs(abs(r["yaw_mean"]) - 180.0) < 0.01, r
    assert abs(r["yaw_std"] - math.sqrt(2)) < 1e-6, r


def test_even_n_boundary_cluster():
    frames = [(178, 0, 0), (179, 0, 0), (-179, 0, 0), (-178, 0, 0)]
    r = aggregate_angles(frames)
    assert r["flip_n"] == 0, r
    mean = r["yaw_mean"]
    assert (179.0 <= mean <= 180.0) or (-180.0 <= mean <= -179.0), r
    assert not math.isnan(r["yaw_std"]) and r["yaw_std"] < 5.0, r


def test_single_flip_excluded():
    frames = [(-5, 0, 0), (-3, 0, 0), (175, 0, 0), (-4, 0, 0)]
    r = aggregate_angles(frames)
    assert r["flip_n"] == 1, r
    assert abs(r["yaw_mean"] - (-4.0)) < 0.01, r
    assert not math.isnan(r["yaw_std"]), r


def test_stable_170s_not_misjudged_as_outlier():
    """head_pose.py 주석 사례: 170도대에서 안정적인 환경을 이상치로 오판하지 않는다."""
    frames = [(170, 0, 0), (172, 0, 0), (171, 0, 0)]
    r = aggregate_angles(frames)
    assert r["flip_n"] == 0, r
    assert abs(r["yaw_mean"] - 171.0) < 0.01, r


def test_frame_level_exclusion_across_axes():
    """pitch 하나만 flip이어도 해당 프레임이 yaw/roll 집계에서도 함께 빠져야 한다."""
    frames = [(0, 0, 0), (0, 0, 0), (5, 170, -3)]
    r = aggregate_angles(frames)
    assert r["flip_n"] == 1, r
    # 세 번째 프레임이 빠지면 yaw/roll은 (0,0) 두 프레임만 남아 mean=0, std=0
    assert r["yaw_mean"] == 0.0 and r["yaw_std"] == 0.0, r
    assert r["roll_mean"] == 0.0 and r["roll_std"] == 0.0, r


def test_empty_frames_all_nan_no_exception():
    r = aggregate_pose_frames([])
    for k in EXPECTED_KEYS - {"pose_valid_n", "pose_flip_n"}:
        assert math.isnan(r[k]), (k, r)
    assert r["pose_valid_n"] == 0, r
    assert r["pose_flip_n"] == 0, r


def test_single_frame_mean_defined_std_nan():
    frame = (10.0, 5.0, -2.0, 100.0, 50.0, 0.5, 0.4)
    r = aggregate_pose_frames([frame])
    assert abs(r["pose_yaw_mean_deg"] - 10.0) < 1e-9, r
    assert abs(r["pose_pitch_mean_deg"] - 5.0) < 1e-9, r
    assert abs(r["pose_roll_mean_deg"] - (-2.0)) < 1e-9, r
    assert math.isnan(r["pose_yaw_std_deg"]), r
    assert math.isnan(r["face_scale_std"]), r
    assert r["pose_flip_n"] == 0, r
    assert r["pose_valid_n"] == 1, r


def test_ddof1_not_ddof0():
    frames = [(0, 0, 0), (2, 0, 0)]
    r = aggregate_angles(frames, ddof=1)
    assert abs(r["yaw_std"] - math.sqrt(2)) < 1e-6, r  # ddof=0이면 1.0이 나와야 함


def test_mean_normalization_wraps_into_range():
    frames = [(176, 0, 0), (184, 0, 0), (184, 0, 0)]
    r = aggregate_angles(frames)
    assert r["flip_n"] == 0, r
    assert -180.0 < r["yaw_mean"] <= 180.0, r
    assert abs(r["yaw_mean"] - (-178.6667)) < 0.01, r


def test_aggregate_pose_frames_key_contract():
    frames = [
        (1.0, 2.0, 3.0, 100.0, 50.0, 0.5, 0.5),
        (1.0, 2.0, 3.0, 100.0, 50.0, 0.5, 0.5),
    ]
    r = aggregate_pose_frames(frames)
    assert set(r.keys()) == EXPECTED_KEYS, set(r.keys()) ^ EXPECTED_KEYS
    assert not any("tz" in k for k in r.keys()), r.keys()
    assert isinstance(r["pose_valid_n"], int), r
    assert isinstance(r["pose_flip_n"], int), r
    for k in EXPECTED_KEYS - {"pose_valid_n", "pose_flip_n"}:
        assert isinstance(r[k], float), (k, r)


def test_scalars_not_filtered_by_angle_flip():
    """각도는 flip이어도 face_scale/face_center는 landmark 직접 산출이라 필터링되지 않는다."""
    frames = [
        (0, 0, 0, 100.0, 50.0, 0.5, 0.5),
        (0, 0, 0, 100.0, 50.0, 0.5, 0.5),
        (170, 0, 0, 130.0, 50.0, 0.5, 0.5),  # yaw만 flip, face_scale은 정상
    ]
    r = aggregate_pose_frames(frames)
    assert r["pose_flip_n"] == 1, r  # 각도 쪽은 flip 프레임 1개 제외됨
    # 스칼라는 3프레임 전부 포함되어야 mean=(100+100+130)/3=110
    assert abs(r["face_scale_mean"] - 110.0) < 1e-9, r


def _run_scalars_smoke_test():
    """aggregate_scalars 단독 호출도 계약대로 동작하는지 최소 확인."""
    r = aggregate_scalars([])
    assert math.isnan(r["scale_mean"]) and math.isnan(r["center_x_std"]), r
    r = aggregate_scalars([(100.0, 0.5, 0.4)])
    assert abs(r["scale_mean"] - 100.0) < 1e-9, r
    assert math.isnan(r["scale_std"]), r


if __name__ == "__main__":
    test_circular_boundary_no_flip()
    test_even_n_boundary_pair()
    test_even_n_boundary_cluster()
    test_single_flip_excluded()
    test_stable_170s_not_misjudged_as_outlier()
    test_frame_level_exclusion_across_axes()
    test_empty_frames_all_nan_no_exception()
    test_single_frame_mean_defined_std_nan()
    test_ddof1_not_ddof0()
    test_mean_normalization_wraps_into_range()
    test_aggregate_pose_frames_key_contract()
    test_scalars_not_filtered_by_angle_flip()
    _run_scalars_smoke_test()
    print("ALL PASSED")
