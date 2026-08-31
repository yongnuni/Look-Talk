import os

from src.calibrations.baseline_manager import load_baseline
from src.common import clock
from src.metrics.csv_export import append_rows


def append_mouth_baseline_history(run_id, saved_path):
    baseline = load_baseline(saved_path)

    if baseline is None:
        print(f"[baseline] 이력 CSV 기록 실패: {saved_path}를 다시 읽지 못함")
        return

    mouth_result = baseline.get("mouth") or {}

    row = {
        "run_id": run_id,
        "ts_ms": clock.now_ms(),
        "saved_at": baseline.get("saved_at"),
    }
    row.update(mouth_result)

    fieldnames = ["run_id", "ts_ms", "saved_at"] + list(mouth_result.keys())

    path = os.path.join(
        "calibration_results",
        "mouth_baseline_history_v2.0.csv"
    )

    history_dir = os.path.dirname(path)
    if history_dir:
        os.makedirs(history_dir, exist_ok=True)

    append_rows(
        path,
        fieldnames,
        [row],
    )


BLINK_HISTORY_FIELDNAMES = [
    "run_id",
    "ts_ms",
    "user_id",
    "open_ear_median",
    "closed_ear_median",
    "close_threshold",
    "open_threshold",
    "total_trials",
    "closed_sample_count",
    "fallback",
]


def append_blink_baseline_history(run_id, saved_path, user_id):
    """baseline.json의 "blink" 키를 별도 이력 CSV로 append한다.

    baseline.json은 저장 때마다 덮어써지므로 blink 개인 임계값은 이전 값이
    보존되지 않는다(docs/cali_review.md 4절). mouth_baseline_history의 스키마와
    저장 로직은 그대로 두고, blink만 별도 파일에 append해 이력을 남긴다.
    """
    baseline = load_baseline(saved_path)

    if baseline is None:
        print(f"[baseline] blink 이력 CSV 기록 실패: {saved_path}를 다시 읽지 못함")
        return

    blink_result = baseline.get("blink")
    if blink_result is None:
        return

    row = {
        "run_id": run_id,
        "ts_ms": clock.now_ms(),
        "user_id": user_id,
        "open_ear_median": blink_result.get("open_ear_median"),
        "closed_ear_median": blink_result.get("closed_ear_median"),
        "close_threshold": blink_result.get("close_threshold"),
        "open_threshold": blink_result.get("open_threshold"),
        "total_trials": blink_result.get("total_trials"),
        "closed_sample_count": blink_result.get("closed_sample_count"),
        "fallback": blink_result.get("calibration_failed_fallback"),
    }

    path = os.path.join(
        "calibration_results",
        "blink_baseline_history_v1.0.csv"
    )

    history_dir = os.path.dirname(path)
    if history_dir:
        os.makedirs(history_dir, exist_ok=True)

    append_rows(
        path,
        BLINK_HISTORY_FIELDNAMES,
        [row],
    )
