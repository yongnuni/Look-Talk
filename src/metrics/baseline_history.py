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
