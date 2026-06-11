import json
import os
from datetime import datetime


DEFAULT_BASELINE_PATH = os.path.join(
    "calibration_results",
    "baseline.json"
)


def save_baseline(
    mouth_result,
    path=DEFAULT_BASELINE_PATH
):
    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    baseline = {
        "calibration_version": "v1",
        "saved_at": datetime.now().isoformat(),
        "mouth": mouth_result
    }

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            baseline,
            f,
            ensure_ascii=False,
            indent=2
        )

    return path


def load_baseline(
    path=DEFAULT_BASELINE_PATH
):
    if not os.path.exists(path):
        return None

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)