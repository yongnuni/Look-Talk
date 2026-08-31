import json
import os
from datetime import datetime


DEFAULT_BASELINE_PATH = os.path.join(
    "calibration_results",
    "baseline.json"
)


def save_baseline(
    mouth_result,
    path=DEFAULT_BASELINE_PATH,
    blink_result=None,
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

    # 기존 mouth-only 호출의 저장 형식은 유지하고, 통합 성능 플로우에서만
    # 같은 baseline 문서에 개인화된 blink EAR 임계값을 함께 보존한다.
    if blink_result is not None:
        baseline["blink"] = blink_result

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
