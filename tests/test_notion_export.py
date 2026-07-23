"""notion_export.py의 BLOCK_EXPERIMENT NaN 직렬화 회귀 테스트.

pytest가 설치되어 있지 않아(2026-07-08 확인) 표준 라이브러리 unittest로 작성.
실행: python -m unittest tests.test_notion_export
"""
import json
import os
import unittest
from unittest.mock import patch

import pandas as pd

from src.analysis.notion_export import _build_block_experiment_section, upload_diagnostic_to_notion


def _block_report_fixture():
    """frozen=False 행이 섞인 block_report — 이 케이스가 실제 크래시를 재현했다."""
    return {
        "variance_decomposition": {
            "ratio": 0.42, "grand_mean": 55.3, "n_blocks_used": 2, "n_sessions_used": 3,
            "icc1": 0.31, "blocks_excluded": [], "sessions_excluded": [],
            "per_block": [{"block_id": "abc12345", "mean": 50.0, "sd": 2.0, "n": 2}],
        },
        "block_session_stats": [
            {"block_id": "abc12345", "test_idx": 1, "session_id": "sess0001",
             "mean_error_px": 50.0, "gain_x": 1.0, "gain_y": 1.0, "clip_n": 0, "frozen": False},
            {"block_id": "abc12345", "test_idx": 2, "session_id": "sess0002",
             "mean_error_px": 55.0, "gain_x": 1.1, "gain_y": 0.9, "clip_n": 1, "frozen": True},
            {"block_id": "cb1ec59d", "test_idx": 1, "session_id": "sess0003",
             "mean_error_px": 60.0, "gain_x": 1.0, "gain_y": 1.0, "clip_n": 0, "frozen": False},
        ],
        "trend": [{"block_id": "cb1ec59d", "n_trials": 2, "spearman_r": float("nan")}],
    }


class BuildBlockExperimentSectionTest(unittest.TestCase):
    """근본 원인(frozen 컬러 매핑) 회귀 테스트."""

    def test_frozen_false_rows_produce_json_serializable_blocks(self):
        blocks = _build_block_experiment_section(_block_report_fixture())
        try:
            json.dumps(blocks, allow_nan=False)
        except ValueError as e:
            self.fail(f"블록 실험 섹션이 JSON 직렬화 불가능한 값을 포함합니다: {e}")


class UploadDiagnosticToNotionBoundaryDefenseTest(unittest.TestCase):
    """payload 전체를 _json_safe로 정제하는 경계 방어 테스트.

    diagnostic.py의 diagnostic_battery/_prepare_scope/build_session_summary_table을
    mock으로 대체해 실제 CSV·matplotlib 없이 requests.post에 전달되는 payload의
    직렬화 가능성만 검증한다 (requests 내부와 동일하게 allow_nan=False).
    """

    def test_payload_is_json_serializable_even_with_block_report(self):
        report_fixture = {
            "correlation": {"r": float("nan"), "n": 2, "table": []},
            "row_compression": [],
            "stuck_at_boundary": {"flagged_rows": [], "screen_w": 1536, "screen_h": 864},
            "axis_anomaly": [],
            "stb_summary": [],
        }
        a_fixture = pd.DataFrame({"session_id": ["sess0001"], "euclidean_error_px": [10.0]})
        df_fixture = pd.DataFrame({"session_id": ["sess0001"], "calib_id": ["abc12345"]})
        sid_to_calib_fixture = pd.Series({"sess0001": "abc12345"})
        summary_df_fixture = pd.DataFrame({
            "session_id": ["sess0001"], "mean_error_px": [10.0],
            "stuck_count": [0], "anomaly_count": [0],
        })

        with patch.dict(os.environ, {"NOTION_TOKEN": "fake-token"}), \
             patch("src.analysis.diagnostic.diagnostic_battery", return_value=report_fixture), \
             patch("src.analysis.diagnostic._prepare_scope",
                   return_value=(df_fixture, a_fixture, sid_to_calib_fixture)), \
             patch("src.analysis.diagnostic.build_session_summary_table",
                   return_value=summary_df_fixture), \
             patch("src.analysis.notion_export.requests.post") as mock_post:
            mock_post.return_value.raise_for_status.return_value = None
            mock_post.return_value.json.return_value = {"url": "https://notion.so/fake"}

            upload_diagnostic_to_notion(
                database_id="fake-db-id",
                sessions=df_fixture, acc=a_fixture, calib_df=pd.DataFrame(),
                session_ids=["sess0001"], label="test",
                include_images=False, block_report=_block_report_fixture(),
            )

        self.assertTrue(mock_post.called)
        payload = mock_post.call_args.kwargs["json"]
        try:
            json.dumps(payload, allow_nan=False)
        except ValueError as e:
            self.fail(f"requests.post에 전달된 payload가 직렬화 불가능합니다: {e}")


if __name__ == "__main__":
    unittest.main()
