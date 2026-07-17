# -*- coding: utf-8 -*-
"""diagnostic.py / notion_export.py 실행 전용 드라이버 스크립트.

실행 (프로젝트 루트에서):
python -m src.analysis.run_export

사용 매뉴얼 — session_ids 지정 방법
----------------------------------
공통 주의사항: 어떤 방식을 쓰든, diagnostic_battery() 호출 전에 print(target_ids)
또는 len(target_ids)로 실제 몇 개가 걸렸는지 먼저 확인할 것
조건이 잘못돼도 빈 리스트나 의도치 않은 세션이 에러 없이 섞일 수 있음.

── 1. 전체 세션 ──
target_ids = sessions["session_id"].tolist()

── 2. 특정 세션 지정 (session_id 앞자리로 매칭) ──
target_ids = sessions[
    sessions["session_id"].str.startswith(("67321a57", "fadd3b55"))
]["session_id"].tolist()

── 3. 조건별 필터링 예시 ──
(a) 날짜 범위로 필터링 (start_timestamp 기준)
ts = pd.to_datetime(sessions["start_timestamp"])
target_ids = sessions[
    (ts >= "2026-07-01") & (ts < "2026-07-05")
]["session_id"].tolist()

(b) 특정 calib_id 하나에 속한 세션들만 (3D on/off처럼 캘리브레이션
    재사용 케이스에서, 그 캘리브레이션을 쓴 세션 전부 보고 싶을 때)
target_ids = sessions[
    sessions["calib_id"] == "dd69b635-xxxx-..."
]["session_id"].tolist()

(c) 특정 세션 제외 (전체에서 불량 세션 하나만 빼고 싶을 때)
target_ids = sessions[
    ~sessions["session_id"].str.startswith("67321a57")
]["session_id"].tolist()

(d) dev_version 기준 필터링
target_ids = sessions[
    sessions["dev_version"] == "v0.3-median"
]["session_id"].tolist()

── 4. 최근 N세션 (session_id를 직접 작성하지 않고) ──
make_report.py의 -n 옵션과 같은 로직.
target_ids = viz.latest_sessions(sessions, n=5)["session_id"].tolist()

── 5. 블록 실험(캘리브레이션 여러 개 x 회차) 세션 한 번에 지정 ──
BLOCK_EXPERIMENT=True로 build_block_experiment_report()를 쓸 때 사용.
(a) 날짜 범위로 그 날 전체 세션 잡기 (가장 간단, calib_id를 몰라도 됨)
ts = pd.to_datetime(sessions["start_timestamp"])
target_ids = sessions[
    (ts >= "2026-07-08") & (ts < "2026-07-09")
]["session_id"].tolist()
print(len(target_ids))  # 예상 세션 수와 일치하는지 먼저 확인할 것

(b) 알고 있는 calib_id 여러 개를 직접 나열 (그 날 다른 목적의 세션이 섞여
    있어 (a)가 부정확할 때 더 정확)
BLOCK_CALIB_IDS = ["xxxxxxxx-...", ...]
target_ids = sessions[sessions["calib_id"].isin(BLOCK_CALIB_IDS)]["session_id"].tolist()

"""

import os

import pandas as pd
from dotenv import load_dotenv

from src.viz import viz
from src.viz import calib_viz
from src.analysis.diagnostic import (
    diagnostic_battery, save_diagnostic_report, build_diagnostic_html, build_block_experiment_report,
)
from src.analysis.notion_export import upload_diagnostic_to_notion

load_dotenv()


# ══════════════════════════════════════════════════════════════
# 데이터 로딩
# ══════════════════════════════════════════════════════════════
SCHEMA_VERSION = "1.3"
CALIB_SCHEMA_VERSION = "1.0"
RESULTS_DIR = "gaze_accuracy_results"

sessions = pd.read_csv(f"{RESULTS_DIR}/sessions_v{SCHEMA_VERSION}.csv", encoding="utf-8-sig")
acc = pd.read_csv(f"{RESULTS_DIR}/gaze_accuracy_v{SCHEMA_VERSION}.csv", encoding="utf-8-sig")
calib_df = pd.read_csv(
    f"{RESULTS_DIR}/{calib_viz.calibration_quality_filename()}", encoding="utf-8-sig"
)


# ══════════════════════════════════════════════════════════════
# ↓↓↓ 실행할 때마다 여기만 고치면 됨 ↓↓↓
# ══════════════════════════════════════════════════════════════
ts = pd.to_datetime(sessions["start_timestamp"])
target_ids = sessions[
    (ts >= "2026-07-08") & (ts < "2026-07-09")
]["session_id"].tolist()
print(len(target_ids))
LABEL = "0708-raw"
GROUP_BY = "session_id"  # 같은 캘리브레이션 재사용 비교 시 "session_id"

# HTML 리포트는 기본적으로 생성하지 않지만, 함수 자체는 diagnostic.py에 남겨둠
# Notion API 장애·계정 문제 등으로 확인이 필요할 때, 또는 Notion 업로드 결과를 교차 검증하고 싶을 때만 True로 켤 것.
GENERATE_HTML = False

# 블록 실험(캘리브레이션 1회 + raw 9점 테스트 N회 = 1블록) 분석을 함께 낼지 여부.
# False면 기존 워크플로와 완전히 동일하게 동작한다.
BLOCK_EXPERIMENT = True
# None(기본값): frozen 세션이 있는 블록 자동 제외. []: 자동 제외 끄고 전체 사용.
# 특정 block_id(=calib_id) 리스트를 직접 넘기면 그 블록들만 수동 제외.
BLOCK_EXCLUDE_IDS = None
# ══════════════════════════════════════════════════════════════


# ── 로컬 산출물 ──
report = diagnostic_battery(sessions, acc, session_ids=target_ids, group_by=GROUP_BY)
save_diagnostic_report(report, label=LABEL)

if GENERATE_HTML:
    build_diagnostic_html(sessions, acc, calib_df, session_ids=target_ids, label=LABEL, group_by=GROUP_BY)

block_report = None
if BLOCK_EXPERIMENT:
    block_report = build_block_experiment_report(
        sessions, acc, session_ids=target_ids, exclude_blocks=BLOCK_EXCLUDE_IDS,
    )
    save_diagnostic_report(block_report, label=f"{LABEL}_block")

# ── Notion 업로드 ──
DATABASE_ID = os.environ["NOTION_DATABASE_ID"]
upload_diagnostic_to_notion(
    DATABASE_ID, sessions, acc, calib_df, target_ids, label=LABEL, group_by=GROUP_BY,
    include_images=True, block_report=block_report,
)