# -*- coding: utf-8 -*-
"""Phase 2: 다중 스키마 버전(sessions_v1.0~1.3) 통합 재분석 드라이버.

run_export.py와 역할이 다르다: run_export.py는 "최신 스키마 단일 버전 +
시각화 + Notion 업로드"용, 이 스크립트는 "스키마 이력 전체를 훑는 1회성
탐색/진단"용. 그림 생성·Notion 업로드 없음.

실행 (프로젝트 루트에서):
    python -m src.analysis.stage0_reanalysis
"""

import os
from datetime import date

from src.analysis.diagnostic import (
    load_all_sessions_history,
    summarize_px_per_cm_values,
    build_session_inventory,
    build_signal_vs_mapping_table,
    build_top_error_targets,
    shorten_session_id,
    save_csv_with_caveat,
    CLIP_CAVEAT_NOTE,
)

RESULTS_DIR = "gaze_accuracy_results"
OUT_DIR = os.path.join("src", "analysis", "diagnostics", f"stage0_{date.today():%Y%m%d}")

# ══ 1) 다중 버전 로드 ══
sessions_hist, acc_hist = load_all_sessions_history(RESULTS_DIR)
print(f"세션 이력 로드 완료: sessions={len(sessions_hist)}행, gaze_accuracy={len(acc_hist)}행")

# ══ 2) 방어 확인: sessions에는 있으나 acc에 타깃 행이 전혀 없는 세션 ══
# (아래 build_session_inventory는 acc_hist.groupby("session_id") 기준이라
# 이런 세션은 조용히 사라지므로 별도로 드러낸다.)
missing_acc = set(sessions_hist["session_id"]) - set(acc_hist["session_id"])
print(f"\n[방어 확인] gaze_accuracy 이력에 타깃 행이 없는 session_id: {len(missing_acc)}건")
if missing_acc:
    print(sorted(missing_acc))

# ══ 3) px_per_cm 고유값 분포 (등급 기준값 재검토용) ══
print("\n[px_per_cm 고유값 분포]")
print(summarize_px_per_cm_values(sessions_hist).to_string())

# ══ 4) 세션 인벤토리 + 비교가능성 2축 + clip_suspect + 게인 ══
inventory = build_session_inventory(sessions_hist, acc_hist)
print(f"\n[세션 인벤토리] {len(inventory)}세션")
print("[비교가능성 2축 조합 분포 (cmp_px x cmp_cm)]")
print(inventory.groupby(["cmp_px", "cmp_cm"]).size().to_string())
print(f"clip_suspect 세션: {int(inventory['clip_suspect'].sum())} / {len(inventory)}")

# ══ 5) 신호 vs 매핑 판별표 (cmp_px=True 세션 전부) ══
signal_vs_mapping = build_signal_vs_mapping_table(inventory, acc_hist)
print(f"\n[신호 vs 매핑] cmp_px=True 세션 {len(signal_vs_mapping)}건")
print(signal_vs_mapping.to_string(index=False))

# ══ 6) 오차 상위 5개 타깃 (cmp_px=True 세션 한정) ══
cmp_px_ids = inventory[inventory["cmp_px"]]["session_id"]
top_errors = build_top_error_targets(acc_hist, session_ids=cmp_px_ids, top_n=5)
print("\n[오차 상위 5개 타깃 (cmp_px=True 세션 한정)]")
print(top_errors.to_string(index=False))

print(f"\n[각주] {CLIP_CAVEAT_NOTE}")

# ══ 7) CSV 3종 저장 (저장 직전에만 session_id 축약) ══
os.makedirs(OUT_DIR, exist_ok=True)
save_csv_with_caveat(shorten_session_id(inventory), os.path.join(OUT_DIR, "inventory.csv"))
save_csv_with_caveat(shorten_session_id(signal_vs_mapping), os.path.join(OUT_DIR, "signal_vs_mapping.csv"))
save_csv_with_caveat(shorten_session_id(top_errors), os.path.join(OUT_DIR, "top_error_targets.csv"))
