# -*- coding: utf-8 -*-
"""세션/캘리브레이션 진단 배터리 + HTML 리포트 생성 도구

팀 공용 시각화 파이프라인(src/viz/viz.py, src/viz/calib_viz.py)과는 별개의 개인 탐색 도구
sessions.csv + gaze_accuracy.csv + calibration_quality.csv를 조인해
상관관계, 좌표 압축 패턴, 화면 경계 고착, 축-선택적 이상치, STB 요약을 한 번에 계산하고,
표+그림을 한 장의 자기완결형 HTML로 묶어준다.

배치 위치: src/analysis/diagnostic.py (프로젝트 루트 기준)
결과 저장 위치: src/analysis/diagnostics/ (.gitignore에 추가 필요)

주의사항
--------
- 9점 테스트(target_index 0~8) 전용.
  run_gaze_accuracy_test가 일반화되어 다른 test_type이 섞이면 이 도구는 그대로 쓸 수 없음
  그 경우 target_index 개수·범위에 맞는 별도 분할 로직이 필요
- axis_anomaly의 z-score 판정은 세션당 9개 표본 기준이라 통계적으로 약함
  "확정된 이상치"가 아니라 "육안 확인이 필요한 후보" 정도로만 취급할 것.
- calib_viz.py/viz.py와 달리 sessions.csv-gaze_accuracy.csv 조인을 사용
  calib_id:session_id 관계가 1:1인 경우와, 같은 캘리브레이션을 재사용하며 여러 세션을 비교하는 1:N 경우를 group_by 인자로 구분한다.

group_by 선택 기준
------------------
- "calib_id" (기본값): 서로 다른 캘리브레이션 품질을 비교할 때 (예: 캘리브레이션마다 조건이 다른 여러 세션을 모아보는 경우)
- "session_id": 같은 캘리브레이션을 재사용하며 조건만 바꿔 여러 번 테스트한 경우
  (예: 3D 모드 on/off 비교, 같은 프로그램 실행 중 재캘리브레이션 없이 여러 번 t키를 누른 경우)
  이 경우 캘리브레이션 RMSE는 같은 값이 여러 session에 반복되어 나타나는 게 정상
- 코드 수정 전후 비교처럼 프로그램 재시작이 필요한 변경은
  재시작 시 캘리브레이션도 자동으로 새로 생성되므로 기본값(calib_id)을 그대로 쓰면 된다.
"""

import os
import io
import json
import base64
import glob
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.viz import viz
from src.viz import calib_viz


# ══════════════════════════════════════════════════════════════
# 1. 진단 배터리 계산
# ══════════════════════════════════════════════════════════════

def diagnostic_battery(sessions, acc, session_ids=None, group_by="calib_id", tol_px=5):
    """5가지 진단(상관관계·압축패턴·경계고착·축이상치·STB요약)을 한 번에 계산.

    Parameters
    ----------
    sessions : pd.DataFrame
        sessions_v{SCHEMA_VERSION}.csv 로드 결과.
    acc : pd.DataFrame
        gaze_accuracy_v{SCHEMA_VERSION}.csv 로드 결과.
    session_ids : list[str], optional
        특정 session_id만 볼 때 지정. 없으면 sessions 전체를 대상으로 한다.
    group_by : {"calib_id", "session_id"}
        위 "group_by 선택 기준" 참고.
    tol_px : int
        화면 경계 '고착' 판정 허용오차(px).

    Returns
    -------
    dict : json.dump로 그대로 저장 가능한 구조.
    """
    assert group_by in ("calib_id", "session_id"), (
        f"group_by는 'calib_id' 또는 'session_id'만 가능합니다: {group_by}"
    )

    df = sessions.copy()
    if session_ids is not None:
        df = df[df["session_id"].isin(session_ids)]

    sid_to_calib = df.set_index("session_id")["calib_id"]
    a = acc[acc["session_id"].isin(df["session_id"])].copy()
    a["calib_id"] = a["session_id"].map(sid_to_calib)

    # ── 9점 테스트 전용 가드 (값 종류 체크) ──
    bad_targets = sorted(set(a["target_index"].unique()) - set(range(9)))
    assert a["target_index"].nunique() == 9 and not bad_targets, (
        f"diagnostic_battery()는 9점 테스트(target_index 0~8) 전용입니다. "
        f"현재 target_index 종류: {sorted(a['target_index'].unique())}. "
        f"5점 축약 테스트 등 다른 test_type이 섞여 있는지 확인하세요."
    )

    # ── 그룹당 행 개수 검증 (다른 세션이 섞여 조용히 풀링되는 것 방지) ──
    group_sizes = a.groupby(group_by).size()
    bad_groups = group_sizes[group_sizes % 9 != 0]
    assert bad_groups.empty, (
        f"9의 배수가 아닌 그룹이 있습니다 (데이터 오염 의심): {bad_groups.to_dict()}"
    )
    pooled = group_sizes[group_sizes > 9]
    if group_by == "calib_id" and not pooled.empty:
        print(
            f"[안내] 아래 calib_id는 여러 세션이 함께 집계됩니다 (1:N 정상 구조, "
            f"세션별로 나눠 보려면 group_by='session_id' 사용): {pooled.to_dict()}"
        )

    # 화면 해상도 역산 (viz.py의 infer_screen_size와 동일 로직, 9점 테스트
    # target 비율(0.1/0.5/0.9) 전제)
    screen_w = round(a["target_x_px"].max() / 0.9)
    screen_h = round(a["target_y_px"].max() / 0.9)

    group_order = list(dict.fromkeys(df.sort_values("start_timestamp")[group_by]))

    results = {"group_by": group_by}

    # ── 1. RMSE vs mean_error 상관관계 ──
    session_summary = a.groupby(group_by)["euclidean_error_px"].mean().reindex(group_order)
    if group_by == "calib_id":
        rmse_map = df.drop_duplicates("calib_id").set_index("calib_id")["calib_reproj_rmse_px"]
    else:  # session_id: sessions.csv에 이미 1행당 1값
        rmse_map = df.set_index("session_id")["calib_reproj_rmse_px"]
    corr_df = pd.DataFrame({"rmse": rmse_map.reindex(group_order), "mean_error": session_summary})
    results["correlation"] = {
        "r": float(corr_df["rmse"].corr(corr_df["mean_error"])),
        "n": int(len(corr_df)),
        "table": corr_df.reset_index().rename(columns={"index": group_by}).to_dict("records"),
    }

    # ── 2. 행(row)별 dx/dy 압축 패턴 (0-2=상단, 3-5=중단, 6-8=하단) ──
    a["dx"] = a["pred_x_px"] - a["target_x_px"]
    a["dy"] = a["pred_y_px"] - a["target_y_px"]
    a["row"] = pd.cut(a["target_index"], bins=[-1, 2, 5, 8], labels=["top", "mid", "bottom"])
    row_pattern = a.groupby([group_by, "row"])[["dx", "dy"]].mean().unstack("row")
    row_pattern = row_pattern.reindex(group_order)
    row_pattern.columns = ["_".join(col) for col in row_pattern.columns]  # (dx, top) -> dx_top
    results["row_compression"] = row_pattern.round(1).reset_index().to_dict("records")

    # ── 3. 화면 경계 고착 탐지 (session_id 항상 포함) ──
    stuck = a[
        (a["pred_x_px"] <= tol_px) | (a["pred_x_px"] >= screen_w - tol_px) |
        (a["pred_y_px"] <= tol_px) | (a["pred_y_px"] >= screen_h - tol_px)
    ]
    id_cols = list(dict.fromkeys(["session_id", group_by]))  # group_by=session_id면 중복 제거
    results["stuck_at_boundary"] = {
        "screen_w": screen_w, "screen_h": screen_h,
        "flagged_rows": stuck[id_cols + ["target_index", "pred_x_px", "pred_y_px"]].to_dict("records"),
    }

    # ── 4. 축-선택적 이상치 (세션 내 z-score, n=9라 참고용 후보일 뿐 확정 아님) ──
    def flag_axis_anomaly(g):
        std_x = g["gaze_std_x_px"].std() or 1
        std_y = g["gaze_std_y_px"].std() or 1
        zx = (g["gaze_std_x_px"] - g["gaze_std_x_px"].mean()) / std_x
        zy = (g["gaze_std_y_px"] - g["gaze_std_y_px"].mean()) / std_y
        out = g[(zx.abs() > 1.5) | (zy.abs() > 1.5)]
        base_cols = ["target_index", "gaze_std_x_px", "gaze_std_y_px"]
        cols = base_cols if group_by == "session_id" else ["session_id"] + base_cols
        return out[cols]

    axis_flags = (
        a.groupby(group_by, group_keys=True)
        .apply(flag_axis_anomaly, include_groups=False)
        .reset_index(level=0)
    )
    results["axis_anomaly"] = axis_flags.to_dict("records")

    # ── 5. STB 요약 ──
    stb_cols = ["stb01_fps", "stb02_landmark_rate", "stb03_face_fail_rate", "stb04_dropout_rate"]
    stb = a.groupby(group_by)[stb_cols].mean().reindex(group_order)
    results["stb_summary"] = stb.round(3).reset_index().to_dict("records")

    return results


def save_diagnostic_report(report, label, out_dir="src/analysis/diagnostics"):
    """진단 결과를 라벨 기준 파일명으로 저장 (고정 이름 덮어쓰기 방지).

    baseline/3D 모드처럼 서로 다른 조건을 비교할 때, 같은 이름으로
    덮어쓰면 이전 결과가 사라져 비교가 불가능해지므로 label을 필수로 받는다.

    프로젝트 루트에서 실행한다는 전제(상대경로). Colab 등 작업 디렉토리가
    다른 환경에서는 os.getcwd()로 위치를 먼저 확인할 것.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{label}_diagnostic_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"저장 완료: {path}")
    return path


# ══════════════════════════════════════════════════════════════
# 2. HTML 리포트 생성 (표 + 그림을 한 장에)
# ══════════════════════════════════════════════════════════════

REPORT_CSS = """
body { font-family: sans-serif; max-width: 1000px; margin: 40px auto; }
table { border-collapse: collapse; margin-bottom: 16px; }
td, th { border: 1px solid #ddd; padding: 4px 8px; text-align: center; }
details { margin-bottom: 16px; border: 1px solid #eee; padding: 8px; border-radius: 4px; }
summary { cursor: pointer; font-weight: bold; }
"""


def _fig_to_base64(fig):
    """matplotlib fig를 파일로 저장하지 않고 base64 문자열로 변환.
    HTML에 <img>로 바로 끼워 넣기 위함."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _style_table(df, highlight_cols=None):
    """발산 정도를 색으로 표시해 표만 봐도 이상 패턴이 보이게 한다.

    axis=0(컬럼별 정규화)을 사용한다 — 컬럼마다 값의 스케일이 달라서
    (예: dy_top은 -400~+200, stuck_count는 0~5) 전체 기준(axis=None)으로
    정규화하면 스케일이 작은 컬럼의 색이 왜곡된다.
    """
    styler = df.style.set_properties(**{
        "border": "1px solid #ddd", "padding": "4px 8px", "text-align": "center"
    })
    if highlight_cols:
        cols = [c for c in highlight_cols if c in df.columns]
        if cols:
            styler = styler.background_gradient(subset=cols, cmap="RdYlGn_r", axis=0)
    return styler.hide(axis="index").to_html()


def _prepare_scope(sessions, acc, session_ids):
    """session_ids로 필터링 + calib_id 매핑 + 추세 그래프용 메타 병합까지
    한 번에 처리. build_diagnostic_html()과 Notion 업로드 함수가 동일한
    전처리를 두 번 구현하지 않도록 공용화했다.
    """
    df = sessions[sessions["session_id"].isin(session_ids)].copy()
    a = acc[acc["session_id"].isin(session_ids)].copy()
    sid_to_calib = df.set_index("session_id")["calib_id"]
    a["calib_id"] = a["session_id"].map(sid_to_calib)

    # viz.plot_session_trend()가 필요로 하는 세션 메타(시간순 정렬,
    # dev_version 색 구분)를 a에 병합해둔다. 이걸 빼먹으면 추세 그래프가
    # 시간순이 아니라 알파벳순(session_id 기준)으로 정렬될 수 있다.
    meta_cols = [c for c in ["dev_version", "start_timestamp", "px_per_cm"] if c in df.columns]
    if meta_cols:
        a = a.merge(df[["session_id"] + meta_cols], on="session_id", how="left")

    return df, a, sid_to_calib


def build_session_summary_table(report, df, a, sid_to_calib):
    """세션당 1행짜리 압축 요약표를 만든다.

    build_diagnostic_html()과 Notion 업로드 함수(notion_export.py)가
    이 함수를 공유한다 — 같은 집계 로직을 두 곳에 따로 구현하면 한쪽만
    고치고 다른 쪽을 깜빡하는 사고가 나기 쉬워서 여기 한 곳으로 모았다.

    Returns
    -------
    pd.DataFrame : columns = [session_id, calib_id, calib_rmse_px,
        mean_error_px, correction_mode, dropout_rate, stuck_count, anomaly_count]
        session_id/calib_id는 앞 8자리로 축약되어 있다.
    """
    session_order = list(dict.fromkeys(df.sort_values("start_timestamp")["session_id"]))

    stuck_rows = report["stuck_at_boundary"]["flagged_rows"]
    anomaly_rows = report["axis_anomaly"]
    stuck_count = (
        pd.Series([r["session_id"] for r in stuck_rows]).value_counts()
        if stuck_rows else pd.Series(dtype="int64")
    )
    anomaly_count = (
        pd.Series([r["session_id"] for r in anomaly_rows]).value_counts()
        if anomaly_rows else pd.Series(dtype="int64")
    )
    mean_err = a.groupby("session_id")["euclidean_error_px"].mean()
    mean_err_cm = a.groupby("session_id")["euclidean_error_cm"].mean()
    dropout = a.groupby("session_id")["stb04_dropout_rate"].mean()
    rmse_map = df.set_index("session_id")["calib_reproj_rmse_px"]
    correction_mode_map = df.set_index("session_id")["correction_mode"]

    summary_rows = []
    for sid in session_order:
        rmse_val = rmse_map.get(sid)
        summary_rows.append({
            "session_id": str(sid)[:8],
            "calib_id": str(sid_to_calib.get(sid))[:8],
            "calib_rmse_px": round(rmse_val, 1) if pd.notna(rmse_val) else None,
            "mean_error_px": round(mean_err.get(sid, float("nan")), 1),
            "correction_mode": correction_mode_map.get(sid, "unknown"),
            "dropout_rate": round(dropout.get(sid, float("nan")), 3),
            "stuck_count": int(stuck_count.get(sid, 0)),
            "anomaly_count": int(anomaly_count.get(sid, 0)),
        })
    return pd.DataFrame(summary_rows)


def collect_figures(df, a, calib_df, screen_w, screen_h):
    """캘리브레이션/세션 오차 지도 그림들을 (category, label, fig) 튜플
    리스트로 만든다. build_diagnostic_html()과 Notion 이미지 업로드
    (notion_export.py)가 이 함수를 공유한다 — "어떤 캘리브레이션/세션을
    그릴지" 결정 로직을 두 곳에 따로 구현하면, 한쪽에서 스코프 계산
    방식을 고칠 때 다른 쪽을 깜빡할 위험이 있다.

    category는 "calibration" 또는 "session". fig는 matplotlib Figure —
    호출한 쪽이 base64 인코딩(HTML용)이든 PNG 바이너리 업로드
    (Notion용)든 원하는 방식으로 소비하면 된다.
    """
    session_order = list(dict.fromkeys(df.sort_values("start_timestamp")["session_id"]))
    calib_ids_in_scope = list(dict.fromkeys(df.sort_values("start_timestamp")["calib_id"]))

    figures = []

    for cid in calib_ids_in_scope:
        c_df = calib_df[calib_df["calib_id"] == cid].sort_values("calib_point_index")
        if len(c_df):
            figures.append(("calibration", str(cid)[:8], calib_viz.plot_calibration_overview(c_df)))

    if len(calib_ids_in_scope) >= 2:
        calib_scope_df = calib_df[calib_df["calib_id"].isin(calib_ids_in_scope)]
        figures.append(("calibration", "추세", calib_viz.plot_calibration_trend(calib_scope_df)))

    for sid in session_order:
        s = viz.get_session(a, sid)
        if len(s):
            figures.append(("session", str(sid)[:8], viz.plot_session_overview(s, screen_w, screen_h)))

    if len(session_order) >= 2:
        figures.append(("session", "추세", viz.plot_session_trend(a)))

    return figures


def build_diagnostic_html(sessions, acc, calib_df, session_ids, label,
                           group_by="calib_id", out_dir="src/analysis/diagnostics",
                           expand_details=False):
    """진단 배터리 + 시각화(테스트 세션·캘리브레이션 모두)를 하나의
    자기완결형 HTML로 묶는다.

    report/ 폴더에 이미 저장된 PNG를 재사용하지 않고, 여기 넘긴 데이터로
    viz.py/calib_viz.py 함수를 그 자리에서 새로 호출해 그린다 — 표와
    그림이 서로 다른 세션 조합을 가리키는 상황을 원천 차단하기 위함.

    최상단에 세션당 1행짜리 압축 요약표를 두고, 상세 표는 <details>로
    접어 필요할 때만 펼쳐보게 한다 (한눈에 보기와 상세 확인을 분리).

    Parameters
    ----------
    calib_df : pd.DataFrame
        calibration_quality_v{CALIB_SCHEMA_VERSION}.csv 로드 결과.
    expand_details : bool
        True면 <details> 섹션을 전부 펼친 상태로 생성한다. PDF 인쇄나
        Notion 아카이빙처럼 상호작용(클릭)이 불가능한 정적 매체로
        내보낼 때 사용 — 접힌 채로 인쇄하면 상세 내역이 통째로 누락된다.
    """
    viz.setup_font()  # matplotlib 한글 폰트 설정. 그림을 그리기 전에 반드시 먼저 호출.

    report = diagnostic_battery(sessions, acc, session_ids=session_ids, group_by=group_by)
    df, a, sid_to_calib = _prepare_scope(sessions, acc, session_ids)

    screen_w, screen_h = viz.infer_screen_size(a)

    stuck_rows = report["stuck_at_boundary"]["flagged_rows"]
    anomaly_rows = report["axis_anomaly"]
    summary_df = build_session_summary_table(report, df, a, sid_to_calib)

    open_attr = " open" if expand_details else ""

    sections = [f"<h1>진단 리포트: {label}</h1>"]
    sections.append("<h2>요약 (세션당 1행)</h2>")
    sections.append(_style_table(
        summary_df, highlight_cols=["mean_error_px", "stuck_count", "anomaly_count"]
    ))

    figures = collect_figures(df, a, calib_df, screen_w, screen_h)

    calib_figs = [f for f in figures if f[0] == "calibration"]
    session_figs = [f for f in figures if f[0] == "session"]

    try:
        # ── 캘리브레이션 시각화 ──
        sections.append("<h2>캘리브레이션</h2>")
        if not calib_figs:
            sections.append("<p>캘리브레이션 데이터 없음</p>")
        for _, fig_label, fig in calib_figs:
            if fig_label == "추세":
                sections.append("<h3>캘리브레이션 추세</h3>")
            sections.append(f'<img src="data:image/png;base64,{_fig_to_base64(fig)}" style="max-width:100%;">')
        if not any(fig_label == "추세" for _, fig_label, _ in calib_figs):
            sections.append("<p>캘리브레이션 1건만 존재 — 추세 생략</p>")

        # ── 테스트 세션 시각화 ──
        sections.append("<h2>테스트 세션</h2>")
        for _, fig_label, fig in session_figs:
            if fig_label == "추세":
                sections.append("<h3>세션 추세</h3>")
            else:
                sections.append(f"<h3>session: {fig_label}</h3>")
            sections.append(f'<img src="data:image/png;base64,{_fig_to_base64(fig)}" style="max-width:100%;">')
    finally:
        # 위 루프 도중 예외가 나면 아직 _fig_to_base64()로 소비되지 않은
        # figure가 열린 채 남아 "More than 20 figures" 경고가 누적되므로,
        # 정상/예외 경로 모두에서 남은 figure를 닫는다.
        for _, _, fig in figures:
            if plt.fignum_exists(fig.number):
                plt.close(fig)

    # ── 상세 표 (접어서 표시) ──
    sections.append("<h2>상세 진단</h2>")

    corr = report["correlation"]
    sections.append(
        f"<details{open_attr}><summary>RMSE-오차 상관관계 (r={corr['r']:.2f}, n={corr['n']})</summary>"
        f"{pd.DataFrame(corr['table']).to_html(index=False)}</details>"
    )

    row_df = pd.DataFrame(report["row_compression"])
    dy_cols = [c for c in row_df.columns if c.startswith("dy_")]
    sections.append(
        f"<details{open_attr}><summary>행별 dx/dy 압축 패턴</summary>"
        f"{_style_table(row_df, highlight_cols=dy_cols)}</details>"
    )

    stuck_html = pd.DataFrame(stuck_rows).to_html(index=False) if stuck_rows else "<p>없음</p>"
    sections.append(
        f"<details{open_attr}><summary>화면 경계 고착 상세 (해상도 {screen_w}x{screen_h})</summary>"
        f"{stuck_html}</details>"
    )

    anomaly_html = pd.DataFrame(anomaly_rows).to_html(index=False) if anomaly_rows else "<p>없음</p>"
    sections.append(
        f"<details{open_attr}><summary>축-선택적 이상치 후보 (참고용, n={len(anomaly_rows)})</summary>"
        f"{anomaly_html}</details>"
    )

    sections.append(
        f"<details{open_attr}><summary>STB 요약</summary>"
        f"{pd.DataFrame(report['stb_summary']).to_html(index=False)}</details>"
    )

    html = f"""<html><head><meta charset="utf-8"><style>{REPORT_CSS}</style>
</head><body>{"".join(sections)}</body></html>"""

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{label}_report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"저장 완료: {path}")
    return path


# ══════════════════════════════════════════════════════════════
# 3. 다중 스키마 버전 재분석 (Phase 2 / stage0_reanalysis 전용)
# ══════════════════════════════════════════════════════════════
# sessions_v1.0~1.3.csv / gaze_accuracy_v1.0~1.3.csv 전체 스키마 이력을 한 번에
# 훑어 "정말로 조건이 동일한 세션끼리 비교하고 있는가"를 검증하기 위한 함수들.
# run_export.py(최신 스키마 단일 버전 + 시각화 + Notion 업로드)와는 역할이
# 다르다 — 이 섹션은 스키마 이력 전체를 훑는 1회성 탐색/진단 전용이며,
# 실행 진입점은 src/analysis/stage0_reanalysis.py로 분리했다.

_VERSION_RE = re.compile(r"_v([\d.]+)\.csv$")


def _version_tuple(version_str, default=(0,)):
    """"1.10" -> (1, 10) 같은 정수 튜플로 파싱해 버전을 비교한다.

    문자열 그대로 비교("1.10" < "1.2")나 float 캐스팅(float("1.10")==1.1)은
    두 자리 이상 마이너 버전에서 순서가 뒤집힐 수 있어 피한다. 파싱 불가(예:
    정규식이 버전을 못 찾아 "unknown"이 된 경우)한 값은 default=(0,) 취급 —
    (1,2)보다 작으므로 구버전 취급되어 0.0이 NaN으로 치환된다(값을 신뢰할 수
    없으니 "측정 안 함"으로 보수적으로 처리).
    """
    try:
        return tuple(int(p) for p in str(version_str).split("."))
    except (ValueError, AttributeError):
        return default


def _concat_versioned_csv(prefix, results_dir):
    """{prefix}_v*.csv 전부를 utf-8-sig로 읽어 concat, 버전은 파일명에서 추출.

    CSV 내부 schema_version 컬럼은 버전마다 dtype이 미묘하게 달라 신뢰하지 않고
    파일명을 진실源으로 삼는다. sorted(glob(...))는 알파벳순 정렬 — 현재 버전
    표기(v1.0~v1.3)는 알파벳순=버전순이 우연히 일치하지만, 두 자리 마이너
    버전(v1.10)이 생기면 깨지므로 그때 재검토 필요.
    """
    paths = sorted(glob.glob(os.path.join(results_dir, f"{prefix}_v*.csv")))
    assert paths, f"{prefix}_v*.csv 파일을 '{results_dir}'에서 찾지 못했습니다."
    frames = []
    for p in paths:
        m = _VERSION_RE.search(os.path.basename(p))
        df = pd.read_csv(p, encoding="utf-8-sig")
        df["source_schema_version"] = m.group(1) if m else "unknown"
        frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False)  # 없는 컬럼은 자동 NaN


def load_all_sessions_history(results_dir="gaze_accuracy_results"):
    """sessions_v*.csv / gaze_accuracy_v*.csv 전 버전을 합친다.

    glob 패턴이 접두사(sessions_v / gaze_accuracy_v)라서 gaze_accuracy_raw_*,
    gaze_accuracy_sqpnp_corrected_* 등 세션별 원본 파일은 자동 제외됨(실측 확인).
    반환하는 session_id는 축약하지 않은 풀 UUID다 — 이후 join에 필요하므로
    stage0_reanalysis.py에서 저장 직전에만 shorten_session_id()로 축약할 것.

    중복 session_id 처리: paths는 파일명 알파벳순(=버전순) 정렬이라 같은
    session_id가 여러 버전 파일에 있으면 나중에 로드된(=더 최신 버전) 행이
    뒤에 온다. keep="last"로 최신 버전 쪽을 채택한다(구버전 스키마는 컬럼이
    적어 정보 손실이 있으므로 최신 우선이 합리적).

    구버전(source_schema_version < 1.2) iris_std 0.0 -> NaN 치환: 타깃별 iris
    프레임 샘플 수집은 v1.2 무렵 추가된 기능이라, 그 이전 버전의
    iris_std_x/y_px는 collector의 n<=1 방어값 관례로 인해 "측정 안 함"이
    정확히 0.0으로 위장되어 있다(실측: v1.0/v1.1 15세션·135행 중 x축 134행,
    y축 129행이 정확히 0.0 — 세션 전체가 아니라 행 단위로 섞여 있어 x/y
    컬럼을 각각 독립적으로 판정해야 한다). 원본 CSV 파일은 변경하지 않고
    이 함수의 반환값에서만 치환한다. 반대로 v1.2 이상인데 세션 전체(9행)가
    양축 다 0.0인 경우(기능이 이미 켜져 있었어야 함)는 치환 대상이 아니며,
    자동 치환 대신 session_id를 경고로만 출력해 별도 확인 대상으로 남긴다.
    """
    sessions_hist = _concat_versioned_csv("sessions", results_dir)
    acc_hist = _concat_versioned_csv("gaze_accuracy", results_dir)

    dup_mask = sessions_hist["session_id"].duplicated(keep=False)
    if dup_mask.any():
        dup_ids = sessions_hist.loc[dup_mask, "session_id"].unique()
        kept_preview = sessions_hist[sessions_hist["session_id"].isin(dup_ids)] \
            .drop_duplicates(subset="session_id", keep="last")
        for sid in dup_ids:
            chosen = kept_preview.loc[kept_preview["session_id"] == sid, "source_schema_version"].iloc[0]
            print(f"[경고] session_id={sid} 중복 등록 — source_schema_version={chosen} (최신) 채택")
        sessions_hist = sessions_hist.drop_duplicates(subset="session_id", keep="last")

    # ── 구버전(<1.2) iris_std 0.0 → NaN 치환 ──
    ver_tuples = acc_hist["source_schema_version"].map(_version_tuple)
    is_pre_1_2 = ver_tuples.map(lambda t: t < (1, 2))

    n_replaced_x = int((is_pre_1_2 & (acc_hist["iris_std_x_px"] == 0.0)).sum())
    n_replaced_y = int((is_pre_1_2 & (acc_hist["iris_std_y_px"] == 0.0)).sum())
    acc_hist.loc[is_pre_1_2 & (acc_hist["iris_std_x_px"] == 0.0), "iris_std_x_px"] = np.nan
    acc_hist.loc[is_pre_1_2 & (acc_hist["iris_std_y_px"] == 0.0), "iris_std_y_px"] = np.nan
    print(f"[정보] 구버전(<1.2) iris_std 0.0 -> NaN 치환: x축 {n_replaced_x}행, y축 {n_replaced_y}행")

    # v1.2 이상인데 세션 전체(9행 모두)가 양축 다 0.0인 경우는 위 치환 대상이
    # 아니다(버전 필터에 안 걸림) — 기능이 이미 켜져 있어야 할 버전에서 여전히
    # 완전히 0인 것은 별도 원인 조사가 필요하므로 치환하지 않고 경고만 남긴다.
    post_1_2 = acc_hist[ver_tuples.map(lambda t: t >= (1, 2))]
    suspect_sessions = [
        sid for sid, g in post_1_2.groupby("session_id")
        if (g["iris_std_x_px"] == 0.0).all() and (g["iris_std_y_px"] == 0.0).all()
    ]
    if suspect_sessions:
        print(f"[경고] v1.2 이상인데 iris_std가 세션 전체(양축) 0.0인 세션 "
              f"{len(suspect_sessions)}건 (치환하지 않음, 별도 확인 필요): {suspect_sessions}")

    return sessions_hist, acc_hist


def summarize_px_per_cm_values(sessions_hist):
    """px_per_cm 고유값별 세션 수(NaN 포함) — 등급 기준값이 여전히 타당한지 확인용."""
    return sessions_hist["px_per_cm"].value_counts(dropna=False).sort_index()


def compute_axis_gain(df_session):
    """target→pred 축별 OLS 게인 회귀(기울기·절편·r²).

    n<3이면 전부 NaN(2점 회귀 금지), n만 실제 값으로 남겨 왜 NaN인지
    표에서 바로 드러나게 함.
    """
    valid = df_session.dropna(subset=["pred_x_px", "pred_y_px"])
    n = len(valid)
    result = {"gain_x": np.nan, "intercept_x": np.nan, "r2_x": np.nan,
              "gain_y": np.nan, "intercept_y": np.nan, "r2_y": np.nan, "n": n}
    if n < 3:
        return result
    for axis in ("x", "y"):
        tcol, pcol = f"target_{axis}_px", f"pred_{axis}_px"
        slope, intercept = np.polyfit(valid[tcol], valid[pcol], 1)
        pred_hat = slope * valid[tcol] + intercept
        ss_res = ((valid[pcol] - pred_hat) ** 2).sum()
        ss_tot = ((valid[pcol] - valid[pcol].mean()) ** 2).sum()
        # ss_tot==0: pred가 완전히 고착(분산 0)돼 r2가 정의 불능인 경우 (경계
        # 고착 세션에서 실제로 발생 가능) — NaN 처리.
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
        result[f"gain_{axis}"], result[f"intercept_{axis}"], result[f"r2_{axis}"] = slope, intercept, r2
    return result


def check_comparability(px_per_cm, screen_w, screen_h,
                         ref_px_per_cm=43.364, ref_resolution=(1536, 864),
                         px_per_cm_tol=1e-3, resolution_tol_px=1):
    """비교가능성 2축 독립 판정.

    비교가능성은 서로 독립인 두 축이다 — 해상도가 같으면 px 단위 지표(오차_px,
    게인 등)를 비교할 수 있고, px_per_cm이 같으면 cm 단위 지표(오차_cm 등)를
    비교할 수 있다. 이 둘을 하나의 등급으로 뭉치면, 해상도는 기준과 일치하지만
    px_per_cm만 다른 세션(실측: px_per_cm=25.697 그룹 4세션)이 통째로 배제되어
    유효한 px 비교 데이터를 잃는다.

    cmp_px  : 추정 해상도가 ref_resolution과 일치(±resolution_tol_px) → px 지표 비교 가능
    cmp_cm  : px_per_cm이 ref_px_per_cm과 일치(±px_per_cm_tol) → cm 지표 비교 가능
    reason  : False인 축에 대한 사유 문자열(둘 다 True면 "")

    경고 — px_per_cm_tol을 0.5 이상으로 키우지 말 것: 42.829(2560x1440 그룹,
    실측 20건)가 43.364(1536x864 그룹) 기준에 흡수되어 서로 다른 해상도의
    세션이 같은 cm 비교군으로 잘못 묶인다. tol은 반드시 1e-3 수준으로 유지.

    resolution_tol_px=1인 이유: 9점 좌표가 SCREEN*0.9 지점이라
    viz.infer_screen_size()의 round(target_max/0.9) 역산이 실제보다 1px
    작게 나오는 반올림 아티팩트가 있음(실측: 1536x864 세션 32건 전부 (1536,863)로
    역산됨). tol=0이면 cmp_px가 하나도 True가 되지 않는다.
    """
    if pd.isna(screen_w) or pd.isna(screen_h):
        cmp_px, px_reason = False, "해상도 역산 불가"
    else:
        cmp_px = (abs(screen_w - ref_resolution[0]) <= resolution_tol_px and
                  abs(screen_h - ref_resolution[1]) <= resolution_tol_px)
        px_reason = "" if cmp_px else f"추정 해상도 {screen_w}x{screen_h} != 기준 {ref_resolution}"

    if pd.isna(px_per_cm):
        cmp_cm, cm_reason = False, "px_per_cm 결측 (v1.0 구버전 스키마)"
    elif px_per_cm <= 0:
        cmp_cm, cm_reason = False, f"px_per_cm={px_per_cm} (측정값 이상, 배제 권장)"
    else:
        cmp_cm = abs(px_per_cm - ref_px_per_cm) < px_per_cm_tol
        cm_reason = "" if cmp_cm else f"px_per_cm={px_per_cm} (기준값 {ref_px_per_cm}과 다름)"

    reason = "; ".join(r for r in (px_reason, cm_reason) if r)
    return cmp_px, cmp_cm, reason


def build_session_inventory(sessions_hist, acc_hist, tol_px=5, clip_ratio_threshold=0.3,
                             ref_px_per_cm=43.364, ref_resolution=(1536, 864)):
    """세션당 1행짜리 인벤토리 + 비교가능성 2축 + clip_suspect + 축별 게인.

    calib_id/correction_mode/calib_reproj_rmse_px를 sessions 메타에서 그대로
    끌어온다 — build_signal_vs_mapping_table()이 calib_reproj_rmse_px를
    참조하고(누락 시 KeyError), correction_mode/calib_id는 이후 raw 세션 판별의
    dev_version 교차 검증용으로 필요하다.

    tol_px=5는 diagnostic_battery()의 stuck_at_boundary 판정과 동일 기본값
    (동일 개념이므로 임계값도 통일).
    clip_ratio_threshold=0.3: 유효 타깃 중 화면 경계(tol_px 이내) 근접 비율이
    이 값 이상이면 clip_suspect=True (실측 60세션 분포 평균 0.139, 75%tile
    0.222 기준 상위 ~15%를 분리하는 지점).

    Returns
    -------
    pd.DataFrame, columns:
        session_id(풀 UUID), source_schema_version, start_timestamp, dev_version,
        calib_id, correction_mode, calib_reproj_rmse_px, px_per_cm, screen_w, screen_h,
        n_targets, n_valid_pred, cmp_px, cmp_cm, comparability_reason,
        clip_ratio, clip_suspect, gain_x, intercept_x, r2_x, gain_y, intercept_y, r2_y, gain_n
    """
    rows = []
    for sid, g in acc_hist.groupby("session_id"):
        smeta_df = sessions_hist[sessions_hist["session_id"] == sid]
        if smeta_df.empty:
            print(f"[경고] session_id={sid}가 sessions 이력에 없어 인벤토리에서 제외합니다.")
            continue
        smeta = smeta_df.iloc[0]

        n_targets = len(g)  # 경계 사례: 9 미만이어도 그대로 진행 (assert 없음).
        # infer_screen_size는 target_x/y_px(항상 존재)만 쓰므로 pred 결측과 무관하게
        # 계산 가능. 단, n_targets<9라서 0.9 지점 타깃 자체가 빠졌다면 해상도가
        # 실제보다 작게 오추정될 수 있음 — n_targets 컬럼으로 그런 세션을 식별해
        # 별도 검토할 것 (이 함수는 그 이상 자동 보정하지 않음).
        screen_w, screen_h = viz.infer_screen_size(g)  # 세션 단위로 필터링된 g에만 적용

        valid = g.dropna(subset=["pred_x_px", "pred_y_px"])  # 경계 사례: pred 결측 타깃 제외
        n_valid = len(valid)
        if n_valid == 0:
            clip_ratio = np.nan
        else:
            near = valid[
                (valid["pred_x_px"] <= tol_px) | (valid["pred_x_px"] >= screen_w - 1 - tol_px) |
                (valid["pred_y_px"] <= tol_px) | (valid["pred_y_px"] >= screen_h - 1 - tol_px)
            ]
            clip_ratio = len(near) / n_valid
        clip_suspect = bool(pd.notna(clip_ratio) and clip_ratio >= clip_ratio_threshold)

        px_per_cm = smeta.get("px_per_cm", np.nan)
        cmp_px, cmp_cm, reason = check_comparability(
            px_per_cm, screen_w, screen_h,
            ref_px_per_cm=ref_px_per_cm, ref_resolution=ref_resolution,
        )
        gain = compute_axis_gain(g)

        rows.append({
            "session_id": sid,
            "source_schema_version": smeta.get("source_schema_version"),
            "start_timestamp": smeta.get("start_timestamp"),
            "dev_version": smeta.get("dev_version"),
            "calib_id": smeta.get("calib_id", np.nan),
            "correction_mode": smeta.get("correction_mode", np.nan),
            "calib_reproj_rmse_px": smeta.get("calib_reproj_rmse_px", np.nan),
            "px_per_cm": px_per_cm,
            "screen_w": screen_w, "screen_h": screen_h,
            "n_targets": n_targets, "n_valid_pred": n_valid,
            "cmp_px": cmp_px, "cmp_cm": cmp_cm, "comparability_reason": reason,
            "clip_ratio": round(clip_ratio, 3) if pd.notna(clip_ratio) else np.nan,
            "clip_suspect": clip_suspect,
            "gain_x": gain["gain_x"], "intercept_x": gain["intercept_x"], "r2_x": gain["r2_x"],
            "gain_y": gain["gain_y"], "intercept_y": gain["intercept_y"], "r2_y": gain["r2_y"],
            "gain_n": gain["n"],
        })

    inv = pd.DataFrame(rows)
    return inv.sort_values("start_timestamp").reset_index(drop=True)


def _format_rank(series, ascending=True):
    """순위를 'k/n' 문자열로 포맷.

    n<=1(비교 대상이 세션 1개뿐)이면 'k/n (비교 대상 없음)'으로 명시해 착시를
    방지한다. NaN 값은 순위 없이 'N/A'로 표기(calib_reproj_rmse_px가 없는
    구버전 스키마 세션이 섞여 있을 수 있음).
    """
    n = int(series.notna().sum())
    ranks = series.rank(ascending=ascending, method="min", na_option="keep")

    def fmt(r):
        if pd.isna(r):
            return "N/A"
        if n <= 1:
            return f"{int(r)}/{n} (비교 대상 없음)"
        return f"{int(r)}/{n}"

    return ranks.map(fmt)


def build_signal_vs_mapping_table(inventory_df, acc_hist):
    """cmp_px==True 세션 대상 신호 vs 매핑 판별표.

    이 표의 지표(mean_error_px, std류, calib_reproj_rmse_px)는 전부 px/무차원
    스케일이라 px_per_cm(cm 축, cmp_cm) 일치 여부와는 무관하다 — 따라서
    스코프는 cmp_px==True 전부(개수 고정 아님)로 한정한다. 자동 판정 로직은
    넣지 않고 수치만 정렬 출력 + 컬럼별 세션간 순위 병기.

    Parameters
    ----------
    inventory_df : pd.DataFrame
        build_session_inventory() 반환값. **session_id가 풀 UUID인 상태에서
        호출할 것** (축약 후 호출 시 acc_hist와 join 불가/충돌 위험).
    """
    scope_sessions = inventory_df[inventory_df["cmp_px"] == True].copy()
    cols = ["session_id", "mean_error_px", "gaze_std_mag_median", "iris_std_mag_median",
            "dropout_rate_mean", "calib_reproj_rmse_px"]
    rank_cols = [f"rank_{c}" for c in cols[1:]]

    if scope_sessions.empty:
        print("[안내] cmp_px=True인 세션이 없어 signal_vs_mapping 표가 비어 있습니다 "
              "(ref_resolution/resolution_tol_px 재검토 필요).")
        return pd.DataFrame(columns=cols + rank_cols)

    a = acc_hist[acc_hist["session_id"].isin(scope_sessions["session_id"])].copy()
    # pred 결측 행은 gaze_std_x/y_px 등도 이미 NaN이라 groupby().mean()/median()의
    # 기본 skipna=True가 자동으로 걸러준다 — 별도 dropna 불필요.
    a["gaze_std_mag"] = np.sqrt(a["gaze_std_x_px"] ** 2 + a["gaze_std_y_px"] ** 2)
    a["iris_std_mag"] = np.sqrt(a["iris_std_x_px"] ** 2 + a["iris_std_y_px"] ** 2)

    agg = a.groupby("session_id").agg(
        mean_error_px=("euclidean_error_px", "mean"),
        gaze_std_mag_median=("gaze_std_mag", "median"),
        iris_std_mag_median=("iris_std_mag", "median"),
        dropout_rate_mean=("dropout_rate", "mean"),
    ).reset_index()

    out = agg.merge(
        scope_sessions[["session_id", "calib_reproj_rmse_px"]], on="session_id", how="left"
        # calib_reproj_rmse_px가 없는 구버전(v1.0/v1.1) 스키마 세션은 NaN으로
        # 남는다 — _format_rank가 'N/A'로 표기해 자연스럽게 드러난다.
    )
    for c in cols[1:]:
        out[f"rank_{c}"] = _format_rank(out[c], ascending=True)  # 전부 "낮을수록 작은 숫자" 방향 정렬

    return out.sort_values("mean_error_px")[cols + rank_cols]


def build_top_error_targets(acc_hist, session_ids=None, top_n=5):
    """오차 상위 top_n개 타깃 보조표.

    session_ids로 세션 범위를 한정할 수 있다(기본은 전 세션). 드라이버에서는
    cmp_px==True 세션으로 한정해서 호출한다 — 해상도가 다른 세션(예:
    2560x1440)을 섞으면 px 오차가 절대적으로 커서 상위권을 해상도 효과로만
    지배하게 되어 진짜 이상 세션을 가리게 되기 때문.

    row_position은 target_index 0-2=상단/3-5=중단/6-8=하단으로 분류한다.
    diagnostic_battery()의 row_compression과 동일한 target_index 3분할 기준을
    재사용한 것 — target_index→행 위치 매핑이 모든 스키마 버전·모든 해상도에서
    불변임을 실측 확인했고, screen_h는 세션마다 863~1440으로 달라 픽셀 기준
    3분할을 쓰면 세션 간 행 정의가 흔들리기 때문에 픽셀 기준으로 재분할하지 않는다.
    """
    scope = acc_hist if session_ids is None else acc_hist[acc_hist["session_id"].isin(session_ids)]

    valid = scope.dropna(subset=["pred_x_px", "pred_y_px"]).copy()  # 경계 사례: pred 결측 타깃 제외
    valid["gaze_std_mag"] = np.sqrt(valid["gaze_std_x_px"] ** 2 + valid["gaze_std_y_px"] ** 2)
    valid["iris_std_mag"] = np.sqrt(valid["iris_std_x_px"] ** 2 + valid["iris_std_y_px"] ** 2)
    valid["row_position"] = pd.cut(valid["target_index"], bins=[-1, 2, 5, 8],
                                    labels=["top", "mid", "bottom"])

    top = valid.nlargest(top_n, "euclidean_error_px")
    return top[["session_id", "target_index", "euclidean_error_px",
                "gaze_std_mag", "iris_std_mag", "row_position"]]


def shorten_session_id(df, col="session_id"):
    """build_session_summary_table()과 동일한 8자리 축약 표시 규칙을
    재사용 가능한 형태로 뽑아낸 헬퍼.

    원본은 변경하지 않고 복사본 반환(풀 UUID가 필요한 후속 join이 남아있을
    수 있어 원본 보존이 안전).
    """
    out = df.copy()
    if col in out.columns:
        out[col] = out[col].astype(str).str.slice(0, 8)
    return out


CLIP_CAVEAT_NOTE = (
    "주의: pred_x_px/pred_y_px는 calibration.map_to_screen()에서 화면 경계로 "
    "clip(0~SCREEN_W-1, 0~SCREEN_H-1)된 값이 그대로 기록된 것이다. 화면 밖으로 "
    "벗어나는 예측의 오차가 클리핑으로 과소평가(중도절단)될 수 있으므로, "
    "clip_suspect=True인 세션의 오차/게인 지표는 실제보다 낮게(또는 왜곡되어) "
    "보일 수 있음에 유의할 것."
)


def save_csv_with_caveat(df, path, note=CLIP_CAVEAT_NOTE):
    """CSV 저장 시 상단에 '# ' 각주 줄을 붙인다.

    pandas read_csv(comment='#')로 다시 읽으면 각주 줄은 자동 무시되어 데이터
    파싱 호환성이 유지된다. utf-8-sig로 저장(Windows Excel에서 한글이 깨지지
    않도록) — 기존 JSON/HTML은 utf-8이지만 CSV는 Excel 호환 목적이 다르므로
    별도 선택.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        for line in note.splitlines():
            f.write(f"# {line}\n")
        df.to_csv(f, index=False)
    print(f"저장 완료: {path}")
    return path


# ══════════════════════════════════════════════════════════════
# 4. 블록 실험 분석 (캘리브레이션 1회 + raw 9점 테스트 N회 = 1블록)
# ══════════════════════════════════════════════════════════════
# 2026-07-08 반복성 실험(캘리브레이션마다 raw 9점 테스트를 4~5회 반복, 6블록)의
# 대화형 1차 분석 결과를 정식 함수로 편입한 것. diagnostic_battery()와 달리
# "블록(캘리브레이션) 간 변동 vs 블록 내(회차 간) 변동"을 분해하고, 재캘리브레이션
# 없이 반복 측정만 했을 때의 드리프트/고착을 진단하는 데 특화되어 있다.

def _json_safe(obj):
    """중첩 dict/list를 재귀적으로 훑어 NaN -> None, numpy 스칼라 -> 파이썬
    네이티브 타입으로 변환한다.

    notion_export.py의 _safe_number()는 스칼라 1개(Notion number 속성 하나)를
    다루는 함수라 재사용할 수 없다 — 여기서 필요한 건 dict 안의 dict, df를
    to_dict("records")한 list[dict] 등 임의 깊이의 중첩 구조 전체를 한 번에
    정리하는 것이라 문제 형태 자체가 다르다. 또한 upload_diagnostic_to_notion()
    안에는 "diagnostic.py가 notion_export.py를 몰라도 되도록 의존 방향을 한쪽
    으로만 유지한다"는 주석이 있어, 이 함수가 notion_export.py를 import하는 것도
    피해야 한다 — 그래서 여기서 독립적으로 정의한다.
    """
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return [_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        return None if obj != obj else obj  # NaN != NaN
    if isinstance(obj, np.floating):
        f = float(obj)
        return None if f != f else f
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def build_block_structure(sessions_df, acc_df, session_ids):
    """session_ids를 calib_id 단위 "블록"으로 묶고, 블록 내부에서 시간순
    회차(test_idx, 1부터)를 매긴다.

    block_id는 calib_id와 값이 동일하다 — 블록 실험이라는 개념을 표현하기
    위해 컬럼명만 새로 붙였을 뿐, 새로운 id 체계를 만든 것은 아니다.

    session_ids 지정 방법(전체/특정/조건별/최근 N세션/블록 실험)은
    run_export.py 모듈 docstring의 "사용 매뉴얼" 절 참고.

    Returns
    -------
    merged : pd.DataFrame
        acc 원본 행(세션 수 x 9) + block_id, test_idx 컬럼.
    block_summary : pd.DataFrame
        columns = [block_id, n_sessions, calib_rmse_px, block_start_ts, block_end_ts]
        block_start_ts 오름차순 정렬.
    """
    df, a, sid_to_calib = _prepare_scope(sessions_df, acc_df, session_ids)

    # 9점 테스트 전용 가드. diagnostic_battery()의 assert와 개념은 같지만
    # 검증 단위가 다르다(diagnostic_battery는 group_by 그룹당 9의 배수 허용,
    # 여기는 세션마다 정확히 9행이어야 test_idx 배정이 안전하다) — 그 함수를
    # 그대로 호출하면 row_compression 등 필요 없는 계산까지 함께 돌게 되므로
    # 가드만 얕게 재검증한다(기존 함수는 변경하지 않음).
    bad_targets = sorted(set(a["target_index"].unique()) - set(range(9)))
    assert a["target_index"].nunique() == 9 and not bad_targets, (
        f"build_block_structure()도 9점 테스트(target_index 0~8) 전용입니다. "
        f"현재 target_index 종류: {sorted(a['target_index'].unique())}"
    )
    per_session_n = a.groupby("session_id").size()
    bad = per_session_n[per_session_n != 9]
    assert bad.empty, f"세션당 9행이 아닌 세션이 있습니다(데이터 오염 의심): {bad.to_dict()}"

    a = a.copy()
    a["block_id"] = a["calib_id"]

    session_order_per_block = (
        df.sort_values("start_timestamp")
          .assign(block_id=lambda d: d["calib_id"])
          .groupby("block_id")["session_id"]
          .apply(lambda s: list(dict.fromkeys(s)))
    )
    test_idx_map = {
        sid: i
        for sids in session_order_per_block
        for i, sid in enumerate(sids, start=1)
    }
    a["test_idx"] = a["session_id"].map(test_idx_map)

    block_summary = (
        df.assign(block_id=df["calib_id"])
          .groupby("block_id")
          .agg(n_sessions=("session_id", "nunique"),
               calib_rmse_px=("calib_reproj_rmse_px", "first"),
               block_start_ts=("start_timestamp", "min"),
               block_end_ts=("start_timestamp", "max"))
          .reset_index()
          .sort_values("block_start_ts")
          .reset_index(drop=True)
    )
    return a, block_summary


def _detect_frozen_session(valid, gaze_std_median_x, gaze_std_median_y,
                            coord_round=0, unique_thresh=2, std_eps=1e-6):
    """세션이 "고착"됐는지 판정: 유효 예측 좌표(정수 px 반올림) 고유값이
    unique_thresh개 이하이고, gaze_std_x/y_px의 세션 내 median이 둘 다
    std_eps 미만이면 고착으로 본다(단순히 좌표가 겹치는 것만으로는 우연의
    일치일 수 있어, 입력 신호 자체가 거의 흔들리지 않았다는 조건을 함께 요구).

    Returns
    -------
    (frozen: bool, frozen_coord: str or None) — frozen_coord는 가장 많이
    나온 좌표를 "(x, y)" 문자열로 표기(고착이 아니면 None).
    """
    if valid.empty:
        return False, None
    coords = list(zip(valid["pred_x_px"].round(coord_round),
                       valid["pred_y_px"].round(coord_round)))
    coord_counts = pd.Series(coords).value_counts()
    is_frozen = bool(
        len(coord_counts) <= unique_thresh
        and pd.notna(gaze_std_median_x) and gaze_std_median_x < std_eps
        and pd.notna(gaze_std_median_y) and gaze_std_median_y < std_eps
    )
    frozen_coord = None
    if is_frozen:
        top_x, top_y = coord_counts.idxmax()
        frozen_coord = f"({int(top_x)}, {int(top_y)})"
    return is_frozen, frozen_coord


def compute_block_session_stats(merged_df):
    """(block_id, test_idx, session_id)당 1행 — 블록×회차 통계표.

    compute_axis_gain()을 그대로 재사용한다(재구현 아님).

    clip_n은 build_session_inventory()와 동일한 경계 허용오차 공식
    (screen_w/h - 1 - tol_px, infer_screen_size의 반올림 아티팩트 보정 포함)을
    따르며, 유효 예측(pred 결측 아님)만 대상으로 한다 — 화면 경계 클리핑과
    추적 실패(결측)는 서로 다른 실패 모드라 n_missing_pred로 별도 집계한다.

    gaze_std/iris_std는 x/y 축별 median과, build_signal_vs_mapping_table()/
    build_top_error_targets()가 이미 쓰는 sqrt(x^2+y^2) 합성 magnitude의
    median을 함께 낸다(신규 관례가 아니라 기존 관례 재사용).

    Returns
    -------
    pd.DataFrame, columns:
        block_id, test_idx, session_id, n_valid, n_missing_pred,
        mean_error_px, gain_x, intercept_x, r2_x, gain_y, intercept_y, r2_y, gain_n,
        clip_n, clip_ratio,
        gaze_std_median_x_px, gaze_std_median_y_px,
        iris_std_median_x_px, iris_std_median_y_px,
        gaze_std_mag_median, iris_std_mag_median,
        frozen, frozen_coord
    """
    tol_px = 5  # build_session_inventory()/diagnostic_battery()와 동일 값 재사용

    rows = []
    for (block_id, test_idx, session_id), g in merged_df.groupby(
        ["block_id", "test_idx", "session_id"], sort=False
    ):
        gain = compute_axis_gain(g)

        screen_w, screen_h = viz.infer_screen_size(g)
        valid = g.dropna(subset=["pred_x_px", "pred_y_px"])
        n_valid = len(valid)
        n_missing_pred = len(g) - n_valid
        if n_valid == 0:
            clip_n, clip_ratio = 0, np.nan
        else:
            near = valid[
                (valid["pred_x_px"] <= tol_px) | (valid["pred_x_px"] >= screen_w - 1 - tol_px) |
                (valid["pred_y_px"] <= tol_px) | (valid["pred_y_px"] >= screen_h - 1 - tol_px)
            ]
            clip_n = len(near)
            clip_ratio = clip_n / n_valid

        gaze_med_x, gaze_med_y = g["gaze_std_x_px"].median(), g["gaze_std_y_px"].median()
        iris_med_x, iris_med_y = g["iris_std_x_px"].median(), g["iris_std_y_px"].median()
        gaze_mag = np.sqrt(g["gaze_std_x_px"] ** 2 + g["gaze_std_y_px"] ** 2)
        iris_mag = np.sqrt(g["iris_std_x_px"] ** 2 + g["iris_std_y_px"] ** 2)

        frozen, frozen_coord = _detect_frozen_session(valid, gaze_med_x, gaze_med_y)

        rows.append({
            "block_id": block_id, "test_idx": test_idx, "session_id": session_id,
            "n_valid": n_valid, "n_missing_pred": n_missing_pred,
            "mean_error_px": g["euclidean_error_px"].mean(),
            "gain_x": gain["gain_x"], "intercept_x": gain["intercept_x"], "r2_x": gain["r2_x"],
            "gain_y": gain["gain_y"], "intercept_y": gain["intercept_y"], "r2_y": gain["r2_y"],
            "gain_n": gain["n"],
            "clip_n": clip_n,
            "clip_ratio": round(clip_ratio, 3) if pd.notna(clip_ratio) else np.nan,
            "gaze_std_median_x_px": gaze_med_x, "gaze_std_median_y_px": gaze_med_y,
            "iris_std_median_x_px": iris_med_x, "iris_std_median_y_px": iris_med_y,
            "gaze_std_mag_median": gaze_mag.median(), "iris_std_mag_median": iris_mag.median(),
            "frozen": frozen, "frozen_coord": frozen_coord,
        })

    return pd.DataFrame(rows).sort_values(["block_id", "test_idx"]).reset_index(drop=True)


def decompose_variance(block_stats, exclude_blocks=None):
    """mean_error_px를 block_id로 묶어 between/within 제곱합(SS)으로 분해한다
    (일원배치 ANOVA 분해, 블록 크기(4 vs 5)를 실제 n_i로 가중).

    ratio = SS_between / SS_total (= SS_between / (SS_between + SS_within),
    eta-squared와 동일한 정의, method="eta_squared") — 값이 클수록 "전체
    분산 중 블록(캘리브레이션) 간 차이로 설명되는 비율이 크다"는 뜻.
    MS_between/(MS_between+MS_within)(자유도로 나눈 뒤 비교하는 버전)도
    함께 계산해 참고용으로 남기지만, 이는 블록 크기가 다르면(4 vs 5) df
    보정이 각 SS를 서로 다른 비율로 깎아 원래의 "설명된 분산 비율"이라는
    의미를 왜곡한다(실측 확인: 2026-07-08 데이터에서 MS 기준 0.91 vs
    SS/SS_total 기준 0.69로 크게 갈림) — 그래서 주 지표(ratio)는
    SS/SS_total 쪽을 쓴다.

    icc1은 불균형 보정 ICC(1)(모집단 급내상관 추정치)을 추가로 계산해
    참고값으로 병기한다:
        n0 = (N - Σ(n_b²)/N) / (k-1)   # 불균형 보정 평균 그룹 크기
        icc1 = (MS_b - MS_w) / (MS_b + (n0-1) * MS_w)
    ratio(η²)는 표본 기술통계, icc1은 모집단 급내상관 추정치 — 통상
    icc1 <= η². k<2이거나 MS_w를 계산할 수 없으면(df_within<=0) None.
    통계적 유의성 검정이 아니다(p-value 없음).

    Parameters
    ----------
    exclude_blocks : list[str] or None
        None(기본값): frozen=True인 세션이 하나라도 있는 block을 자동 제외한다
        (고착 세션의 mean_error_px는 "고착 좌표-타깃 거리"일 뿐 추적 성능을
        반영하지 않아 블록 평균을 오염시키므로). 자동 제외를 끄고 싶으면 빈
        리스트 []를 명시적으로 넘길 것 — None과 []는 의미가 다르다.

    Returns
    -------
    dict : grand_mean, n_blocks_used, n_sessions_used, ss_between, ss_within,
        ss_total, df_between, df_within, ms_between, ms_within,
        ratio(=SS_between/SS_total), method("eta_squared"), icc1(불균형 보정
        ICC(1), 계산 불능 시 None), per_block(list[dict]: block_id, n, mean, sd),
        blocks_used, blocks_excluded, sessions_excluded.
    """
    if exclude_blocks is None:
        auto_excluded = sorted(block_stats.loc[block_stats["frozen"], "block_id"].unique().tolist())
    else:
        auto_excluded = list(exclude_blocks)

    excluded_sessions = block_stats.loc[
        block_stats["block_id"].isin(auto_excluded), "session_id"
    ].tolist()

    scope = block_stats[~block_stats["block_id"].isin(auto_excluded)].dropna(subset=["mean_error_px"])

    N = len(scope)
    k = scope["block_id"].nunique()
    grand_mean = scope["mean_error_px"].mean() if N else np.nan

    per_block = scope.groupby("block_id")["mean_error_px"].agg(
        n="count", mean="mean", sd=lambda s: s.std(ddof=1)
    )

    if k >= 1 and N:
        ss_between = (per_block["n"] * (per_block["mean"] - grand_mean) ** 2).sum()
        block_means = scope["block_id"].map(per_block["mean"])
        ss_within = ((scope["mean_error_px"] - block_means) ** 2).sum()
    else:
        ss_between, ss_within = np.nan, np.nan

    df_between = k - 1
    df_within = N - k
    ms_between = ss_between / df_between if df_between > 0 else np.nan
    ms_within = ss_within / df_within if df_within > 0 else np.nan

    ss_total = ss_between + ss_within if pd.notna(ss_between) and pd.notna(ss_within) else np.nan
    ratio = ss_between / ss_total if pd.notna(ss_total) and ss_total != 0 else np.nan

    # ── 불균형 보정 ICC(1) (참고값, 주 지표는 위 ratio) ──
    if df_between > 0 and df_within > 0 and pd.notna(ms_between) and pd.notna(ms_within):
        sum_nb2 = (per_block["n"] ** 2).sum()
        n0 = (N - sum_nb2 / N) / df_between
        icc1_denom = ms_between + (n0 - 1) * ms_within
        icc1 = (ms_between - ms_within) / icc1_denom if icc1_denom != 0 else np.nan
    else:
        icc1 = np.nan

    return {
        "grand_mean": grand_mean, "n_blocks_used": int(k), "n_sessions_used": int(N),
        "ss_between": ss_between, "ss_within": ss_within, "ss_total": ss_total,
        "df_between": df_between, "df_within": df_within,
        "ms_between": ms_between, "ms_within": ms_within,
        "ratio": ratio, "method": "eta_squared", "icc1": icc1,
        "per_block": per_block.reset_index().to_dict("records"),
        "blocks_used": sorted(scope["block_id"].unique().tolist()),
        "blocks_excluded": auto_excluded,
        "sessions_excluded": excluded_sessions,
    }


def detect_block_trend(block_stats):
    """블록별 test_idx-mean_error_px 스피어만 상관 "추세 지표".

    통계적 유의성 검정이 아니다(p-value 없음, "유의함" 주장 없음) — 블록당
    표본이 4~5개뿐이라 유의성 검정 자체가 무의미해 의도적으로 배제했다.
    pandas 내장 Series.corr(method="spearman")만 쓴다(scipy 의존성 불필요).

    Returns
    -------
    pd.DataFrame, columns:
        block_id, n_trials, spearman_r(n_trials<=3이면 NaN, 계산 안 함),
        monotonic({"increasing","decreasing","flat","none"}),
        mean_error_first, mean_error_last, delta_first_last
    """
    rows = []
    for block_id, g in block_stats.groupby("block_id"):
        g = g.dropna(subset=["mean_error_px"]).sort_values("test_idx")
        n = len(g)
        r = g["test_idx"].corr(g["mean_error_px"], method="spearman") if n > 3 else np.nan

        diffs = g["mean_error_px"].diff().dropna()
        if n < 2:
            monotonic = "none"
        elif (diffs >= 0).all():
            monotonic = "increasing" if (diffs > 0).any() else "flat"
        elif (diffs <= 0).all():
            monotonic = "decreasing" if (diffs < 0).any() else "flat"
        else:
            monotonic = "none"

        rows.append({
            "block_id": block_id, "n_trials": n, "spearman_r": r, "monotonic": monotonic,
            "mean_error_first": g["mean_error_px"].iloc[0] if n else np.nan,
            "mean_error_last": g["mean_error_px"].iloc[-1] if n else np.nan,
            "delta_first_last": (g["mean_error_px"].iloc[-1] - g["mean_error_px"].iloc[0]) if n else np.nan,
        })
    return pd.DataFrame(rows).sort_values("block_id").reset_index(drop=True)


def build_block_experiment_report(sessions_df, acc_df, session_ids, exclude_blocks=None):
    """블록 실험(캘리브레이션 1회 + raw 9점 테스트 N회 = 1블록) 분석을 한 번에
    계산해 JSON-직렬화 가능한 dict로 반환한다.

    build_block_structure/compute_block_session_stats/decompose_variance/
    detect_block_trend를 순서대로 호출해 묶는 통합 함수 — save_diagnostic_report()로
    저장하거나 notion_export.upload_diagnostic_to_notion()의 block_report
    인자로 그대로 넘길 수 있다.

    Returns
    -------
    dict with keys: n_sessions, n_blocks, block_summary(list[dict]),
        block_session_stats(list[dict]), variance_decomposition(dict), trend(list[dict])
    """
    merged, block_summary = build_block_structure(sessions_df, acc_df, session_ids)
    block_stats = compute_block_session_stats(merged)
    variance = decompose_variance(block_stats, exclude_blocks=exclude_blocks)
    trend = detect_block_trend(block_stats)

    report = {
        "n_sessions": len(session_ids),
        "n_blocks": int(block_summary["block_id"].nunique()),
        "block_summary": block_summary.to_dict("records"),
        "block_session_stats": block_stats.to_dict("records"),
        "variance_decomposition": variance,
        "trend": trend.to_dict("records"),
    }
    return _json_safe(report)


# ══════════════════════════════════════════════════════════════
# 5. 실행 방법
# ══════════════════════════════════════════════════════════════
# 이 파일은 라이브러리 모듈이다 (diagnostic_battery, save_diagnostic_report,
# build_diagnostic_html, load_all_sessions_history 등을 정의만 함). 실제 실행은
# 이 파일에 두지 않고 두 개의 드라이버로 분리했다 — 라이브러리 코드와 매번
# 값이 바뀌는 실행 설정을 한 파일에 섞으면, 드라이버와 여기 두 곳에 같은
# 예시 코드가 중복되어 한쪽만 고치고 잊어버리는 사고가 나기 쉽다.
#
# - 최신 스키마 단일 버전 + 시각화 + Notion 업로드: python -m src.analysis.run_export
# - 스키마 이력 전체 재분석(Phase 2): python -m src.analysis.stage0_reanalysis
#
# session_ids를 정하는 방법(전체/특정/조건별/최근 N세션)은 run_export.py
# 모듈 docstring의 "사용 매뉴얼" 절 참고. (참고: 이 파일은 src.viz.viz,
# src.tracking.calibration 등 로컬 프로젝트 모듈을 import하므로 Colab
# 등 프로젝트 폴더 밖 환경에서는 그대로 실행할 수 없다.)

if __name__ == "__main__":
    print("이 파일은 라이브러리 모듈입니다. 실행은 'python -m src.analysis.run_export' 또는 "
          "'python -m src.analysis.stage0_reanalysis'를 사용하세요.")
