# -*- coding: utf-8 -*-
"""캘리브레이션 품질 진단 지표(STB-11/12/13) 시각화 모듈.

calibration_quality.csv(캘리브레이션 포인트당 1행, 16점/calib_id)를 읽어
최신 캘리브레이션의 재투영 오차 지도·입력 안정성·수집 품질을 시각화한다.

viz.py(9점 테스트 전용, session_id·px 절대좌표 기준)와는 컬럼명·좌표계·
선택 키가 모두 달라 별도 모듈로 분리했다. setup_font()만 viz.py에서 가져온다.

설계 원칙
- 그래프 함수는 plt.show()를 직접 호출하지 않고 fig를 반환한다.
  호출하는 쪽에서 fig.savefig(저장) 또는 plt.show(팝업)를 선택.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.viz.viz import setup_font
from src.tracking.calibration import Calibrator


# ── 데이터 로딩 / 정리 ─────────────────────────────────────

def calibration_quality_filename():
    """현재 코드의 CALIB_SCHEMA_VERSION에 대응하는 calibration_quality CSV 파일명."""
    return f"calibration_quality_v{Calibrator.CALIB_SCHEMA_VERSION}.csv"


def load_calibration_data(results_dir):
    """calibration_quality_vN.N.csv를 읽어 DataFrame으로 반환.

    9점 테스트의 sessions.csv 같은 병합 대상이 없으므로 단순 로딩만 한다.
    """
    path = os.path.join(results_dir, calibration_quality_filename())
    return pd.read_csv(path, encoding="utf-8-sig")


def latest_calibration(df):
    """가장 최근 calib_id의 16개 포인트를 calib_point_index 순으로 반환.

    calibration_quality.csv는 append 전용이라 파일 내 등장 순서가 곧 시간
    순서다. 따라서 행 개수나 별도 타임스탬프 없이 "마지막으로 처음 등장한
    calib_id"만으로 최신 캘리브레이션을 판단할 수 있다.
    """
    seen = list(dict.fromkeys(df["calib_id"].tolist()))
    if not seen:
        raise ValueError("calibration_quality.csv에 데이터가 없습니다.")
    latest_id = seen[-1]
    return df[df["calib_id"] == latest_id].sort_values("calib_point_index")


def list_calib_ids(df):
    """df에 들어 있는 calib_id를 등장(=시간) 순서로 중복 없이 반환."""
    return list(dict.fromkeys(df["calib_id"].tolist()))


def recent_calibrations(df, n=None):
    """등장 순서 기준으로 가장 최근 n개 calib_id의 전체 행만 반환.

    viz.py의 latest_sessions(df, n)과 동일한 패턴을 calib_id 기준으로 옮긴 것.
    calibration_quality.csv에는 timestamp 컬럼이 없어 list_calib_ids()가 쓰는
    파일 등장 순서를 그대로 시간 순서로 취급한다. n이 없거나 전체 개수보다
    크면 전체를 반환한다.
    """
    seen = list_calib_ids(df)
    if n is None or n > len(seen):
        recent_ids = seen
    else:
        recent_ids = seen[-n:]
    return df[df["calib_id"].isin(recent_ids)].copy()


def compute_calib_rmse(single_calib_df):
    """한 calib_id의 포인트 행들에서 reproj_error_px(STB-11)로 RMSE를 재계산.

    calibration.py의 calib_reproj_rmse_px와 동일한 공식(sqrt(mean(x**2)))을
    쓰되, reproj_error_px가 None인 포인트는 제외한다. findHomography가
    성공한 정상 케이스는 16포인트 전부 유효해 결과가 sessions.csv의
    calib_reproj_rmse_px와 (반올림 차이 수준까지) 일치하고, findHomography가
    통째로 실패한 케이스는 n_valid=0이 되어 rmse=None을 반환한다.
    """
    valid = single_calib_df["reproj_error_px"].dropna()
    n_valid = len(valid)
    failed_count = len(single_calib_df) - n_valid
    rmse = float(np.sqrt(np.mean(valid ** 2))) if n_valid > 0 else None
    return {"rmse": rmse, "failed_count": int(failed_count), "n_valid": int(n_valid)}


def get_calibration_by_prefix(df, calib_id_prefix):
    """calib_id_prefix로 시작하는 단일 캘리브레이션을 calib_point_index 순으로 반환.

    plot_calibration_overview()는 한 번에 하나의 calib_id(16포인트)만 의미 있게
    표현하므로, 매칭이 여러 개면 "전부 보여주기"를 하지 않고 ValueError로 명확히
    알린다. 서로 다른 캘리브레이션의 포인트가 섞이면 좌표는 있지만 의미 없는
    그래프가 되기 때문이다.
    """
    matched = [cid for cid in list_calib_ids(df) if str(cid).startswith(calib_id_prefix)]

    if not matched:
        raise ValueError(f"'{calib_id_prefix}'로 시작하는 calib_id가 없습니다.")
    if len(matched) > 1:
        shown = [str(cid)[:8] for cid in matched]
        raise ValueError(
            f"'{calib_id_prefix}'로 시작하는 calib_id가 여러 개입니다: {shown}. "
            "더 긴 prefix로 다시 지정하세요."
        )

    return df[df["calib_id"] == matched[0]].sort_values("calib_point_index")


# ── 재투영 오차 공간 지도 ───────────────────────────────────

def _draw_reproj_map(ax, calib_df):
    """캘리브레이션 16포인트의 재투영 오차(STB-11)를 공간에 표시한다.

    point_x_norm/point_y_norm(해당 포인트의 화면상 목표 위치, 0~1 정규화)에
    산점도를 그리고 reproj_error_px 크기를 색으로 인코딩한다. 9점 테스트의
    target-예측 화살표와 달리 여기서는 목표 위치 자체에 오차 크기를 얹는
    방식이라 그대로 베끼지 않았다.

    findHomography 실패로 reproj_error_px가 비어있는(None) 포인트는 값으로
    표현할 수 없으므로 회색 X 마커로 구분해 "재투영 실패"임을 알린다.
    """
    df = calib_df
    valid = df[df["reproj_error_px"].notna()]
    failed = df[df["reproj_error_px"].isna()]

    sc = None
    if len(valid):
        sc = ax.scatter(
            valid["point_x_norm"], valid["point_y_norm"],
            c=valid["reproj_error_px"], cmap="YlOrRd",
            s=220, edgecolors="black", linewidths=0.6, zorder=3,
        )
    if len(failed):
        ax.scatter(
            failed["point_x_norm"], failed["point_y_norm"],
            c="gray", marker="X", s=220, linewidths=0,
            label="재투영 실패", zorder=3,
        )

    for _, row in df.iterrows():
        ax.annotate(
            str(int(row["calib_point_index"])),
            (row["point_x_norm"], row["point_y_norm"]),
            textcoords="offset points", xytext=(8, 8),
            fontsize=8, color="dimgray",
        )

    if sc is not None:
        cbar = ax.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("reproj_error_px (px)")

    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(1.08, -0.08)  # 화면 좌표 관례: 위가 0 → y축 뒤집기
    ax.set_title("Calibration Reprojection Error Map (16 points)")
    ax.set_xlabel("point_x_norm (0~1)")
    ax.set_ylabel("point_y_norm (0~1)")
    if len(failed):
        ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)


# ── 입력 신호 안정성 (iris_std) ─────────────────────────────

def _draw_iris_std_bars(ax, calib_df):
    """캘리브레이션 입력 신호 안정성(STB-12)을 포인트별 막대로 표시한다.

    iris_std_x_px/iris_std_y_px는 이름에 _px가 붙어 있지만 실제로는
    캘리브레이션 시점 정규화 좌표(0~1) 기준 표준편차다 — gaze_accuracy.csv의
    동일 이름 컬럼(카메라 프레임 px 기준, STB-10)과 단위가 다른 알려진
    이슈(CLAUDE.md 참고)라서, 축 라벨에 반드시 "정규화 좌표 기준"을 명시해
    두 지표가 섞여 비교되지 않도록 한다.
    """
    df = calib_df.sort_values("calib_point_index")
    idx = df["calib_point_index"].astype(int)
    x = np.arange(len(idx))
    width = 0.35

    ax.bar(x - width / 2, df["iris_std_x_px"], width,
           label="iris_std_x", color="steelblue")
    ax.bar(x + width / 2, df["iris_std_y_px"], width,
           label="iris_std_y", color="darkorange")

    ax.set_xticks(x)
    ax.set_xticklabels(idx)
    ax.set_xlabel("calib_point_index")
    ax.set_ylabel("iris_std (정규화 좌표 기준, 0~1 스케일)")
    ax.set_title("Calibration Input Stability (iris_std_x/y)")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)


# ── 수집 품질 (sample_count / mean_confidence) ──────────────

def _draw_sample_quality_bars(ax, calib_df):
    """캘리브레이션 포인트 수집 품질(STB-13)을 막대+선으로 표시한다.

    sample_count(개수)와 mean_confidence(0~1 비율)는 단위가 달라 같은
    막대에 얹지 않고, sample_count는 막대(왼쪽 축), mean_confidence는
    선(오른쪽 축, twinx)으로 분리해 표현한다. mean_confidence가 None인
    포인트는 NaN으로 바꿔 선이 그 지점에서 자연스럽게 끊기도록 한다.
    """
    df = calib_df.sort_values("calib_point_index")
    idx = df["calib_point_index"].astype(int)
    x = np.arange(len(idx))

    ax.bar(x, df["sample_count"], color="steelblue", alpha=0.8,
           label="sample_count")
    ax.set_xticks(x)
    ax.set_xticklabels(idx)
    ax.set_xlabel("calib_point_index")
    ax.set_ylabel("sample_count (개)", color="steelblue")
    ax.tick_params(axis="y", labelcolor="steelblue")

    ax2 = ax.twinx()
    mean_conf = pd.to_numeric(df["mean_confidence"], errors="coerce")
    ax2.plot(x, mean_conf, color="darkred", marker="o",
             label="mean_confidence")
    ax2.set_ylabel("mean_confidence (0~1 비율)", color="darkred")
    ax2.tick_params(axis="y", labelcolor="darkred")
    ax2.set_ylim(0, 1.05)

    ax.set_title("Calibration Collection Quality (sample_count / mean_confidence)")
    ax.grid(True, axis="y", alpha=0.3)

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="upper right")


# ── 재투영 오차 시계열 추세 ──────────────────────────────────

def _draw_calibration_trend(ax, labels, rmse_values, failed_counts):
    """calib_id별 RMSE를 시간순 막대로 그린다.

    정상 케이스(rmse 값 있음)와 전체 실패 케이스(rmse=None, 해당
    calib_id의 16포인트가 모두 reproj_error_px 없음)를 명확히 구분한다.
    실패 막대는 실제 오차 크기가 아니라 "측정 불가" placeholder이므로
    고정 숫자 대신 정상 막대들의 y축 오토스케일 결과(ylim) 기준 상대
    위치에 회색 해치로 그리고, 축 상단에 빨간 "failed" 텍스트를 고정해
    낮은 정상 RMSE와 절대 헷갈리지 않게 한다 (_draw_reproj_map이 실패
    포인트를 회색 X로 구분하는 것과 같은 원칙).
    """
    x = np.arange(len(labels))

    normal_idx = [i for i, r in enumerate(rmse_values) if r is not None]
    failed_idx = [i for i, r in enumerate(rmse_values) if r is None]

    if normal_idx:
        ax.bar([x[i] for i in normal_idx],
               [rmse_values[i] for i in normal_idx],
               color="steelblue")

    # 정상 막대를 먼저 그린 뒤의 오토스케일 결과를 기준으로 실패 막대 높이를 잡는다.
    if not normal_idx:
        ax.set_ylim(0, 1)
    fallback_height = ax.get_ylim()[1] * 0.8

    if failed_idx:
        ax.bar([x[i] for i in failed_idx],
               [fallback_height] * len(failed_idx),
               color="lightgray", hatch="//", edgecolor="gray")
        for i in failed_idx:
            ax.text(x[i], 0.95, "failed", transform=ax.get_xaxis_transform(),
                    ha="center", va="top", color="crimson",
                    fontsize=8, fontweight="bold")

    # 부분 실패(정상 케이스인데 일부 포인트만 없음) — 현재 calibration.py
    # 로직상 발생하지 않지만 방어적으로 처리.
    for i in normal_idx:
        if failed_counts[i] > 0:
            ax.text(x[i], rmse_values[i], f"{failed_counts[i]} failed",
                    ha="center", va="bottom", fontsize=7, color="dimgray")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, fontsize=8)
    ax.set_xlabel("calib_id (chronological)")
    ax.set_ylabel("calib_reproj_rmse_px (px)")
    ax.set_title("Calibration Reprojection Error Trend (RMSE)")
    ax.grid(True, axis="y", alpha=0.3)

    if failed_idx:
        from matplotlib.patches import Patch
        legend_items = [Patch(facecolor="lightgray", edgecolor="gray",
                               hatch="//", label="재투영 실패")]
        # "failed" 텍스트가 축 상단에 고정되므로, 실패 막대가 오른쪽 끝(최신
        # calib_id)에 있으면 upper right 범례와 겹친다. loc="best"로 겹침을
        # 피한다.
        ax.legend(handles=legend_items, fontsize=8, loc="best")


def plot_calibration_trend(df, n=None, meta_df=None):
    """calib_id별 재투영 오차(STB-11) RMSE를 시간순 막대로 나열해 추세를 본다.

    sessions.csv를 조인하지 않고 calibration_quality.csv의 reproj_error_px
    에서 compute_calib_rmse()로 직접 RMSE를 재계산한다. calib_id 순서는
    recent_calibrations()/list_calib_ids()가 쓰는 파일 등장 순서를 그대로
    따른다 (latest_calibration()과 동일한 전제).

    meta_df는 이번에는 사용하지 않는다(내부에서 무시). dev_version별 색
    구분 등이 필요해지면 이 인자로 (calib_id, dev_version) 매핑을 받아
    확장할 수 있도록 시그니처만 열어둔다.
    """
    subset = recent_calibrations(df, n)
    ids = list_calib_ids(subset)

    labels = [str(cid)[:8] for cid in ids]
    rmse_values = []
    failed_counts = []

    for cid in ids:
        result = compute_calib_rmse(subset[subset["calib_id"] == cid])
        rmse_values.append(result["rmse"])
        failed_counts.append(result["failed_count"])

    fig, ax = plt.subplots(figsize=(10, 5))
    _draw_calibration_trend(ax, labels, rmse_values, failed_counts)
    fig.tight_layout()
    return fig


# ── 합본 ────────────────────────────────────────────────────

def plot_calibration_overview(calib_df):
    """한 캘리브레이션(calib_id)의 재투영 오차 지도·입력 안정성·수집 품질을
    세로 3단으로 한 fig에 그린다.

    지도가 정사각형에 가까워 위쪽에 가장 크게 배치하고, 막대그래프 2개는
    아래로 쌓는다. fig만 반환하며 저장(savefig)/팝업(show)은 호출하는
    쪽이 선택한다 (viz.py의 plot_session_overview와 동일한 관례).
    """
    df = calib_df
    calib_id = df["calib_id"].iloc[0] if len(df) else "?"

    fig, (ax_top, ax_mid, ax_bot) = plt.subplots(
        3, 1, figsize=(10, 14),
        gridspec_kw={"height_ratios": [5.0, 3.0, 3.0]}
    )

    _draw_reproj_map(ax_top, df)
    _draw_iris_std_bars(ax_mid, df)
    _draw_sample_quality_bars(ax_bot, df)

    rmse_info = compute_calib_rmse(df)
    if rmse_info["rmse"] is not None:
        title_suffix = f" | RMSE: {rmse_info['rmse']:.2f}px"
        if rmse_info["failed_count"] > 0:
            title_suffix += f" ({rmse_info['failed_count']} failed)"
    else:
        title_suffix = " | RMSE: 측정 불가 (전체 실패)"

    fig.suptitle(f"calibration: {str(calib_id)[:8]}{title_suffix}",
                 fontsize=10, color="dimgray")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return fig
