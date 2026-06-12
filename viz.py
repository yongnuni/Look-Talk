# -*- coding: utf-8 -*-
"""시선 정확도 평가 지표 시각화 공통 모듈.

코랩에서 수동으로 돌리던 시각화 코드를 재구성한 것으로,
make_report.py(저장용)와 main.py 종료 팝업(즉시 확인용)이
같은 함수를 공유한다.

설계 원칙
- 그래프 함수는 plt.show()를 직접 호출하지 않고 fig를 반환한다.
  호출하는 쪽에서 fig.savefig(저장) 또는 plt.show(팝업)를 선택.
- 세션 선택은 id 복붙이 아니라 "최근 N개"로 자동화.
- 경로·폰트는 OS에 의존하지 않도록 인자/분기로 처리.
"""

import os
import platform

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# ── 폰트 설정 (OS별 한글 폰트) ─────────────────────────────

def setup_font():
    """실행 환경 OS에 맞는 한글 폰트를 설정한다.

    Windows: 맑은 고딕 / macOS: AppleGothic / Linux: NanumGothic.
    폰트를 못 찾아도 그래프는 그려지도록 조용히 넘어간다.
    """
    system = platform.system()
    if system == "Windows":
        plt.rcParams["font.family"] = "Malgun Gothic"
    elif system == "Darwin":  # macOS
        plt.rcParams["font.family"] = "AppleGothic"
    else:  # Linux 등
        try:
            import matplotlib.font_manager as fm
            for path in [
                "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
                "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
            ]:
                if os.path.isfile(path):
                    fm.fontManager.addfont(path)
                    plt.rcParams["font.family"] = "NanumGothic"
                    break
        except Exception:
            pass
    plt.rcParams["axes.unicode_minus"] = False


# ── 데이터 로딩 / 정리 ─────────────────────────────────────

def load_data(results_dir):
    """gaze_accuracy.csv + sessions.csv를 읽어 병합한 DataFrame을 반환.

    타깃 단위 지표(gaze_accuracy)에 세션 메타(sessions)를 session_id로 붙인다.
    start_timestamp 기준으로 정렬해 항상 측정 시간순을 보장한다.
    """
    acc_path = os.path.join(results_dir, "gaze_accuracy.csv")
    sess_path = os.path.join(results_dir, "sessions.csv")

    acc = pd.read_csv(acc_path, encoding="utf-8-sig")

    # sessions.csv가 있으면 메타(버전·시각)를 붙인다. 없어도 동작.
    if os.path.isfile(sess_path):
        sess = pd.read_csv(sess_path, encoding="utf-8-sig")
        keep = [c for c in ["session_id", "dev_version", "start_timestamp",
                            "px_per_cm"] if c in sess.columns]
        acc = acc.merge(sess[keep], on="session_id", how="left")

    # 시간순 정렬 (start_timestamp 없으면 원래 순서 유지)
    if "start_timestamp" in acc.columns:
        acc = acc.sort_values("start_timestamp").reset_index(drop=True)

    return acc


def filter_sessions(df, exclude_ids=None):
    """분석에서 제외할 세션(불량 데이터)을 걸러낸다. 원본은 보존.

    exclude_ids: 영구 제외할 session_id 리스트. 기본은 아무것도 제외 안 함.
    """
    if not exclude_ids:
        return df
    return df[~df["session_id"].isin(exclude_ids)].copy()


def latest_sessions(df, n=1):
    """측정 시간순으로 가장 최근 n개 세션만 반환.

    session_id를 직접 지정하지 않고 "최근 N개"로 자동 선택한다.
    start_timestamp가 없으면 등장 순서 기준.
    """
    if "start_timestamp" in df.columns:
        order = df.sort_values("start_timestamp")
    else:
        order = df

    # 세션의 등장(정렬) 순서를 보존하며 고유 id 추출 → 뒤에서 n개
    seen = list(dict.fromkeys(order["session_id"].tolist()))
    recent_ids = seen[-n:]
    return df[df["session_id"].isin(recent_ids)].copy()


def get_session(df, session_id):
    """단일 세션을 target_index 순으로 정렬해 반환."""
    s = df[df["session_id"] == session_id].sort_values("target_index")
    return s


def list_session_ids(df):
    """df에 들어 있는 세션 id를 측정 시간순으로 반환."""
    if "start_timestamp" in df.columns:
        order = df.sort_values("start_timestamp")
    else:
        order = df
    return list(dict.fromkeys(order["session_id"].tolist()))


# ── 텍스트 요약 (새 지표 cm·STB-01~04 포함) ────────────────

def summarize_session(session_df):
    """한 세션의 핵심 지표를 dict로 요약. 새 지표(cm·STB)를 포함한다.

    그래프 없이 콘솔 한 줄 출력이나 판정에 쓰기 좋은 형태.
    유효 예측이 없는(전실패) 타깃은 평균에서 자동 제외(NaN 무시).
    """
    s = session_df
    sid = s["session_id"].iloc[0] if len(s) else "?"

    def _mean(col):
        return float(s[col].mean()) if col in s.columns else float("nan")

    summary = {
        "session_id": sid,
        "short_id": str(sid)[:8],
        "n_targets": int(len(s)),
        "mean_error_px": _mean("euclidean_error_px"),
        "mean_error_cm": _mean("euclidean_error_cm"),
        "mean_fps": _mean("stb01_fps"),
        "mean_landmark_rate": _mean("stb02_landmark_rate"),
        "mean_face_fail": _mean("stb03_face_fail_rate"),
        "mean_dropout": _mean("stb04_dropout_rate"),
    }
    if "dev_version" in s.columns and len(s):
        summary["dev_version"] = s["dev_version"].iloc[0]
    return summary


def format_summary_line(summary):
    """summarize_session() 결과를 사람이 읽는 한 줄 문자열로."""
    parts = [f"[{summary['short_id']}]"]
    if summary.get("dev_version"):
        parts.append(str(summary["dev_version"]))
    parts.append(f"평균 {summary['mean_error_px']:.1f}px")
    if not np.isnan(summary["mean_error_cm"]):
        parts.append(f"({summary['mean_error_cm']:.2f}cm)")
    if not np.isnan(summary["mean_fps"]):
        parts.append(f"FPS {summary['mean_fps']:.1f}")
    if not np.isnan(summary["mean_landmark_rate"]):
        parts.append(f"얼굴 {summary['mean_landmark_rate']*100:.0f}%")
    if not np.isnan(summary["mean_dropout"]):
        parts.append(f"dropout {summary['mean_dropout']*100:.1f}%")
    return " | ".join(parts)


# ── 그래프 그리기: 주어진 ax에 그리는 내부 함수 ─────────────
# 그리는 로직을 ax 단위로 분리해, 단독 출력과 합본(한 장에 두 칸)
# 양쪽에서 같은 코드를 재사용한다.

def _draw_error_map(ax, session_df, screen_w, screen_h):
    """오차 지도를 주어진 ax에 그린다. (화면 좌표: 위가 0 → y축 뒤집기)"""
    s = session_df
    sid = s["session_id"].iloc[0] if len(s) else "?"
    valid = s.dropna(subset=["pred_x_px", "pred_y_px"])

    ax.scatter(valid["target_x_px"], valid["target_y_px"],
               c="green", s=120, label="정답 (타깃)", zorder=3)
    ax.scatter(valid["pred_x_px"], valid["pred_y_px"],
               c="red", s=80, label="예측 시선", zorder=3)

    for _, row in valid.iterrows():
        ax.annotate("",
            xy=(row["pred_x_px"], row["pred_y_px"]),
            xytext=(row["target_x_px"], row["target_y_px"]),
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.2, alpha=0.7))
        ax.text(row["target_x_px"], row["target_y_px"] - 15,
                str(int(row["target_index"])), fontsize=9, ha="center")

    margin = 100
    rect = patches.Rectangle((0, 0), screen_w, screen_h, linewidth=1,
                             edgecolor="gray", linestyle="--", facecolor="none")
    ax.add_patch(rect)
    ax.set_xlim(-margin, screen_w + margin)
    ax.set_ylim(screen_h + margin, -margin)   # y축 뒤집기는 여기서만
    ax.set_aspect("equal")

    ax.set_title(f"Gaze Accuracy Map ({len(valid)} points)")
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)


def _draw_error_bars(ax, session_df):
    """타깃별 Euclidean 오차 막대 + 떨림(std) 에러바를 주어진 ax에 그린다."""
    s = session_df
    valid = s.dropna(subset=["euclidean_error_px"])

    idx = valid["target_index"].astype(int)
    err = valid["euclidean_error_px"]
    std = np.sqrt(valid["gaze_std_x_px"]**2 + valid["gaze_std_y_px"]**2)

    ax.bar(idx, err, yerr=std, capsize=4, color="steelblue", alpha=0.8,
           error_kw=dict(ecolor="darkred", lw=1.2))

    mean_err = err.mean()
    ax.axhline(mean_err, color="gray", linestyle="--", alpha=0.7,
               label=f"평균 {mean_err:.0f}px")

    ax.set_title("Euclidean Error per Target (error bar = gaze std)")
    ax.set_xlabel("target_index")
    ax.set_ylabel("error (px)")
    ax.set_xticks(idx)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)


# ── 합본: 오차 지도 + 막대를 한 장(위/아래 두 칸)에 ─────────

def plot_session_overview(session_df, screen_w, screen_h):
    """한 세션의 오차 지도(위)와 타깃별 오차 막대(아래)를 한 fig에 그린다.

    make_report 저장과 main.py 종료 팝업이 공유하는 기본 출력.
    fig 한 개만 반환하므로 파일도 한 장, 팝업 창도 한 개.
    """
    s = session_df
    sid = s["session_id"].iloc[0] if len(s) else "?"

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(10, 10),
        gridspec_kw={"height_ratios": [5.6, 4.0]}
    )

    _draw_error_map(ax_top, s, screen_w, screen_h)
    _draw_error_bars(ax_bot, s)

    # 세션 id는 그림 전체 제목으로 한 번만 표시
    fig.suptitle(f"session: {str(sid)[:8]}", fontsize=10, color="dimgray")
    fig.tight_layout(rect=[0, 0, 1, 0.98])  # suptitle 자리 확보
    return fig


# ── 단독 출력 (기존 호환: 필요 시 개별 그래프만) ───────────

def plot_error_map(session_df, screen_w, screen_h):
    """오차 지도만 단독 fig로. (합본은 plot_session_overview 사용)"""
    fig, ax = plt.subplots(figsize=(10, 5.6))
    _draw_error_map(ax, session_df, screen_w, screen_h)
    fig.tight_layout()
    return fig


def plot_error_bars(session_df):
    """타깃별 오차 막대만 단독 fig로. (합본은 plot_session_overview 사용)"""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    _draw_error_bars(ax, session_df)
    fig.tight_layout()
    return fig


# ── 그래프 3: 세션별 추세 (다세션, 시간순) ─────────────────

def plot_session_trend(df):
    """세션별 평균 오차를 측정 시간순으로 막대화. dev_version별 색 구분. fig 반환.

    groupby는 id 알파벳순으로 섞이므로, 집계 후 start_timestamp로 재정렬한다.
    """
    agg = {
        "mean_error": ("euclidean_error_px", "mean"),
        "std_error": ("euclidean_error_px", "std"),
    }
    summary = df.groupby("session_id").agg(**agg).reset_index()

    # 메타(시각·버전)를 붙여 시간순 정렬 — groupby의 알파벳순 정렬 교정
    meta_cols = [c for c in ["session_id", "dev_version", "start_timestamp"]
                 if c in df.columns]
    meta = df[meta_cols].drop_duplicates("session_id")
    summary = summary.merge(meta, on="session_id", how="left")

    if "start_timestamp" in summary.columns:
        summary = summary.sort_values("start_timestamp").reset_index(drop=True)

    x = range(len(summary))

    # dev_version별 색
    if "dev_version" in summary.columns:
        versions = summary["dev_version"].dropna().unique()
        cmap = plt.cm.tab10(np.linspace(0, 1, max(len(versions), 1)))
        color_map = dict(zip(versions, cmap))
        bar_colors = [color_map.get(v, "steelblue") for v in summary["dev_version"]]
    else:
        versions = []
        color_map = {}
        bar_colors = "steelblue"

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x, summary["mean_error"], yerr=summary["std_error"], capsize=4,
           color=bar_colors, alpha=0.8, error_kw=dict(ecolor="gray", lw=1))

    overall = summary["mean_error"].mean()
    ax.axhline(overall, color="red", linestyle="--", alpha=0.6,
               label=f"overall mean {overall:.0f}px")

    ax.set_title("Mean Error per Session (over time)")
    ax.set_xlabel("session (chronological)")
    ax.set_ylabel("mean error (px)")
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(s)[:6] for s in summary["session_id"]],
                       rotation=45, fontsize=8)

    # 범례: 버전별 색 + 평균선
    from matplotlib.patches import Patch
    legend_items = [Patch(color=color_map[v], label=str(v)) for v in versions]
    legend_items.append(plt.Line2D([0], [0], color="red", linestyle="--",
                                   label=f"mean {overall:.0f}px"))
    if legend_items:
        ax.legend(handles=legend_items, fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


# ── 화면 해상도 역산 ───────────────────────────────────────

def infer_screen_size(df):
    """타깃 좌표(0.9 비율 지점)로 화면 px 해상도를 역산.

    9점 테스트가 target = SCREEN * 0.9로 바깥 점을 찍으므로,
    최댓값을 0.9로 나누면 원래 해상도가 나온다. 기기 무관 자동.
    """
    screen_w = round(df["target_x_px"].max() / 0.9)
    screen_h = round(df["target_y_px"].max() / 0.9)
    return screen_w, screen_h
