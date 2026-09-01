"""
캘리브레이션 유(calibrated) / 무(no_calibration) 조건 시선 분산표 생성 스크립트.

1회성 보고서용 스크립트다. gaze_accuracy_v{SCHEMA}.csv에서 지정한 두 run_id의
9점 테스트 결과(target_x_px/y_px, pred_x_px/y_px, gaze_std_x_px/y_px)를 읽어
타깃별 [정답점 / 응시 평균점 / 표준편차 타원]을 그린다.

**조건별로 별도 PNG 2장**을 만든다(한 그림에 겹쳐 그리지 않음) — 범례로 색을
구분하지 않아도 두 이미지를 나란히 놓고 보는 것만으로 분산 차이가 바로
보이도록, 두 그림의 축 범위·스케일·종횡비를 동일하게 맞춘다.

주의 (사전 조사에서 확인된 제약):
- gaze_accuracy CSV는 프레임 단위 원시 좌표가 아니라 타깃별 평균+표준편차만
  담고 있다(src/metrics/collector.py의 end_target()이 원시 샘플을 통계 계산
  직후 폐기함). 따라서 이 그림은 실제 개별 프레임 점들의 산포가 아니라
  "평균 응시점 + 분산 타원"이다.
- canonical gaze_accuracy_v{SCHEMA}.csv 자체에는 mode 컬럼이 없어, 각 run_id가
  어느 조건인지 알려면 sessions_v{SCHEMA}.csv의 dev_version과 join해야 한다.
  이 스크립트가 그 join을 대신 해준다.

사용 예:
    python -m scripts.plot_gaze_dispersion_comparison \\
        --calibrated-run-id 23decfd5-c6a0-4a02-916d-d2927e453cdf \\
        --no-calibration-run-id 8b8a29f7-b3e2-4704-bd7e-9fb1e8c8e682
"""

import argparse
import csv
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

from src.viz.viz import setup_font

DEFAULT_RESULTS_DIR = "gaze_accuracy_results"
DEFAULT_SCHEMA_VERSION = "1.9"

COLOR_POINT = "#2563eb"       # 파랑 (평균 응시점 / 타원)
COLOR_TARGET = "#111111"      # 검정 (정답 좌표)


def load_rows(path):
    if not os.path.isfile(path):
        raise SystemExit(f"파일을 찾을 수 없습니다: {path}")
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def get_session_row(sessions_rows, run_id):
    for row in sessions_rows:
        if row["run_id"] == run_id:
            return row
    return None


def to_float_or_none(value):
    if value is None or value == "":
        return None
    return float(value)


def plot_single_condition(rows, cond_label, screen_w, screen_h):
    """한 조건(run_id)만 담은 별도의 figure를 만들어 반환한다.

    두 조건 그림을 나란히 놓고 눈으로 바로 비교할 수 있도록, 호출하는
    쪽에서 screen_w/h(두 run 공통값)로 축 범위·종횡비를 강제 통일한다.
    """
    fig, ax = plt.subplots(figsize=(7, 6.5))

    missing_targets = []

    for row in sorted(rows, key=lambda r: int(r["target_index"])):
        target_x = float(row["target_x_px"])
        target_y = float(row["target_y_px"])
        target_index = row["target_index"]

        ax.scatter(
            [target_x], [target_y],
            marker="x", color=COLOR_TARGET, s=70, linewidths=2, zorder=5,
        )
        ax.annotate(
            target_index, (target_x, target_y),
            textcoords="offset points", xytext=(6, 6),
            fontsize=8, color=COLOR_TARGET,
        )

        pred_x = to_float_or_none(row["pred_x_px"])
        pred_y = to_float_or_none(row["pred_y_px"])

        if pred_x is None or pred_y is None:
            missing_targets.append(target_index)
            continue

        ax.scatter([pred_x], [pred_y], color=COLOR_POINT, s=45, zorder=4)

        std_x = to_float_or_none(row["gaze_std_x_px"])
        std_y = to_float_or_none(row["gaze_std_y_px"])

        if std_x is not None and std_y is not None:
            ellipse = Ellipse(
                (pred_x, pred_y),
                width=2 * std_x,
                height=2 * std_y,
                edgecolor=COLOR_POINT,
                facecolor=COLOR_POINT,
                alpha=0.15,
                linewidth=1.5,
                zorder=3,
            )
            ax.add_patch(ellipse)

    if screen_w and screen_h:
        ax.set_xlim(0, screen_w)
        ax.set_ylim(screen_h, 0)  # 화면 좌표계(y 아래로 증가)에 맞춰 y축 반전
    else:
        ax.invert_yaxis()

    ax.set_xlabel("화면 x (px)")
    ax.set_ylabel("화면 y (px)")
    ax.set_title(f"시선 분산표 — {cond_label}\n(9점 테스트, 평균 응시점 ± 1σ 타원)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.3)

    legend_elements = [
        Line2D([0], [0], marker="x", color=COLOR_TARGET, linestyle="None",
               markersize=9, markeredgewidth=2, label="타깃(정답 좌표)"),
        Line2D([0], [0], marker="o", color=COLOR_POINT, linestyle="None",
               markersize=8, label="평균 응시점 (± 1σ 타원)"),
    ]
    ax.legend(handles=legend_elements, loc="upper center",
              bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=9)

    fig.tight_layout()

    return fig, missing_targets


def print_summary_table(condition_label, rows):
    print(f"\n[{condition_label}] 타깃별 오프셋/분산 (px)")
    print(f"{'idx':>3} {'target_x':>9} {'target_y':>9} {'pred_x':>9} {'pred_y':>9} "
          f"{'error':>8} {'std_x':>7} {'std_y':>7} {'n':>4}")
    for row in sorted(rows, key=lambda r: int(r["target_index"])):
        pred_x = row["pred_x_px"] or "-"
        pred_y = row["pred_y_px"] or "-"
        error = row["euclidean_error_px"] or "-"
        std_x = row["gaze_std_x_px"] or "-"
        std_y = row["gaze_std_y_px"] or "-"
        print(
            f"{row['target_index']:>3} {row['target_x_px']:>9} {row['target_y_px']:>9} "
            f"{pred_x:>9} {pred_y:>9} {error:>8} {std_x:>7} {std_y:>7} {row['sample_count']:>4}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibrated-run-id", required=True)
    parser.add_argument("--no-calibration-run-id", required=True)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--schema-version", default=DEFAULT_SCHEMA_VERSION)
    parser.add_argument("--out-dir", default=None,
                         help="PNG 2장을 저장할 디렉터리 (기본값: --results-dir)")
    args = parser.parse_args()

    accuracy_path = os.path.join(
        args.results_dir, f"gaze_accuracy_v{args.schema_version}.csv"
    )
    sessions_path = os.path.join(
        args.results_dir, f"sessions_v{args.schema_version}.csv"
    )

    accuracy_rows = load_rows(accuracy_path)
    sessions_rows = load_rows(sessions_path)

    conditions = [
        ("calibrated", args.calibrated_run_id),
        ("no_calibration", args.no_calibration_run_id),
    ]

    setup_font()

    out_dir = args.out_dir or args.results_dir
    os.makedirs(out_dir, exist_ok=True)

    # 두 그림의 축 범위를 동일하게 맞추기 위해 먼저 screen_w/h를 확정한다.
    screen_w, screen_h = None, None
    for _, run_id in conditions:
        session_row = get_session_row(sessions_rows, run_id)
        if session_row is None:
            continue
        sw = to_float_or_none(session_row.get("screen_w"))
        sh = to_float_or_none(session_row.get("screen_h"))
        if sw and sh:
            if screen_w is not None and (screen_w != sw or screen_h != sh):
                print(
                    "[경고] 두 run의 화면 해상도가 다릅니다 "
                    f"({screen_w}x{screen_h} vs {sw}x{sh}) — 두 그림을 나란히 "
                    "비교하는 전제(공정 비교 조건)가 깨졌을 수 있습니다."
                )
            screen_w, screen_h = sw, sh

    for cond_name, run_id in conditions:
        rows = [r for r in accuracy_rows if r["run_id"] == run_id]
        if len(rows) != 9:
            print(
                f"[경고] run_id={run_id} ({cond_name})의 gaze_accuracy 행이 "
                f"9개가 아니라 {len(rows)}개입니다."
            )
        if not rows:
            raise SystemExit(
                f"gaze_accuracy_v{args.schema_version}.csv에 "
                f"run_id={run_id} 데이터가 없습니다."
            )

        session_row = get_session_row(sessions_rows, run_id)
        if session_row is None:
            print(f"[경고] sessions_v{args.schema_version}.csv에서 run_id={run_id}를 "
                  "찾지 못했습니다 — 제목에 run_id를 그대로 씁니다.")
            cond_label = run_id
        else:
            cond_label = session_row.get("dev_version") or run_id

        fig, missing = plot_single_condition(rows, cond_label, screen_w, screen_h)
        if missing:
            print(f"[{cond_label}] 유효 추적 샘플이 없어 평균점을 못 그린 타깃: {missing}")

        print_summary_table(cond_label, rows)

        out_path = os.path.join(out_dir, f"gaze_dispersion_{cond_name}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"저장 완료: {out_path}")


if __name__ == "__main__":
    main()
