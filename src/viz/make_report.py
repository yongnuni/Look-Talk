# -*- coding: utf-8 -*-
"""성능 지표 리포트 자동 생성 스크립트 (저장용).

gaze_accuracy_results/ 의 CSV를 읽어 최근 N세션의
오차 지도·타깃별 막대·세션 추세 그래프를 PNG로 저장하고,
각 세션의 핵심 지표를 콘솔에 한 줄씩 요약 출력한다.

사용 예 (프로젝트 루트에서 터미널에 입력):
    python -m src.viz.make_report                 # 최근 1세션
    python -m src.viz.make_report -n 3            # 최근 3세션
    python -m src.viz.make_report --results-dir gaze_accuracy_results --out report
    python -m src.viz.make_report --session sessionid   # 특정 세션(앞자리 매칭)
    python -m src.viz.make_report --calib                # 최신 캘리브레이션 품질 그래프
    python -m src.viz.make_report --calib calibid       # 특정 캘리브레이션(앞자리 매칭)
"""

import os
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import src.viz.viz as viz
import src.viz.calib_viz as calib_viz


def save_session_figures(df, session_id, out_dir, screen_w, screen_h):
    """한 세션의 오차 지도 + 막대그래프를 한 장의 PNG로 저장."""
    s = viz.get_session(df, session_id)
    short = str(session_id)[:8]

    fig = viz.plot_session_overview(s, screen_w, screen_h)
    path = os.path.join(out_dir, f"{short}_overview.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    return path


def main():
    parser = argparse.ArgumentParser(
        description="시선 정확도 지표 리포트 생성 (PNG 저장 + 요약 출력)"
    )
    parser.add_argument("--results-dir", default="gaze_accuracy_results",
                        help="CSV가 있는 폴더 (기본: gaze_accuracy_results)")
    parser.add_argument("-n", "--num", type=int, default=1,
                        help="최근 몇 세션을 볼지 (기본: 1)")
    parser.add_argument("--out", default="report",
                        help="PNG 저장 폴더 (기본: report)")
    parser.add_argument("--session", default=None,
                        help="특정 세션만 (session_id 앞자리로 매칭)")
    parser.add_argument("--exclude", nargs="*", default=None,
                        help="분석 제외할 session_id 목록 (불량 데이터)")
    parser.add_argument("--calib", nargs="?", const="latest", default=None,
                        help="캘리브레이션 품질 그래프 저장. 값 없이 --calib만 쓰면 최신 "
                             "캘리브레이션, --calib <ID>로 특정 calib_id(앞자리) 지정 가능")
    args = parser.parse_args()

    viz.setup_font()

    # 데이터 로딩
    try:
        df = viz.load_data(args.results_dir)
    except FileNotFoundError:
        print(f"[오류] CSV를 찾을 수 없습니다: {os.path.join(args.results_dir, viz.accuracy_filename())}")
        return

    if len(df) == 0:
        print("[경고] 데이터가 비어 있습니다.")
        return

    df = viz.filter_sessions(df, args.exclude)

    os.makedirs(args.out, exist_ok=True)
    screen_w, screen_h = viz.infer_screen_size(df)

    # 대상 세션 결정
    if args.session:
        all_ids = viz.list_session_ids(df)
        matched = [sid for sid in all_ids if str(sid).startswith(args.session)]
        if not matched:
            print(f"[오류] '{args.session}'로 시작하는 세션이 없습니다.")
            print("사용 가능한 세션:", [str(s)[:8] for s in all_ids])
            return
        target_ids = matched
        scope_df = df[df["session_id"].isin(target_ids)].copy()
    else:
        scope_df = viz.latest_sessions(df, args.num)
        target_ids = viz.list_session_ids(scope_df)

    # 헤더
    print("=" * 60)
    print(f"리포트 대상: {len(target_ids)}개 세션  |  해상도 역산 {screen_w}x{screen_h}")
    print(f"저장 폴더: {os.path.abspath(args.out)}")
    print("=" * 60)

    # 세션별 그래프 저장 + 요약 출력
    for sid in target_ids:
        s = viz.get_session(df, sid)
        summary = viz.summarize_session(s)
        print(viz.format_summary_line(summary))
        save_session_figures(df, sid, args.out, screen_w, screen_h)

    # 세션 추세 그래프 (2개 이상일 때만)
    if len(target_ids) >= 2:
        fig_trend = viz.plot_session_trend(scope_df)
        trend_path = os.path.join(args.out, "session_trend.png")
        fig_trend.savefig(trend_path, dpi=130, bbox_inches="tight")
        plt.close(fig_trend)
        print("-" * 60)
        print(f"세션 추세 그래프 저장: {trend_path}")

    # 캘리브레이션 품질 그래프 (--calib 시에만)
    # sessions.csv와의 1:N 조인이 아니라 calibration_quality.csv 단독 조회라
    # --session/-n/--exclude로 어떤 세션을 보든 이 부분은 영향받지 않는다.
    # --calib (값 없음) -> 최신 캘리브레이션, --calib <id-prefix> -> 지정한 캘리브레이션
    calib_saved = False
    if args.calib is not None:
        try:
            calib_df = calib_viz.load_calibration_data(args.results_dir)
            if args.calib == "latest":
                target_calib = calib_viz.latest_calibration(calib_df)
                calib_scope_msg = "(조회 중인 세션과 무관, 최신 캘리브레이션 기준)"
            else:
                target_calib = calib_viz.get_calibration_by_prefix(calib_df, args.calib)
                calib_scope_msg = "(지정한 calib_id 기준)"
        except FileNotFoundError:
            print(f"[경고] calibration_quality CSV를 찾을 수 없습니다: "
                  f"{os.path.join(args.results_dir, calib_viz.calibration_quality_filename())}")
        except ValueError as e:
            print(f"[경고] {e}")
        else:
            fig_calib = calib_viz.plot_calibration_overview(target_calib)
            calib_id = target_calib["calib_id"].iloc[0]
            short_calib = str(calib_id)[:8]
            calib_path = os.path.join(args.out, f"{short_calib}_calib_overview.png")
            fig_calib.savefig(calib_path, dpi=130, bbox_inches="tight")
            plt.close(fig_calib)
            print("-" * 60)
            print(f"[{short_calib}] 캘리브레이션 품질 그래프 저장: {calib_path}")
            print(calib_scope_msg)
            calib_saved = True

    print("=" * 60)
    n_png = len(target_ids) + (1 if len(target_ids) >= 2 else 0) + (1 if calib_saved else 0)
    print(f"완료. PNG {n_png}개 저장됨 -> {args.out}/")


if __name__ == "__main__":
    main()
