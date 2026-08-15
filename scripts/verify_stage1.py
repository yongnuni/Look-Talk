"""1단계(식별자 통합 및 공유 모듈 배선) 이후 생성된 CSV들의 정합성을 검사한다.

사용법 (프로젝트 루트에서):
    python scripts/verify_stage1.py
    python scripts/verify_stage1.py --results-dir gaze_accuracy_results --calib-dir calibration_results

먼저 main.py를 캘리브레이션 → 일반 키보드 사용(몇 글자 입력) → t(9점 테스트)
→ y(타겟팅 테스트, 완료 또는 ESC) → 종료 순서로 최소 1회 실행해 CSV를 만든
뒤 이 스크립트를 돌린다. 웹캠 실행 자체는 이 스크립트가 하지 않는다 — 이미
디스크에 쌓인 CSV만 읽어서 검사한다.

검사 항목
0. 필수 파일(sessions, gaze_accuracy)이 존재하는가 - 없으면 스킵이 아니라 실패
1. 대상 run_id 하나를 골라(기본: sessions.t0_utc가 가장 늦은 실행, --run-id로
   직접 지정 가능) 각 파일에 그 run_id의 행이 있는지 확인
2. sessions.calib_id <-> calibration_quality.calib_id 조인이 성립하는가
3. sessions.config_hash가 비어 있지 않고 12자 hex인가
4. mapper_session_log의 ts_ms가 (run_id 그룹 내에서) 단조 증가하는가
5. 고아 run_id 확인 - mapper_session_log 외 파일에 mapper_session_log에는 없는
   run_id가 있으면 배선 누락 신호로 보고

sessions/gaze_accuracy는 MetricsCollector가 매 실행마다 반드시 만들어야 하는
필수 산출물이라 파일 자체가 없으면 실패로 처리한다. CSV는 append 누적이라
여러 번의 실행 결과가 한 파일에 섞여 있고, 실행마다 어디까지 진행했는지
(캘리만 함/9점까지/타겟팅까지)가 다르므로 "모든 파일에 공통 존재하는 run_id"를
찾는 방식은 정상 상황도 실패로 오판한다. 그래서 검사 1은 "run_id 1개를 골라 그 실행에서 무엇을 했는지"를 확인하는
방식으로 바꿨다. 대상 run_id는 sessions.csv에서 뽑으므로(t0_utc 최신값) sessions
존재는 선택 방법 자체가 보장한다. mapper_session_log는 main.py 메인 루프에
진입하면 매 프레임 무조건 기록되므로(collector 존재 여부와 무관), 대상 run_id가
거기 없다는 것은 로깅이 실제로 깨졌다는 뜻이라 필수로 둔다. 나머지
(calibration_quality/gaze_accuracy/targeting_results/mouth_baseline_history)는
없으면 "미수행"으로만 표시하고 통과 처리한다.
"""

import argparse
import os
import re
import sys

import pandas as pd

from src.metrics.collector import MetricsCollector
from src.tracking.calibration import Calibrator
from src.metrics.session_logger import SessionLogger

CONFIG_HASH_RE = re.compile(r"^[0-9a-f]{12}$")


def _load_csv(path, label, required=False):
    if not os.path.isfile(path):
        tag = "실패" if required else "스킵"
        print(f"[{tag}] {label} 파일 없음: {path}")
        return None
    df = pd.read_csv(path, encoding="utf-8-sig")
    print(f"[로드] {label}: {path} ({len(df)}행)")
    return df


def check_required_files(frames, required_labels):
    """필수 파일이 실제로 로드됐는지 확인한다. 누락은 스킵이 아니라 실패다."""
    print("\n=== 0. 필수 파일 존재 확인 ===")

    ok = True
    for label in required_labels:
        if frames.get(label) is None:
            print(f"  [실패] 필수 파일 누락: {label}")
            ok = False
        else:
            print(f"  [통과] {label} 존재 ({len(frames[label])}행)")

    return ok


REQUIRED_FOR_TARGET_RUN = ["sessions", "mapper_session_log"]


def select_target_run_id(sessions_df, explicit_run_id):
    """검사 대상 run_id 1개를 고른다.

    --run-id가 주어지면 그대로 쓰고, 아니면 sessions.csv에서 t0_utc가 가장 늦은
    행의 run_id를 쓴다(= sessions에 존재함이 선택 방법 자체로 보장됨).
    반환값: (run_id 또는 None, 선택 사유 설명)
    """
    if explicit_run_id:
        return explicit_run_id, "명시 지정(--run-id)"

    if sessions_df is None:
        return None, "sessions.csv 없음"

    if "run_id" not in sessions_df.columns or "t0_utc" not in sessions_df.columns:
        return None, "sessions.csv에 run_id 또는 t0_utc 컬럼 없음"

    candidates = sessions_df.dropna(subset=["run_id", "t0_utc"]).copy()

    if candidates.empty:
        return None, "sessions.csv에 run_id/t0_utc 값이 있는 행이 없음"

    candidates["_t0_parsed"] = pd.to_datetime(
        candidates["t0_utc"], utc=True, errors="coerce"
    )
    candidates = candidates.dropna(subset=["_t0_parsed"])

    if candidates.empty:
        return None, "sessions.csv의 t0_utc를 시각으로 파싱할 수 없음"

    latest = candidates.loc[candidates["_t0_parsed"].idxmax()]
    return latest["run_id"], "sessions.t0_utc 최신값"


def check_target_run(frames, target_run_id, selection_note):
    """대상 run_id가 각 파일에 존재하는지 확인한다.

    sessions/mapper_session_log는 필수 - 없으면 실패. 나머지는 실행 방식에 따라
    자연스럽게 빠질 수 있으므로 없으면 "미수행"으로만 표시하고 통과 처리한다.
    """
    print("\n=== 1. 대상 run_id 커버리지 ===")

    if target_run_id is None:
        print(f"  [실패] 대상 run_id를 정할 수 없음: {selection_note}")
        return False

    print(f"  대상 run_id: {target_run_id} (선정 기준: {selection_note})")

    ok = True
    for label, df in frames.items():
        if df is None:
            present = False
        elif "run_id" not in df.columns:
            present = False
        else:
            present = target_run_id in set(df["run_id"].dropna().unique())

        required = label in REQUIRED_FOR_TARGET_RUN

        if present:
            print(f"  [통과] {label}: 이 실행의 행 존재")
        elif required:
            print(f"  [실패] {label}: 이 실행의 행 없음 (필수)")
            ok = False
        else:
            print(f"  [정보] {label}: 미수행 (이 실행에서 해당 단계를 진행하지 않음)")

    return ok


def check_orphan_run_ids(frames):
    """mapper_session_log에 없는데 다른 파일에만 존재하는 run_id를 찾는다.

    mapper_session_log는 main.py 메인 루프 진입 시 매 프레임 무조건 기록되므로,
    실제로 실행된 run_id라면 반드시 여기 남는다. 다른 파일에 있는 run_id가
    mapper_session_log에 없다면 run_id가 잘못 배선됐거나 다른 실행의 값이
    잘못 붙었다는 신호다. mapper_session_log 쪽만 있고 다른 파일에는 없는
    run_id(캘리만 하고 바로 끈 실행 등)는 정상이므로 실패로 치지 않는다.
    """
    print("\n=== 5. 고아 run_id 확인 ===")

    log_df = frames.get("mapper_session_log")

    if log_df is None or "run_id" not in log_df.columns:
        print("  [스킵] mapper_session_log가 없어 기준 집합을 만들 수 없음")
        return None

    log_ids = set(log_df["run_id"].dropna().unique())

    checked_any = False
    any_orphan = False

    for label, df in frames.items():
        if label == "mapper_session_log" or df is None or "run_id" not in df.columns:
            continue

        checked_any = True
        ids = set(df["run_id"].dropna().unique())
        orphans = ids - log_ids

        if orphans:
            any_orphan = True
            preview = sorted(orphans)[:3]
            print(f"  [실패] {label}: mapper_session_log에 없는 run_id {len(orphans)}개: "
                  f"{preview}{' ...' if len(orphans) > 3 else ''}")
        else:
            print(f"  [통과] {label}: 전부 mapper_session_log에 존재")

    if not checked_any:
        print("  [스킵] run_id 컬럼을 가진 다른 파일이 없음")
        return None

    return not any_orphan


def summarize_all_runs(frames):
    """run_id별로 각 파일에 몇 행씩 있는지 실행 단위 표로 출력한다 (--all 전용, 통과/실패 없음)."""
    print("\n" + "=" * 60)
    print("실행 단위 요약 (--all)")
    print("=" * 60)

    labels = list(frames.keys())

    all_ids = set()
    for df in frames.values():
        if df is not None and "run_id" in df.columns:
            all_ids |= set(df["run_id"].dropna().unique())

    if not all_ids:
        print("  run_id를 가진 행이 어떤 파일에도 없음")
        return

    t0_by_run = {}
    sessions_df = frames.get("sessions")
    if sessions_df is not None and "run_id" in sessions_df.columns and "t0_utc" in sessions_df.columns:
        tmp = sessions_df.dropna(subset=["run_id"]).copy()
        tmp["_t0_parsed"] = pd.to_datetime(tmp["t0_utc"], utc=True, errors="coerce")
        for _, row in tmp.iterrows():
            t0_by_run[row["run_id"]] = row["_t0_parsed"]

    def sort_key(run_id):
        t0 = t0_by_run.get(run_id)
        if t0 is None or pd.isna(t0):
            return pd.Timestamp.min.tz_localize("UTC")
        return t0

    ordered_ids = sorted(all_ids, key=sort_key)

    run_id_col_width = 41  # UUID 36자 + 앞 2칸 여백 + 뒤 최소 3칸 구분 여백
    count_col_width = 16

    header = "  run_id".ljust(run_id_col_width) + "".join(
        label[:14].ljust(count_col_width) for label in labels
    )
    print(header)

    for run_id in ordered_ids:
        row_str = f"  {str(run_id)}".ljust(run_id_col_width)
        for label in labels:
            df = frames[label]
            if df is not None and "run_id" in df.columns:
                count = int((df["run_id"] == run_id).sum())
            else:
                count = 0
            row_str += str(count).ljust(count_col_width)
        print(row_str)


def check_calib_join(sessions_df, calib_quality_df):
    print("\n=== 2. sessions.calib_id ↔ calibration_quality.calib_id 조인 ===")

    if sessions_df is None or calib_quality_df is None:
        print("  [스킵] 두 파일 중 하나가 없음")
        return None

    if "calib_id" not in sessions_df.columns or "calib_id" not in calib_quality_df.columns:
        print("  [실패] calib_id 컬럼이 한쪽에 없음")
        return False

    session_calib_ids = set(sessions_df["calib_id"].dropna().unique())
    quality_calib_ids = set(calib_quality_df["calib_id"].dropna().unique())

    if not session_calib_ids:
        print("  [스킵] sessions.csv에 calib_id 값이 없음")
        return None

    missing = session_calib_ids - quality_calib_ids

    if missing:
        print(f"  [실패] calibration_quality에 없는 calib_id {len(missing)}개: "
              f"{sorted(missing)[:3]}")
        return False

    print(f"  [통과] sessions의 calib_id {len(session_calib_ids)}개 전부 "
          "calibration_quality에 존재")
    return True


def check_config_hash(sessions_df):
    print("\n=== 3. sessions.config_hash 형식 ===")

    if sessions_df is None or "config_hash" not in sessions_df.columns:
        print("  [스킵] sessions.csv 또는 config_hash 컬럼 없음")
        return None

    values = sessions_df["config_hash"].dropna()

    if len(values) == 0:
        print("  [실패] config_hash 값이 전부 비어 있음")
        return False

    bad = [v for v in values if not CONFIG_HASH_RE.match(str(v))]

    if bad:
        print(f"  [실패] 12자리 hex 형식이 아닌 값 {len(bad)}개: {bad[:3]}")
        return False

    print(f"  [통과] config_hash {len(values)}개 전부 12자리 hex 형식")
    return True


def check_ts_ms_monotonic(session_log_df):
    print("\n=== 4. mapper_session_log.ts_ms 단조 증가 (run_id별) ===")

    if session_log_df is None:
        print("  [스킵] mapper_session_log 파일 없음")
        return None

    if "ts_ms" not in session_log_df.columns:
        print("  [실패] ts_ms 컬럼이 없음 (구버전 timestamp 컬럼일 수 있음)")
        return False

    if "run_id" not in session_log_df.columns:
        print("  [실패] run_id 컬럼이 없어 실행 단위로 나눌 수 없음")
        return False

    all_ok = True
    # 파일은 여러 번의 실행에 걸쳐 누적된다. 전체 파일 기준으로 단조 증가를
    # 검사하면 새 실행이 시작될 때마다 ts_ms가 0 근처로 리셋되어 항상
    # 실패하므로, run_id별로 원래 행 순서를 보존한 채 나눠서 그룹 내부에서만 검사한다.
    for run_id, group in session_log_df.groupby("run_id", sort=False):
        ts = group["ts_ms"].to_numpy()
        diffs = ts[1:] - ts[:-1]
        if len(diffs) > 0 and (diffs < 0).any():
            bad_at = int((diffs < 0).argmax())
            print(f"  [실패] run_id={run_id}: {bad_at}번째 지점에서 ts_ms 역행 "
                  f"({ts[bad_at]:.1f} -> {ts[bad_at + 1]:.1f})")
            all_ok = False
        else:
            print(f"  [통과] run_id={run_id}: {len(ts)}행, 단조 증가")

    return all_ok


def main():
    parser = argparse.ArgumentParser(description="1단계 CSV 정합성 검사")
    parser.add_argument("--results-dir", default="gaze_accuracy_results")
    parser.add_argument("--calib-dir", default="calibration_results")
    parser.add_argument(
        "--run-id", default=None,
        help="검사 대상 run_id를 직접 지정 (기본: sessions.t0_utc 최신 실행)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="통과/실패 판정 대신 run_id별 실행 단위 요약 표만 출력"
    )
    args = parser.parse_args()

    sessions_path = os.path.join(
        args.results_dir, f"sessions_v{MetricsCollector.SCHEMA_VERSION}.csv"
    )
    accuracy_path = os.path.join(
        args.results_dir, f"gaze_accuracy_v{MetricsCollector.SCHEMA_VERSION}.csv"
    )
    calib_quality_path = os.path.join(
        args.results_dir, f"calibration_quality_v{Calibrator.CALIB_SCHEMA_VERSION}.csv"
    )
    session_log_path = os.path.join(
        args.results_dir, f"mapper_session_log_v{SessionLogger.SCHEMA_VERSION}.csv"
    )
    targeting_path = os.path.join(args.results_dir, "targeting_results_v1.0.csv")
    mouth_history_path = os.path.join(args.calib_dir, "mouth_baseline_history_v1.0.csv")

    print("=" * 60)
    print("1단계 검증: 생성된 CSV 로딩")
    print("=" * 60)

    REQUIRED_FILES = ["sessions", "gaze_accuracy"]

    frames = {
        "sessions": _load_csv(sessions_path, "sessions", required=True),
        "gaze_accuracy": _load_csv(accuracy_path, "gaze_accuracy", required=True),
        "calibration_quality": _load_csv(calib_quality_path, "calibration_quality"),
        "mapper_session_log": _load_csv(session_log_path, "mapper_session_log"),
        "targeting_results": _load_csv(targeting_path, "targeting_results"),
        "mouth_baseline_history": _load_csv(mouth_history_path, "mouth_baseline_history"),
    }

    if args.all:
        summarize_all_runs(frames)
        return

    target_run_id, selection_note = select_target_run_id(frames["sessions"], args.run_id)

    results = {
        "required_files_present": check_required_files(frames, REQUIRED_FILES),
        "target_run_coverage": check_target_run(frames, target_run_id, selection_note),
        "calib_join": check_calib_join(frames["sessions"], frames["calibration_quality"]),
        "config_hash_format": check_config_hash(frames["sessions"]),
        "ts_ms_monotonic": check_ts_ms_monotonic(frames["mapper_session_log"]),
        "no_orphan_run_ids": check_orphan_run_ids(frames),
    }

    print("\n" + "=" * 60)
    print("요약")
    print("=" * 60)
    for name, result in results.items():
        status = "통과" if result is True else ("스킵" if result is None else "실패")
        print(f"  {name}: {status}")

    if any(r is False for r in results.values()):
        print("\n[결론] 일부 검사 실패 - 위 로그에서 원인을 확인하세요.")
        sys.exit(1)

    print("\n[결론] 실패한 검사 없음 (스킵 항목은 해당 CSV가 아직 없다는 뜻일 수 있습니다).")


if __name__ == "__main__":
    main()
