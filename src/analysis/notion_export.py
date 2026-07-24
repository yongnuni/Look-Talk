# -*- coding: utf-8 -*-
"""진단 리포트를 Notion 데이터베이스에 자동 업로드하는 모듈

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
실행 매뉴얼
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 실제 실행은 src/analysis/run_export.py에서 한다.
   이 파일(및 diagnostic.py)은 라이브러리 모듈이라 __main__에 실행 코드를 두지 않음
   run_export.py 안의 "Notion 업로드" 부분 주석을 풀고 DATABASE_ID를 채운 뒤
       python -m src.analysis.run_export
   로 실행한다.

2. label 짓는 규칙: JSON/HTML 파일명과 동일한 label을 쓰는 습관을 들일 것
   나중에 Notion 페이지와 로컬 산출물(diagnostics/ 폴더의
   {label}_diagnostic_report.json, {label}_report.html)을 한눈에 대응시켜 찾기 위함.

3. ⚠ 재실행 시 "덮어쓰기"가 아니라 "새 행 추가"
   save_diagnostic_report()/build_diagnostic_html()은 label 기준 파일을 덮어쓰지만,
   이 함수는 Notion pages.create를 호출하므로 같은 label로 두 번 실행하면 데이터베이스에 같은 이름의 행이 두 개 남음
   계산 로직을 고친 뒤 재업로드하는 경우, 이전 행을 Notion에서 직접 지우거나 보관(archive) 처리한 뒤 재실행할 것.

4. 자주 만나는 에러:
   - 400 Bad Request: properties의 키 이름·타입이 Notion 속성과 불일치. 에러 메시지에 어떤 속성이 문제인지 나오니 확인.
   - 404 Not Found: database_id가 틀렸거나, 데이터베이스가 통합과 공유되지 않음
   - 401 Unauthorized: NOTION_TOKEN 환경변수 미설정 또는 만료.
   - 429 Too Many Requests: 이미지가 많아 짧은 시간에 API 호출이 몰린 경우. 재시도 로직은 아직 없음. 발생하면 잠시 후 다시 실행할 것.

5. include_images=False로 호출하면 이미지 없이 텍스트·표만 업로드됨 (기본값은 True)
"""

import os

import pandas as pd
import requests

from src.viz import viz
from src.viz import calib_viz

NOTION_API_BASE = "https://api.notion.com/v1"
NOTION_VERSION = "2026-03-11"


def _notion_headers():
    token = os.environ.get("NOTION_TOKEN")
    if not token:
        raise RuntimeError(
            "환경변수 NOTION_TOKEN이 설정되어 있지 않습니다. "
            "Notion 통합 토큰을 발급받아 환경변수로 등록하세요."
        )
    return {
        "Authorization": f"Bearer {token}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }


def _rt(text, color=None, bold=False):
    """Notion rich_text 객체 하나 생성.

    bold=True는 블록 실험의 frozen(고착) 세션처럼, 기존 3단계 색상
    팔레트(_color_bucket)로는 구분되지 않는 별도 상태를 표시하기 위해
    추가했다. 기존 호출부는 전부 bold를 넘기지 않으므로(기본 False)
    annotations 내용이 이전과 동일하게 유지된다.
    """
    obj = {"type": "text", "text": {"content": "" if text is None else str(text)}}
    annotations = {}
    if color:
        annotations["color"] = color
    if bold:
        annotations["bold"] = True
    if annotations:
        obj["annotations"] = annotations
    return obj


def _safe_number(value, ndigits=None):
    """NaN/None을 Notion number 속성이 받아들이는 형태(None -> null)로 변환.

    Correlation r은 표본이 1개뿐이거나(표준편차 계산 불가), group_by=
    "session_id"로 같은 캘리브레이션을 재사용한 세션끼리 비교할 때
    (rmse가 모든 행에서 동일 -> 분산 0) 계산 자체가 불가능해 NaN이
    된다. Mean Error Avg도 모든 타깃이 추적 실패한 세션만 있으면
    같은 이유로 NaN이 될 수 있다. requests 라이브러리는 NaN을 JSON
    으로 보내는 것 자체를 거부하므로(allow_nan=False), 값이 없다는
    의미의 None(Notion number 속성의 null)으로 명시 변환해 페이지
    생성 요청 자체가 실패하는 것을 막는다.
    """
    if value is None or (isinstance(value, float) and value != value):  # NaN != NaN
        return None
    return round(value, ndigits) if ndigits is not None else value


def _color_bucket(series, higher_is_worse=True, use_abs=False):
    """시리즈 값을 3단계(낮음/중간/높음)로 나눠 Notion 배경색 이름을 매핑.

    값의 실제 크기(최솟값~최댓값 범위를 균등 3분할, pd.cut)를 기준으로
    나눈다 — "개수가 같도록" 나누는 pd.qcut은 쓰지 않는다.

    이전 버전은 rank(method="first") + qcut을 썼는데, 두 가지 실제
    버그가 있었다:
    1. rank(method="first")는 동점(tie)을 원본 행 순서로 강제로
       갈라놓는다. 그 결과 완전히 같은 값(예: 0, 0, 0)이 서로 다른
       색으로 칠해지는 문제가 있었다.
    2. rank는 순위만 볼 뿐 값 사이의 실제 격차를 무시한다. 예를 들어
       [9, 1, 0, 0, 0]에서 1은 0보다 살짝 클 뿐인데도 "두 번째로 높은
       순위"라는 이유만으로 9와 나란히 나쁜 색이 칠해졌다.
    pd.cut(값의 실제 크기 기준)으로 바꾸면 두 문제 모두 해결된다 —
    동일 값은 항상 같은 구간에 들어가고, 이상치(9) 하나만 도드라지고
    나머지(0, 1처럼 서로 가까운 값)는 자연스럽게 같은 색으로 묶인다.

    use_abs=True면 부호 있는 편향값(dy_top/dy_mid/dy_bottom 등)에 쓴다.
    이런 값은 "크다=나쁘다"가 아니라 "0에서 멀수록(부호 무관) 나쁘다"가
    맞는 해석이라, 절댓값 기준으로 3분할한다. 이 경우 higher_is_worse는
    의미가 없으므로(절댓값은 항상 클수록 나쁨) 내부적으로 True로
    고정한다.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    if use_abs:
        numeric = numeric.abs()
        higher_is_worse = True

    color_map = {"low": "green_background", "mid": "yellow_background", "high": "red_background"}
    if not higher_is_worse:
        color_map = {"low": "red_background", "mid": "yellow_background", "high": "green_background"}

    if numeric.nunique(dropna=True) < 2:
        # 값이 전부 같으면 3단계로 나누는 것 자체가 의미가 없다 -> 중립(mid).
        return pd.Series(["mid"] * len(numeric), index=numeric.index).map(color_map)

    try:
        bins = pd.cut(numeric, 3, labels=["low", "mid", "high"], include_lowest=True)
    except ValueError:
        bins = pd.Series(["mid"] * len(numeric), index=numeric.index)
    # bins가 Categorical이라 값이 한쪽에 쏠리면 실제로는 low/high 두
    # 구간만 쓰이는 경우도 있다(예: 값이 딱 2종류일 때) — .map()은
    # 그 경우에도 문제없이 동작한다.
    mapped = bins.astype(object).map(color_map)
    values = [v if v == v else None for v in mapped.tolist()]
    return pd.Series(values, index=mapped.index, dtype=object)


def color_bucket_report(series, higher_is_worse=True, use_abs=False):
    """_color_bucket()과 동일한 로직으로 (원본값, 비교기준값, 예상색상)을
    정렬된 표로 반환한다 — Notion 업로드 결과를 눈으로 검증할 때, 셀
    색을 하나하나 암산으로 추측하지 않고 이 표와 나란히 대조하기 위함.

    사용 예 (업로드 전/후 언제든 확인 가능):
        from src.analysis.notion_export import color_bucket_report
        print(color_bucket_report(summary_df["stuck_count"]))
        print(color_bucket_report(row_df["dy_top"], use_abs=True))

    검증 시 확인할 것 3가지:
    1. 단조성 — compare_basis를 오름차순 정렬했을 때 expected_color가
       한 방향으로만 바뀌는지(중간에 역전되면 버그).
    2. 동점 일관성 — compare_basis 값이 같은 행은 항상 같은 색인지.
    3. 극값 검증 — 가장 나쁜/좋은 값이 가장 나쁜/좋은 색인지.
    """
    colors = _color_bucket(series, higher_is_worse=higher_is_worse, use_abs=use_abs)
    compare_basis = series.abs() if use_abs else series
    return pd.DataFrame({
        "value": series,
        "compare_basis": compare_basis,
        "expected_color": colors,
    }).sort_values("compare_basis")


def _df_to_table_block(df, color_cols=None, bold_cols=None):
    """DataFrame을 Notion table 블록(children에 table_row 포함)으로 변환.

    color_cols : dict[str, pd.Series]
        컬럼명 -> (df와 같은 인덱스의) Notion 배경색 이름 Series.
        지정된 컬럼의 셀에만 배경색이 적용된다.
    bold_cols : dict[str, pd.Series[bool]]
        컬럼명 -> (df와 같은 인덱스의) bool Series. True인 행만 굵게 표시한다.
        frozen(고착) 세션처럼 3단계 색상(_color_bucket)으로는 구분되지 않는
        "완전히 다른 종류의 이상"을 표시하는 4번째 시각 상태로 쓴다.
        기본 None이면 기존 호출부와 동일하게 동작한다.
    """
    color_cols = color_cols or {}
    bold_cols = bold_cols or {}
    header_row = {
        "object": "block", "type": "table_row",
        "table_row": {"cells": [[_rt(c)] for c in df.columns]},
    }
    rows = [header_row]
    for idx, row in df.iterrows():
        cells = []
        for col in df.columns:
            color = color_cols[col].get(idx) if col in color_cols else None
            bold = bool(bold_cols[col].get(idx, False)) if col in bold_cols else False
            cells.append([_rt(row[col], color=color, bold=bold)])
        rows.append({"object": "block", "type": "table_row", "table_row": {"cells": cells}})

    return {
        "object": "block",
        "type": "table",
        "table": {
            "table_width": len(df.columns),
            "has_column_header": True,
            "has_row_header": False,
            "children": rows,
        },
    }


def _heading(text, level=2):
    key = f"heading_{level}"
    return {"object": "block", "type": key, key: {"rich_text": [_rt(text)]}}


def _toggle(summary_text, children_blocks):
    return {
        "object": "block", "type": "toggle",
        "toggle": {"rich_text": [_rt(summary_text)], "children": children_blocks},
    }


def _paragraph(text):
    return {"object": "block", "type": "paragraph", "paragraph": {"rich_text": [_rt(text)]}}


# ══════════════════════════════════════════════════════════════
# 이미지 업로드 (Notion 공식 Direct Upload, single-part 흐름)
# ══════════════════════════════════════════════════════════════

def _fig_to_png_bytes(fig):
    """matplotlib fig를 PNG 바이너리로 변환 (base64 아님 — 실제 업로드용).

    build_diagnostic_html의 _fig_to_base64()와 저장 설정(dpi, bbox_inches)은
    동일하게 맞춰서, HTML 리포트와 Notion에 올라가는 그림 화질이
    달라지지 않도록 했다.
    """
    import io
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _create_file_upload(filename, content_type="image/png"):
    """1/2단계: Notion File Upload 객체 생성. file_upload_id를 반환.

    문서상 mode 기본값(single_part)이 20MB 이하 파일에 맞으므로 명시하지
    않는다 (우리 그림은 훨씬 작음).
    """
    payload = {"filename": filename, "content_type": content_type}
    resp = requests.post(f"{NOTION_API_BASE}/file_uploads", headers=_notion_headers(), json=payload)
    resp.raise_for_status()
    return resp.json()["id"]


def _send_file_upload(file_upload_id, filename, content_bytes, content_type="image/png"):
    """2/2단계: 생성된 업로드 객체에 실제 바이너리 전송.

    이 엔드포인트만 multipart/form-data를 쓴다 (다른 Notion API는 전부
    JSON). 그래서 _notion_headers()를 그대로 쓰지 않고 Content-Type을
    빼서 별도로 헤더를 구성한다 — requests가 files= 인자를 받으면
    boundary가 포함된 Content-Type을 알아서 채워주므로, 여기서 직접
    'application/json'을 넣으면 오히려 요청이 깨진다.
    """
    token = os.environ.get("NOTION_TOKEN")
    if not token:
        raise RuntimeError(
            "환경변수 NOTION_TOKEN이 설정되어 있지 않습니다. "
            "Notion 통합 토큰을 발급받아 환경변수로 등록하세요."
        )
    headers = {"Authorization": f"Bearer {token}", "Notion-Version": NOTION_VERSION}
    files = {"file": (filename, content_bytes, content_type)}
    resp = requests.post(
        f"{NOTION_API_BASE}/file_uploads/{file_upload_id}/send",
        headers=headers, files=files,
    )
    resp.raise_for_status()
    return resp.json()


def _upload_figure(fig, filename):
    """matplotlib fig 하나를 Notion에 업로드하고 file_upload_id를 반환.

    업로드만 하고 아무 블록에도 붙이지 않은 채 1시간이 지나면 자동
    archived 처리되므로(Notion 정책), 호출한 쪽에서 반환된 id를 곧바로
    _image_block()에 넣어 children에 포함시켜야 한다.
    """
    content = _fig_to_png_bytes(fig)
    file_upload_id = _create_file_upload(filename, content_type="image/png")
    _send_file_upload(file_upload_id, filename, content, content_type="image/png")
    return file_upload_id


def _image_block(file_upload_id, caption=None):
    return {
        "object": "block", "type": "image",
        "image": {
            "type": "file_upload",
            "file_upload": {"id": file_upload_id},
            "caption": [_rt(caption)] if caption else [],
        },
    }


def _group_calibration_sessions(df, figures):
    """collect_figures()가 반환한 (category, label, fig) 리스트를
    "캘리브레이션 - 그 아래 관련 세션들" 순서로 재구성한다 (배치 방식 2).

    calibration_quality.csv와 새로 조인하지 않는다 — 이미 df(sessions.csv
    필터링 결과)에 있는 calib_id 컬럼만으로 그룹을 만들므로, 오늘
    계속 경계해온 "위험한 조인"과는 무관하다. group_by="session_id"로
    같은 캘리브레이션을 재사용한 여러 세션(예: 3D 모드 on/off)을 보는
    경우, 그 세션들이 자연스럽게 한 그룹으로 묶인다.

    Returns
    -------
    trend_figs : [(category, "추세", fig), ...] (0~2개)
    groups : [(calib_id_short, calib_fig_or_None, [(session_id_short, fig), ...]), ...]
        캘리브레이션 시간순. calib_fig가 None이면 해당 calib_id의
        캘리브레이션 그림이 없다는 뜻(calibration_quality.csv에 데이터
        없음 등 엣지 케이스) — 세션은 그래도 정상적으로 포함시킨다.
    """
    trend_figs = [(c, l, f) for c, l, f in figures if l == "추세"]
    calib_order_figs = [(l, f) for c, l, f in figures if c == "calibration" and l != "추세"]
    session_figs = [(l, f) for c, l, f in figures if c == "session" and l != "추세"]

    sid_short_to_calib_short = {
        str(row.session_id)[:8]: str(row.calib_id)[:8] for row in df.itertuples()
    }

    calib_short_to_sessions = {}
    for sid_short, fig in session_figs:
        cid_short = sid_short_to_calib_short.get(sid_short)
        calib_short_to_sessions.setdefault(cid_short, []).append((sid_short, fig))

    groups = []
    seen = set()
    for cid_short, calib_fig in calib_order_figs:
        groups.append((cid_short, calib_fig, calib_short_to_sessions.get(cid_short, [])))
        seen.add(cid_short)

    # 캘리브레이션 그림은 없지만 세션은 있는 경우(드문 엣지 케이스)도
    # 뒤에 추가해 누락시키지 않는다.
    for cid_short, sess_list in calib_short_to_sessions.items():
        if cid_short not in seen:
            groups.append((cid_short, None, sess_list))

    return trend_figs, groups


def _build_block_experiment_section(block_report):
    """diagnostic.build_block_experiment_report()의 반환 dict를 Notion 블록
    리스트로 변환한다. upload_diagnostic_to_notion()이 children 맨 앞에
    이어붙인다(block_report가 주어졌을 때만).

    frozen=True 행은 3단계 색상(_color_bucket)과 구분되는 4번째 시각 상태로
    빨강 배경 + 굵게를 함께 적용한다.
    """
    from src.analysis.diagnostic import shorten_session_id  # 지연 import, diagnostic.py 기존 관례

    var = block_report["variance_decomposition"]
    blocks = [_heading("블록 실험 요약", level=2)]

    ratio_display = f"{var['ratio']:.2f}" if var["ratio"] is not None else "계산 불가"
    grand_mean_display = f"{var['grand_mean']:.1f}" if var["grand_mean"] is not None else "N/A"
    icc1_display = f"ICC(1)={var['icc1']:.2f}" if var.get("icc1") is not None else None
    summary_text = (
        f"블록간/전체 분산 설명 비율(η²): {ratio_display} "
        f"(그랜드평균 {grand_mean_display}px, 사용 블록 {var['n_blocks_used']}개, "
        f"세션 {var['n_sessions_used']}개)"
    )
    if icc1_display:
        summary_text += f"\n{icc1_display} (불균형 보정 급내상관 추정치, 참고용)"
    blocks.append(_paragraph(summary_text))

    per_block_df = pd.DataFrame(var["per_block"])
    if not per_block_df.empty:
        per_block_df = shorten_session_id(per_block_df, col="block_id")
    blocks.append(_toggle("블록별 평균±SD", [_df_to_table_block(per_block_df)]))

    if var["blocks_excluded"]:
        blocks.append(_paragraph(
            f"제외된 블록: {[str(b)[:8] for b in var['blocks_excluded']]} "
            f"(제외 세션: {[str(s)[:8] for s in var['sessions_excluded']]})"
        ))

    stats_df = pd.DataFrame(block_report["block_session_stats"])[
        ["block_id", "test_idx", "session_id", "mean_error_px", "gain_x", "gain_y", "clip_n", "frozen"]
    ].reset_index(drop=True)
    if not stats_df.empty:
        stats_df = shorten_session_id(stats_df, col="block_id")
        stats_df = shorten_session_id(stats_df, col="session_id")

        frozen_series = stats_df["frozen"]
        colors = {
            "mean_error_px": _color_bucket(stats_df["mean_error_px"], higher_is_worse=True),
            "clip_n": _color_bucket(stats_df["clip_n"], higher_is_worse=True),
            # gain은 1.0에서 멀수록 나쁘다 — dx/dy 편향값과 동일한 use_abs 관례 재사용
            "gain_x": _color_bucket(stats_df["gain_x"] - 1.0, use_abs=True),
            "gain_y": _color_bucket(stats_df["gain_y"] - 1.0, use_abs=True),
            # frozen_series.map({True: ..., False: None})은 pandas 3.x에서 문자열+None
            # 혼합 결과의 dtype을 자동으로 string형으로 추론해 None을 다시 np.nan으로
            # 바꿔버린다(_color_bucket()이 244-245번째 줄에서 dtype=object를 명시하는
            # 이유와 동일). 그 결과 _rt()의 `if color:` 검사를 통과한 채(bool(nan)==True)
            # raw float NaN이 그대로 JSON payload에 박혀 requests.post 직렬화가 실패했다.
            # dtype=object를 명시해 None이 NaN으로 바뀌는 것을 막는다.
            "frozen": pd.Series(
                ["red_background" if v else None for v in frozen_series.tolist()],
                index=frozen_series.index, dtype=object,
            ),
        }
        bold_cols = {c: frozen_series for c in stats_df.columns}
        blocks.append(_df_to_table_block(stats_df, color_cols=colors, bold_cols=bold_cols))
    else:
        blocks.append(_paragraph("블록×회차 통계 없음"))

    trend_df = pd.DataFrame(block_report["trend"])
    if not trend_df.empty:
        trend_df = shorten_session_id(trend_df, col="block_id")
    blocks.append(_toggle(
        "블록별 추세 지표 (참고용, 유의성 검정 아님)",
        [_df_to_table_block(trend_df)] if not trend_df.empty else [_paragraph("없음")],
    ))

    return blocks


# ══════════════════════════════════════════════════════════════
# 진단 리포트 업로드
# ══════════════════════════════════════════════════════════════

def upload_diagnostic_to_notion(database_id, sessions, acc, calib_df, session_ids, label,
                                 group_by="calib_id", include_images=True, block_report=None):
    """diagnostic_battery() 결과를 Notion 데이터베이스에 새 항목(페이지)으로 추가.

    diagnostic.py의 diagnostic_battery()/build_session_summary_table()/
    collect_figures()를 그대로 재사용한다 — HTML 리포트와 다른 계산식·
    스코프 결정 로직을 쓰면 두 결과가 서로 다른 값을 보여줄 위험이 있어,
    반드시 같은 함수를 공유해야 한다.

    Parameters
    ----------
    calib_df : pd.DataFrame
        calibration_quality_v{CALIB_SCHEMA_VERSION}.csv 로드 결과.
        include_images=True일 때 캘리브레이션 오차 지도를 그리는 데 쓰인다.
    include_images : bool
        False면 텍스트·표·토글·색상만 업로드한다 (예전 1단계와 동일한
        결과). 그림 1장당 API 호출이 2번 추가되므로, 빠르게 값만
        확인하고 싶을 때는 False로 두면 더 빠르다.
    block_report : dict, optional
        diagnostic.build_block_experiment_report()의 반환값. 기본 None —
        이 경우 기존 동작과 완전히 동일하다(분기 자체를 타지 않음). 값이
        있으면 "블록 실험 요약" 섹션을 children 맨 앞에 추가한다.
    """
    # 순환 참조를 피하기 위해 함수 안에서 import (diagnostic.py가
    # notion_export.py를 몰라도 되도록, 의존 방향을 한쪽으로만 유지)
    from src.analysis.diagnostic import (
        diagnostic_battery, _prepare_scope, build_session_summary_table, collect_figures,
        _json_safe,
    )

    report = diagnostic_battery(sessions, acc, session_ids=session_ids, group_by=group_by)
    df, a, sid_to_calib = _prepare_scope(sessions, acc, session_ids)
    summary_df = build_session_summary_table(report, df, a, sid_to_calib)

    total_stuck = len(report["stuck_at_boundary"]["flagged_rows"])
    total_anomaly = len(report["axis_anomaly"])
    mean_error_avg = float(a["euclidean_error_px"].mean())

    # ── 속성(Notion 페이지 상단 필드) ──
    properties = {
        "Name": {"title": [_rt(label)]},
        "Date": {"date": {"start": pd.Timestamp.now().strftime("%Y-%m-%d")}},
        "Group By": {"select": {"name": group_by}},
        "N Sessions": {"number": len(session_ids)},
        "Correlation r": {"number": _safe_number(report["correlation"]["r"], 3)},
        "Mean Error Avg (px)": {"number": _safe_number(mean_error_avg, 1)},
        "Total Stuck": {"number": total_stuck},
        "Total Anomaly": {"number": total_anomaly},
    }

    # ── 요약표 (3단계 색상 근사 적용) ──
    summary_colors = {
        "mean_error_px": _color_bucket(summary_df["mean_error_px"], higher_is_worse=True),
        "stuck_count": _color_bucket(summary_df["stuck_count"], higher_is_worse=True),
        "anomaly_count": _color_bucket(summary_df["anomaly_count"], higher_is_worse=True),
    }
    children = [
        _heading("요약 (세션당 1행)", level=2),
        _df_to_table_block(summary_df, color_cols=summary_colors),
    ]
    if block_report is not None:
        children = _build_block_experiment_section(block_report) + children

    # ── 시각화 (캘리브레이션 - 그 아래 관련 세션 순서로 그룹화, 배치 방식 2) ──
    if include_images:
        import matplotlib.pyplot as plt

        viz.setup_font()  # matplotlib 한글 폰트. 그림을 그리기 전에 반드시 먼저 호출.
        screen_w, screen_h = viz.infer_screen_size(a)
        figures = collect_figures(df, a, calib_df, screen_w, screen_h)

        try:
            if figures:
                children.append(_heading("시각화", level=2))

                trend_figs, groups = _group_calibration_sessions(df, figures)

                # 추세 그래프(있으면)를 맨 위에 — 개별 그룹보다 먼저 전체
                # 흐름을 보여주는 것이 "한눈에 파악"에 더 적합하다고 판단.
                for category, fig_label, fig in trend_figs:
                    category_kr = "캘리브레이션" if category == "calibration" else "세션"
                    caption = f"{category_kr}: {fig_label}"
                    file_upload_id = _upload_figure(fig, f"{category}_trend.png")
                    children.append(_heading(caption, level=3))
                    children.append(_image_block(file_upload_id, caption=caption))

                # 캘리브레이션별 그룹 — 헤딩은 캘리브레이션 id에만 부여하고,
                # 그 아래 세션들은 헤딩 없이 캡션만 붙여 블록 수를 줄인다.
                for calib_id_short, calib_fig, session_list in groups:
                    children.append(_heading(f"캘리브레이션: {calib_id_short}", level=3))
                    if calib_fig is not None:
                        calib_caption = f"캘리브레이션: {calib_id_short}"
                        file_upload_id = _upload_figure(calib_fig, f"calibration_{calib_id_short}.png")
                        children.append(_image_block(file_upload_id, caption=calib_caption))
                    else:
                        children.append(_paragraph(
                            "캘리브레이션 이미지 없음 (calibration_quality.csv에 데이터 없음)"
                        ))

                    for session_id_short, session_fig in session_list:
                        session_caption = f"세션: {session_id_short}"
                        file_upload_id = _upload_figure(session_fig, f"session_{session_id_short}.png")
                        children.append(_image_block(file_upload_id, caption=session_caption))
        finally:
            # 업로드 도중 예외(네트워크 오류 등)가 나면 아직 소비되지 않은
            # figure가 열린 채로 남아 "More than 20 figures" 경고가 누적되므로,
            # 정상/예외 경로 모두에서 남은 figure를 닫는다. 이미 _upload_figure()
            # 안에서 닫힌 figure는 plt.fignum_exists()가 False라 다시 닫지 않는다.
            for _, _, fig in figures:
                if plt.fignum_exists(fig.number):
                    plt.close(fig)
    else:
        children.append(_paragraph(
            "캘리브레이션·세션 오차 지도 등 시각화는 포함되지 않았습니다 "
            "(include_images=False). 같은 label로 생성한 HTML 리포트를 "
            "함께 참고하세요."
        ))

    # ── 상세 진단 (토글로 접어서) ──
    children.append(_heading("상세 진단", level=2))

    corr = report["correlation"]
    corr_df = pd.DataFrame(corr["table"])
    r_value = corr["r"]
    r_display = f"{r_value:.2f}" if r_value == r_value else "계산 불가(rmse 분산 0)"
    children.append(_toggle(
        f"RMSE-오차 상관관계 (r={r_display}, n={corr['n']})",
        [_df_to_table_block(corr_df)],
    ))

    row_df = pd.DataFrame(report["row_compression"])
    signed_cols = [c for c in row_df.columns if c.startswith("dx_") or c.startswith("dy_")]
    row_colors = {c: _color_bucket(row_df[c], use_abs=True) for c in signed_cols}
    children.append(_toggle(
        "행별 dx/dy 압축 패턴",
        [_df_to_table_block(row_df, color_cols=row_colors)],
    ))

    stuck_rows = report["stuck_at_boundary"]["flagged_rows"]
    stuck_children = (
        [_df_to_table_block(pd.DataFrame(stuck_rows))] if stuck_rows else [_paragraph("없음")]
    )
    children.append(_toggle(
        f"화면 경계 고착 상세 (해상도 {report['stuck_at_boundary']['screen_w']}x"
        f"{report['stuck_at_boundary']['screen_h']})",
        stuck_children,
    ))

    anomaly_rows = report["axis_anomaly"]
    anomaly_children = (
        [_df_to_table_block(pd.DataFrame(anomaly_rows))] if anomaly_rows else [_paragraph("없음")]
    )
    children.append(_toggle(f"축-선택적 이상치 후보 (참고용, n={total_anomaly})", anomaly_children))

    stb_df = pd.DataFrame(report["stb_summary"])
    children.append(_toggle("STB 요약", [_df_to_table_block(stb_df)]))

    payload = {
        "parent": {"database_id": database_id},
        "properties": properties,
        "children": children,
    }
    # 경계 방어: 개별 값 발생지에서 NaN을 놓치는 경로가 또 생기더라도
    # 네트워크 호출 직전에 한 번 더 걸러지도록 payload 전체를 정제한다.
    payload = _json_safe(payload)
    resp = requests.post(f"{NOTION_API_BASE}/pages", headers=_notion_headers(), json=payload)
    resp.raise_for_status()
    page_url = resp.json().get("url")
    print(f"Notion 업로드 완료: {page_url}")
    return page_url


if __name__ == "__main__":
    print("이 파일은 라이브러리 모듈입니다. 실행은 'python -m src.analysis.run_export'를 사용하세요.")