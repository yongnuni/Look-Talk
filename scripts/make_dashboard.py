# -*- coding: utf-8 -*-
"""make_dashboard.py v4.0 — 입력 방식 추천 결과 리포트 (일반인용 기본 / 기술 상세 --detail)

사용법 (레포 루트에서, --performance-flow 완주 후):
    python scripts/make_dashboard.py                  # 최신 완주 세션 자동 선택 → 일반인용(public) 리포트
    python scripts/make_dashboard.py --run-id e7c0cb9a  # run_id 앞 8자리로 지정
    python scripts/make_dashboard.py --detail          # 기술 지표 매트릭스(EAR/MAR/RMSE/FPS 등) 그대로 렌더
    python scripts/make_dashboard.py --open            # 생성 직후 브라우저로 열기
    python scripts/make_dashboard.py --png             # PNG 동시 저장 (선택 의존성)
출력: dashboard_<run앞8자리>.html (--detail 시 dashboard_<run앞8자리>_detail.html)
      (+ --png 시 동명 .png, 기본 2배 해상도) (레포 루트에 생성 — .gitignore에 dashboard_*.html 권장)
의존성: pandas (requirements.txt 포함) · PNG 저장 시에만 playwright 추가 설치
        (pip install playwright && playwright install chromium)

[수집 규칙] 대시보드 품질을 위해 반드시 지킬 것:
  - 실행 전 src/config.py의 MONITOR_DIAGONAL_INCH를 자기 모니터 값으로 설정
  - python main.py --performance-flow --user-id <자기이름> [--condition-label 라벨]
  - DASHBOARD 화면까지 완주해야 summary CSV가 생성됨 (중도 종료 시 이 도구가 세션을 찾지 못함)
추천 카드는 PerformanceTestFlow.get_recommended_input_mode()와 동일 규칙을 오프라인
재계산해 저장값과 대조한다. 코드 규칙 변경 시 recommend()도 갱신할 것.

[기본(public) 지표 정의 — 모두 CSV 실측에서 계산, 수기 상수 없음]
  인식 정확도(m) = 성공 시도 수 / 전체 시도 수 × 100
      전체 시도 수·성공 시도 수는 input_method_test_results 원시 행에서 집계.
      summary의 {m}_success_rate_percent와 값이 같아야 하며, 불일치 시 콘솔 경고.
  오타율(m)      = {m}_incorrect_attempts / 전체 시도 수 × 100  (낮을수록 좋음 → 게이지 색 반전)
  입력 속도(m)   = min(완료된 모드들의 평균 입력 시간) / {m}_average_input_time_sec × 100
      완료 모드가 1개뿐이면 그 모드만 100.
  입력 안정성(m):
      gaze  = gaze_mean_confidence × 100
      blink = (blink_open_ear_median − blink_close_threshold)
              / (blink_open_ear_median − blink_closed_ear_median) × 100   (임계 분리 여유)
      mouth = (1 − mouth_false_trigger_rate) × mouth_consistency × 100
  4개 지표 모두, 해당 모드의 1음절 테스트가 완료(completed)되지 않았거나 원시 시도 행이
  없으면 "측정 안 됨"으로 표기하고 %를 표시하지 않는다(스킵 폴백).

  추천 점수(m, "N점") = 위 4개 지표의 평균. 오타율은 (100 − 오타율)로 반전해
      "값이 클수록 좋음" 방향으로 통일한 뒤 평균한다(total_score()). 사용자 노출용
      요약 점수이며, 원인 진단은 아래 '상세 지표' 막대그래프(지표별 값 그대로)로 확인한다.
"""
import argparse, random, webbrowser
from pathlib import Path
import pandas as pd

GRID, AXIS = "#E8EAED", "#80868B"
BANDS = {"nat_max": 0.25, "int_min": 0.30, "int_max": 0.80}
GATE_PX = 150.0
MODES = ["gaze", "blink", "mouth"]
MODE_KO = {"gaze": "시선", "blink": "깜빡임", "mouth": "입벌림"}
MODE_SUBTITLE = {
    "gaze": "화면을 응시하여 입력",
    "blink": "눈을 깜빡여 입력",
    "mouth": "입 움직임으로 입력",
}
MODE_COLOR = {"gaze": "#4285F4", "blink": "#12B5CB", "mouth": "#34A853"}
# public 대시보드 전용 모드 명칭(기술 상세 --detail의 MODE_KO/"입벌림" 표기는 유지)
PUBLIC_MODE_LABEL = {**MODE_KO, "mouth": "입 움직임"}
COLS = {"c1": "#4285F4", "c2": "#12B5CB", "c3": "#34A853", "c4": "#FA7B17"}

# public 대시보드 행 정의: (지표 key, 라벨, 한 줄 설명, invert(낮을수록 좋음이면 True))
ROWS = [
    ("error", "오타율", "잘못 입력한 비율(낮을수록 좋음)", True),
    ("speed", "입력 속도", "가장 빠른 방식을 100으로 본 상대 속도", False),
    ("accuracy", "인식 정확도", "의도한 글자가 정확히 입력된 비율", False),
    ("stability", "입력 안정성", "오작동 없이 일관되게 동작한 정도", False),
]

def read(p, need=True):
    if not p.exists():
        if need: raise SystemExit(f"[없음] {p} — --performance-flow 완주 후 실행하세요")
        return None
    return pd.read_csv(p, encoding="utf-8-sig")

def pick(df, run_prefix):
    if run_prefix:
        df = df[df.run_id.astype(str).str.startswith(run_prefix)]
        if df.empty: raise SystemExit(f"run_id '{run_prefix}*' 요약 없음")
    return df.iloc[-1]

def recommend(S):
    order = {m: i for i, m in enumerate(MODES)}
    cand = [(-float(S[f"{m}_success_rate_percent"]), float(S[f"{m}_average_input_time_sec"]), order[m], m)
            for m in MODES if str(S[f"{m}_status"]) == "completed"]
    if not cand: return None, "완료된 테스트 없음"
    best = min(cand)
    tied = len({c[0] for c in cand}) == 1
    return best[3], ("확인 성공률 동률 → 소요시간 최단" if tied else "확인 성공률 최상위")

def plain_reason(S, reco, comp):
    """recommend()의 판정을 일반인이 이해할 수 있는 평문 근거로 옮긴다. 기술 용어 없음."""
    done = [m for m in MODES if comp[m]]
    if not done or reco is None:
        return "완료된 테스트가 없어 추천할 수 없습니다."
    rates = {m: float(S[f"{m}_success_rate_percent"]) for m in done}
    top = max(rates.values())
    tied = [m for m in done if rates[m] == top]
    if len(tied) == 1:
        return f"{MODE_KO[reco]} 방식이 가장 정확하게 입력되었습니다."
    names = "·".join(MODE_KO[m] for m in tied)
    if len(tied) == len(MODES) and top >= 100:
        return f"세 방식 모두 정확히 입력했고, {MODE_KO[reco]} 방식이 가장 빨랐습니다."
    return f"{names} 방식이 똑같이 정확했고, 그중 {MODE_KO[reco]} 방식이 가장 빨랐습니다."

def compute_public_metrics(S, tests, rid, comp):
    """public 대시보드 4지표 산출. 반환: (metrics[mode][key] -> float|None, warnings, total_attempts[mode]).
    산출식은 모듈 docstring의 [기본(public) 지표 정의]를 그대로 구현한다.
    """
    metrics = {m: {"accuracy": None, "error": None, "speed": None, "stability": None} for m in MODES}
    warnings = []
    total_attempts = {m: 0 for m in MODES}
    tm = None
    if tests is not None:
        tm = tests[tests.run_id.astype(str) == rid]
        for m in MODES:
            total_attempts[m] = int((tm.input_mode == m).sum())

    for m in MODES:
        if not comp[m] or total_attempts[m] == 0:
            continue
        total = total_attempts[m]
        success_ct = int(tm[tm.input_mode == m]["success"].astype(bool).sum())
        accuracy = success_ct / total * 100
        stored = float(S[f"{m}_success_rate_percent"])
        if abs(accuracy - stored) > 0.5:
            warnings.append(
                f"[경고] {MODE_KO[m]} 인식 정확도 재계산({accuracy:.1f}%)이 저장값({stored:.1f}%)과 불일치"
            )
        metrics[m]["accuracy"] = accuracy
        metrics[m]["error"] = float(S[f"{m}_incorrect_attempts"]) / total * 100

    done_times = {m: float(S[f"{m}_average_input_time_sec"]) for m in MODES if comp[m] and total_attempts[m] > 0}
    if done_times:
        fastest = min(done_times.values())
        for m, t in done_times.items():
            metrics[m]["speed"] = fastest / t * 100

    if comp["gaze"] and total_attempts["gaze"] > 0:
        metrics["gaze"]["stability"] = float(S.gaze_mean_confidence) * 100
    if comp["blink"] and total_attempts["blink"] > 0:
        denom = float(S.blink_open_ear_median) - float(S.blink_closed_ear_median)
        if denom > 0:
            metrics["blink"]["stability"] = (
                (float(S.blink_open_ear_median) - float(S.blink_close_threshold)) / denom * 100
            )
        else:
            warnings.append("[경고] blink 안정성 계산 불가 (open_ear_median <= closed_ear_median)")
    if comp["mouth"] and total_attempts["mouth"] > 0:
        metrics["mouth"]["stability"] = (1 - float(S.mouth_false_trigger_rate)) * float(S.mouth_consistency) * 100

    return metrics, warnings, total_attempts

def total_score(mode_metrics):
    """'추천 점수' = 4개 상세 지표의 평균. 오타율은 (100-오타율)로 반전해
    '값이 클수록 좋음' 방향으로 통일한 뒤 평균한다(수기 상수 없음, ROWS 정의 재사용).
    측정된 지표가 하나도 없으면 None(= '측정 안 됨')."""
    vals = []
    for key, _label, _desc, invert in ROWS:
        pct = mode_metrics[key]
        if pct is None: continue
        vals.append((100 - pct) if invert else pct)
    return sum(vals) / len(vals) if vals else None

# ── public 대시보드 막대 게이지 ─────────────────────────────
def gauge_color(pct, invert=False):
    if pct is None: return AXIS
    v = (100 - pct) if invert else pct
    if v >= 80: return "#34A853"
    if v >= 50: return "#F9AB00"
    return "#EA4335"

def metric_hbars(mode_metrics, w=250, h=140):
    """모드 1개의 4개 상세 지표를 가로 막대로 표시. x축=퍼센티지(0~100), y축=지표 4종.
    ROWS 순서(오타율/입력 속도/인식 정확도/입력 안정성) 그대로 사용."""
    pl = 62; W = w - pl - 38; n = len(ROWS); rh = (h - 8) / n; o = []
    for i, (key, label, _desc, invert) in enumerate(ROWS):
        y = 6 + i * rh
        o.append(f'<text x="{pl-6}" y="{y+rh/2+3:.1f}" font-size="10.5" fill="#3C4043" text-anchor="end">{label}</text>')
        pct = mode_metrics[key]
        if pct is None:
            o.append(f'<rect x="{pl}" y="{y+rh*0.22:.1f}" width="{W:.1f}" height="{rh*0.5:.1f}" rx="3" fill="#F1F3F4"/>')
            o.append(f'<text x="{pl+8}" y="{y+rh/2+3:.1f}" font-size="9.5" fill="#9AA0A6">측정 안 됨</text>')
        else:
            bw = max(0.0, min(100.0, pct)) / 100 * W
            col = gauge_color(pct, invert)
            o.append(f'<rect x="{pl}" y="{y+rh*0.22:.1f}" width="{bw:.1f}" height="{rh*0.5:.1f}" rx="3" fill="{col}"/>')
            o.append(f'<text x="{pl+bw+6:.1f}" y="{y+rh/2+3:.1f}" font-size="10" fill="#3C4043" font-weight="600">{round(pct)}%</text>')
    return f'<svg viewBox="0 0 {w} {h}" width="100%">' + "".join(o) + "</svg>"

# ── 기술 상세(detail) 차트 (셀 폭 250 기준) ─────────────────────────────
def vbars(pairs, w=250, h=118, unit="px", gate=None):
    """[(label, v, color)] 세로 막대."""
    cap = max(v for _, v, _ in pairs) * 1.25
    if gate: cap = max(cap, gate * 1.18)  # 게이트 선이 상단 라벨과 겹치지 않게
    n = len(pairs); pl, pb, pt = 30, 16, 8; cw = (w-pl)/n; o = []
    for i in range(4):
        y = pt + (h-pb-pt)*(1-i/3)
        o.append(f'<line x1="{pl}" y1="{y:.1f}" x2="{w}" y2="{y:.1f}" stroke="{GRID}"/>')
        o.append(f'<text x="{pl-5}" y="{y+3:.1f}" font-size="8.5" fill="{AXIS}" text-anchor="end">{cap*i/3:.0f}</text>')
    for i, (lab, v, col) in enumerate(pairs):
        bh = min(v, cap)/cap*(h-pb-pt)
        o.append(f'<rect x="{pl+i*cw+cw*0.22:.1f}" y="{h-pb-bh:.1f}" width="{cw*0.56:.1f}" height="{bh:.1f}" rx="2" fill="{col}"/>')
        o.append(f'<text x="{pl+i*cw+cw/2:.1f}" y="{h-pb-bh-4:.1f}" font-size="9" fill="#3C4043" text-anchor="middle" font-weight="600">{v:g}</text>')
        o.append(f'<text x="{pl+i*cw+cw/2:.1f}" y="{h-4}" font-size="8.5" fill="{AXIS}" text-anchor="middle">{lab}</text>')
    if gate and gate < cap:
        gy = pt + (h-pb-pt)*(1-gate/cap)
        o.append(f'<line x1="{pl}" y1="{gy:.1f}" x2="{w}" y2="{gy:.1f}" stroke="#EA4335" stroke-width="1.2" stroke-dasharray="4 3"/>')
        o.append(f'<text x="{w-2}" y="{gy-3:.1f}" font-size="8" fill="#EA4335" text-anchor="end">게이트 {gate:.0f}</text>')
    return f'<svg viewBox="0 0 {w} {h}" width="100%">' + "".join(o) + "</svg>"

def ear_bullet(S, w=250, h=112):
    mx = max(0.6, float(S.blink_open_ear_median)*1.15); pl = 8; W = w-pl-6
    x = lambda v: pl + float(v)/mx*W
    o = [f'<rect x="{pl}" y="48" width="{W}" height="11" rx="5.5" fill="#F1F3F4"/>',
         f'<rect x="{x(S.blink_close_threshold):.1f}" y="48" width="{x(S.blink_open_threshold)-x(S.blink_close_threshold):.1f}" height="11" fill="#FDE293"/>',
         f'<line x1="{x(S.blink_close_threshold):.1f}" y1="40" x2="{x(S.blink_close_threshold):.1f}" y2="67" stroke="#EA4335" stroke-width="2"/>',
         f'<text x="{x(S.blink_close_threshold):.1f}" y="36" font-size="8.5" fill="#EA4335" text-anchor="middle">close {S.blink_close_threshold:.3f}</text>',
         f'<line x1="{x(S.blink_open_threshold):.1f}" y1="40" x2="{x(S.blink_open_threshold):.1f}" y2="67" stroke="#F9AB00" stroke-width="2"/>',
         f'<text x="{x(S.blink_open_threshold):.1f}" y="80" font-size="8.5" fill="#B06000" text-anchor="middle">open {S.blink_open_threshold:.3f}</text>',
         f'<circle cx="{x(S.blink_closed_ear_median):.1f}" cy="53.5" r="5" fill="#fff" stroke="#5F6368" stroke-width="2.3"/>',
         f'<text x="{x(S.blink_closed_ear_median):.1f}" y="97" font-size="8.5" fill="#5F6368" text-anchor="middle">감은눈 {S.blink_closed_ear_median:.3f}</text>',
         f'<circle cx="{x(S.blink_open_ear_median):.1f}" cy="53.5" r="5" fill="#fff" stroke="#12B5CB" stroke-width="2.5"/>',
         f'<text x="{x(S.blink_open_ear_median):.1f}" y="22" font-size="8.5" fill="#0E8CA0" text-anchor="middle">뜬눈 {S.blink_open_ear_median:.3f}</text>']
    return f'<svg viewBox="0 0 {w} {h}" width="100%">' + "".join(o) + "</svg>"

def mar_bullet(S, w=250, h=112):
    mx = max(1.2, float(S.mouth_open_mar)*1.15); pl = 8; W = w-pl-6
    x = lambda v: pl + float(v)/mx*W
    o = [f'<rect x="{pl}" y="48" width="{W}" height="11" rx="5.5" fill="#F1F3F4"/>',
         f'<rect x="{pl}" y="48" width="{x(S.mouth_open_mar)-pl:.1f}" height="11" rx="5.5" fill="#CEEAD6"/>',
         f'<line x1="{x(S.mouth_open_threshold):.1f}" y1="40" x2="{x(S.mouth_open_threshold):.1f}" y2="67" stroke="#EA4335" stroke-width="2"/>',
         f'<text x="{x(S.mouth_open_threshold):.1f}" y="36" font-size="8.5" fill="#EA4335" text-anchor="middle">open {S.mouth_open_threshold:.3f}</text>',
         f'<line x1="{x(S.mouth_close_threshold):.1f}" y1="40" x2="{x(S.mouth_close_threshold):.1f}" y2="67" stroke="#F9AB00" stroke-width="2"/>',
         f'<text x="{x(S.mouth_close_threshold):.1f}" y="80" font-size="8.5" fill="#B06000" text-anchor="middle">close {S.mouth_close_threshold:.3f}</text>',
         f'<circle cx="{x(S.mouth_mar_baseline):.1f}" cy="53.5" r="5" fill="#fff" stroke="#5F6368" stroke-width="2.3"/>',
         f'<text x="{x(S.mouth_mar_baseline)+4:.1f}" y="97" font-size="8.5" fill="#5F6368">다묾 {S.mouth_mar_baseline:.3f}</text>',
         f'<circle cx="{x(S.mouth_open_mar):.1f}" cy="53.5" r="5" fill="#fff" stroke="#34A853" stroke-width="2.5"/>',
         f'<text x="{x(S.mouth_open_mar):.1f}" y="22" font-size="8.5" fill="#137333" text-anchor="middle">벌림 {S.mouth_open_mar:.3f}</text>']
    return f'<svg viewBox="0 0 {w} {h}" width="100%">' + "".join(o) + "</svg>"

def time_hbars(S, reco, comp, w=250, h=118):
    done = [(m, float(S[f"{m}_average_input_time_sec"])) for m in MODES if comp[m]]
    mx = (max(v for _, v in done) if done else 1.0) * 1.18; pl = 44; W = w-pl-34; rh = (h-8)/3; o = []
    for i, m in enumerate(MODES):
        y = 6 + i*rh
        o.append(f'<text x="{pl-6}" y="{y+rh/2+3:.1f}" font-size="9.5" fill="#3C4043" text-anchor="end">{MODE_KO[m]}</text>')
        if comp[m]:
            v = float(S[f"{m}_average_input_time_sec"]); col = "#137333" if m == reco else "#FA7B17"
            o.append(f'<rect x="{pl}" y="{y+rh*0.18:.1f}" width="{v/mx*W:.1f}" height="{rh*0.55:.1f}" rx="3" fill="{col}" opacity="{1 if m==reco else 0.55}"/>')
            o.append(f'<text x="{pl+v/mx*W+5:.1f}" y="{y+rh/2+3:.1f}" font-size="9" fill="#3C4043" font-weight="600">{v:.1f}s</text>')
        else:
            o.append(f'<rect x="{pl}" y="{y+rh*0.18:.1f}" width="{W:.1f}" height="{rh*0.55:.1f}" rx="3" fill="#F1F3F4"/>')
            o.append(f'<text x="{pl+8}" y="{y+rh/2+3:.1f}" font-size="9" fill="#80868B">건너뜀 (미측정)</text>')
    return f'<svg viewBox="0 0 {w} {h}" width="100%">' + "".join(o) + "</svg>"

def blinkscatter(bb, w=250, h=118):
    MX = 1.0; pl = 6; W = w-pl-4; x = lambda s: pl + s/MX*W
    K = {"NATURAL": "#4285F4", "INTENTIONAL": "#34A853", "LONG_CLOSURE": "#A142F4"}
    o = []
    for s, e, col, lab in [(0, BANDS["nat_max"], "#4285F4", "자연"), (BANDS["nat_max"], BANDS["int_min"], "#9AA0A6", ""),
                           (BANDS["int_min"], BANDS["int_max"], "#34A853", "의도"), (BANDS["int_max"], MX, "#A142F4", "장기")]:
        o.append(f'<rect x="{x(s):.1f}" y="0" width="{x(e)-x(s):.1f}" height="{h-18}" fill="{col}" opacity="0.08"/>')
        if lab: o.append(f'<text x="{(x(s)+x(e))/2:.1f}" y="12" font-size="8.5" fill="{col}" text-anchor="middle">{lab}</text>')
    random.seed(4)
    for r in bb.itertuples():
        o.append(f'<circle cx="{x(min(r.duration_ms/1000, MX)):.1f}" cy="{26+random.uniform(0, h-52):.1f}" r="3.6" fill="{K.get(r.kind, AXIS)}" opacity="0.85"/>')
    for s in [0, 0.25, 0.5, 0.8, 1.0]:
        o.append(f'<text x="{x(s):.1f}" y="{h-4}" font-size="8" fill="{AXIS}" text-anchor="middle">{s:g}s</text>')
    return f'<svg viewBox="0 0 {w} {h}" width="100%">' + "".join(o) + "</svg>"


def export_png(html_path, scale=2):
    """HTML을 PNG로 캡처. playwright 미설치 시 안내만 하고 넘어감."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("[안내] PNG 저장에는 playwright가 필요합니다:")
        print("       pip install playwright && playwright install chromium")
        print("       (또는 브라우저 DevTools의 'Capture full size screenshot' 사용)")
        return None
    out = str(Path(html_path).with_suffix(".png"))
    try:
        with sync_playwright() as pw:
            b = pw.chromium.launch()
            pg = b.new_page(viewport={"width": 1280, "height": 900}, device_scale_factor=scale)
            pg.goto(Path(html_path).resolve().as_uri())
            pg.wait_for_timeout(600)
            pg.locator("body").screenshot(path=out)
            b.close()
        return out
    except Exception as e:
        print(f"[경고] PNG 캡처 실패({type(e).__name__}: {e}) — HTML은 정상 생성됨.")
        print("       처음이라면 'playwright install chromium' 실행 여부를 확인하세요.")
        return None

# ── 렌더: public(기본, 일반인용) ─────────────────────────────
def render_public(S, rid, comp, metrics, reco, reason_text):
    heads = "".join(
        f'<div class="phead" style="background:{MODE_COLOR[m]}">'
        f'<div class="pht">{PUBLIC_MODE_LABEL[m]} 입력</div><div class="phs">{MODE_SUBTITLE[m]}</div></div>'
        for m in MODES
    )
    def score_cell(m):
        score = total_score(metrics[m])
        if score is None:
            return '<div class="pscore na">측정 안 됨</div>'
        return f'<div class="pscore" style="color:{gauge_color(score)}">{round(score)}<span>점</span></div>'

    rows_html = '<div class="plabel"><div class="plt">추천 점수</div></div>'
    rows_html += "".join(f'<div class="pcell">{score_cell(m)}</div>' for m in MODES)
    rows_html += '<div class="plabel"><div class="plt">상세 지표</div></div>'
    rows_html += "".join(f'<div class="pcell">{metric_hbars(metrics[m])}</div>' for m in MODES)

    if reco:
        badge_bg, badge_border, icon = "#E6F4EA", "#34A853", "👍"
        accent, mode_text = MODE_COLOR[reco], PUBLIC_MODE_LABEL[reco]
    else:
        badge_bg, badge_border, icon = "#F1F3F4", "#9AA0A6", "❓"
        accent, mode_text = "#9AA0A6", "추천 보류"

    return f"""<!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8"><title>Look Talk 입력 방식 추천 결과</title><style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:"Roboto","Noto Sans KR","Malgun Gothic",sans-serif;background:#F8F9FA;color:#202124;max-width:1160px;margin:0 auto;padding-bottom:18px}}
.appbar{{background:#fff;border-bottom:1px solid #DADCE0;padding:14px 24px;display:flex;align-items:center;gap:14px}}
.appbar .logo{{font-size:19px;font-weight:700;display:flex;align-items:center;gap:8px}}
.appbar .logo img{{height:40px;width:auto;display:block;margin:3px}}
.appbar h1{{font-size:15px;font-weight:500;color:#5F6368}}
.daterange{{margin-left:auto;border:1px solid #DADCE0;border-radius:4px;padding:6px 12px;font-size:12.5px;color:#3C4043}}
.filters{{padding:10px 24px;display:flex;gap:8px;flex-wrap:wrap}}
.fchip{{border:1px solid #DADCE0;border-radius:16px;background:#fff;padding:5px 12px;font-size:12px;color:#3C4043}}
.fchip b{{color:#1A73E8;font-weight:500}}
.pgrid{{margin:14px 24px;display:grid;grid-template-columns:220px repeat(3,1fr);border:1px solid #DADCE0;border-radius:10px;overflow:hidden;background:#fff}}
.pgrid>div{{border-bottom:1px solid #E8EAED;border-right:1px solid #E8EAED}}
.pcorner{{background:#F8F9FA}}
.phead{{padding:16px 12px;text-align:center}}
.pht{{font-size:22px;font-weight:700;color:#fff}}
.phs{{font-size:11.5px;color:rgba(255,255,255,0.9);margin-top:4px}}
.plabel{{background:#F8F9FA;padding:14px 18px;display:flex;flex-direction:column;justify-content:center;align-items:center;text-align:center}}
.plt{{font-size:22px;font-weight:700;color:#202124}}
.pcell{{padding:14px;display:flex;align-items:center;justify-content:center}}
.pscore{{font-size:55px;font-weight:800;line-height:1;margin:10px}}
.pscore span{{font-size:16px;font-weight:600;margin-left:2px;color:#5F6368}}
.pscore.na{{font-size:13px;font-weight:500;color:#9AA0A6}}
.reco2{{margin:14px 24px;background:#fff;border:1px solid #DADCE0;border-left:7px solid {accent};border-radius:10px;padding:20px 26px;display:flex;align-items:center;gap:22px;box-shadow:0 2px 10px rgba(60,64,67,0.12)}}
.rbadge{{width:72px;height:72px;border-radius:50%;background:{badge_bg};display:grid;place-items:center;font-size:32px;border:3px solid {badge_border};flex-shrink:0}}
.rtext{{flex:1}}
.rlabel{{font-size:12.5px;font-weight:600;color:#5F6368;letter-spacing:.02em}}
.rmode{{font-size:55px;font-weight:800;color:{accent};line-height:1.15;margin-top:2px}}
.ractions{{display:flex;flex-direction:column;gap:8px;flex-shrink:0}}
.abtn{{border:1px solid #1A73E8;color:#1A73E8;border-radius:16px;padding:7px 14px;font-size:15px;font-weight:500;white-space:nowrap}}
.abtn.ghost{{border-color:#DADCE0;color:#5F6368}}
</style></head><body>
<div class="appbar"><div class="logo"><img src="assets/Logo.png" alt="Look Talk"></div><h1>입력 방식 추천 결과</h1>
<div class="daterange">📅 {str(S.t0_utc)[:10]}</div></div>
<div class="filters"><span class="fchip">사용자 <b>{S.user_id} ▾</b></span><span class="fchip">세션 <b>{rid[:8]} ▾</b></span></div>
<div class="reco2">
<div class="rbadge">{icon}</div>
<div class="rtext"><div class="rlabel">자동 추천 방식</div><div class="rmode">{mode_text}</div></div>
<div class="ractions"><span class="abtn ghost">다시 검사하기</span></div>
</div>
<div class="pgrid"><div class="pcorner"></div>{heads}{rows_html}</div>
</body></html>"""

# ── 렌더: detail(기존 기술 매트릭스, --detail 플래그) ─────────────────────────────
def render_detail(S, rid, bb, cnt, comp, reco, reason, meta):
    n_done = sum(comp.values())
    def sv(x, fmt="{:.1f}", na="—"):
        try:
            v = float(x)
            return na if v != v else fmt.format(v)
        except (TypeError, ValueError):
            return na
    tgt = lambda m: (str(S[f"{m}_target_character"]) if str(S[f"{m}_target_character"]) not in ("nan","None","") else "—")

    kv = lambda k, v: f'<div class="kv"><span>{k}</span><b>{v}</b></div>'
    big = lambda v, u, c, d: f'<div class="big" style="color:{c}">{v}<span>{u}</span></div><div class="delta">{d}</div>'

    cells = {
      "info": [
        kv("포인트", "16 / 16")+kv("샘플 사용", f"{int(S.gaze_used_sample_count)}/{int(S.gaze_sample_count)}")+kv("재시도 / 폴백", f"{int(S.gaze_calibration_retry_count)}회 / {'유' if bool(S.gaze_calibration_fallback_used) else '무'}"),
        kv("시행", f"{int(S.blink_total_trials)}회")+kv("감은눈 샘플", f"{int(S.blink_closed_sample_count)}/{int(S.blink_total_trials)}")+kv("폴백", "유" if bool(S.blink_calibration_fallback) else "무"),
        kv("시행", f"{int(S.mouth_total_trials)}회")+kv("성공", f"{int(S.mouth_success_count)}/{int(S.mouth_total_trials)}")+kv("오작동", f"{int(S.mouth_false_trigger_count)}회"),
        kv("모드", "3개 (동일 프로토콜)")+kv("음절", " · ".join(tgt(m) for m in MODES))+kv("재시도", "무제한 · 잠금 350ms"),
      ],
      "key": [
        big(f"{S.calib_reproj_rmse_px:.0f}", "px RMSE", COLS["c1"], f"게이트 {GATE_PX:.0f}px {'통과' if S.calib_reproj_rmse_px < GATE_PX else '초과'}"),
        big(f"{S.blink_close_threshold:.3f}", "EAR 개인 임계", COLS["c2"], "기본 0.18 → 개인화 적용"),
        big(f"{S.mouth_contrast_ratio:.0f}", "배 대비율", COLS["c3"], f"오작동 {S.mouth_false_trigger_rate:.0%}"),
        (big("전원 100%", "", COLS["c4"], "3개 모드 확인 성공")
         if n_done == 3 and all(float(S[f"{m}_success_rate_percent"]) >= 100 for m in MODES)
         else big(f"{n_done}/3", "모드 완료", COLS["c4"] if n_done else "#80868B",
                  ("건너뜀: " + ", ".join(MODE_KO[m] for m in MODES if not comp[m])) if n_done else "전 모드 건너뜀 — 추천 보류")),
      ],
      "chart": [
        vbars([("가장자리", round(float(S.edge_mean_reproj_error_px),1), "#AECBFA"), ("중앙", round(float(S.center_mean_reproj_error_px),1), "#1A73E8"), ("RMSE", round(float(S.calib_reproj_rmse_px),1), "#4285F4")], gate=GATE_PX),
        ear_bullet(S), mar_bullet(S), time_hbars(S, reco, comp),
      ],
      "detail": [
        kv("시선 신뢰도", f"{S.gaze_mean_confidence:.3f}")+kv("홍채 흔들림 x/y", f"{S.iris_std_x_norm_mean:.4f} / {S.iris_std_y_norm_mean:.4f}")+kv("FPS (세션 전체)", f"{S.stb01_fps:.1f}")+kv("랜드마크 / 유실", f"{S.stb02_landmark_rate:.0%} / {S.stb04_dropout_rate:.1%}"),
        blinkscatter(bb)+f'<div class="mini">실측 {len(bb)}건 · 자연 {cnt.get("NATURAL",0)} · 의도 {cnt.get("INTENTIONAL",0)} · 장기 {cnt.get("LONG_CLOSURE",0)}</div>',
        kv("기준선 (다묾→벌림)", f"{S.mouth_mar_baseline:.3f} → {S.mouth_open_mar:.3f}")+kv("일관성", f"{S.mouth_consistency:.2f}")+kv("활성 유지", f"{S.mouth_activation_duration_mean:.2f}초")+kv("성공률", f"{S.mouth_success_rate:.0%}"),
        "".join((kv(f"{MODE_KO[m]} ({tgt(m)}→{S[f'{m}_selected_character']})", f"{S[f'{m}_success_rate_percent']:.0f}% · {S[f'{m}_average_input_time_sec']:.1f}s · 오답 {int(S[f'{m}_incorrect_attempts'])}") if comp[m] else kv(f"{MODE_KO[m]}", f"건너뜀 ({S[f'{m}_status']})")) for m in MODES),
      ],
    }
    heads = [("16점 시선 캘리브레이션", COLS["c1"]), ("깜빡임 개인화", COLS["c2"]), ("입벌림 개인화", COLS["c3"]), ("1음절 입력 테스트", COLS["c4"])]
    rails = [("🗓", "시행 정보"), ("🎯", "핵심 결과"), ("📊", "측정 차트"), ("📄", "상세 결과")]

    grid = "".join(f'<div class="chead" style="background:{c}">{t}</div>' for t, c in heads)
    for (icon, rail), key in zip(rails, ["info", "key", "chart", "detail"]):
        grid += f'<div class="rail"><div class="ric">{icon}</div><div>{rail}</div></div>'
        grid += "".join(f'<div class="cell">{cells[key][i]}</div>' for i in range(4))

    return f"""<!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8"><title>Look Talk 캘리브레이션·초기 측정 리포트</title><style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:"Roboto","Noto Sans KR","Malgun Gothic",sans-serif;background:#F8F9FA;color:#202124;max-width:1280px;margin:0 auto;padding-bottom:18px}}
.appbar{{background:#fff;border-bottom:1px solid #DADCE0;padding:14px 24px;display:flex;align-items:center;gap:14px}}
.appbar .logo{{font-size:19px;font-weight:700;display:flex;align-items:center;gap:8px}}
.appbar .logo img{{height:26px;width:auto;display:block}}
.appbar h1{{font-size:15px;font-weight:500;color:#5F6368}}
.daterange{{margin-left:auto;border:1px solid #DADCE0;border-radius:4px;padding:6px 12px;font-size:12.5px;color:#3C4043}}
.filters{{padding:10px 24px;display:flex;gap:8px;flex-wrap:wrap}}
.fchip{{border:1px solid #DADCE0;border-radius:16px;background:#fff;padding:5px 12px;font-size:12px;color:#3C4043}}
.fchip b{{color:#1A73E8;font-weight:500}}
.matrix{{margin:6px 24px;display:grid;grid-template-columns:118px repeat(4,1fr);border:1px solid #DADCE0;border-radius:8px;overflow:hidden;background:#fff}}
.matrix>div{{border-bottom:1px solid #E8EAED;border-right:1px solid #E8EAED}}
.chead{{grid-column:auto;color:#fff;font-size:13px;font-weight:600;padding:11px 12px;text-align:center;letter-spacing:.01em}}
.matrix>.chead:first-of-type{{grid-column:2}}
.rail{{background:#F8F9FA;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:6px;font-size:11.5px;font-weight:600;color:#3C4043;padding:12px 6px;text-align:center}}
.ric{{width:34px;height:34px;border-radius:50%;background:#fff;border:1px solid #DADCE0;display:grid;place-items:center;font-size:15px}}
.cell{{padding:12px 14px;display:flex;flex-direction:column;justify-content:center;gap:2px}}
.kv{{display:flex;justify-content:space-between;font-size:11.5px;padding:3.5px 0;border-bottom:1px dashed #F1F3F4;color:#5F6368}}
.kv b{{color:#202124;font-weight:600;text-align:right}}
.big{{font-size:25px;font-weight:600;line-height:1.1}} .big span{{font-size:11.5px;color:#5F6368;font-weight:400;margin-left:4px}}
.delta{{font-size:10.5px;color:#137333;margin-top:6px}}
.mini{{font-size:10px;color:#80868B;margin-top:4px}}
.reco{{margin:12px 24px 0;background:#fff;border:1px solid #DADCE0;border-radius:8px;padding:16px 20px;display:flex;gap:24px;align-items:center}}
.reco .rv{{font-size:45px;font-weight:700;color:#137333}} .reco .rr{{font-size:12px;color:#5F6368;line-height:1.75;margin-top:5px}}
.reco h2{{font-size:13.5px;font-weight:500}} .reco .cs{{font-size:11px;color:#5F6368;margin:2px 0 4px}}
.badge{{width:80px;height:80px;border-radius:50%;background:#E6F4EA;display:grid;place-items:center;font-size:32px;border:3px solid #34A853}}
.foot{{padding:10px 24px;font-size:10.5px;color:#80868B;line-height:1.7}}
</style></head><body>
<div class="appbar"><div class="logo"><img src="assets/Logo.png" alt="Look Talk"> Look Talk</div><h1>캘리브레이션 · 초기 입력 측정 리포트</h1>
<div class="daterange">📅 {str(S.t0_utc)[:10]}</div></div>
<div class="filters"><span class="fchip">참가자 <b>{S.user_id} ▾</b></span><span class="fchip">세션 <b>{rid[:8]} ▾</b></span>
<span class="fchip">레이아웃 <b>{S.keyboard_layout} ▾</b></span></div>
<div class="matrix"><div style="background:#F8F9FA"></div>{grid}</div>
<div class="reco"><div style="flex:1"><h2>초기 입력 방식 추천 <span class="mini">시스템 추천값: {S.recommended_input_mode}</span></h2>
<div class="rv">{(MODE_KO[reco] + f" ({reco})") if reco else "추천 보류"}</div></div>
<div class="badge">👍</div></div>
</body></html>"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="gaze_accuracy_results")
    ap.add_argument("--calib-dir", default="calibration_results")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--detail", action="store_true", help="기술 지표 매트릭스(EAR/MAR/RMSE/FPS 등) 렌더")
    ap.add_argument("--open", action="store_true", help="생성 후 브라우저로 열기")
    ap.add_argument("--png", action="store_true", help="PNG로도 저장 (playwright 필요)")
    ap.add_argument("--scale", type=int, default=2, help="PNG 해상도 배율 (기본 2x)")
    a = ap.parse_args()
    R = Path(a.results_dir)

    S = pick(read(R/"performance_flow_summary_v1.1.csv"), a.run_id)
    rid = str(S.run_id)
    bev = read(R/"blink_events_v1.0.csv", need=False)
    bb = bev[bev.run_id.astype(str) == rid] if bev is not None else pd.DataFrame(columns=["kind", "duration_ms", "ts_ms"])
    # 대시보드 종속 원칙: 이벤트 로그는 blink 1음절 테스트 구간으로만 한정한다
    # (16점 캘리브레이션 중 자연 깜빡임·blink 캘리브레이션의 의도적 감음·대시보드 열람 구간 배제)
    tests = read(R/"input_method_test_results_v1.1.csv", need=False)
    blink_window = None
    if tests is not None:
        tb = tests[(tests.run_id.astype(str) == rid) & (tests.input_mode == "blink")]
        if len(tb):
            blink_window = (float(tb.input_started_at_ms.min()), float(tb.selection_completed_at_ms.max()))
    if blink_window and len(bb):
        bb = bb[(bb.ts_ms >= blink_window[0]) & (bb.ts_ms <= blink_window[1])]
    elif len(bb):
        bb = bb.iloc[0:0]  # 구간 확정 불가(테스트 스킵 등) → 미귀속 이벤트는 표시하지 않음
    ses = read(R/"sessions_v1.9.csv", need=False)
    meta = {"condition_label": "", "git_commit": ""}
    if ses is not None:
        m = ses[ses.run_id.astype(str) == rid]
        if len(m):
            clean = lambda v: "" if pd.isna(v) else str(v)
            meta = {"condition_label": clean(m.iloc[-1].condition_label), "git_commit": clean(m.iloc[-1].git_commit)}
    reco, reason = recommend(S)
    cnt = bb.kind.value_counts().to_dict() if len(bb) else {}
    comp = {m: str(S[f"{m}_status"]) == "completed" for m in MODES}

    if a.detail:
        html = render_detail(S, rid, bb, cnt, comp, reco, reason, meta)
        out = f"dashboard_{rid[:8]}_detail.html"
    else:
        metrics, warnings, total_attempts = compute_public_metrics(S, tests, rid, comp)
        for w in warnings:
            print(w)
        reason_text = plain_reason(S, reco, comp)
        html = render_public(S, rid, comp, metrics, reco, reason_text)
        out = f"dashboard_{rid[:8]}.html"

    Path(out).write_text(html, encoding="utf-8")
    ok = S.recommended_input_mode == reco
    print(f"완료 → {out}  (시스템 추천={S.recommended_input_mode}, 재계산={reco}, 일치={ok})")
    if not ok: print("[경고] 저장된 추천값과 규칙 재계산 불일치 — 코드 규칙 변경 여부 확인")
    if a.png:
        png = export_png(out, scale=a.scale)
        if png: print(f"PNG 저장 → {png}")
    if a.open: webbrowser.open(Path(out).resolve().as_uri())

if __name__ == "__main__":
    main()
