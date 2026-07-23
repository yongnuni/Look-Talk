# src/analysis/ — Notion 자동 업로드 가이드

`sessions.csv` / `gaze_accuracy.csv` / `calibration_quality.csv`를 조인해 심화 진단(상관관계·압축패턴·경계고착·축이상치 등)을 계산하고,
결과를 로컬 HTML 또는 팀 Notion DB로 내보내는 자동 분석 도구 모음
팀 공용 시각화 파이프라인 (`src/viz/viz.py`, `src/viz/calib_viz.py`)과는 별개

---

## 📁 파일별 설명

| 파일 | 역할 |
|---|---|
| **`diagnostic.py`** | 계산 라이브러리. `sessions.csv`+`gaze_accuracy.csv`+`calibration_quality.csv`를 조인해 상관관계·dx/dy 압축패턴·화면경계고착·축이상치·STB요약 5종 진단(`diagnostic_battery()`)과, 세션당 1행 요약표(`build_session_summary_table()`), 시각화(`collect_figures()`), 자기완결형 HTML 리포트(`build_diagnostic_html()`)를 만든다. 후반부엔 여러 스키마 버전(v1.0~1.3) 통합 재분석용 함수들과 블록 실험(캘리브레이션 1회+9점테스트 N회) 분석 함수들도 있음. `__main__` 실행 코드 없음(라이브러리 전용). |
| **`notion_export.py`** | `diagnostic.py`가 만든 리포트를 Notion DB 페이지로 자동 업로드(`upload_diagnostic_to_notion()`). 표·토글·3단계 배경색·캘리브레이션/세션 오차 지도 이미지까지 포함. Notion Direct Upload API(3단계: 업로드 객체 생성 → 바이너리 전송 → 블록 참조) 사용. 파일 상단에 **사전 준비(통합 생성, DB 속성 스펙, 공유 설정)**와 자주 만나는 에러(400/404/401/429) 대응법이 상세히 문서화되어 있음. |
| **`run_export.py`** | 실제 실행 진입점(드라이버). CSV 로딩 → `session_ids`/`LABEL` 등 설정 → 로컬 진단 리포트 저장 → (옵션) Notion 업로드까지 한 번에 수행. 파일 상단에 `session_ids` 지정 방법 매뉴얼(전체/특정/조건별/최근N/블록실험)이 있음. |
| **`stage0_reanalysis.py`** | 1회성 탐색용 드라이버. 최신 버전만 보는 `run_export.py`와 달리 `sessions_v1.0~1.3.csv` 전체 스키마 이력을 훑어 세션 인벤토리, 비교가능성(해상도/px_per_cm) 판정, 신호 vs 매핑 판별표 등을 CSV로 저장. 그림 생성·Notion 업로드는 하지 않음. |
| **`diagnostics/`** (폴더) | 위 도구들의 JSON/HTML 산출물. **git에는 올라가지 않음** — `.gitignore`의 `*.json`/`*.png` 규칙에 걸려 로컬에만 생성됨. |

git으로 실제 추적되는 파일은 `diagnostic.py`, `notion_export.py`, `run_export.py`, `stage0_reanalysis.py` 4개뿐이며, `diagnostics/` 산출물과 `.env`는 자동으로 제외된다 (`.gitignore`: `*.json`, `*.png`, `.env`).

---

## 🔑 `.env` / API 키 안내

`.env`는 `.gitignore`에 등록되어 있어 저장소에 올라가지 않는다. 노션 업로드 기능을 쓰려면
**프로젝트 루트에 본인 `.env` 파일을 직접 만들어야** 한다:

```
NOTION_TOKEN = 노션 API 토큰
NOTION_DATABASE_ID = 팀 공용 DB ID
```

- `NOTION_TOKEN`, `NOTION_DATABASE_ID` 값은 **팀 노션 보고서에 별도로 추가**해두었으니 그걸 참고해서 각자 `.env`에 채워 넣을 것.
- 코드에 절대 하드코딩하지 말 것. (gitignore에 .env 추가 필수)

---

## ▶️ `run_export.py` 사용 가이드

### 0. 사전 준비 (최초 1회)
1. `pip install python-dotenv requests` — 프로젝트에 `requirements.txt`가 따로 없어서 이 둘은 개별 설치 필요.
2. 루트에 `.env` 생성 후 `NOTION_TOKEN`, `NOTION_DATABASE_ID` 채우기 (위 항목 참고)

### 1. 실행 전 설정 (매번 여기만 수정)
`run_export.py`의 "↓↓↓ 실행할 때마다 여기만 고치면 됨 ↓↓↓" 구간:
- `target_ids` — 분석 대상 세션 지정 (전체 / 특정 session_id prefix / 날짜범위 / calib_id / dev_version / 최근 N세션 / 블록실험 등 — 파일 상단 docstring에 예시 코드 전부 있음)
- `LABEL` — Notion 페이지 제목 겸 로컬 파일명(JSON/HTML)에 쓰이는 라벨. **JSON/HTML과 동일 label을 쓰는 습관** 권장(나중에 대응 찾기 쉬움)
- `GROUP_BY` — `"calib_id"`(기본, 서로 다른 캘리브레이션 비교) 또는 `"session_id"`(같은 캘리브레이션 재사용, 예: 3D on/off 비교)
- `GENERATE_HTML` — 기본 False. Notion 장애 시 교차검증용으로만 True
- `BLOCK_EXPERIMENT` — 캘리브레이션 1회+raw 9점테스트 N회 블록 구조 분석 여부

실행 전 `print(len(target_ids))`로 의도한 세션 수가 맞는지 먼저 확인할 것 (조건이 잘못돼도 에러 없이 빈 리스트나 엉뚱한 세션이 섞일 수 있다).

### 2. 실행
프로젝트 루트에서:
```
python -m src.analysis.run_export
```

### 3. 결과 확인
- 로컬: `src/analysis/diagnostics/{LABEL}_diagnostic_report.json` (및 `BLOCK_EXPERIMENT=True`면 `{LABEL}_block_diagnostic_report.json`)
- Notion: 실행 로그에 찍히는 업로드 완료 URL로 바로 확인

### 4. 주의사항
- **재실행 = 새 행 추가, 덮어쓰기 아님.** 같은 `LABEL`로 두 번 돌리면 Notion DB에 같은 이름 행이 중복 생성된다. 계산 로직 수정 후 재업로드할 땐 이전 행을 Notion에서 직접 지우거나 보관 처리할 것.
- 자주 만나는 에러: `400`(속성 이름/타입 불일치) / `404`(database_id 오류 또는 통합 미공유) / `401`(`NOTION_TOKEN` 미설정·만료) / `429`(이미지 많을 때 rate limit — 잠시 후 재시도, 재시도 로직 없음).
- 그림 없이 텍스트·표만 빠르게 확인하고 싶으면 `include_images=False` 옵션도 있음(기본 True).
